import json
import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.inference.engine import InferenceEngine
from src.utils.key_manager import KeyManager


def create_app(
    engine: InferenceEngine,
    api_key: str | None = None,
    rate_limit: int = 60,
) -> FastAPI:
    if api_key is None:
        api_key = os.environ.get("NEXUS_API_KEY")

    app = FastAPI(title="Nexus API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    key_manager = KeyManager()

    # --- Auth ---
    async def check_auth(request: Request):
        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        query_key = request.query_params.get("api_key", "")

        provided = ""
        if auth_header.startswith("Bearer "):
            provided = auth_header[7:]
        elif api_key_header:
            provided = api_key_header
        elif query_key:
            provided = query_key

        # If a master/environment api_key is configured, it acts as a valid key.
        if api_key and provided == api_key:
            return

        # Otherwise, check the SQLite database
        if provided and key_manager.validate_key(provided):
            return

        raise HTTPException(401, "unauthorized")

    # --- Rate limiter ---
    rate_store: dict[str, list[float]] = defaultdict(list)

    async def check_rate_limit(request: Request):
        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        query_key = request.query_params.get("api_key", "")

        key = "anonymous"
        if auth_header.startswith("Bearer "):
            key = auth_header[7:]
        elif api_key_header:
            key = api_key_header
        elif query_key:
            key = query_key

        now = time.time()
        window = 60.0
        timestamps = rate_store[key]
        rate_store[key] = [t for t in timestamps if now - t < window]
        if len(rate_store[key]) >= rate_limit:
            raise HTTPException(429, "rate_limit_exceeded")
        rate_store[key].append(now)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path not in ("/health", "/models", "/v1/keys/generate", "/v1/keys"):
            try:
                await check_auth(request)
                await check_rate_limit(request)
            except HTTPException as e:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        return await call_next(request)

    # --- Models ---
    class GenerateKeyRequest(BaseModel):
        name: str

    class RevokeKeyRequest(BaseModel):
        key: str

    class ChatRequest(BaseModel):
        message: str
        max_tokens: int = 512
        temperature: float = 0.7
        thinking: bool = False

    class GenerateRequest(BaseModel):
        prompt: str

        max_tokens: int = 512
        temperature: float = 0.7

    def _usage(prompt_text: str, response_text: str) -> dict:
        return {
            "prompt_tokens": len(engine.tokenizer.encode(prompt_text)),
            "completion_tokens": len(engine.tokenizer.encode(response_text)),
            "total_tokens": len(engine.tokenizer.encode(prompt_text + response_text)),
        }

    # --- Endpoints ---

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": "nexus",
            "device": str(engine.device),
        }

    @app.get("/models")
    async def list_models():
        return {"models": ["nexus-3b"]}

    @app.post("/v1/keys/generate")
    async def generate_key(req: GenerateKeyRequest, request: Request):
        admin_key = os.environ.get("NEXUS_ADMIN_KEY") or api_key
        if admin_key:
            auth_header = request.headers.get("Authorization", "")
            provided = auth_header[7:] if auth_header.startswith("Bearer ") else request.headers.get("X-API-Key", "")
            if provided != admin_key:
                raise HTTPException(401, "unauthorized to generate keys")
        new_key = key_manager.generate_key(req.name)
        return {"key": new_key, "name": req.name}

    @app.get("/v1/keys")
    async def list_keys(request: Request):
        admin_key = os.environ.get("NEXUS_ADMIN_KEY") or api_key
        if admin_key:
            auth_header = request.headers.get("Authorization", "")
            provided = auth_header[7:] if auth_header.startswith("Bearer ") else request.headers.get("X-API-Key", "")
            if provided != admin_key:
                raise HTTPException(401, "unauthorized to list keys")
        return key_manager.list_keys()

    @app.post("/v1/keys/revoke")
    async def revoke_key(req: RevokeKeyRequest, request: Request):
        await check_auth(request)
        success = key_manager.revoke_key(req.key)
        if not success:
            raise HTTPException(404, "key not found or already revoked")
        return {"status": "success", "message": "Key revoked successfully"}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        output = engine.generate(
            req.message,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            thinking_mode=req.thinking,
        )
        thinking = None
        response = output
        if req.thinking and isinstance(output, tuple):
            response, thinking = output

        return {
            "response": response,
            "thinking": thinking,
            "usage": _usage(req.message, response),
        }

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        return StreamingResponse(
            _stream_chat(engine, req),
            media_type="text/event-stream",
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        output = engine.generate(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return {
            "text": output,
            "usage": _usage(req.prompt, output),
        }

    async def _stream_chat(engine: InferenceEngine, req):
        for token_str in engine.generate(
            req.message,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
            thinking_mode=req.thinking,
        ):
            yield f"data: {json.dumps({'token': token_str})}\n\n"
        yield "data: [DONE]\n\n"

    return app


def main():
    import uvicorn

    from src.inference.cli import load_model

    engine, _ = load_model("checkpoints/model.pt", "tokenizer/tokenizer.model", None, "cuda")
    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

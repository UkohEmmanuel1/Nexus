import json
import os
import secrets
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.crawl_scheduler import CrawlScheduler
from src.agents.web_crawler import CrawlTask, WebCrawler

if TYPE_CHECKING:
    from src.inference.engine import InferenceEngine
from src.rag.web_index import WebIndex
from src.utils.key_manager import KeyManager

crawler = WebCrawler()
scheduler = CrawlScheduler(crawler)
web_index = WebIndex(os.environ.get("NEXUS_WEB_INDEX_DB", "nexus_web_index.db"))


def create_app(
    engine: "InferenceEngine",
    api_key: str | None = None,
    rate_limit: int = 60,
) -> FastAPI:
    if api_key is None:
        api_key = os.environ.get("NEXUS_API_KEY")

    admin_key = os.environ.get("NEXUS_ADMIN_KEY") or api_key
    if not admin_key:
        raise ValueError("Set NEXUS_ADMIN_KEY or NEXUS_API_KEY before starting the server")
    if rate_limit < 1:
        raise ValueError("rate_limit must be positive")
    app = FastAPI(
        title="Nexus API", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            s.strip()
            for s in os.environ.get("NEXUS_CORS_ORIGINS", "http://localhost:3000").split(",")
            if s.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )

    key_manager = KeyManager()
    engine_lock = threading.Lock()

    @contextmanager
    def generation_slot():
        if not engine_lock.acquire(blocking=False):
            raise HTTPException(503, "model busy; retry shortly")
        try:
            yield
        finally:
            engine_lock.release()

    def provided_key(request: Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return request.headers.get("X-API-Key", "")

    async def check_admin(request: Request):
        if not secrets.compare_digest(provided_key(request).encode(), admin_key.encode()):
            raise HTTPException(401, "admin authorization required")

    async def check_auth(request: Request):
        provided = provided_key(request)
        if api_key and secrets.compare_digest(provided.encode(), api_key.encode()):
            return
        if provided and key_manager.validate_key(provided):
            return
        raise HTTPException(401, "unauthorized")

    rate_store: dict[str, list[float]] = defaultdict(list)

    async def check_rate_limit(request: Request):
        key = KeyManager._digest(provided_key(request))
        now = time.monotonic()
        window = 60.0
        for stale in list(rate_store):
            if not rate_store[stale] or now - rate_store[stale][-1] >= window:
                del rate_store[stale]
        if key not in rate_store and len(rate_store) >= 10000:
            raise HTTPException(429, "rate_limit_exceeded")
        timestamps = rate_store[key]
        rate_store[key] = [t for t in timestamps if now - t < window]
        if len(rate_store[key]) >= rate_limit:
            raise HTTPException(429, "rate_limit_exceeded")
        rate_store[key].append(now)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        try:
            path = request.url.path
            is_api = path.startswith(("/chat", "/generate", "/crawl", "/v1/"))
            if is_api and request.method != "OPTIONS":
                if path.startswith(("/v1/keys", "/crawl")):
                    await check_admin(request)
                else:
                    await check_auth(request)
                await check_rate_limit(request)
                if path.startswith("/crawl") and os.environ.get("NEXUS_ENABLE_CRAWLER") != "1":
                    raise HTTPException(403, "crawler disabled")
                # Bound actual bytes read, including chunked requests.
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise HTTPException(413, "request too large")
                request._body = bytes(body)
            response = await call_next(request)
        except HTTPException as e:
            response = JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; "
            "connect-src 'self' https: http://localhost:8000; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if is_api:
            response.headers["Cache-Control"] = "no-store"
        if os.environ.get("RENDER") or request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    class GenerateKeyRequest(BaseModel):
        name: str = Field(min_length=1, max_length=100)

    class RevokeKeyRequest(BaseModel):
        key: str = Field(min_length=1, max_length=256)

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1, max_length=16000)
        max_tokens: int = Field(default=512, ge=1, le=2048)
        temperature: float = Field(default=0.7, gt=0, le=2)
        thinking: bool = False

    class GenerateRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=16000)
        max_tokens: int = Field(default=512, ge=1, le=2048)
        temperature: float = Field(default=0.7, gt=0, le=2)

    class CrawlRequest(BaseModel):
        url: str | None = None
        query: str | None = None
        search_query: str | None = None
        max_depth: int = Field(default=1, ge=0, le=2)
        batch_urls: list[str] | None = Field(default=None, max_length=10)
        schedule: bool = False
        interval_seconds: int = Field(default=300, ge=60, le=86400)

    class ScheduleRequest(BaseModel):
        seed_urls: list[str] = Field(min_length=1, max_length=10)
        interval_seconds: int = Field(default=300, ge=60, le=86400)
        max_depth: int = Field(default=1, ge=0, le=2)
        task_name: str | None = None

    def _usage(prompt_text: str, response_text: str) -> dict:
        return {
            "prompt_tokens": len(engine.tokenizer.encode(prompt_text)),
            "completion_tokens": len(engine.tokenizer.encode(response_text)),
            "total_tokens": len(engine.tokenizer.encode(prompt_text + response_text)),
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": "nexus",
        }

    @app.get("/models")
    async def list_models():
        return {"models": ["nexus-3b"]}

    @app.post("/v1/keys/generate")
    async def generate_key(req: GenerateKeyRequest, request: Request):
        await check_admin(request)
        new_key = key_manager.generate_key(req.name)
        return {"key": new_key, "name": req.name}

    @app.get("/v1/keys")
    async def list_keys(request: Request):
        await check_admin(request)
        return key_manager.list_keys()

    @app.post("/v1/keys/revoke")
    async def revoke_key(req: RevokeKeyRequest, request: Request):
        await check_admin(request)
        success = (
            key_manager.revoke_by_id(req.key)
            if req.key.startswith("sha256:")
            else key_manager.revoke_key(req.key)
        )
        if not success:
            raise HTTPException(404, "key not found or already revoked")
        return {"status": "success", "message": "Key revoked successfully"}

    @app.post("/chat")
    def chat(req: ChatRequest):
        with generation_slot():
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
        return {"response": response, "thinking": thinking, "usage": _usage(req.message, response)}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        return StreamingResponse(_stream_chat(engine, req), media_type="text/event-stream")

    @app.post("/generate")
    def generate(req: GenerateRequest):
        with generation_slot():
            output = engine.generate(
                req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
            )
        return {"text": output, "usage": _usage(req.prompt, output)}

    def _stream_chat(engine, req):
        with generation_slot():
            for token_str in engine.generate(
                req.message,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=True,
                thinking_mode=req.thinking,
            ):
                yield f"data: {json.dumps({'token': token_str})}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/crawl")
    async def crawl(req: CrawlRequest):
        try:
            if req.batch_urls:
                results = await crawler.crawl_batch(req.batch_urls, max_depth=req.max_depth)
            elif req.search_query:
                results = await crawler.crawl_search_results(
                    req.search_query, num_results=5, max_depth=req.max_depth
                )
            elif req.url:
                task = CrawlTask(url=req.url, depth=0, max_depth=req.max_depth)
                results = [await crawler.crawl(task)]
            else:
                raise HTTPException(400, "Provide url, batch_urls, or search_query")

            if req.schedule:
                scheduler.schedule_crawl(
                    seed_urls=[r.url for r in results if r.status_code == 200],
                    interval_seconds=req.interval_seconds,
                    max_depth=req.max_depth,
                )

            web_index.index_results(results)
            return {
                "status": "success",
                "results": [
                    {"url": r.url, "title": r.title, "status_code": r.status_code, "error": r.error}
                    for r in results
                ],
            }
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(500, "Crawl failed") from None

    @app.post("/crawl/schedule")
    async def schedule_crawl(req: ScheduleRequest):
        task_id = scheduler.schedule_crawl(
            seed_urls=req.seed_urls,
            interval_seconds=req.interval_seconds,
            max_depth=req.max_depth,
            task_name=req.task_name,
        )
        return {"status": "scheduled", "task_id": task_id, "seed_urls": req.seed_urls}

    @app.get("/crawl/status")
    async def crawl_status():
        return {"scheduler": scheduler.get_status(), "stats": web_index.get_stats()}

    @app.get("/crawl/data")
    async def crawl_data(query: str | None = None, limit: int = 10):
        if query:
            results = web_index.search(query, limit=limit)
        else:
            urls = web_index.get_all_urls()
            results = [{"url": u} for u in urls]
        return {"data": results}

    @app.post("/crawl/stream")
    async def crawl_stream(req: CrawlRequest):
        async def _stream():
            try:
                if req.search_query:
                    results = await crawler.crawl_search_results(
                        req.search_query, num_results=5, max_depth=req.max_depth
                    )
                elif req.url:
                    task = CrawlTask(url=req.url, depth=0, max_depth=req.max_depth)
                    results = [await crawler.crawl(task)]
                else:
                    yield f"data: {json.dumps({'error': 'provide url or search_query'})}\n\n"
                    return
                for r in results:
                    yield f"data: {json.dumps({'url': r.url, 'title': r.title, 'status_code': r.status_code})}\n\n"  # noqa: E501
                web_index.index_results(results)
                yield "data: [DONE]\n\n"
            except Exception:
                yield f"data: {json.dumps({'error': 'Crawl failed'})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    static_dir = os.environ.get("NEXUS_STATIC_DIR")
    if static_dir:
        if not Path(static_dir, "index.html").is_file():
            raise ValueError("NEXUS_STATIC_DIR must contain the built frontend")
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


def main():
    import uvicorn

    if not (os.environ.get("NEXUS_ADMIN_KEY") or os.environ.get("NEXUS_API_KEY")):
        raise ValueError("Set NEXUS_ADMIN_KEY or NEXUS_API_KEY before starting the server")
    if os.environ.get("RENDER"):
        for name in ("NEXUS_ADMIN_KEY", "NEXUS_API_KEY"):
            if len(os.environ.get(name, "")) < 32:
                raise ValueError(f"{name} must be at least 32 characters on Render")
    if os.environ.get("NEXUS_MODEL_BACKEND") == "gguf":
        from src.inference.gguf import GGUFEngine

        engine = GGUFEngine.from_environment()
    else:
        from src.inference.cli import load_model

        engine, _ = load_model(
            os.environ.get("MODEL_PATH", "checkpoints/model.pt"),
            os.environ.get("TOKENIZER_PATH", "tokenizer/tokenizer.model"),
            os.environ.get("MODEL_CONFIG"),
            os.environ.get("DEVICE", "cpu"),
        )
    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), proxy_headers=False)


if __name__ == "__main__":
    main()

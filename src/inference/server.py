import json
import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.crawl_scheduler import CrawlScheduler
from src.agents.web_crawler import CrawlTask, WebCrawler
from src.inference.engine import InferenceEngine
from src.rag.web_index import WebIndex
from src.utils.key_manager import KeyManager

crawler = WebCrawler()
scheduler = CrawlScheduler(crawler)
web_index = WebIndex()


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
        if api_key and provided == api_key:
            return
        if provided and key_manager.validate_key(provided):
            return
        raise HTTPException(401, "unauthorized")

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
        if request.url.path not in (
            "/health",
            "/models",
            "/v1/keys/generate",
            "/v1/keys",
            "/crawl/status",
        ):
            try:
                await check_auth(request)
                await check_rate_limit(request)
            except HTTPException as e:
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        return await call_next(request)

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

    class CrawlRequest(BaseModel):
        url: str | None = None
        query: str | None = None
        search_query: str | None = None
        max_depth: int = 1
        batch_urls: list[str] | None = None
        schedule: bool = False
        interval_seconds: int = 300

    class ScheduleRequest(BaseModel):
        seed_urls: list[str]
        interval_seconds: int = 300
        max_depth: int = 1
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
            "device": str(engine.device),
            "web_crawler": "active",
            "crawl_scheduler": "active",
        }

    @app.get("/models")
    async def list_models():
        return {"models": ["nexus-3b"]}

    @app.post("/v1/keys/generate")
    async def generate_key(req: GenerateKeyRequest, request: Request):
        admin_key = os.environ.get("NEXUS_ADMIN_KEY") or api_key
        if admin_key:
            auth_header = request.headers.get("Authorization", "")
            provided = (
                auth_header[7:]
                if auth_header.startswith("Bearer ")
                else request.headers.get("X-API-Key", "")
            )
            if provided != admin_key:
                raise HTTPException(401, "unauthorized to generate keys")
        new_key = key_manager.generate_key(req.name)
        return {"key": new_key, "name": req.name}

    @app.get("/v1/keys")
    async def list_keys(request: Request):
        admin_key = os.environ.get("NEXUS_ADMIN_KEY") or api_key
        if admin_key:
            auth_header = request.headers.get("Authorization", "")
            provided = (
                auth_header[7:]
                if auth_header.startswith("Bearer ")
                else request.headers.get("X-API-Key", "")
            )
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
        return {"response": response, "thinking": thinking, "usage": _usage(req.message, response)}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        return StreamingResponse(_stream_chat(engine, req), media_type="text/event-stream")

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        output = engine.generate(
            req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
        )
        return {"text": output, "usage": _usage(req.prompt, output)}

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
        except Exception as e:
            raise HTTPException(500, f"Crawl failed: {str(e)}")

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
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app


def main():
    import uvicorn

    from src.inference.cli import load_model

    engine, _ = load_model("checkpoints/model.pt", "tokenizer/tokenizer.model", None, "cuda")
    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

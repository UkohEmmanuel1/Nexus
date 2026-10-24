import asyncio
import json
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import TYPE_CHECKING

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from src.agents.crawl_scheduler import CrawlScheduler
from src.agents.web_crawler import CrawlTask, WebCrawler
from src.inference.security import WindowLimiter
from src.inference.providers import PROVIDERS, EXAMPLE_MODELS, complete_byok

if TYPE_CHECKING:
    from src.inference.engine import InferenceEngine
from src.rag.web_index import WebIndex
from src.utils.key_manager import KeyManager

logger = logging.getLogger(__name__)


def create_app(
    engine: "InferenceEngine",
    api_key: str | None = None,
    rate_limit: int | None = None,
) -> FastAPI:
    if api_key is None:
        api_key = os.environ.get("NEXUS_API_KEY")

    admin_key = os.environ.get("NEXUS_ADMIN_KEY") or api_key
    if not admin_key:
        raise ValueError("Set NEXUS_ADMIN_KEY or NEXUS_API_KEY before starting the server")
    if rate_limit is None:
        rate_limit = int(os.environ.get("NEXUS_RATE_LIMIT", "60"))
    if rate_limit < 1:
        raise ValueError("rate_limit must be positive")
    crawler_enabled = os.environ.get("NEXUS_ENABLE_CRAWLER") == "1"
    crawler = WebCrawler()
    scheduler = CrawlScheduler(crawler)
    web_index = (
        WebIndex(os.environ.get("NEXUS_WEB_INDEX_DB", "nexus_web_index.db"))
        if crawler_enabled
        else None
    )

    @asynccontextmanager
    async def lifespan(app):
        worker = None
        if crawler_enabled:
            scheduler.set_on_crawl_callback(web_index.index_results)
            worker = asyncio.create_task(scheduler.start())
        try:
            yield
        finally:
            await scheduler.stop()
            if worker is not None:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
            await crawler.close()

    app = FastAPI(
        title="Vonk API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            s.strip() for s in os.environ.get("NEXUS_CORS_ORIGINS", "").split(",") if s.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "X-Provider-Key", "Content-Type"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    key_manager = KeyManager()
    engine_lock = threading.Lock()
    model_id = getattr(engine, "model_id", "vonk")
    if not isinstance(model_id, str):
        model_id = "vonk"

    @contextmanager
    def generation_slot():
        if not engine_lock.acquire(blocking=False):
            raise HTTPException(503, "model busy; retry shortly", headers={"Retry-After": "5"})
        try:
            yield
        finally:
            engine_lock.release()

    def provided_key(request: Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided = auth_header[7:]
        else:
            provided = request.headers.get("X-API-Key", "")
        if len(provided) > 256:
            raise HTTPException(401, "unauthorized")
        return provided

    async def check_admin(request: Request):
        if not secrets.compare_digest(provided_key(request).encode(), admin_key.encode()):
            raise HTTPException(401, "admin authorization required")

    async def check_auth(request: Request):
        provided = provided_key(request)
        if api_key and secrets.compare_digest(provided.encode(), api_key.encode()):
            return
        if provided and await anyio.to_thread.run_sync(key_manager.validate_key, provided):
            return
        raise HTTPException(401, "unauthorized")

    credential_limiter = WindowLimiter(rate_limit)
    authentication_limiter = WindowLimiter(300)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Pydantic's default response includes the submitted input, including keys.
        errors = [
            {"loc": error["loc"], "type": error["type"], "msg": "Invalid value"}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        is_api = False
        try:
            path = request.url.path
            is_api = path.startswith(("/chat", "/generate", "/crawl", "/v1/"))
            if is_api and request.method != "OPTIONS":
                peer = request.client.host if request.client else "unknown"
                # Count failures only: valid users behind a shared proxy must
                # not consume a single global authentication-attempt budget.
                try:
                    if path.startswith(("/v1/keys", "/crawl")):
                        await check_admin(request)
                    else:
                        await check_auth(request)
                except HTTPException:
                    authentication_limiter.check(peer)
                    raise
                credential_limiter.check(KeyManager._digest(provided_key(request)))
                if path.startswith("/crawl") and not crawler_enabled:
                    raise HTTPException(403, "crawler disabled")
                # Bound actual bytes read, including chunked requests.
                body = bytearray()
                try:
                    with anyio.fail_after(10):
                        async for chunk in request.stream():
                            if len(body) + len(chunk) > 65536:
                                raise HTTPException(413, "request too large")
                            body.extend(chunk)
                except TimeoutError:
                    raise HTTPException(408, "request body timed out") from None
                request._body = bytes(body)
            response = await call_next(request)
        except HTTPException as e:
            response = JSONResponse(
                status_code=e.status_code, content={"detail": e.detail}, headers=e.headers
            )
        except Exception as exc:
            logger.error("request_id=%s failure=%s", request_id, type(exc).__name__)
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        if is_api:
            response.headers["Cache-Control"] = "no-store"
        if os.environ.get("RENDER") or request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        # Do not log request bodies, credentials, or query strings.
        logger.info(
            "request_id=%s method=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            response.status_code,
            (time.monotonic() - started) * 1000,
        )
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

    class ProviderMessage(BaseModel):
        role: str = Field(pattern="^(system|user|assistant)$")
        content: str = Field(min_length=1, max_length=16000)

    class ProviderChatRequest(BaseModel):
        provider: str = Field(min_length=1, max_length=40)
        model: str = Field(min_length=1, max_length=200)
        messages: list[ProviderMessage] = Field(min_length=1, max_length=30)
        max_tokens: int = Field(default=512, ge=1, le=2048)
        temperature: float = Field(default=0.7, ge=0, le=2)

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
        prompt_tokens = len(engine.tokenizer.encode(prompt_text))
        completion_tokens = len(engine.tokenizer.encode(response_text))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @app.get("/", include_in_schema=False)
    async def service_info():
        return {
            "service": "Vonk API",
            "version": app.version,
            "health": "/health",
            "readiness": "/ready",
            "openapi": "/openapi.json",
            "chat": "/chat",
            "stream": "/chat/stream",
            "authentication": "Bearer API key; request a personal key from the operator",
        }

    @app.get("/ready", include_in_schema=False)
    def readiness():
        try:
            with key_manager._connect() as connection:
                connection.execute("SELECT 1 FROM api_keys LIMIT 1").fetchone()
        except Exception:
            raise HTTPException(503, "credential storage unavailable") from None
        return {"status": "ready", "model": model_id}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": model_id,
        }

    @app.get("/models")
    async def list_models():
        return {"models": [model_id]}

    @app.post("/v1/keys/generate")
    def generate_key(req: GenerateKeyRequest):
        new_key = key_manager.generate_key(req.name)
        return {"key": new_key, "name": req.name}

    @app.get("/v1/keys")
    def list_keys():
        return key_manager.list_keys()

    @app.post("/v1/keys/revoke")
    def revoke_key(req: RevokeKeyRequest):
        success = (
            key_manager.revoke_by_id(req.key)
            if req.key.startswith("sha256:")
            else key_manager.revoke_key(req.key)
        )
        if not success:
            raise HTTPException(404, "key not found or already revoked")
        return {"status": "success", "message": "Key revoked successfully"}

    @app.get("/v1/providers")
    async def list_providers():
        return {"providers": [
            {"id": key, "example_models": EXAMPLE_MODELS.get(key, [])}
            for key in PROVIDERS
        ], "note": "Examples only; model availability depends on provider account."}

    @app.post("/v1/chat/byok")
    async def provider_chat(req: ProviderChatRequest, request: Request):
        # User-supplied API key is forwarded for this request only.
        # Do not persist it or include it in request bodies or logs.
        provider_key = request.headers.get("X-Provider-Key", "")
        if not provider_key:
            raise HTTPException(400, "Missing X-Provider-Key header")
        return await complete_byok(
            provider=req.provider,
            model=req.model,
            messages=[message.model_dump() for message in req.messages],
            provider_key=provider_key,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

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
            usage = _usage(req.message, response)
        return {"response": response, "thinking": thinking, "usage": usage}

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        if not engine_lock.acquire(blocking=False):
            raise HTTPException(503, "model busy; retry shortly", headers={"Retry-After": "5"})
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                engine_lock.release()

        try:
            tokens = iter(
                engine.generate(
                    req.message,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=True,
                    thinking_mode=req.thinking,
                )
            )
            first = next(tokens, None)
        except Exception:
            release()
            raise

        def stream():
            try:
                if first is not None:
                    yield f"data: {json.dumps({'token': first})}\n\n"
                for token in tokens:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                yield f"data: {json.dumps({'error': 'Generation failed'})}\n\n"
            finally:
                try:
                    close = getattr(tokens, "close", None)
                    if close is not None:
                        close()
                finally:
                    release()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            background=BackgroundTask(release),
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post("/generate")
    def generate(req: GenerateRequest):
        with generation_slot():
            output = engine.generate(
                req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
            )
            usage = _usage(req.prompt, output)
        return {"text": output, "usage": usage}

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
    async def crawl_data(
        query: str | None = Query(default=None, max_length=1000),
        limit: int = Query(default=10, ge=1, le=100),
    ):
        if query:
            results = web_index.search(query, limit=limit)
        else:
            urls = web_index.get_all_urls(limit=limit)
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

    @app.get("/openapi.json", include_in_schema=False)
    async def api_schema():
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKey"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "Personal API key. Key management and crawling require the admin key.",
        }
        for path, operations in schema["paths"].items():
            if path.startswith(("/chat", "/generate", "/crawl", "/v1/")):
                for operation in operations.values():
                    operation["security"] = [{"ApiKey": []}]
        return schema

    return app


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=False,
        server_header=False,
        access_log=False,
        limit_concurrency=100,
        backlog=128,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=120,
    )


if __name__ == "__main__":
    main()

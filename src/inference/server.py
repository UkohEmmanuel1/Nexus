import json
import time
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.model import Transformer, ModelConfig
from src.tokenizer import Tokenizer
from src.inference.engine import InferenceEngine

app = FastAPI(title="Nexus API", version="0.1.0")
engine: Optional[InferenceEngine] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[Dict]
    usage: Dict


@app.on_event("startup")
async def startup():
    global engine
    config = ModelConfig()
    model = Transformer(config)
    tokenizer = Tokenizer("tokenizer/tokenizer.model")
    engine = InferenceEngine(model, tokenizer)


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "nexus-3b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if engine is None:
        raise HTTPException(500, "Model not loaded")

    prompt = engine._format_chat([m.model_dump() for m in request.messages])

    if request.stream:
        return StreamingResponse(
            _stream_chat(prompt, request),
            media_type="text/event-stream",
        )

    output = engine.generate(
        prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": output},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": len(engine.tokenizer.encode(prompt)),
            "completion_tokens": len(engine.tokenizer.encode(output)),
            "total_tokens": len(engine.tokenizer.encode(prompt + output)),
        },
    )


async def _stream_chat(prompt: str, request: ChatRequest):
    for token_str in engine.generate(
        prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=True,
    ):
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "nexus-3b",
            "choices": [{"delta": {"content": token_str}, "index": 0}],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(prompt: str, request: ChatRequest):
    if engine is None:
        raise HTTPException(500, "Model not loaded")

    output = engine.generate(
        prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "choices": [{"text": output, "index": 0, "finish_reason": "stop"}],
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

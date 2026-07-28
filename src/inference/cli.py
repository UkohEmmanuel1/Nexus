import argparse
import os

import torch

from src.inference.engine import InferenceEngine
from src.model import ModelConfig, Transformer
from src.tokenizer import Tokenizer


def load_model(
    model_path: str,
    tokenizer_path: str,
    config_path: str | None,
    device: str,
) -> tuple[InferenceEngine, ModelConfig]:
    print("Loading model...")
    config = ModelConfig()
    if config_path:
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            for k, v in cfg.get("model", {}).items():
                if hasattr(config, k):
                    setattr(config, k, v)

    model = Transformer(config)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    tokenizer = Tokenizer(tokenizer_path)
    engine = InferenceEngine(model, tokenizer, device=device)

    print(f"Model loaded. {sum(p.numel() for p in model.parameters()):,} parameters")
    return engine, config


def run_chat(engine: InferenceEngine, args: argparse.Namespace) -> None:
    print("Chat initialized. Type 'exit' to quit, '/clear' to clear history.\n")
    history = []
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "/clear":
            history = []
            print("History cleared.\n")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        prompt = engine._format_chat(history)

        print("AI: ", end="", flush=True)
        response = ""
        for token_str in engine.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=True,
            thinking_mode=args.thinking,
            thinking_budget=args.thinking_budget,
        ):
            print(token_str, end="", flush=True)
            response += token_str
        print("\n")

        history.append({"role": "assistant", "content": response})


def run_server(engine: InferenceEngine, args: argparse.Namespace) -> None:
    import uvicorn

    from src.inference.server import create_app

    app = create_app(engine, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="Nexus CLI — Chat or serve your model")
    parser.add_argument(
        "--model", type=str, default="checkpoints/model.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizer/tokenizer.model", help="Path to tokenizer model"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Server mode
    parser.add_argument(
        "--serve", action="store_true", help="Start API server instead of interactive chat"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8000, help="Server bind port")
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for authentication (env: NEXUS_API_KEY)"
    )

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")

    # Thinking mode
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--thinking-budget", type=int, default=2048, help="Thinking token budget")

    args = parser.parse_args()

    if args.api_key is None:
        args.api_key = os.environ.get("NEXUS_API_KEY")

    engine, _ = load_model(args.model, args.tokenizer, args.config, args.device)

    if args.serve:
        run_server(engine, args)
    else:
        run_chat(engine, args)


if __name__ == "__main__":
    main()

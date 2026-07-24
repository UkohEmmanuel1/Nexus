import argparse
try:
    import readline
except ImportError:
    pass
from pathlib import Path

import torch

from src.model import Transformer, ModelConfig
from src.tokenizer import Tokenizer
from src.inference.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Nexus CLI Chat")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer model")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    print("Loading model...")
    config = ModelConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            for k, v in cfg.get("model", {}).items():
                if hasattr(config, k):
                    setattr(config, k, v)

    model = Transformer(config)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    tokenizer = Tokenizer(args.tokenizer)
    engine = InferenceEngine(model, tokenizer, device=args.device)

    print(f"Model loaded. {sum(p.numel() for p in model.parameters()):,} parameters")
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
        for token_str in engine.generate(prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, stream=True):
            print(token_str, end="", flush=True)
            response += token_str
        print("\n")

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

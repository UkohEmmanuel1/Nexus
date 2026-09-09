import struct
import json
import sys
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFReader

from src.model import ModelConfig, Transformer
from src.model.config import MoEConfig

GGUF_PATH = "checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf"
OUTPUT_PATH = "checkpoints/model.pt"

def load_gguf_tensors(path: str) -> dict:
    reader = GGUFReader(path)
    tensors = {}
    for tensor_info in reader.tensors:
        name = tensor_info.name
        data = np.frombuffer(tensor_info.data, dtype=tensor_info.dtype)
        shape = list(tensor_info.shape)
        tensor = torch.from_numpy(data.reshape(shape)).clone()
        if tensor_info.dtype in [np.float16, np.float32, np.float64]:
            tensor = tensor.to(torch.float32)
        elif tensor_info.dtype == np.int8:
            tensor = tensor.to(torch.int8)
        elif tensor_info.dtype == np.int4:
            tensor = tensor.to(torch.int4)
        tensors[name] = tensor
    reader.close()
    return tensors

def build_nexus_config() -> ModelConfig:
    return ModelConfig(
        dim=1536,
        n_layers=24,
        n_heads=12,
        n_kv_heads=2,
        ffn_mult=4912 / 1536,
        vocab_size=151936,
        max_seq_len=4096,
        norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        moe=MoEConfig(enabled=False),
        use_flash_attn=False,
        dtype="float32",
    )

def create_weight_mapping(tensors: dict) -> dict:
    weight_map = {}

    qwen_prefix = "model."
    nexus_prefix = ""

    embed_map = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "model.norm.weight": "norm.weight",
    }

    for qwen_name, nexus_name in embed_map.items():
        if qwen_name in tensors:
            weight_map[qwen_name] = nexus_name

    for i in range(24):
        layer_prefix = f"model.layers.{i}."
        nexus_layer_prefix = f"layers.{i}."

        layer_map = {
            f"{layer_prefix}self_attn.q_proj.weight": f"{nexus_layer_prefix}attention.wq.weight",
            f"{layer_prefix}self_attn.k_proj.weight": f"{nexus_layer_prefix}attention.wk.weight",
            f"{layer_prefix}self_attn.v_proj.weight": f"{nexus_layer_prefix}attention.wv.weight",
            f"{layer_prefix}self_attn.o_proj.weight": f"{nexus_layer_prefix}attention.wo.weight",
            f"{layer_prefix}mlp.gate_proj.weight": f"{nexus_layer_prefix}feed_forward.swiglu.w1.weight",
            f"{layer_prefix}mlp.up_proj.weight": f"{nexus_layer_prefix}feed_forward.swiglu.w3.weight",
            f"{layer_prefix}mlp.down_proj.weight": f"{nexus_layer_prefix}feed_forward.swiglu.w2.weight",
            f"{layer_prefix}input_layernorm.weight": f"{nexus_layer_prefix}attention_norm.weight",
            f"{layer_prefix}post_attention_layernorm.weight": f"{nexus_layer_prefix}ffn_norm.weight",
        }
        weight_map.update(layer_map)

    lm_head_name = None
    for name in tensors:
        if "output" in name.lower() or "lm_head" in name or "output_projs" in name:
            lm_head_name = name
            break

    if lm_head_name:
        weight_map[lm_head_name] = "output.weight"
    else:
        weight_map["model.output_projs.weight"] = "output.weight"

    return weight_map

def convert(tensors: dict, weight_map: dict, config: ModelConfig) -> dict:
    model = Transformer(config)
    state_dict = model.state_dict()

    missing = []
    mismatched = []

    for qwen_name, nexus_name in weight_map.items():
        if qwen_name not in tensors:
            missing.append(qwen_name)
            continue

        if nexus_name not in state_dict:
            missing.append(f"nexus:{nexus_name}")
            continue

        src = tensors[qwen_name]
        dst = state_dict[nexus_name]

        if src.shape != dst.shape:
            mismatched.append((qwen_name, src.shape, dst.shape))
            continue

        state_dict[nexus_name] = src.to(dtype=dst.dtype, device=dst.device)

    model.load_state_dict(state_dict, strict=False)

    output = {"model_state_dict": state_dict, "config": config}

    print(f"Converted weights: {len(weight_map) - len(missing)}/{len(weight_map)}")
    if missing:
        print(f"Missing tensors: {missing[:20]}")
    if mismatched:
        print(f"Mismatched shapes: {mismatched[:10]}")

    return output

def main():
    print(f"Loading GGUF: {GGUF_PATH}")
    tensors = load_gguf_tensors(GGUF_PATH)
    print(f"Loaded {len(tensors)} tensors from GGUF")

    print("Building Nexus config...")
    config = build_nexus_config()
    print(f"Config: dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")

    print("Creating weight mapping...")
    weight_map = create_weight_mapping(tensors)
    print(f"Mapping {len(weight_map)} weights")

    print("Converting weights...")
    output = convert(tensors, weight_map, config)

    print(f"Saving to {OUTPUT_PATH}...")
    torch.save(output, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")

    from src.inference.cli import load_model
    engine, loaded_config = load_model(OUTPUT_PATH, "tokenizer/tokenizer.model", None, "cpu")
    print(f"Model loaded successfully: {sum(p.numel() for p in engine.model.parameters()):,} params")

if __name__ == "__main__":
    main()

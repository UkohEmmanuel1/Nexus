from llama_cpp import Llama
try:
    llm = Llama(
        model_path="checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_ctx=1,
        n_gpu_layers=0,
        vocab_only=False,
    )
    print("Model loaded successfully")
    print(f"n_vocab: {llm.model.n_vocab}")
    print(f"n_embd: {llm.model.n_embd}")
    print(f"n_layer: {llm.model.n_layer}")
    print(f"n_head: {llm.model.n_head}")
    print(f"n_embd_head: {llm.model.n_embd_head}")
except Exception as e:
    print(f"Error: {e}")

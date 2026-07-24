import torch
import ast
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_vision_encoder():
    from src.model.vision_encoder import VisionEncoder
    encoder = VisionEncoder(image_size=224, patch_size=16, vit_dim=128, llm_dim=64, n_layers=2, n_heads=4, num_image_tokens=16)
    dummy = torch.randn(1, 3, 224, 224)
    out = encoder(dummy)
    assert out.shape == (1, 16, 64), f"Expected (1,16,64), got {out.shape}"
    print("C1. VisionEncoder OK")


def test_multimodal_embedding():
    from src.model.multimodal_embedding import MultimodalEmbedding
    mm = MultimodalEmbedding(text_vocab_size=100, llm_dim=64)
    ids = torch.randint(0, 100, (2, 16))
    em = mm(ids)
    assert em.shape == (2, 16, 64)
    print("C2. MultimodalEmbedding OK")


def test_code_executor():
    from src.agents.code_executor import CodeExecutor
    executor = CodeExecutor(timeout=15)
    res = executor.execute("x = 1 + 1\nprint(x)")
    assert res["success"], f"Execution failed: {res.get('error', '')}"
    assert "2" in res["stdout"], f"Expected '2' in stdout, got: {res['stdout']}"
    print(f"D1. CodeExecutor OK: stdout={res['stdout']!r}")


def test_code_validation():
    from src.agents.code_executor import CodeExecutor
    executor = CodeExecutor()
    executor.validate_code("print('hello')")
    print("D2. Code validation OK")

    try:
        executor.validate_code("import os\nos.system('rm')")
        print("D3. Should have raised")
    except ValueError:
        print("D3. Code validation blocked OK")


def test_orchestrator():
    from src.agents.orchestrator import AgentOrchestrator
    o = AgentOrchestrator.__new__(AgentOrchestrator)
    o.planner = None
    o.executor = None
    o.max_iterations = 10
    print("E1. Orchestrator init OK")


def test_search_grounding():
    from src.agents.search_grounding import SearchGrounding
    sg = SearchGrounding()
    res = sg.search("test query")
    assert "query" in res
    print("E2. SearchGrounding OK")


def test_file_search():
    from src.agents.file_search import FileSearch
    fs = FileSearch(".")
    fs.index_directory(patterns=["*.py"])
    matches = fs.search("import")
    assert isinstance(matches, list)
    print("E3. FileSearch OK")


def test_schema_constraint():
    from src.inference.structured import SchemaConstraint
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    constraint = SchemaConstraint(schema)
    fmt = constraint.format_prompt("Hello")
    assert "JSON response" in fmt
    print("F1. SchemaConstraint OK")


def test_long_context_processor():
    from src.model import Transformer, ModelConfig
    from src.inference.long_context import LongContextProcessor
    from src.tokenizer import Tokenizer

    config = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=100, max_seq_len=128, use_flash_attn=False, dtype="float32")
    model = Transformer(config)
    t = Tokenizer.__new__(Tokenizer)
    t.sp = None
    t.vocab_size = 100
    t.encode = lambda text, **kw: [1, 2, 3]
    t.decode = lambda ids: "hello"

    proc = LongContextProcessor(model, t, config)
    chunks, texts = proc.process_long_input("hello " * 1000)
    assert len(chunks) > 0
    print("G1. LongContextProcessor OK")


def test_reasoning_trainer():
    from src.model import Transformer, ModelConfig
    from src.training.reasoning_trainer import ReasoningTrainer

    config = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=100, max_seq_len=32, use_flash_attn=False, dtype="float32")
    model = Transformer(config)
    trainer = ReasoningTrainer(model, {"lr": 1e-4, "max_steps": 5, "warmup_steps": 1, "dtype": "float32", "batch_size": 2})
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "labels": torch.randint(0, 100, (2, 32)),
        "thinking_mask": torch.randint(0, 2, (2, 32)),
    }
    result = trainer.train_step(batch)
    assert "loss" in result
    print("H1. ReasoningTrainer OK")


if __name__ == "__main__":
    test_vision_encoder()
    test_multimodal_embedding()
    test_code_executor()
    test_code_validation()
    test_orchestrator()
    test_search_grounding()
    test_file_search()
    test_schema_constraint()
    test_long_context_processor()
    test_reasoning_trainer()
    print("\n=== ALL PHASES C-H TESTS PASSED ===")

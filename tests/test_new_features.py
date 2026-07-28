import pytest
import torch


class TestMultimodal:
    def test_vision_encoder(self):
        from src.model.vision_encoder import VisionEncoder

        encoder = VisionEncoder(
            image_size=224,
            patch_size=16,
            vit_dim=128,
            llm_dim=64,
            n_layers=2,
            n_heads=4,
            num_image_tokens=16,
        )
        dummy_images = torch.randn(1, 3, 224, 224)
        output = encoder(dummy_images)
        assert output.shape == (1, 16, 64)

    def test_multimodal_embedding(self):
        from src.model.multimodal_embedding import MultimodalEmbedding

        mm = MultimodalEmbedding(text_vocab_size=100, llm_dim=64)
        ids = torch.randint(0, 100, (2, 16))
        embeds = mm(ids)
        assert embeds.shape == (2, 16, 64)


class TestCodeExecutor:
    def test_simple_execution(self):
        from src.agents.code_executor import CodeExecutor

        executor = CodeExecutor(timeout=10)
        result = executor.execute("x = 1 + 1\nprint(x)")
        assert result["success"]
        assert "2" in result["stdout"]

    def test_validation(self):
        from src.agents.code_executor import CodeExecutor

        executor = CodeExecutor()
        with pytest.raises(ValueError):
            executor.validate_code("import os\nos.system('rm -rf /')")


class TestOrchestrator:
    def test_orchestrator_init(self):
        from src.agents.orchestrator import AgentOrchestrator

        o = AgentOrchestrator.__new__(AgentOrchestrator)
        o.planner = None
        o.executor = None
        o.max_iterations = 10
        assert o.max_iterations == 10


class TestStructuredOutput:
    def test_schema_constraint(self):
        from src.inference.structured import SchemaConstraint

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        constraint = SchemaConstraint(schema)
        formatted = constraint.format_prompt("Hello")
        assert "JSON response" in formatted


class TestSearchGrounding:
    def test_search_grounding(self):
        from src.agents.search_grounding import SearchGrounding

        sg = SearchGrounding()
        result = sg.search("test query")
        assert "query" in result


class TestFileSearch:
    def test_file_search(self):
        from src.agents.file_search import FileSearch

        fs = FileSearch(".")
        fs.index_directory(patterns=["*.py"])
        results = fs.search("import")
        assert isinstance(results, list)


class TestLongContextProcessor:
    def test_long_context_processor(self, mock_tokenizer):
        from src.inference.long_context import LongContextProcessor
        from src.model import ModelConfig, Transformer

        config = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=100,
            max_seq_len=128,
            use_flash_attn=False,
            dtype="float32",
        )
        model = Transformer(config)

        processor = LongContextProcessor(model, mock_tokenizer, config)
        chunks, texts = processor.process_long_input("Hello world " * 100)
        assert len(chunks) > 0


class TestReasoningTrainer:
    def test_reasoning_trainer(self):
        from src.model import ModelConfig, Transformer
        from src.training.reasoning_trainer import ReasoningTrainer

        config = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=100,
            max_seq_len=32,
            use_flash_attn=False,
            dtype="float32",
        )
        model = Transformer(config)
        trainer = ReasoningTrainer(
            model,
            {
                "lr": 1e-4,
                "max_steps": 5,
                "warmup_steps": 1,
                "dtype": "float32",
                "batch_size": 2,
                "output_dir": "./tests/tmp_checkpoints",
            },
        )
        batch = {
            "input_ids": torch.randint(0, 100, (2, 32)),
            "labels": torch.randint(0, 100, (2, 32)),
            "thinking_mask": torch.randint(0, 2, (2, 32)),
        }
        result = trainer.train_step(batch)
        assert "loss" in result

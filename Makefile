.PHONY: install dev test lint format clean train eval chat server docker

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/
	ruff format --check src/

format:
	ruff format src/

train:
	python -m src.training.trainer

eval:
	python -m src.evaluation.runner --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model

chat:
	python -m src.inference.cli --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model

server:
	python -m src.inference.server

tokenizer-train:
	python -m src.tokenizer.train --input data/corpus.txt --model-prefix tokenizer/tokenizer --vocab-size 128000

docker-build:
	docker build -t opencode-llm:latest -f docker/Dockerfile .

docker-compose:
	docker-compose -f docker/docker-compose.yml up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

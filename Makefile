.PHONY: install dev test lint format clean train eval chat server docker precommit

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

train:
	python -m src.training.trainer

eval:
	python -m src.evaluation.runner --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model

chat:
	nexus

server:
	nexus --serve

tokenizer-train:
	python -m src.tokenizer.train --input data/corpus.txt --model-prefix tokenizer/tokenizer --vocab-size 128000

docker-build:
	docker build -t opencode-llm:latest -f docker/Dockerfile .

docker-compose:
	docker-compose -f docker/docker-compose.yml up

precommit:
	pre-commit install

clean:
	Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force 2>$null
	Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force 2>$null
	Remove-Item -Recurse -Force -Path "build", "dist", "*.egg-info" -ErrorAction SilentlyContinue

# Contributing

We welcome contributions! This document outlines the development workflow, coding standards, and how to get started.

## Development Setup

```bash
# Clone
git clone https://github.com/your-org/nexus.git
cd nexus

# Install with dev dependencies (recommended)
pip install -e ".[dev]"

# Or minimal install
pip install -e .
```

## Code Style

- **Format**: `black` with 100 char line length
- **Imports**: `isort` (black-compatible profile)
- **Types**: Full type annotations for all function signatures
- **Docstrings**: Google-style for public APIs only
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants

Pre-commit hooks (`pre-commit install`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core_model.py -v

# Run with coverage
pytest --cov=src tests/

# Quick smoke test
make test
```

### Test Structure
```
tests/
├── conftest.py
├── test_attention.py
├── test_long_context.py
├── test_model.py
├── test_new_features.py
├── test_server.py
├── test_tokenizer.py
└── test_training.py
```

## Pull Request Process

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** following code style
4. **Write tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Run lint**: `pre-commit run --all-files`
7. **Commit**: `git commit -m "feat: description of change"`
8. **Push**: `git push origin feature/your-feature`
9. **Open PR** against `main` branch

### PR Guidelines
- One logical change per PR
- Keep PRs small (<500 lines when possible)
- Link related issues
- Update docs if API changes
- Add tests for new features

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add vision encoder module
fix: correct RoPE position calculation for odd dimensions
docs: update deployment guide
test: add attention mask unit tests
refactor: unify loss functions into single module
chore: update dependency versions
```

## Adding Kernels

When contributing custom CUDA/Triton kernels:
- Place in `src/kernels/`
- Include Python fallback in `src/model/` for CPU testing
- Add kernel tests in `tests/test_kernels.py`
- Profile: `python -m tests.benchmark_kernel`

## Documentation

- Add docstrings to public functions only
- Update relevant `docs/*.md` when changing functionality
- Run `python -m docs.build` to verify docs build
- Use examples in docstrings

## Code Review Checklist

- [ ] Tests pass
- [ ] Lint passes (black, isort)
- [ ] Type annotations complete
- [ ] No hardcoded paths or secrets
- [ ] Error handling appropriate
- [ ] Performance implications considered
- [ ] API backwards compatible (or migration path provided)

## Getting Help

- Open an issue for bugs or feature requests
- Tag with appropriate label: `bug`, `enhancement`, `question`
- Provide minimum reproducible examples for bugs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

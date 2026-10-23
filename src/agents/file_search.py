import fnmatch
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class FileSearch:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.index: dict[str, list[str]] = {}

    def index_directory(self, path: str | None = None, patterns: list[str] = None):
        search_path = (self.root_path / path).resolve() if path else self.root_path
        if not search_path.is_relative_to(self.root_path):
            raise ValueError("Search path must be inside the configured root")
        patterns = patterns or [
            "*.py",
            "*.js",
            "*.ts",
            "*.md",
            "*.txt",
            "*.yaml",
            "*.json",
            "*.html",
            "*.css",
        ]
        excluded = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".next",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "checkpoints",
        }
        self.index = {}
        for folder, directories, filenames in os.walk(search_path, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name not in excluded
                and not (Path(folder) / name).is_symlink()
                and (Path(folder) / name).resolve().is_relative_to(self.root_path)
            ]
            for name in filenames:
                if name.startswith(".env") or name.endswith((".pem", ".key", ".db")):
                    continue
                if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    continue
                file_path = Path(folder) / name
                try:
                    if (
                        file_path.is_symlink()
                        or not file_path.resolve().is_relative_to(self.root_path)
                        or file_path.stat().st_size > 2 * 1024 * 1024
                    ):
                        continue
                    rel_path = str(file_path.relative_to(self.root_path))
                    self.index[rel_path] = file_path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                except OSError as error:
                    logger.debug("Could not read file (%s)", type(error).__name__)
                if len(self.index) >= 10000:
                    return

        logger.info(f"Indexed {len(self.index)} files from {search_path}")

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        results = []
        query_lower = query.lower()

        for filepath, lines in self.index.items():
            score = 0
            matches = []
            for i, line in enumerate(lines, 1):
                if query_lower in line.lower():
                    score += 1
                    matches.append(
                        {
                            "line": i,
                            "content": line.strip()[:200],
                        }
                    )

            if score > 0:
                results.append(
                    {
                        "file": filepath,
                        "score": score,
                        "matches": matches[:5],
                    }
                )

        results.sort(key=lambda x: -x["score"])
        return results[:max_results]

    def read_file(self, filepath: str, max_lines: int = None) -> str | None:
        full_path = (self.root_path / filepath).resolve()
        if not full_path.is_relative_to(self.root_path):
            raise ValueError("File must be inside the configured root")
        if full_path.is_file() and full_path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("File exceeds the 2 MiB reading limit")
        if full_path.exists() and full_path.is_file():
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            if max_lines:
                lines = content.splitlines()
                return "\n".join(lines[:max_lines])
            return content
        return None

    def find_files(self, pattern: str) -> list[str]:
        matches = []
        for filepath in self.index:
            if fnmatch.fnmatch(filepath, pattern):
                matches.append(filepath)
        return matches


file_search = FileSearch()

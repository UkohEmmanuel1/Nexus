import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileSearch:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.index: dict[str, list[str]] = {}

    def index_directory(self, path: str | None = None, patterns: list[str] = None):
        search_path = Path(path) if path else self.root_path
        if patterns is None:
            patterns = [
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

        self.index = {}
        for pattern in patterns:
            for file_path in search_path.rglob(pattern):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self.root_path))
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        self.index[rel_path] = content.splitlines()
                    except Exception as e:
                        logger.debug(f"Could not read {rel_path}: {e}")

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
        full_path = self.root_path / filepath
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

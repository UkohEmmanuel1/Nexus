import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

from src.agents.web_crawler import CrawlResult

logger = logging.getLogger(__name__)


class WebIndex:
    def __init__(self, db_path: str = "nexus_web_index.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                links TEXT,
                status_code INTEGER,
                error TEXT,
                crawled_at TEXT,
                indexed_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_at ON pages(crawled_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON pages(title)")
        conn.commit()
        conn.close()

    def index_results(self, results: list[CrawlResult]) -> int:
        conn = sqlite3.connect(self.db_path)
        count = 0
        for r in results:
            try:
                conn.execute(
            "INSERT OR REPLACE INTO pages (url, title, content, links, "
            "status_code, error, crawled_at, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                        r.url,
                        r.title,
                        r.content,
                        json.dumps(r.links),
                        r.status_code,
                        r.error,
                        r.crawled_at.isoformat(),
                        datetime.now().isoformat(),
                    ),
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to index {r.url}: {e}")
        conn.commit()
        conn.close()
        logger.info(f"Indexed {count} pages")
        return count

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT url, title, content, crawled_at FROM pages WHERE title LIKE ? OR content LIKE ? "  # noqa: E501
            "ORDER BY crawled_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_all_urls(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT url FROM pages WHERE status_code = 200 ORDER BY crawled_at DESC"
        )
        urls = [row[0] for row in cursor.fetchall()]
        conn.close()
        return urls

    def get_stats(self) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        success = conn.execute("SELECT COUNT(*) FROM pages WHERE status_code = 200").fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM pages WHERE status_code != 200 AND status_code != 304"
        ).fetchone()[0]
        conn.close()
        return {"total": total, "success": success, "errors": errors}

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM pages")
        conn.commit()
        conn.close()
        logger.info("Web index cleared")


web_index = WebIndex()

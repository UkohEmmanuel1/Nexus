import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta

from src.agents.web_crawler import CrawlResult, WebCrawler

logger = logging.getLogger(__name__)


class CrawlScheduler:
    def __init__(self, crawler: WebCrawler | None = None):
        self.crawler = crawler or WebCrawler()
        self._tasks: list[dict] = []
        self._running = False
        self._on_crawl_callback: Callable | None = None

    def schedule_crawl(
        self,
        seed_urls: list[str],
        interval_seconds: int = 300,
        max_depth: int = 1,
        task_name: str | None = None,
    ) -> str:
        task_id = f"crawl_{datetime.now().timestamp()}"
        self._tasks.append(
            {
                "id": task_id,
                "name": task_name or f"Crawl {len(seed_urls)} URLs",
                "seed_urls": seed_urls,
                "interval": interval_seconds,
                "max_depth": max_depth,
                "last_run": None,
                "next_run": datetime.now(),
                "status": "scheduled",
            }
        )
        return task_id

    def set_on_crawl_callback(self, callback: Callable[[list[CrawlResult]], None]) -> None:
        self._on_crawl_callback = callback

    async def start(self) -> None:
        self._running = True
        logger.info("Crawl scheduler started")
        while self._running:
            now = datetime.now()
            for task in self._tasks:
                if task["status"] != "scheduled":
                    continue
                if task["next_run"] and now >= task["next_run"]:
                    await self._run_task(task)
            await asyncio.sleep(10)

    async def _run_task(self, task: dict) -> None:
        task["status"] = "running"
        task["last_run"] = datetime.now()
        logger.info(f"Running scheduled crawl: {task['name']}")
        try:
            results = await self.crawler.crawl_batch(task["seed_urls"], max_depth=task["max_depth"])
            task["status"] = "completed"
            task["next_run"] = datetime.now() + timedelta(seconds=task["interval"])
            if self._on_crawl_callback:
                self._on_crawl_callback(results)
            logger.info(f"Crawl completed: {len(results)} pages fetched")
        except Exception as e:
            task["status"] = "failed"
            logger.error(f"Crawl task failed: {e}")

    async def stop(self) -> None:
        self._running = False
        logger.info("Crawl scheduler stopped")

    def get_status(self) -> list[dict]:
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "status": t["status"],
                "last_run": t["last_run"].isoformat() if t["last_run"] else None,
                "next_run": t["next_run"].isoformat() if t["next_run"] else None,
                "seed_urls": t["seed_urls"],
            }
            for t in self._tasks
        ]

    def get_crawled_data(self) -> list[CrawlResult]:
        return self.crawler.get_results()


scheduler = CrawlScheduler()

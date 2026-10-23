import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.agents.crawl_scheduler import CrawlScheduler


def test_recurring_crawls_run_again_after_completion(monkeypatch):
    async def scenario():
        crawler = MagicMock()
        crawler.crawl_batch = AsyncMock(return_value=[])
        scheduler = CrawlScheduler(crawler)
        scheduler.schedule_crawl(["https://example.com"], interval_seconds=0)
        callback = MagicMock()
        scheduler.set_on_crawl_callback(callback)
        original_sleep = asyncio.sleep

        async def tick(delay):
            if crawler.crawl_batch.await_count >= 2:
                await scheduler.stop()
            await original_sleep(0)

        monkeypatch.setattr("src.agents.crawl_scheduler.asyncio.sleep", tick)
        await scheduler.start()
        assert crawler.crawl_batch.await_count == 2
        assert callback.call_count == 2
        crawler.crawl_batch.assert_awaited_with(["https://example.com"], max_depth=1, refresh=True)

    asyncio.run(scenario())

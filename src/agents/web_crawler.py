import asyncio
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    url: str
    title: str = ""
    content: str = ""
    links: list[str] = field(default_factory=list)
    status_code: int = 200
    error: str | None = None
    crawled_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrawlTask:
    url: str
    depth: int = 0
    max_depth: int = 2
    timeout: int = 15
    max_content_length: int = 50000


class WebCrawler:
    def __init__(
        self,
        base_url: str | None = None,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.5,
        user_agent: str = "NexusCrawler/1.0",
    ):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.user_agent = user_agent
        self._visited: set[str] = set()
        self._results: list[CrawlResult] = []
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=10.0),
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        return self._client

    def _normalize_url(self, url: str, base: str | None = None) -> str | None:
        url, _ = urldefrag(url)
        if not url.startswith(("http://", "https://")):
            if base:
                url = urljoin(base, url)
            else:
                return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        if self.base_url and parsed.netloc != urlparse(self.base_url).netloc:
            return None
        return url

    def _is_visited(self, url: str) -> bool:
        return url in self._visited

    def _mark_visited(self, url: str) -> None:
        self._visited.add(url)

    async def crawl(self, task: CrawlTask) -> CrawlResult:
        url = task.url
        if self._is_visited(url):
            return CrawlResult(url=url, status_code=304, error="already visited")
        if task.depth > task.max_depth:
            return CrawlResult(url=url, status_code=403, error="max depth exceeded")
        self._mark_visited(url)
        client = self._get_client()
        try:
            resp = await client.get(url, timeout=task.timeout)
            if resp.status_code != 200:
                return CrawlResult(
                    url=url, status_code=resp.status_code, error=f"HTTP {resp.status_code}"
                )
            content = resp.text[: task.max_content_length]
            soup = BeautifulSoup(content, "lxml")
            title = soup.title.string if soup.title and soup.title.string else ""
            text = soup.get_text(separator="\n", strip=True)
            links = [a.get("href", "") for a in soup.find_all("a", href=True)]
            normalized_links = []
            for link in links:
                normalized = self._normalize_url(link, url)
                if normalized and not self._is_visited(normalized):
                    normalized_links.append(normalized)
            result = CrawlResult(
                url=url,
                title=title,
                content=text[: task.max_content_length],
                links=normalized_links,
                status_code=resp.status_code,
            )
            self._results.append(result)
            return result
        except asyncio.TimeoutError:
            return CrawlResult(url=url, status_code=408, error="timeout")
        except Exception as e:
            logger.error(f"Crawl failed for {url}: {e}")
            return CrawlResult(url=url, status_code=500, error=str(e))

    async def crawl_batch(self, urls: list[str], max_depth: int = 1) -> list[CrawlResult]:
        tasks = [
            CrawlTask(url=url, depth=0, max_depth=max_depth)
            for url in urls
            if not self._is_visited(url)
        ]
        return await self._crawl_worker_pool(tasks)

    async def crawl_search_results(
        self, query: str, num_results: int = 5, max_depth: int = 1
    ) -> list[CrawlResult]:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        try:
            client = self._get_client()
            resp = await client.get(search_url, timeout=15)
            soup = BeautifulSoup(resp.text, "lxml")
            search_results = []
            for a in soup.find_all("a", class_="result__snippet"):
                href = a.get("href", "")
                if href:
                    search_results.append(href)
            for a in soup.find_all("a", class_="result__url"):
                href = a.get("href", "")
                if href and href not in search_results:
                    search_results.append(href)
            if not search_results:
                for a in soup.find_all("a"):
                    href = a.get("href", "")
                    if href and "http" in href and query.lower() in href.lower():
                        search_results.append(href)
            search_results = search_results[:num_results]
            tasks = [
                CrawlTask(url=sr, depth=0, max_depth=max_depth)
                for sr in search_results
                if not self._is_visited(sr)
            ]
            return await self._crawl_worker_pool(tasks)
        except Exception as e:
            logger.error(f"Search crawl failed for query '{query}': {e}")
            return []

    async def _crawl_worker_pool(self, tasks: list[CrawlTask]) -> list[CrawlResult]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def sem_crawl(task: CrawlTask) -> CrawlResult:
            async with semaphore:
                await asyncio.sleep(self.rate_limit_delay)
                return await self.crawl(task)

        results = await asyncio.gather(*[sem_crawl(t) for t in tasks], return_exceptions=True)
        parsed_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Crawl error: {r}")
            elif isinstance(r, CrawlResult):
                parsed_results.append(r)
        return parsed_results

    async def continuous_crawl(
        self, seed_urls: list[str], interval_seconds: int = 300, max_depth: int = 1
    ) -> Generator[list[CrawlResult], None, None]:
        while True:
            for url in seed_urls:
                if not self._is_visited(url):
                    task = CrawlTask(url=url, depth=0, max_depth=max_depth)
                    result = await self.crawl(task)
                    if result.status_code == 200:
                        child_results = await self._crawl_worker_pool(
                            [CrawlTask(url=link, depth=1, max_depth=max_depth) for link in result.links],  # noqa: E501
                        )
                        yield [result] + child_results
                    else:
                        yield [result]
            await asyncio.sleep(interval_seconds)

    def get_results(self) -> list[CrawlResult]:
        return self._results

    def get_visited_urls(self) -> list[str]:
        return list(self._visited)

    def get_unvisited(self, urls: list[str]) -> list[str]:
        return [u for u in urls if not self._is_visited(u)]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


crawler = WebCrawler()

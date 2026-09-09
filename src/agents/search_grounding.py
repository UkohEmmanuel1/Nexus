import logging

logger = logging.getLogger(__name__)


class SearchGrounding:
    def __init__(self, search_func=None):
        self.search_func = search_func or self._default_search

    def _default_search(self, query: str) -> list[dict[str, str]]:
        try:
            import requests

            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10,
            )
            data = resp.json()
            results = []
            abstract = data.get("AbstractText", "")
            if abstract:
                results.append(
                    {
                        "title": data.get("Heading", "Result"),
                        "snippet": abstract,
                        "source": data.get("AbstractSource", "web"),
                    }
                )
            for topic in data.get("RelatedTopics", [])[:5]:
                if "Text" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "").split(" - ")[0]
                            if " - " in topic.get("Text", "")
                            else "",
                            "snippet": topic.get("Text", ""),
                            "source": topic.get("FirstURL", ""),
                        }
                    )
            return results
        except ImportError:
            logger.warning("requests not available for web search")
            return [{"title": "Note", "snippet": "Web search unavailable", "source": "local"}]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [{"title": "Error", "snippet": str(e), "source": "error"}]

    def search(self, query: str, top_k: int = 5) -> dict:
        results = self.search_func(query)
        results = results[:top_k]
        return {
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    def ground(self, query: str) -> str:
        search_result = self.search(query)
        if not search_result["results"]:
            return f"No search results found for: {query}"

        context = "Search results:\n\n"
        for i, r in enumerate(search_result["results"], 1):
            context += f"[{i}] {r.get('title', '')}\n"
            context += f"    {r.get('snippet', '')}\n"
            context += f"    Source: {r.get('source', '')}\n\n"
        return context


search_grounding = SearchGrounding()

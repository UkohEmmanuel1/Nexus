# Agents

The agent system enables LLM-driven autonomous task completion through planning, tool use, code execution, search grounding, and memory.

## Architecture

```
User Task
    │
    ▼
Planner ──→ Decompose into sub-tasks
    │
    ▼
Orchestrator ──→ Execute sub-tasks
    │
    ├── ReAct Loop (Think → Act → Observe)
    ├── Function Calling (structured tool use)
    ├── Code Execution (sandboxed Python)
    ├── Search Grounding (web search + citations)
    └── File Search (recursive file system access)
    │
    └── Memory (conversation + summarization)
    │
    ▼
Final Answer
```

## Components

### ReAct (`react.py`)
Thought/Action/Observation loop that interleaves reasoning with tool use:

```
Input: "Calculate 15! and tell me the result"

Thought: I need to compute 15 factorial. I can use Python code execution.
Action: python_repl
Action Input: {"code": "import math\nprint(math.factorial(15))"}
Observation: 1307674368000
Thought: The factorial of 15 is 1307674368000. I can now answer.
Final Answer: 15! = 1,307,674,368,000
```

### Function Calling (`function_calling.py`)
Structured protocol for tool invocation:

```
<function_call>
{"name": "python_repl", "args": {"code": "print(sum(range(100)))"}}
</function_call>
```

### Tools (`tools.py`)
Built-in tool set:

| Tool | Description |
|------|-------------|
| `search_web` | Search the web via DuckDuckGo instant answer API |
| `python_repl` | Run Python in sandbox |
| `calculator` | Safe arithmetic evaluation |
| `current_datetime` | Get current date/time |

### Planner (`planner.py`)
Breaks complex tasks into sequential sub-steps:

```python
from src.agents import TaskPlanner, StepExecutor

planner = TaskPlanner(engine, tokenizer)
plan = planner.plan("Research the population of Tokyo, then write a summary")
# Returns: [
#   "Search: Tokyo population 2024",
#   "Read: top 3 results",
#   "Synthesize: write 200-word summary",
# ]
```

### Orchestrator (`orchestrator.py`)
Coordinates multi-step agent execution:

```python
orchestrator = AgentOrchestrator(engine, tools)
result = orchestrator.run("Compare Python and JavaScript popularity in 2024")
```

Orchestrator handles:
1. Planning → sub-tasks
2. ReAct loop per sub-task
3. Context management across sub-tasks
4. Final synthesis

### Memory (`memory.py`)
Maintains conversation history with an optional summarizer and recent-window trimming:

```python
from src.agents.memory import ConversationMemory

memory = ConversationMemory(max_tokens=8192)

# Add turns
memory.add("user", "What's the capital of France?")
memory.add("assistant", "Paris.")

# Get context for LLM (recent window, prepends summary if set)
context = memory.get_context()

# Reset the conversation
memory.clear()
```

An optional `summarizer` callable can be supplied to compress older history into a
system-prompt summary included in `get_context()`.

### Web Crawler (`web_crawler.py`)
Asynchronous web crawler that fetches pages, extracts text/links, and crawls DuckDuckGo search results with concurrency + rate limiting:

```python
import asyncio
from src.agents.web_crawler import CrawlTask, WebCrawler

crawler = WebCrawler(
    base_url=None,        # Restrict to a domain when set
    max_concurrent=5,     # Parallel page fetches
    rate_limit_delay=0.5, # Delay between requests
)

async def main():
    # Crawl a single task with depth limit
    result = await crawler.crawl(CrawlTask(url="https://example.com", max_depth=2))
    print(result.title, result.content[:500])

    # Crawl a batch of URLs
    results = await crawler.crawl_batch(
        ["https://example.com/a", "https://example.com/b"], max_depth=1
    )

    # Crawl top search results for a query
    results = await crawler.crawl_search_results("Nexus LLM", num_results=5)

    # Continuous re-crawl on an interval
    async for batch in crawler.continuous_crawl(
        ["https://example.com"], interval_seconds=300, max_depth=1
    ):
        print(f"Fetched {len(batch)} pages")

asyncio.run(main())
```

Key features:
- Visited-URL deduplication + same-domain restriction via `base_url`
- Async worker pool bounded by `max_concurrent` with `rate_limit_delay`
- HTML → text extraction via BeautifulSoup, link normalization
- `crawl_search_results` seeds crawls from DuckDuckGo HTML results

### Crawl Scheduler (`crawl_scheduler.py`)
Schedules periodic crawling of seed URLs and triggers callbacks with results:

```python
import asyncio
from src.agents.crawl_scheduler import CrawlScheduler
from src.agents.web_crawler import WebCrawler

scheduler = CrawlScheduler(crawler=WebCrawler())

def on_crawl(results):
    for r in results:
        print(f"Crawled {r.url} ({r.status_code})")

scheduler.set_on_crawl_callback(on_crawl)
task_id = scheduler.schedule_crawl(
    seed_urls=["https://example.com"],
    interval_seconds=3600,
    max_depth=1,
    task_name="Daily docs refresh",
)

async def main():
    await scheduler.start()  # Runs until scheduler.stop()

asyncio.run(main())

# Inspect scheduled task status
print(scheduler.get_status())
```

## Usage

### Full Agent
```python
from src.agents import AgentOrchestrator, create_orchestrator, get_function_map
from src.inference.engine import InferenceEngine

engine = InferenceEngine(model, tokenizer)
tools = get_function_map()  # search_web, python_repl, calculator, current_datetime

# Option 1: helper that wires default tools
orchestrator = create_orchestrator(engine)

# Option 2: explicit construction with custom tool map
orchestrator = AgentOrchestrator(engine, tools, max_iterations=10)

result = orchestrator.run("What was the weather in Tokyo yesterday?")
print(result["final_answer"])
```

### Standalone ReAct
```python
from src.agents.react import ReActAgent
from src.agents.tools import get_function_map

agent = ReActAgent(engine, get_function_map(), max_steps=10)
answer = agent.run("Calculate sqrt(144) + sqrt(81)")
print(answer)  # Final answer string (tool-verified)

# Stream intermediate steps
for event in orchestrator.run_stream("Calculate sqrt(144) + sqrt(81)"):
    print(event)  # {"type": "plan" | "step_start" | "step_result" | "final", "data": ...}
```

### REST API

```json
POST /v1/agent/run
{
  "task": "Calculate 15! and web search the meaning of life",
  "max_steps": 10
}
```

Web crawl endpoints are served by the API server (see [Deployment](deployment.md) for the full list):

```json
POST /crawl      { "search_query": "Nexus LLM", "max_depth": 1 }
POST /crawl/schedule  { "seed_urls": ["https://example.com"], "interval_seconds": 3600 }
GET  /crawl/data ?query=llm&limit=10
GET  /crawl/status
```

## Configuration

```yaml
agent:
  max_steps: 10
  max_history_tokens: 8000
  temperature: 0.7
  top_p: 0.9
  tools:
    - search_web
    - python_repl
    - calculator
    - current_datetime
```

```yaml
crawler:
  max_concurrent: 5
  rate_limit_delay: 0.5
  user_agent: "NexusCrawler/1.0"
  scheduler:
    default_interval_seconds: 300
    default_max_depth: 1
```

## Best Practices

1. **Limit steps**: 10–15 steps max to avoid loops
2. **Use planning**: Complex tasks should be decomposed first
3. **Monitor tokens**: Each step consumes context; use memory summarization
4. **Combine with thinking**: Agents work well with thinking mode for complex reasoning
5. **Rate limiting**: Add delays between web search calls

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
Action: execute_python({"code": "import math\nprint(math.factorial(15))"})
Observation: 1307674368000
Thought: The factorial of 15 is 1307674368000. I can now answer.
Action: answer({"answer": "15! = 1,307,674,368,000"})
```

### Function Calling (`function_calling.py`)
Structured protocol for tool invocation:

```
<function_call>
{"name": "execute_python", "args": {"code": "print(sum(range(100)))"}}
</function_call>
```

### Tools (`tools.py`)
Built-in tool set:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo/SERP API |
| `execute_python` | Run Python in sandbox |
| `calculate` | Safe arithmetic evaluation |
| `datetime_now` | Get current date/time |
| `file_search` | Search files on filesystem |

### Planner (`planner.py`)
Breaks complex tasks into sequential sub-tasks:

```python
planner = Planner(engine, tokenizer)
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
Maintains conversation history with automatic summarization:

```python
memory = Memory(max_tokens=8192)

# Add turns
memory.add("user", "What's the capital of France?")
memory.add("assistant", "Paris.")

# Summarize long history
summary = memory.summarize()
print(summary)  # "User asked about France's capital. Answer: Paris."

# Get context for LLM
context = memory.get_context()
```

## Usage

### Full Agent
```python
from src.agents import AgentOrchestrator, get_default_tools
from src.inference.engine import InferenceEngine

engine = InferenceEngine(model, tokenizer)
tools = get_default_tools()  # web_search, execute_python, calculate, datetime
orchestrator = AgentOrchestrator(
    engine, tools,
    max_steps=10,      # Max ReAct steps
    max_history=8000,  # Context window per step
)

result = orchestrator.run("What was the weather in Tokyo yesterday?")
```

### Standalone ReAct
```python
from src.agents.react import ReActAgent

agent = ReActAgent(engine, tools)
replies = agent.run("Calculate sqrt(144) + sqrt(81)")
for reply in replies:
    if reply.type == "thought":
        print(f"🤔 {reply.content}")
    elif reply.type == "action":
        print(f"🛠️ {reply.content}")
    elif reply.type == "observation":
        print(f"👁️ {reply.content}")
    elif reply.type == "answer":
        print(f"✅ {reply.content}")
```

### REST API
```json
POST /v1/agent/run
{
  "task": "Calculate 15! and web search the meaning of life",
  "max_steps": 10
}
```

## Configuration

```yaml
agent:
  max_steps: 10
  max_history_tokens: 8000
  temperature: 0.7
  top_p: 0.9
  tools:
    - web_search
    - execute_python
    - calculate
    - datetime_now
    - file_search
```

## Best Practices

1. **Limit steps**: 10–15 steps max to avoid loops
2. **Use planning**: Complex tasks should be decomposed first
3. **Monitor tokens**: Each step consumes context; use memory summarization
4. **Combine with thinking**: Agents work well with thinking mode for complex reasoning
5. **Rate limiting**: Add delays between web search calls

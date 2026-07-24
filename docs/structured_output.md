# Structured Output

Generate JSON and list outputs that conform to a user-defined schema, enforcing type constraints and structural rules directly in the sampling loop.

## How It Works

Instead of generating free text and parsing (which can produce invalid output), structured output constrains the generation distribution at each token step:

1. A schema defines the desired output structure
2. During generation, only tokens that produce valid (prefix-valid) output are considered
3. The probability distribution is masked to exclude invalid tokens
4. Generation continues until the schema is satisfied

## Schema Constraint Engine

```python
from src.inference.structured import SchemaConstraint, generate_json

# Define a schema
schema = SchemaConstraint({
    "name": str,
    "age": int,
    "scores": [float],
    "metadata": {
        "created": str,
        "tags": [str],
    }
})

# Generate constrained output
result = generate_json(
    model, tokenizer, input_ids,
    schema=schema,
    max_new_tokens=512,
    temperature=0.7,
)
# {"name": "Alice", "age": 30, "scores": [95.5, 88.0], "metadata": {"created": "2024-01-15", "tags": ["exam", "final"]}}
```

## Supported Types

| Type | Description | Example |
|------|-------------|---------|
| `str` | String | `"hello"` |
| `int` | Integer | `42` |
| `float` | Float | `3.14` |
| `bool` | Boolean | `true` |
| `[T]` | List of type T | `[1, 2, 3]` |
| `{k: T}` | Dict with value type T | `{"a": 1}` |
| Nested `dict` | Structured object | `{"x": 1, "y": 2}` |
| `Optional[T]` | Nullable field | `null` or value |
| `Literal["a"]` | Enum | must match one of values |

## Usage

### Python API
```python
# Simple JSON
schema = SchemaConstraint({"name": str, "score": int})
result = generate_json(model, tokenizer, input_ids, schema)

# Nested JSON
schema = SchemaConstraint({
    "user": {"id": int, "name": str},
    "items": [{"product": str, "price": float}],
})
result = generate_json(model, tokenizer, input_ids, schema)

# List output
result = generate_list(model, tokenizer, input_ids, element_type=int, max_items=5)
# [1, 2, 3, 4, 5]

# List of objects
result = generate_list(
    model, tokenizer, input_ids,
    element_type={"name": str, "value": float},
    max_items=3,
)
# [{"name": "a", "value": 1.0}, {"name": "b", "value": 2.0}]

# With streaming
for chunk in generate_json_stream(
    model, tokenizer, input_ids, schema
):
    print(chunk)
```

### REST API
```json
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "List 3 fruits with prices"}],
  "response_format": {
    "type": "json_schema",
    "schema": {
      "type": "object",
      "properties": {
        "fruits": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "price": {"type": "number"}
            }
          }
        }
      }
    }
  }
}
```

### Agent Integration
```python
from src.agents.orchestrator import AgentOrchestrator

result = orchestrator.run_with_schema(
    "Extract all dates and amounts from this text",
    output_schema={
        "transactions": [{"date": str, "amount": float}],
    },
)
```

## How Constraint Enforcement Works

```
1. Parse schema → create state machine
2. For each generation step:
   a. Get logits from model
   b. Compute valid next tokens based on current state:
      - JSON opening/closing brackets
      - Valid string terminators
      - Number vs string continuation
   c. Mask invalid tokens (set logits to -inf)
   d. Sample from constrained distribution
   e. Update parser state
3. Return when schema is fully matched
```

## Best Practices

1. **Keep schemas small**: Large schemas (>100 fields) increase generation time due to repeated validation.

2. **Use Optional for nullable fields**: Without Optional, null values will be considered invalid.

3. **Prefer smaller max_new_tokens**: Reduces unnecessary generation beyond the schema.

4. **Temperature**: Lower temperature (0.1–0.5) for more deterministic structured output.

5. **Testing**:
```python
# Validate output against schema
result_schema = SchemaConstraint({"name": str, "age": int})
try:
    result_schema.validate(result)
    print("Valid")
except ValueError as e:
    print(f"Invalid: {e}")
```

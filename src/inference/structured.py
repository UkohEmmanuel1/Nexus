import json
import re
from typing import Any

from src.inference.engine import InferenceEngine


class SchemaConstraint:
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self._validate_schema()

    def _validate_schema(self):
        if "type" not in self.schema:
            raise ValueError("Schema must have a 'type' field")

    def format_prompt(self, prompt: str) -> str:
        schema_str = json.dumps(self.schema, indent=2)
        return (
            f"{prompt}\n\n"
            f"Respond with a valid JSON object matching this schema:\n{schema_str}\n\n"
            "JSON response:"
        )

    def parse_response(self, response: str) -> dict:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("Could not parse JSON response")


class StructuredOutput:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def generate(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_retries: int = 3,
        temperature: float = 0.2,
        **kwargs,
    ) -> dict:
        constraint = SchemaConstraint(schema)
        formatted_prompt = constraint.format_prompt(prompt)

        for attempt in range(max_retries):
            response = self.engine.generate(
                formatted_prompt,
                temperature=temperature,
                max_new_tokens=2048,
                **kwargs,
            )
            try:
                return constraint.parse_response(response)
            except ValueError:
                if attempt == max_retries - 1:
                    raise
                continue
        return {}

    def generate_json(
        self,
        prompt: str,
        properties: dict[str, str],
        required: list[str] = None,
        **kwargs,
    ) -> dict:
        schema = {
            "type": "object",
            "properties": {k: {"type": v} for k, v in properties.items()},
        }
        if required:
            schema["required"] = required
        return self.generate(prompt, schema, **kwargs)

    def generate_list(self, prompt: str, item_schema: dict, **kwargs) -> list:
        schema = {
            "type": "array",
            "items": item_schema,
        }
        result = self.generate(prompt, schema, **kwargs)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in result:
                if isinstance(result[key], list):
                    return result[key]
        return [result]


structured_output = StructuredOutput

#TODO: Implement model class

from .data_types import Message
from typing import List, Dict
from openai import OpenAI
import inspect
import json

try:
    _client_probe = OpenAI()
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


def _python_type_to_json_schema(py_type):
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is str:
        return {"type": "string"}
    if py_type is bool:
        return {"type": "boolean"}
    # Fallbacks
    return {"type": "string"}


def _function_to_tool_spec(fn):
    signature = inspect.signature(fn)
    annotations = getattr(fn, "__annotations__", {})

    properties: Dict[str, Dict] = {}
    required: List[str] = []

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        annotation = annotations.get(param_name, str)
        properties[param_name] = _python_type_to_json_schema(annotation)
        if param.default is inspect._empty:
            required.append(param_name)

    description = (fn.__doc__ or "").strip()

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _prepare_tools(tools):
    if not tools:
        return None
    prepared = []
    for t in tools:
        if callable(t):
            prepared.append(_function_to_tool_spec(t))
        else:
            prepared.append(t)
    return prepared


class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()

    def chat(self, messages: List[Message], tools: list = None):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI Python client is not installed/configured. Install with: pip install openai, and ensure OPENAI_API_KEY is set."
            )

        prepared_tools = _prepare_tools(tools)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=prepared_tools,
            tool_choice="auto" if prepared_tools else "none",
            reasoning_effort="minimal"
        )

        message = response.choices[0].message
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
        }


if __name__ == "__main__":
    model = Model("gpt-5")

    def add_numbers(num1: int, num2: int):
        """Adds two numbers together
        Args:
            num1: int
            num2: int
        Returns:
            int: The sum of the two numbers
        """
        return num1 + num2

    messages = [{"role": "user", "content": "What is 4+4?"}]

    response = model.chat(messages, tools=[add_numbers])
    print(messages)
    messages.append(response)

    tool_args_raw = response["tool_calls"][0]["function"]["arguments"]
    tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
    tool_output = add_numbers(**tool_args)

    print(messages)
    messages.append({
        "role": "tool",
        "tool_call_id": response["tool_calls"][0]["id"],
        "content": str(tool_output),
    })

    response = model.chat(messages, tools=[add_numbers])
    print(messages)
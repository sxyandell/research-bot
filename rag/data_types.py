from typing import Optional, List, Literal, TypedDict


class FunctionCall(TypedDict):
    name: str
    arguments: dict


class ToolCall(TypedDict):
    function: FunctionCall


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None

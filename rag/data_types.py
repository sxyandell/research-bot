from typing import Optional, List, Literal, TypedDict, Union


class FunctionCall(TypedDict):
    name: str
    arguments: Union[dict, str]


class ToolCall(TypedDict, total=False):
    id: str
    function: FunctionCall


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    tool_calls: Optional[List[ToolCall]]
    tool_call_id: Optional[str]
    name: Optional[str]

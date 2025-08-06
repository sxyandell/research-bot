#TODO: Implement model class


from typing import Optional, List, TypedDict, Literal


class ToolCall(TypedDict):
    name: str
    args: dict


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def chat(self, messages: List[Message]):
        pass


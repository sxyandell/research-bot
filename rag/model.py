#TODO: Implement model class


from typing import Optional, List, TypedDict, Literal
from ollama import chat, ChatResponse


class ToolCall(TypedDict):
    name: str
    args: dict


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def chat(self, messages: List[Message], tools: dict = None):
        response: ChatResponse = chat(model=self.model_name, messages=messages, tools=tools)
        return response.message.content


if __name__ == "__main__":
    model = Model("phi3:mini")
    messages = [{"role": "user", "content": "Why is the sky blue?"}]
    print(model.chat(messages))
    


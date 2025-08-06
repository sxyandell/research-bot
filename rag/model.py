#TODO: Implement model class


from typing import Optional, List, TypedDict, Literal

import ollama


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
        url = "http://127.0.0.1:11434/api/generate"
        response = requests.post(url, json={"model": self.model_name, "messages": messages, "tools": tools})
        return response.json()
        #TODO: Make it actually work

    


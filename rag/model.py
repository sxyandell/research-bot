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
        response: ChatResponse = chat(model=self.model_name, messages=messages, tools=tools, think=False)
        return response.message


if __name__ == "__main__":
    model = Model("qwen3:8b")
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
    tool_args = response.tool_calls[0].function.arguments
    tool_output = add_numbers(**tool_args)
    print(messages)
    messages.append({"role": "tool", "content": str(tool_output)})
    response = model.chat(messages, tools=[add_numbers])
    print(messages)


















[{"role": "user", "content": "What is 4+4?", "tools": [add_numbers]}, 
 {"role": "assistant", "content": "", "tool_calls": [{"name": "add_numbers", "args": {"num1": 4, "num2": 4}}]},
 {"role": "tool", "content": "8"},
 {
    "role": "assistant",
    "content": "8",
    "tool_calls": [
        {
            "name": "add_numbers",
            "args": {"num1": 4, "num2": 4}
        }
    ]
}]
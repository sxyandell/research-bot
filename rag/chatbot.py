from .prompts import SYSTEM_PROMPT
from typing import Callable, List
from .tools import tool_dict
from .model import Model
from .data_types import Message, ToolCall


class Chatbot:
    def __init__(self, model_name: str, tools: dict[str, Callable] = tool_dict):
        self.tools = tools
        self.model = Model(model_name)
        self.messages: List[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, query: str = None):
        if query:
            self.messages.append(Message(role="user", content=query))
        
        response = self.model.chat(self.messages, tools=list(self.tools.values()))
        
        if response.get('tool_calls'):
            self.messages.append(response)
            for tool_call in response['tool_calls']:
                function_response = self.execute_tool(tool_call)
                self.messages.append(Message(role="tool", name=tool_call['function']['name'], content=str(function_response)))
            return self.chat()
        else:
            self.messages.append(response)
            return response['content']
        
    def execute_tool(self, tool_call: ToolCall):
        function = tool_call['function']
        tool_name = function['name']
        tool_args = function['arguments']
        print(f"Calling tool: {tool_name} with args: {tool_args}")
        tool = self.tools[tool_name]
        function_response = tool(**tool_args)
        return function_response

if __name__ == "__main__":
    chat = Chatbot("qwen3:8b", tool_dict)
    query = input("Enter a query: ")
    while query != "exit":
        print(chat.chat(query))
        query = input("Enter a query: ")
    print(chat.messages)


    
    
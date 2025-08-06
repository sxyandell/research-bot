from prompts import SYSTEM_PROMPT, USER_PROMPT

from typing import Callable
from tools import tool_dict
from model import Model

class Chat:
    def __init__(self, model_name: str, tools: dict[str, Callable] = tool_dict):
        self.tools = tools
        self.model = Model(model_name, tools.values())
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, query: str = None):
        if query:
            self.messages.append({"role": "user", "content": query})
        response = self.model.generate(self.messages)
        self.messages.append(response.message)
        tool_calls = response.message.tool_calls
        if tool_calls:
            self.execute_tools(tool_calls)
            return self.chat(self.messages) #isnt this going to return all the messages in history
        return response.message.content
    
    def execute_tools(self, response):
        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                tool = self.tools[tool_name]
                function_response = tool(**tool_args)
                self.messages.append({"role": "tool", "content": function_response})
        return None



    
    
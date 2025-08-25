#source /home/khwillis@ad.wisc.edu/research-bot/.venv/bin/activate
#python -m dotenv -f .env.local run -- python - <<'PY'
#from rag.chatbot import Chatbot
#chat = Chatbot("gpt-5")
#print(chat.chat("What is the human homolog for the mouse gene Trp53?"))
#PY

from .prompts import SYSTEM_PROMPT
from typing import Callable, List
from .tools import tool_dict
from .model import Model
from .data_types import Message, ToolCall
import json


class Chatbot:
    def __init__(self, model_name: str, tools: dict[str, Callable] = tool_dict):
        self.tools = tools
        self.model = Model(model_name)
        self.messages: List[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Map callable __name__ to the tool callable for resolution when model returns function names
        self._tools_by_callable_name = {fn.__name__: fn for fn in self.tools.values()}

    def chat(self, query: str = None):
        if query:
            self.messages.append(Message(role="user", content=query))
        
        # First attempt: allow the model to choose among all tools
        response = self.model.chat(self.messages, tools=list(self.tools.values()))
        
        if response.get('tool_calls'):
            self.messages.append(response)
            for tool_call in response['tool_calls']:
                function_response = self.execute_tool(tool_call)
                self.messages.append(Message(role="tool", name=tool_call['function']['name'], tool_call_id=tool_call.get('id'), content=str(function_response)))
            return self.chat()
        
        # Fallback: if no tool was selected, force search by exposing only the 'search' tool
        if 'search' in self.tools:
            response_fallback = self.model.chat(self.messages, tools=[self.tools['search']])
            if response_fallback.get('tool_calls'):
                self.messages.append(response_fallback)
                for tool_call in response_fallback['tool_calls']:
                    function_response = self.execute_tool(tool_call)
                    self.messages.append(Message(role="tool", name=tool_call['function']['name'], tool_call_id=tool_call.get('id'), content=str(function_response)))
                return self.chat()
            else:
                self.messages.append(response_fallback)
                return response_fallback.get('content')
        
        # No tools available to force; just return the assistant content
        self.messages.append(response)
        return response.get('content')
        
    def execute_tool(self, tool_call: ToolCall):
        function = tool_call['function']
        tool_name = function['name']
        tool_args_raw = function['arguments']
        tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
        print(f"Calling tool: {tool_name} with args: {tool_args}")
        # Resolve tool by either registered key or callable name
        tool = self.tools.get(tool_name) or self._tools_by_callable_name.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool not found: {tool_name}")
        function_response = tool(**tool_args)
        return function_response

if __name__ == "__main__":
    chat = Chatbot("gpt-5", tool_dict)
    query = input("Enter a query: ")
    while query != "exit":
        print(chat.chat(query))
        query = input("Enter a query: ")
    print(chat.messages)


    
    
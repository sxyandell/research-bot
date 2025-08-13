#TODO: Implement model class

from data_types import Message
from typing import List
from ollama import Client


class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Client()

    def chat(self, messages: List[Message], tools: dict = None):

        # Convert messages to the format expected by Ollama 0.5.1
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                # Handle Message objects if they exist
                formatted_messages.append({
                    "role": getattr(msg, 'role', 'user'),
                    "content": getattr(msg, 'content', str(msg))
                })
        
        # Prepare the chat request
        chat_request = {
            "model": self.model_name,
            "messages": formatted_messages
        }
        
        # Add tools if provided
        if tools:
            chat_request["tools"] = tools
        
        # Make the chat request
        response = self.client.chat(**chat_request)
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
    tool_args = response['tool_calls'][0]['function']['arguments']
    tool_output = add_numbers(**tool_args)
    print(messages)
    messages.append({"role": "tool", "content": str(tool_output)})
    response = model.chat(messages, tools=[add_numbers])
    print(messages)
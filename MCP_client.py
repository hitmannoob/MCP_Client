"""
a weather MCP client
"""

import asyncio
import sys
from typing import Optional
from ast import literal_eval    
from contextlib import AsyncExitStack
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


load_dotenv()  # load environment variables from .env



print("Starting the MCP Client")
class MCPClient:
    """creating a new MCP cle"""

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()
        self.memory = []
        

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query and return the response"""
        human_msg = HumanMessage(query)
        self.memory.append(human_msg)
        conversation_context = []
        
        for msg in self.memory:
            if isinstance(msg, HumanMessage):
                conversation_context.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation_context.append({"role": "assistant", "content": msg.content})

        response = await self.session.list_tools()
        print(response)
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]


        final_text = []
        messages = conversation_context
        assistant_message_content = []


        while True:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=messages,  
                tools=available_tools,  
                tool_choice="auto",  
            )
            if response.choices[0].message.content:
                ai_reply = response.choices[0].message.content
                final_text.append(ai_reply)
                assistant_message_content.append(ai_reply)
                ai_msg = AIMessage(ai_reply)
                messages.append({"role": "assistant", "content": ai_reply})
                break
            elif response.choices[0].message.tool_calls:

                tool_calls = response.choices[0].message.tool_calls

                for tool_call in tool_calls:
                    # Execute the tool
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args),
                                    },
                                    "type": "function",
                                }
                            ],
                        }
                    )
                    # Call the tool
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(
                        f"[Calling tool {tool_name} with args {tool_args}]"
                    )
                    result_string = [res.text for res in result.content]
                    assistant_message_content.append(result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": " ".join(result_string),
                        }
                    )
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

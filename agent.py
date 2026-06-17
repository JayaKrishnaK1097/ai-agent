from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from datetime import date

from tools.database import query_database
#from tools.file_reader import read_file
from tools.mcp_filesystem import get_filesystem_tools
from tools.web_search import search_tool

load_dotenv()
filesystem_tools = get_filesystem_tools()
# Step 1: Define the tools the agent can use
tools = [search_tool, *filesystem_tools, query_database]  # Add the filesystem tools and query_database tools to the list of tools available to the agent

# Step 2: Create the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Step 3: Build a system prompt with today's date computed dynamically
today = date.today().strftime("%B %d, %Y")

system_prompt = f"""You are a helpful assistant.

Today's date is {today}.

You have access to tools that can search the web, read local files, and query a company database. Choose tools based on the question:
- For current events or general web information: use web search
- For file or directory operations: use the filesystem tools (read_text_file to read a file, list_directory to browse, search_files to find files)
- For questions about employees, departments, salaries, or hire dates: use the database query tool

Trust the data returned by your tools. If a tool returns information dated in the future relative to your training data, accept it — your training cutoff is older than today's date."""
# Step 4: Create the agent
agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

# Step 4: Run a test question
if __name__ == "__main__":
    import asyncio

    async def main():
        question = "What files are in the current directory?"
        print(f"Question: {question}\n")

        result = await agent.ainvoke({"messages": [("user", question)]})

        # Print every step the agent took
        print("=" * 60)
        print("Agent execution trace:")
        print("=" * 60)
        for message in result["messages"]:
            msg_type = type(message).__name__
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"\n[{msg_type}] — calling tool(s):")
                for call in message.tool_calls:
                    print(f"  -> {call['name']}({call['args']})")
            elif msg_type == "ToolMessage":
                content_preview = str(message.content)[:200]
                print(f"\n[{msg_type}] — tool returned:")
                print(f"  {content_preview}...")
            else:
                content = message.content if isinstance(message.content, str) else str(message.content)[:300]
                print(f"\n[{msg_type}]: {content}")

        print("\n" + "=" * 60)
        print("Final answer:")
        print("=" * 60)
        final_message = result["messages"][-1]
        content = final_message.content
        if isinstance(content, list):
            # New API returns list of content blocks
            text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            print("\n".join(text_parts))
        else:
            print(content)
    
    asyncio.run(main())
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import date

load_dotenv()

@tool
def read_file(filename: str) -> str:
    """Read the contents of a local text file from the current directory.

    Use this tool when the user asks about the contents of a specific file
    or asks you to summarize, analyze, or quote from a local file.

    Args:
        filename: The name of the file to read (e.g. 'notes.txt')

    Returns:
        The full text contents of the file, or an error message if the file
        cannot be read.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Step 1: Define the tools the agent can use
search_tool = TavilySearch(max_results=3)
tools = [search_tool, read_file]  # Add the read_file tool to the list of tools available to the agent

# Step 2: Create the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Step 3: Build a system prompt with today's date computed dynamically
today = date.today().strftime("%B %d, %Y")

system_prompt = f"""You are a helpful assistant with access to a web search tool.

Today's date is {today}. 

You have access to tools that can search the web and read local files. Choose tools based on the question:
- For current events or general information not in local data: use web search
- For questions about specific local files: use the file reader

Trust the data returned by your tools. If a tool returns information dated in the future relative to your training data, accept it — your training cutoff is older than today's date."""

# Step 4: Create the agent
agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

# Step 4: Run a test question
if __name__ == "__main__":
    question = "What is in sample_notes.txt and what's the weather in Dublin Ohio?"
    print(f"Question: {question}\n")

    result = agent.invoke({"messages": [("user", question)]})

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
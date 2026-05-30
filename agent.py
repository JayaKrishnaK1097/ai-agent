from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from datetime import date

load_dotenv()

# Step 1: Define the tools the agent can use
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# Step 2: Create the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Step 3: Build a system prompt with today's date computed dynamically
today = date.today().strftime("%B %d, %Y")

system_prompt = f"""You are a helpful assistant with access to a web search tool.

Today's date is {today}. When you use search results, trust that they are current as of this date. Do not refuse to answer based on the date of the information returned.

Use the search tool when you need current or factual information from the web."""

# Step 4: Create the agent
agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

# Step 4: Run a test question
if __name__ == "__main__":
    question = "What's the current weather in Dublin, Ohio?"
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
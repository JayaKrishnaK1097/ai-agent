from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import date
import os
import pyodbc

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

@tool
def query_database(sql_query: str) -> str:
    """Execute a SQL SELECT query against the company database and return the results.

    Use this tool when the user asks about employee data — names, departments, roles,
    salaries, or hire dates. The database has one table called 'employees' with columns:
    id (int), name (text), department (text), role (text), salary (int), hire_date (date).

    Only SELECT queries are allowed. Any INSERT, UPDATE, DELETE, or DROP will be rejected.

    Args:
        sql_query: A valid SQL SELECT statement (e.g., "SELECT name, salary FROM employees WHERE department = 'Engineering'")

    Returns:
        Query results formatted as text, or an error message if the query fails.
    """
    # Safety guard: only allow SELECT queries
    query_upper = sql_query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed. INSERT, UPDATE, DELETE, and DROP are blocked."

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE"]
    for word in forbidden:
        if word in query_upper:
            return f"Error: Query contains forbidden keyword '{word}'."

    try:
        conn_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('DB_SERVER')};"
            f"DATABASE={os.getenv('DB_NAME')};"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASSWORD')}"
        )
        with pyodbc.connect(conn_string) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                return "Query executed successfully but returned no rows."

            # Format as readable text
            result = " | ".join(columns) + "\n"
            result += "-" * len(result) + "\n"
            for row in rows:
                result += " | ".join(str(v) for v in row) + "\n"
            return result
    except Exception as e:
        return f"Database error: {str(e)}"

# Step 1: Define the tools the agent can use
search_tool = TavilySearch(max_results=3)
tools = [search_tool, read_file, query_database]  # Add the read_file and query_database tools to the list of tools available to the agent

# Step 2: Create the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Step 3: Build a system prompt with today's date computed dynamically
today = date.today().strftime("%B %d, %Y")

system_prompt = f"""You are a helpful assistant.

Today's date is {today}.

You have access to tools that can search the web, read local files, and query a company database. Choose tools based on the question:
- For current events or general web information: use web search
- For questions about specific local files: use the file reader
- For questions about employees, departments, salaries, or hire dates: use the database query tool

Trust the data returned by your tools. If a tool returns information dated in the future relative to your training data, accept it — your training cutoff is older than today's date."""
# Step 4: Create the agent
agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

# Step 4: Run a test question
if __name__ == "__main__":
    question = "Who is the highest-paid employee in the Engineering department, and when were they hired?"
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
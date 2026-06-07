import os
import pyodbc
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def query_database(sql_query: str) -> str:
    """Execute a SQL SELECT query against the company database in Microsoft SQL Server and return the results.

    Use this tool when the user asks about employee data — names, departments, roles,
    salaries, or hire dates. The database has a table called 'employees' with columns:
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
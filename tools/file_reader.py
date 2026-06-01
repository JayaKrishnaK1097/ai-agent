from langchain_core.tools import tool
from dotenv import load_dotenv

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
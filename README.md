# AI Agent

An AI agent that picks the right tool based on the question. Built with Google Gemini, LangChain, and FastAPI.

## What it does

- Searches the web via Tavily
- Reads local text files
- Queries a SQL Server database with natural language
- Auto-detects user location for location-based questions
- Exposes everything as a REST API

## Example questions

| Question | Tools called |
|----------|------|
| "What's the weather in Dublin Ohio?" | web search |
| "What's in sample_notes.txt?" | file reader |
| "Who is the highest-paid Engineer?" | database query |
| "Summarize the file and tell me the weather here" | file reader + web search |

## Tech stack

Python, LangChain v1, Google Gemini 2.5 Flash, Tavily, Microsoft SQL Server, FastAPI.

## Setup

You need Python 3.11+, a SQL Server instance, and free API keys from Google AI Studio and Tavily.

```bash
git clone https://github.com/JayaKrishnaK1097/ai-agent.git
cd ai-agent
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-google-genai langchain-tavily langchain-classic langgraph fastapi uvicorn pyodbc python-dotenv requests
```

Create a `.env` file with your own keys and DB credentials.

## Run it

```bash
# CLI
python agent.py

# API
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test.

## API example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the highest paid Engineer?"}'
```

Response:

```json
{
  "answer": "Marcus Chen is the highest-paid Engineer.",
  "tools_used": ["query_database"]
}
```

## License

MIT
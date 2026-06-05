from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent
from cache import ResponseCache

app = FastAPI(
    title="AI Agent API",
    description="A multi-tool AI agent that uses web search, file reading, and database queries to answer questions"
)
response_cache = ResponseCache(ttl=300)  # Cache entries expire after 5 minutes

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    cache_hit: bool
    tools_used: list[str]

def normalize_question(question: str) -> str:
    cleaned = question.strip().lower()
    cleaned = cleaned.rstrip("?!.")
    return cleaned

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/cache-stats")
async def cache_stats():
    return response_cache.stats()

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = response_cache.get(normalize_question(request.question))
    if result is not None:
        return AnswerResponse(answer=result["answer"], cache_hit=True, tools_used=result.get("tools_used", []))
    
    try:    
        result = agent.invoke({"messages": [("user", request.question)]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    
    tools_used = []
    for message in result["messages"]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for call in message.tool_calls:
                tools_used.append(call["name"])

    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, list):
        text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        answer_text = "\n".join(text_parts)
    else:
        answer_text = str(content)

    response_cache.set(normalize_question(request.question), {"answer": answer_text, "tools_used": tools_used})
        
    return AnswerResponse(answer=answer_text, cache_hit=False, tools_used=tools_used)
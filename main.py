from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent

# Cache implementation
from cache import ResponseCache

# Logging Setup
import logging
import uuid
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai-agent")

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
    
    # Generate a unique request ID for logging and traceability
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    logger.info(f"event=request_started request_id=[{request_id}] question={request.question!r}")

    result = response_cache.get(normalize_question(request.question))
    if result is not None:
        
        # Log the cache hit with latency and request ID for traceability.
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"event=request_completed request_id=[{request_id}] cache_hit=true tools_used=none latency={latency:.2f} ms")

        return AnswerResponse(answer=result["answer"], cache_hit=True, tools_used=result.get("tools_used", []))
    
    try:    
        result = agent.invoke({"messages": [("user", request.question)]})
    except Exception as e:

        # Log the error with latency and request ID for debugging.  
        latency = (time.perf_counter() - start_time) * 1000
        logger.error(f"event=request_failed request_id=[{request_id}] error_message={str(e)} latency={latency:.2f} ms")
        
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

    # Log the cache miss with latency, tools used, and request ID for traceability.
    latency = (time.perf_counter() - start_time) * 1000
    logger.info(f"event=request_completed request_id=[{request_id}] cache_hit=false tools_used={tools_used} latency={latency:.2f} ms")

    return AnswerResponse(answer=answer_text, cache_hit=False, tools_used=tools_used)
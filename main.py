import time, os, uuid, psutil
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from agents.main_agent import Agent
from observability.monitoring import process_request
from guardrails.safety import validate_input, validate_output

_START_TIME = time.time()
_REQUEST_COUNT = 0
_AGENT_NAME = "Agent"  # replace with actual agent name
_AGENT_VERSION = "1.0.0"

app = FastAPI(title=_AGENT_NAME)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

agent = Agent()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class RunRequest(BaseModel):
    input: str
    context: Optional[Dict[str, Any]] = {}

@app.get("/health")
@process_request()
async def health():
    return {"status": "ok", "agent": _AGENT_NAME, "version": _AGENT_VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.post("/chat")
@process_request()
async def chat(req: ChatRequest):
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    session_id = req.session_id or str(uuid.uuid4())
    message = validate_input(req.message)
    response = await agent.chat(message)
    response = validate_output(response)
    return {"response": response, "session_id": session_id}

@app.post("/run")
@process_request()
async def run(req: RunRequest):
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    task_id = str(uuid.uuid4())
    input = validate_input(req.input)
    result = await agent.chat(input)
    result = validate_output(result)
    return {"ok": True, "task_id": task_id, "result": result,
            "input": req.input, "completed_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/info")
@process_request()
def info():
    from tools.tool_manager import get_tools
    tools = [tool.__name__ for tool in get_tools()]
    return {
        "name": _AGENT_NAME,
        "version": _AGENT_VERSION,
        "description": "AI agent powered by the KRE platform. RAG Based Chatbot answers questions using retrieved information from a knowledge base, providing accurate and contextually relevant responses. It leverages retrieval-augmented generation to ground its answers in reliable sources.",
        "capabilities": ["chat", "task_execution", "rag", "tool_use"],
        "tools": tools,
        "endpoints": [
            {"method": "GET",  "path": "/health", "description": "Liveness check"},
            {"method": "POST", "path": "/chat",   "description": "Chat with the agent"},
            {"method": "POST", "path": "/run",    "description": "Run an agent task"},
            {"method": "GET",  "path": "/info",   "description": "Agent metadata"},
            {"method": "GET",  "path": "/status", "description": "Runtime status"},
        ],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/status")
@process_request()
def status():
    uptime = time.time() - _START_TIME
    try:
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1_048_576
    except Exception:
        mem_mb = 0.0
    return {
        "status": "running",
        "agent": _AGENT_NAME,
        "version": _AGENT_VERSION,
        "uptime_seconds": round(uptime, 2),
        "memory_mb": round(mem_mb, 2),
        "requests_total": _REQUEST_COUNT,
        "started_at": datetime.utcfromtimestamp(_START_TIME).isoformat() + "Z",
    }

if __name__ == "__main__":
    import uvicorn, os, socket
    def _port_free(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", p)) != 0
    port = int(os.environ.get("PORT", 8081))
    while not _port_free(port):
        print(f"Port {port} is already in use.")
        try:
            ans = input(f"Try port {port + 1} instead? [Y/n]: ").strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("", "y", "yes"):
            port += 1
        else:
            print("Exiting. Free the port and try again.")
            raise SystemExit(1)
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
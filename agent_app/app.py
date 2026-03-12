"""
FastAPI server for the Telecom Retention Agent Databricks App.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import create_agent

app = FastAPI(title="Telecom Retention Agent", version="1.0.0")

# Create agent once at startup
agent = create_agent()


class ChatRequest(BaseModel):
    messages: list[dict]


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = agent.invoke({"messages": request.messages})
        response_text = result["messages"][-1].content
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

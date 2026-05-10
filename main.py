"""
main.py

FastAPI app. Two endpoints:
  GET  /health  -> {"status": "ok"}
  POST /chat    -> agent response

"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel, field_validator # type: ignore

from search import ensure_index
from agent import chat


# Request / Response models 
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# App lifecycle 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm up the embedding model + FAISS index at startup
    # this takes a few seconds but means the first request is fast
    print("[startup] building FAISS index...")
    ensure_index()
    print("[startup] ready")
    yield
    # nothing to clean up


app = FastAPI(
    title="SHL Assessment Recommender",
    lifespan=lifespan,
)


# Endpoints

@app.get("/health")
def health():
    """Simple readiness check. Returns 200 when the service is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint. Takes full conversation history, returns agent reply.

    The request carries ALL messages so far (user + assistant turns).
    We're stateless — we don't store anything between calls.
    """
    # convert pydantic models to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # basic sanity check — last message should be from the user
    if messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from the user"
        )

    try:
        result = chat(messages)
    except Exception as e:
        import traceback
        print(f"[chat] unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal error generating response"
        )

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**r) for r in result["recommendations"]
        ],
        end_of_conversation=result["end_of_conversation"],
    )

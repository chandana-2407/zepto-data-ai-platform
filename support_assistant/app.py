from fastapi import FastAPI
from pydantic import BaseModel

from graph import graph
from schemas import SupportResponse


# Create FastAPI application
app = FastAPI(
    title="Zepto Support Assistant",
    description="GenAI Support Assistant for Zepto",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    query: str


# ============================================================
# POST /ask
# ============================================================

@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest):

    # Run LangGraph
    result = graph.invoke({
        "query": request.query
    })

    # Convert graph result into required JSON schema
    response = SupportResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )

    return response


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Zepto Support Assistant is running"
    }
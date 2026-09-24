import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from assistant import search_documents


# ============================================================
# MOCK LLM TOGGLE
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# STATE
# ============================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    documents: list
    answer: str
    sources: list
    confidence: float


# ============================================================
# NODE 1 - CLASSIFY INTENT
# ============================================================

def classify_intent(state: SupportState):

    query = state["query"].lower()

    if MOCK_LLM == "1":

        policy_keywords = [
            "delivery",
            "return",
            "refund",
            "membership",
            "tracking",
            "cancel",
            "gift card",
            "support hours"
        ]

        if any(keyword in query for keyword in policy_keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:
        # Optional real LLM extension
        # For the graded baseline, MOCK_LLM=1 is used.
        intent = "policy_question"

    return {
        "intent": intent
    }


# ============================================================
# NODE 2 - RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: SupportState):

    query = state["query"]

    # Retrieval always runs for policy questions
    results = search_documents(query, top_k=3)

    documents = results.get("documents", [[]])[0]

    # Get document IDs
    ids = results.get("ids", [[]])[0]

    if MOCK_LLM == "1":

        if documents:

            top_chunk = documents[0]

            # Short snippet from retrieved document
            print("DEBUG DOCUMENT:", repr(top_chunk))
            snippet = top_chunk[:200]

            answer = f"Based on the retrieved context: {snippet}"

            sources = ids

            confidence = 1.0

        else:

            answer = "No relevant policy information was found."

            sources = []

            confidence = 0.0

    else:
        # Optional real LLM extension
        answer = (
            "The retrieved policy information should be used "
            "to generate the final answer."
        )

        sources = ids
        confidence = 1.0 if documents else 0.0

    return {
        "documents": documents,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


# ============================================================
# NODE 3 - DIRECT ANSWER
# ============================================================

def direct_answer(state: SupportState):

    if MOCK_LLM == "1":

        answer = "I can only answer questions about Zepto policies right now."

    else:

        # Optional real LLM extension
        answer = (
            "I can help answer your question using the available "
            "Zepto support information."
        )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_question(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(SupportState)

# Add required 3 nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

# Start -> classify
builder.add_edge(START, "classify_intent")

# Conditional routing
builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

# End nodes
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

# Compile graph
graph = builder.compile()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = "What is the return policy?"

    result = graph.invoke({
        "query": question
    })

    print("\nUser Question:")
    print(question)

    print("\nIntent:")
    print(result["intent"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    print("\nConfidence:")
    print(result["confidence"])
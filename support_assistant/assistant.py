import os
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Get existing collection
collection = client.get_collection(name="zepto_policies")


def search_documents(query, top_k=3):
    # Convert user question into embedding
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results


def classify_intent(query):
    """
    Classify the user query as:
    - policy_question
    - general_question

    MOCK_LLM graded baseline:
    No LLM call is made.
    """

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

    query_lower = query.lower()

    for keyword in policy_keywords:
        if keyword in query_lower:
            return "policy_question"

    return "general_question"

def retrieve_and_answer(query):
    """
    Retrieve the top-3 similar chunks from ChromaDB
    and generate the required mock-mode answer.
    """

    results = search_documents(query, top_k=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Use the most similar chunk
    top_chunk = documents[0]

    # Short snippet: first 200 characters
    top_chunk_snippet = top_chunk[:200]

    answer = f"Based on the retrieved context: {top_chunk_snippet}"

    # Collect source document IDs
    sources = []

    for metadata in metadatas:
        sources.append(metadata["source"])

    return {
        "answer": answer,
        "sources": sources,
        "confidence": 1.0
    }

# Test
if __name__ == "__main__":
    question = "What is the return policy?"

    intent = classify_intent(question)

    print("\nUser Question:")
    print(question)

    print("\nIntent:")
    print(intent)

    if intent == "policy_question":
        results = search_documents(question)

        print("\nRelevant Documents:")

        for i, document in enumerate(results["documents"][0]):
            print(f"\n--- Result {i + 1} ---")
            print(document)
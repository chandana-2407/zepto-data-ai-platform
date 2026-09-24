import os
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Load embedding model locally
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create/get collection
collection = client.get_or_create_collection(
    name="zepto_policies"
)

# Read all documents
documents = []
ids = []
metadatas = []

for filename in sorted(os.listdir(DOCS_DIR)):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        documents.append(text)
        ids.append(filename.replace(".txt", ""))
        metadatas.append({"source": filename})

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Successfully indexed {len(documents)} documents.")
print(f"Collection count: {collection.count()}")
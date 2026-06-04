import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

print("--- Starting Search Test (Python 3.13 Optimized) ---")

# 1. Initialize the client
client = QdrantClient(path="data/qdrant_db")

# 2. Load the model
print("Loading model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 3. Define the query
query = "What are the reporting requirements for money market funds?"
print(f"Searching for: {query}")

# 4. Generate the vector
query_vector = model.encode(query).tolist()

# 5. Search using the modern 'query_points' method
# This method is the new standard and avoids the AttributeError in Python 3.13
response = client.query_points(
    collection_name="lux_circulars",
    query=query_vector,
    limit=3
)

# 6. Results
# In query_points, the results are in the '.points' attribute
if not response.points:
    print("No results found.")
else:
    print(f"\n✅ Found {len(response.points)} matches:")
    for i, res in enumerate(response.points):
        print(f"\nMatch {i+1} (Score: {res.score:.4f})")
        source = res.payload.get("source", "Unknown")
        page = res.payload.get("page", "Unknown")
        text = res.payload.get("text", "")
        
        print(f"File: {source} | Page: {page}")
        print(f"Text Snippet: {text[:300]}...")
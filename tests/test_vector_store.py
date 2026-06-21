import chromadb

client = chromadb.PersistentClient(
    path="vector_store/chroma_db"
)

collection = client.get_collection(
    "cfpb_complaints"
)

results = collection.query(
    query_texts=[
        "credit card fees"
    ],
    n_results=3,
)

print(results["documents"][0])
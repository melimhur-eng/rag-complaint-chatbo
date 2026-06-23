import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_STORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store"))

def get_retriever(vector_store_dir=DEFAULT_VECTOR_STORE_DIR, k=5):
    """
    Loads the persisted ChromaDB vector store and returns a retriever instance.
    
    Args:
        vector_store_dir (str): Path to the persisted vector database directory.
        k (int): Number of top relevant text chunks to retrieve.
        
    Returns:
        VectorStoreRetriever: LangChain retriever object configured for similarity search.
    """
    if not os.path.exists(vector_store_dir):
        raise FileNotFoundError(
            f"Vector store directory not found at: {vector_store_dir}. "
            f"Please verify your Task 2 execution or your pre-built data path."
        )
        
    print(f"[Retriever] Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"[Retriever] Loading persisted vector store from: {vector_store_dir}...")
    vector_store = Chroma(
        persist_directory=vector_store_dir,
        embedding_function=embeddings
    )
    
    # Configure similarity search to return the top-k most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever

if __name__ == "__main__":
    # Quick standalone test of the retriever functionality
    try:
        sample_retriever = get_retriever()
        sample_query = "Why are people unhappy with Credit Cards?"
        print(f"\nTesting retriever with query: '{sample_query}'")
        retrieved_docs = sample_retriever.invoke(sample_query)
        
        print(f"Successfully retrieved {len(retrieved_docs)} document chunks:")
        for idx, doc in enumerate(retrieved_docs, 1):
            metadata = doc.metadata
            print(f"  Chunk {idx} | Product: {metadata.get('product_category', 'N/A')} | Complaint ID: {metadata.get('complaint_id', 'N/A')}")
            print(f"  Text excerpt: {doc.page_content[:100]}...\n")
    except Exception as e:
        print(f"Retriever test failed: {e}")
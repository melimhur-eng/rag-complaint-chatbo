## Task 2: Text Chunking, Embedding and Indexing

### Sampling Strategy

A proportional stratified sample of 12,000 complaints was created from the filtered CFPB dataset to preserve the original product distribution.

### Chunking Strategy

Complaint narratives were split using LangChain's RecursiveCharacterTextSplitter with:

- chunk_size = 500
- chunk_overlap = 50

### Embedding Model

sentence-transformers/all-MiniLM-L6-v2

The model generates 384-dimensional embeddings suitable for semantic retrieval while remaining computationally efficient.

### Vector Store

ChromaDB was used as the persistent vector database.

Each chunk stores:

- complaint_id
- product_category
- chunk_index
- total_chunks
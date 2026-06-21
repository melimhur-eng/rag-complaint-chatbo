import pandas as pd
import chromadb
from tqdm import tqdm

from chunking import (
    get_text_splitter,
    create_chunk_records,
)

from embedding import (
    load_embedding_model,
    generate_embeddings,
)


def build_vector_store():

    print("Loading sample dataset...")

    df = pd.read_csv(
        "data/sample_complaints.csv"
    )

    print(f"Loaded {len(df)} complaints")

    splitter = get_text_splitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    print("Creating chunks...")

    records = create_chunk_records(
        df,
        splitter,
    )

    print(f"Created {len(records)} chunks")

    model = load_embedding_model()

    client = chromadb.PersistentClient(
        path="vector_store/chroma_db"
    )

    collection = client.get_or_create_collection(
        name="cfpb_complaints"
    )

    BATCH_SIZE = 1000

    for start in tqdm(
        range(0, len(records), BATCH_SIZE)
    ):

        batch = records[
            start:start + BATCH_SIZE
        ]

        texts = [
            r["text"]
            for r in batch
        ]

        ids = [
            r["chunk_id"]
            for r in batch
        ]

        metadata = [
            {
                "complaint_id": str(
                    r["complaint_id"]
                ),
                "product_category":
                    r["product_category"],
                "chunk_index":
                    r["chunk_index"],
                "total_chunks":
                    r["total_chunks"],
            }
            for r in batch
        ]

        embeddings = generate_embeddings(
            texts,
            model,
        )

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata,
        )

    print(
        "Vector store created successfully!"
    )


if __name__ == "__main__":
    build_vector_store()
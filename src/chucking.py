# src/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_splitter(
    chunk_size=500,
    chunk_overlap=50,
):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


def create_chunk_records(df, splitter):
    records = []

    for _, row in df.iterrows():

        complaint_id = row["Complaint ID"]
        product = row["Product"]
        text = str(row["Consumer complaint narrative"])

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):

            records.append(
                {
                    "chunk_id": f"{complaint_id}_{idx}",
                    "text": chunk,
                    "complaint_id": complaint_id,
                    "product_category": product,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            )

    return records
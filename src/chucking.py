# src/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_text_splitter(
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
            "",
        ],
    )
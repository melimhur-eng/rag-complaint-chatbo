import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration Paths (assuming execution from within the 'src' directory)
DATA_PATH = "./data/processed/filtered_complaints.csv"
VECTOR_STORE_PATH = "./vector_store"

def create_stratified_sample(df, sample_size=10000):
    """
    Creates a stratified sample to ensure proportional representation 
    across product categories based on the 'Product' column.
    """
    # Calculate proportions using the exact header name 'Product'
    proportions = df['Product'].value_counts(normalize=True)
    
    sampled_df = pd.DataFrame()
    for category, proportion in proportions.items():
        category_df = df[df['Product'] == category]
        n_samples = int(sample_size * proportion)
        
        # Only sample if there are enough records, otherwise take all available
        if len(category_df) >= n_samples:
            sampled_df = pd.concat([sampled_df, category_df.sample(n=n_samples, random_state=42)])
        else:
            sampled_df = pd.concat([sampled_df, category_df])
            
    return sampled_df

def main():
    print("Loading filtered complaints dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find file at {DATA_PATH}. Ensure Task 1 saved it there.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Standardize key tracking columns into metadata fields required by Task 2 & 3
    print("Aligning metadata column fields...")
    df['product_category'] = df['Product']
    df['complaint_id'] = df['Complaint ID']
    
    # 1. Stratified Sampling (10K - 15K target size)
    print("Creating a stratified sample of 10,000 complaints...")
    sampled_df = create_stratified_sample(df, sample_size=10000)
    
    # 2. Document Loading & Text Chunking
    print("Loading narratives into LangChain documents...")
    # Uses your exact text column header: 'Consumer complaint narrative'
    loader = DataFrameLoader(
        sampled_df, 
        page_content_column="Consumer complaint narrative"
    )
    documents = loader.load()
    
    print("Splitting long text narratives into chunks...")
    # Using specifications from the challenge document: chunk size 500, overlap 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     
        chunk_overlap=50,   
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully generated {len(chunks):,} text chunks.")

    # 3. Choose Embedding Model
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4 & 5. Generate Embeddings and Save Vector Store
    print("Generating embeddings and building ChromaDB index. This may take a few minutes...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print(f"Vector store successfully built and persisted to {VECTOR_STORE_PATH}!")

if __name__ == "__main__":
    # Ensure the target directory exists
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    main()
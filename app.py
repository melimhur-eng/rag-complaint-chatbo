import streamlit as st
import sys
import os

# Ensure the app can find modules inside the 'src' directory
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# --- CRITICAL: IMPORT YOUR WORKING LOGIC FROM TASK 3 ---
# Replace these with your actual function/variable names from your src files
# Example:
# from your_retriever_module import retriever 
# from generator import run_rag_generation

# For demonstration, placeholder functions are provided below. 
# Connect these to your actual vector store retriever and LLM chain logic.

# --- CONNECT YOUR ACTUAL TASK 3 BACKEND LOGIC HERE ---
# (Adjust these import statements to match your exact filenames in src/)
# from evaluate_rag import retriever         # Or wherever your initialized vector database retriever lives
# from generator import run_rag_generation   # Your working conversational RAG chain function


import streamlit as st
import sys
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # Swap to FAISS if you used FAISS in Task 3

# Ensure system path can see your src folder for your custom LLM endpoint
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import HuggingFaceChatEndpoint  # Import your custom LLM class from Task 3

# --- CACHED RETRIEVER INITIALIZATION ---
@st.cache_resource
def load_retriever():
    """
    Loads the pre-built vector store once and keeps it warm in memory.
    """
    # 1. Instantiate the exact same embedding model used in Tasks 2 & 3
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load the database from the designated project path
    persist_directory = os.path.join(os.path.dirname(__file__), "vector_store")
    
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    # 3. Configure it as a retriever (fetching top k=5 context blocks)
    return db.as_retriever(search_kwargs={"k": 5})

# Initialize the warm retriever instance
retriever = load_retriever()


# --- STREAMLIT EXECUTION FUNCTION ---
def get_rag_context_and_answer(query):
    try:
        # 1. Fetch live chunks matching the question from your real ChromaDB instance
        retrieved_docs = retriever.invoke(query)
        
        # 2. Re-initialize your working custom LLM configuration from Task 3
        # Ensure you have your environment token set up!
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        
        llm = HuggingFaceChatEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            max_new_tokens=512,
            huggingfacehub_api_token=token
        )
        
        # 3. Build your prompt template exactly as defined in your Task 3 specifications
        prompt_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""

        # Format context blocks seamlessly into the template
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        full_prompt = prompt_template.format(context=context_text, question=query)
        
        # 4. Generate context-grounded response
        answer = llm.invoke(full_prompt)
        
        return answer, retrieved_docs
        
    except Exception as e:
        st.error(f"Error during live RAG execution: {e}")
        return f"An error occurred while running the pipeline: {str(e)}", []
    

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="wide")

st.title("💼 CrediTrust Intelligent Complaint Analysis Bot")
st.markdown(
    "Empowering Product, Support, and Compliance teams to seamlessly analyze "
    "customer pain points across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers[cite: 25]."
)

# Initialize Session State to track active responses across user actions
if "generated_answer" not in st.session_state:
    st.session_state.generated_answer = ""
if "source_chunks" not in st.session_state:
    st.session_state.source_chunks = []

# --- RUBRIC ITEM #2: Text Input Box ---
user_query = st.text_input(
    label="Ask a plain-English question about customer complaints:",
    placeholder="e.g., Is there any mention of fraudulent security breaches on user accounts?",
    key="query_input"
)

# Create side-by-side action buttons
col1, col2, _ = st.columns([2, 2, 8])

with col1:
    # --- RUBRIC ITEM #2: Submit/Ask Button ---
    submit_pressed = st.button("Submit Question", use_container_width=True)

with col2:
    # --- RUBRIC ITEM #5: 'Clear' Button to reset conversation ---
    clear_pressed = st.button("Clear Conversation", use_container_width=True)

# Process Clear Action (Rubric Item #5)
if clear_pressed:
    st.session_state.generated_answer = ""
    st.session_state.source_chunks = []
    st.rerun()

# Process Submit Action
if submit_pressed and user_query:
    if user_query.strip() == "":
        st.warning("Please enter a valid question before submitting.")
    else:
        with st.spinner("Searching database and analyzing customer complaints..."):
            # Execute your Task 3 core pipeline logic
            answer, sources = get_rag_context_and_answer(user_query)
            
            # Persist results in session state
            st.session_state.generated_answer = answer
            st.session_state.source_chunks = sources

# --- DISPLAY LOGIC ---
if st.session_state.generated_answer:
    st.markdown("---")
    
    # --- RUBRIC ITEM #3: Display Area for AI-Generated Answer ---
    st.subheader("💡 AI-Generated Answer")
    st.info(st.session_state.generated_answer)
    
    # --- RUBRIC ITEM #4: Retrieved Source Text Chunks Displayed BELOW ---
    st.subheader("🔍 Grounding Sources (Retrieved Text Chunks)")
    st.markdown(
        "*The data chunks below were extracted from the vector store and used "
        "by the LLM to synthesize the response above.*"
    )
    
    # Render source blocks cleanly for verification and audit transparency
    for idx, doc in enumerate(st.session_state.source_chunks, 1):
        # Gracefully handles both LangChain Document objects or standard dictionaries
        content = doc.page_content if hasattr(doc, 'page_content') else doc.get("page_content", "")
        metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get("metadata", {})
        c_id = metadata.get("complaint_id", "N/A")
        prod = metadata.get("product_category", "General")
        
        with st.expander(f"Source Chunk #{idx} | Product: {prod} (Complaint ID: {c_id}) [cite: 48, 49]"):
            st.write(content)
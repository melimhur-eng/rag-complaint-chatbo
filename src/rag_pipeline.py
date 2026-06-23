import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient

class HuggingFaceChatEndpoint(LLM):
    """
    A custom LangChain LLM wrapper that forces Hugging Face to use 
    the chat/conversational endpoint instead of legacy text-generation.
    """
    repo_id: str
    temperature: float = 0.1
    max_new_tokens: int = 512
    huggingfacehub_api_token: str

    @property
    def _llm_type(self) -> str:
        return "hf_chat_endpoint"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        # Initialize the native Hugging Face client
        client = InferenceClient(api_key=self.huggingfacehub_api_token)
        
        # This natively maps to task: conversational (chat.completions)
        response = client.chat.completions.create(
            model=self.repo_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content

# -------------------------------------------------------------------------
# PATH AND CONFIGURATION SETUP
# -------------------------------------------------------------------------
# Dynamically locate the project root directory from this file's position
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Free serverless endpoint; ensures factual responses with low temperature
# LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3" 
LLM_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct" 

# -------------------------------------------------------------------------
# RUBRIC 1: Retriever Implementation
# -------------------------------------------------------------------------
def get_retriever(k=5):
    """
    Initializes the embedding model and loads the pre-built vector store 
    from the root directory to perform similarity searches.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store directory not found at: {VECTOR_STORE_PATH}. "
            f"Ensure the pre-built ChromaDB folder is unzipped directly into the repository root."
        )
    
    # Re-use the exact same embedding model from Task 2
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Load the persisted Chroma store
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=embeddings
    )
    
    # Return as a retriever configured for the top-k most relevant text chunks
    return vector_store.as_retriever(search_kwargs={"k": k})

# -------------------------------------------------------------------------
# RUBRIC 2: Defined Prompt Template
# -------------------------------------------------------------------------
def get_prompt_template():
    """
    Returns the strict prompt template instructing the model to act as a 
    financial analyst assistant for CrediTrust and restrict answers to context.
    """
    template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""
    return PromptTemplate.from_template(template)

# -------------------------------------------------------------------------
# RUBRIC 3: Generator Implementation
# -------------------------------------------------------------------------
def generate_response(question, retriever, llm_token=None):
    """
    Combines the prompt, user question, and retrieved chunks to send them 
    to the LLM and return the generated answer along with the source context records.
    """
    # 1. Retrieve the source documents
    retrieved_docs = retriever.invoke(question)
    
    # Formatter to join page contents for context placeholder
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. Set up the LLM generator model
    # Token fallback checks environment variable if not passed explicitly
    token = llm_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Hugging Face API token is missing. Set HUGGINGFACEHUB_API_TOKEN in your environment.")
        
    llm = HuggingFaceChatEndpoint(
        repo_id=LLM_REPO_ID,
        temperature=0.1,  # Strict adherence to facts
        max_new_tokens=512,
        huggingfacehub_api_token=token
    )
    
    # 3. Construct the chain using LangChain Expression Language (LCEL)
    prompt = get_prompt_template()
    chain = prompt | llm | StrOutputParser()
    
    # 4. Invoke the generation process
    answer = chain.invoke({"context": context_text, "question": question})
    
    return {
        "answer": answer.strip(),
        "source_documents": retrieved_docs
    }
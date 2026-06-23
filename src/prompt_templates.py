from langchain_core.prompts import PromptTemplate

# Strict prompt template aligning perfectly with the challenge rubric
RAG_PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""

def get_rag_prompt():
    """
    Returns the compiled LangChain PromptTemplate instance.
    """
    return PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
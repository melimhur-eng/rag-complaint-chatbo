import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def get_prompt_template():
    """
    Defines the prompt template grounded in context matching Task 3 specs.
    """
    return ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.\n"
            "Use the following retrieved complaint excerpts to formulate your answer.\n"
            "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}")
    ])

def run_rag_generation(question, context_documents):
    """
    Executes generation through Hugging Face's OpenAI-compatible conversational router.
    """
    # Format retrieved LangChain document chunks into a single text block
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is missing.")
        
    # Use ChatOpenAI pointing to Hugging Face's router. 
    # This automatically guarantees the 'conversational' task endpoint is targeted.
    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
        model="meta-llama/Llama-3.1-8B-Instruct",  # Works flawlessly here
        temperature=0.1,
        max_tokens=512
    )
    
    prompt = get_prompt_template()
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain.invoke({"context": context_text, "question": question})
import os
from dotenv import  load_dotenv
from rag_pipeline import get_retriever, generate_response


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__abspath__ if '__abspath__' in locals() else __file__)), ".env"))


def run_evaluation():
    print("=" * 60)
    print("CREDITRUST RAG SYSTEM EVALUATION RUNNER")
    print("=" * 60)
    
    # Initialize the retriever
    try:
        retriever = get_retriever(k=5)
    except Exception as e:
        print(f"[Initialization Error] {e}")
        return

    # Representative test queries targeting different product categories
    test_queries = [
        "Why are people unhappy with Credit Cards?",
        "Are there recurring issues with missing money transfers?",
        "What are the main complaints regarding personal loan interest rates?",
        "What customer difficulties are reported regarding savings account withdrawals?",
        "Is there any mention of fraudulent security breaches on user accounts?"
    ]
    
    # Execute and log each query
    for idx, query in enumerate(test_queries, 1):
        print(f"\nQuery #{idx}: '{query}'")
        print("-" * 50)
        
        try:
            result = generate_response(query, retriever)
            
            print(f"Generated Answer:\n{result['answer']}\n")
            
            print("Retrieved Source Chunks:")
            for s_idx, doc in enumerate(result['source_documents'], 1):
                p_cat = doc.metadata.get('product_category', 'Unknown')
                c_id = doc.metadata.get('complaint_id', 'Unknown')
                text_snippet = doc.page_content[:120].replace('\n', ' ')
                print(f"  [{s_idx}] [Prod: {p_cat} | ID: {c_id}] {text_snippet}...")
                
        except Exception as e:
            print(f"Execution Error processing query: {e}")
            
        print("=" * 60)

if __name__ == "__main__":
    # Example token set up if you choose not to use your terminal environment
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_token_here"
    run_evaluation()
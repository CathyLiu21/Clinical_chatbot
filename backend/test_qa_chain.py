from qa_chain import QAChain
import time

def main():
    # Initialize QA chain
    print("Initializing QA chain...")
    qa_chain = QAChain()

    # Test questions
    test_questions = [
        "What is the main finding of this study regarding HER-2/neu amplification and breast cancer outcomes?",
        "How was HER-2/neu gene amplification measured in this study?",
        "What is the relationship between HER-2/neu amplification and lymph node status?",
        "What are the clinical implications of these findings for breast cancer treatment?"
    ]

    print("\nTesting Q&A for each question...")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        start_time = time.time()
        result = qa_chain.answer(question)
        query_time = time.time() - start_time
        
        print(f"Answer generated in {query_time:.2f} seconds:")
        print(result['answer'])
        print("\nSources:")
        for src in result['sources']:
            print(f"\n[{src['metadata']['source'].upper()}] {src['content'][:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()

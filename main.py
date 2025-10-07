import os
import hashlib
from dotenv import load_dotenv
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    from langgraph_checkpoint_sqlite import SqliteSaver
from graphs.workflow import build_workflow

# Task E: Load environment variables for tracing
load_dotenv()

# Verify tracing configuration is loaded
print("=== Configuration Status ===")
print(f"LangSmith Tracing: {os.getenv('LANGCHAIN_TRACING_V2', 'Not Set')}")
print(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT', 'Not Set')}")
print(f"Gemini API Key: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not Set'}")
print(f"Tavily API Key: {'Set' if os.getenv('TAVILY_API_KEY') else 'Not Set'}")
print()


def run_qa_workflow(question: str):
    """Execute the Q&A workflow with persistent memory and tracing"""
    
    # Task D: Setup SQLite checkpointer for persistent memory
    with SqliteSaver.from_conn_string(
        conn_string="checkpoints/langgraph_memory.sqlite"
    ) as checkpointer:
        
        # Compile workflow with checkpointer
        workflow = build_workflow().compile(checkpointer=checkpointer)
        
        # Task D: Generate unique thread_id from question for persistence
        thread_id = hashlib.sha256(question.encode()).hexdigest()[:10]
        
        print(f"Thread ID: {thread_id}")
        print(f"Question: {question}\n")
        print("=" * 60)
        print("Starting reflective Q&A workflow...")
        print("=" * 60)
        
        # Execute workflow with config for persistence
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Initialize state with retry_count
        initial_state = {
            "question": question,
            "retry_count": 0
        }
        
        # Invoke the workflow
        result = workflow.invoke(initial_state, config=config)
    
    # Display results
    print("\n" + "=" * 60)
    print("=== WORKFLOW COMPLETE ===")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"  - Total Research Iterations: {result.get('retry_count', 0)}")
    print(f"  - Answer Accepted: {'Yes' if result.get('is_acceptable', False) else 'No'}")
    print(f"  - Thread ID (for replay): {thread_id}")
    
    if result.get('reflection'):
        print(f"\n💭 Final Reflection: {result['reflection']}")
    
    print(f"\n" + "=" * 60)
    print("=== FINAL ANSWER ===")
    print("=" * 60)
    print(result.get("final_answer", "No answer generated."))
    print()
    
    # LangSmith trace information
    if os.getenv('LANGCHAIN_TRACING_V2') == 'true':
        print("\n🔍 View detailed trace in LangSmith:")
        print(f"   Project: {os.getenv('LANGCHAIN_PROJECT')}")
        print(f"   Check your LangSmith dashboard for node-by-node execution trace")
    
    return result


if __name__ == "__main__":
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    
    # Get question from user or use default
    question = input("\nAsk me a question: ").strip()
    
    if not question:
        question = "What are the latest developments in quantum computing in 2024?"
        print(f"Using default question: {question}")
    
    # Run the workflow
    run_qa_workflow(question)

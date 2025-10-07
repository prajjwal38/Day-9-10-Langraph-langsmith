
import os
import hashlib
from dotenv import load_dotenv

# Import necessary components for memory (Task D)
from langgraph.checkpoint.sqlite import SqliteSaver

# Note: We must ensure the import path for the workflow is correct based on the file structure.
from graphs.workflow import build_workflow

# Load environment variables for API keys and tracing setup (Task E)
load_dotenv()

# --- Execution Function ---
def run_workflow():
    """Builds and runs the reflective Q-A workflow with persistent memory."""

    # Task D: Initialize SqliteSaver for persistent memory
    # Creates a database file named 'langgraph_memory.sqlite' in the 'checkpoints' directory.
    checkpointer = SqliteSaver.from_conn_string(conn_string="checkpoints/langgraph_memory.sqlite")

    # Compile the workflow, passing the checkpointer to enable state saving (Task D)
    workflow_app = build_workflow().compile(checkpointer=checkpointer)

    print("--- LangGraph Reflective Q-A Assistant ---")

    # 1. Get user input
    question = input("Ask a question: ")
    if not question:
        question = "What is the primary function of the LangGraph SqliteSaver checkpointer?"
        print(f"Using default question: {question}")

    # 2. Get Thread ID for Memory (Task D)
    # A unique ID is needed to save and resume the thread's state.
    # Use a hash of the question for simple, deterministic thread IDs.
    thread_id = hashlib.sha256(question.encode()).hexdigest()[:10]
    print(f"Generated Thread ID for persistence: {thread_id}")

    # 3. Define the Configuration
    # This config enables checkpointing (via thread_id) and tracing (via env vars - Task E)
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Initial state, including the question and starting the retry count at 0
    # Note: If this thread_id exists, the state will be loaded from the checkpoint.
    initial_state = {"question": question, "retry_count": 0}

    # Invoke the workflow
    # LangSmith tracing (Task E) is automatically enabled here if environment variables are set.
    for step in workflow_app.stream(initial_state, config=config):
        # We can stream the steps, but for a concise output, we'll process the final result after the loop completes.
        pass

    # Retrieve the final state from the checkpointer after execution finishes
    final_state = workflow_app.get_state(config).values

    # --- Print Final Results ---

    final_answer = final_state.get("final_answer")
    final_draft = final_state.get("draft_answer")

    # Determine the output to display
    if final_answer:
        output = final_answer
        status = "Accepted and Finalized"
    elif final_draft:
        output = final_draft
        status = "Best Draft after Max Retries"
    else:
        output = "Execution Complete (Final Answer Missing)"
        status = "Completed"

    print("\n" + "="*50)
    print(f"Execution Summary (ID: {thread_id})")
    print(f"Status: {status}")
    print(f"Total Research/Refinement Cycles: {final_state.get('retry_count', 0)}")
    print("\n[ FINAL ANSWER ]")
    print(output)
    print("="*50)

    print(f"\n💡 Note: The state is saved under ID '{thread_id}'. You can use this ID to resume or inspect the trace in LangSmith (if configured).")


if __name__ == "__main__":
    # Ensure the checkpoints directory exists for the SqliteSaver
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # The user is expected to run this file. We assume 'graphs/workflow.py' is complete.
    run_workflow()

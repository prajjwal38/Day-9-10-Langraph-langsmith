import os
import json
from typing import TypedDict, Optional

# --- FIX 1: Update Pydantic Import (Removes Deprecation Warning) ---
from pydantic import BaseModel, Field 

# LangChain/LangGraph Imports
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
# --- FIX 2: Update with_structured_output Import (Resolves ImportError) ---
# Trying the most common alternative import path for modern versions:
from langchain_core.prompts import with_structured_output


# --- 1. Define the Structured Output Schema (Task A) ---
class CritiqueSchema(BaseModel):
    """Schema for the Critique/Reflection Step."""
    is_acceptable: bool = Field(
        ...,
        description="MUST be True if the answer is clear, factually grounded, and complete. MUST be False if more research or refinement is needed."
    )
    reflection: str = Field(
        ...,
        description="A detailed analysis of the current draft. If is_acceptable is False, describe exactly what new research is needed or how the answer should be improved."
    )


# --- 2. Define the Workflow State (Task D/Memory & General) ---
class QAState(TypedDict, total=False):
    """
    Represents the state of the Q-A workflow.
    - research_data: The output from the web search tool (Task C).
    - reflection: The text critique from the LLM (Task B).
    - retry_count: Counter to prevent infinite loops (Critique b).
    - draft_answer: The initial or re-generated answer.
    - final_answer: The final accepted answer.
    - question: The user's original question.
    """
    question: str
    research_data: Optional[str]
    reflection: Optional[str]
    retry_count: int
    draft_answer: Optional[str]
    final_answer: Optional[str]


# --- 3. Define Graph Nodes ---

def research_node(state: QAState) -> QAState:
    """Performs web search using Tavily and updates the state. (Task C)"""
    # Note: Print statements are kept here to show node invocation (Critique e)
    print("--- 🔍 RESEARCH NODE: Executing web search... ---")
    
    # Initialize Tavily Tool (TAVILY_API_KEY must be in .env)
    tavily_tool = TavilySearchResults(k=3) # Gets top 3 results
    
    question = state["question"]
    
    # Use the reflection to refine the search query if it's a retry
    reflection = state.get("reflection", "")
    
    if reflection:
        search_query = f"Based on this critique: '{reflection}', refine the search query for: {question}"
    else:
        search_query = question

    search_results = tavily_tool.invoke(search_query)
    
    # Format results for the next LLM call
    formatted_results = "\n\n".join(
        [f"Source {i+1} ({r['source']}): {r['content']}" for i, r in enumerate(search_results)]
    )

    # Increment retry counter and update research data
    current_retry = state.get("retry_count", 0) + 1
    
    return {
        "research_data": formatted_results,
        "retry_count": current_retry
    }


def generate_answer_node(state: QAState) -> QAState:
    """Generates an initial or refined draft answer using research data."""
    print("--- ✍️ GENERATE NODE: Drafting answer... ---")
    
    # Use gemini-2.5-flash for speed
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert Q-A assistant. Generate a detailed, professional, and well-structured answer based *only* on the provided research context."),
            ("user", "Question: {question}\n\nResearch Context:\n{research_data}\n\nDraft the full answer:")
        ]
    )

    chain = prompt_template | llm
    
    # Get inputs from state
    question = state["question"]
    research_data = state["research_data"]

    draft = chain.invoke({
        "question": question,
        "research_data": research_data
    }).content
    
    return {"draft_answer": draft}


def critique_answer_node(state: QAState) -> QAState:
    """Critiques the draft, generates a reflection, and decides if the answer is acceptable. (Task A/B)"""
    print("--- ✨ CRITIQUE NODE: Reviewing draft... ---")
    
    # Use gemini-2.5-pro for reliable structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
    
    # Model configured for structured output (Task A)
    structured_llm = llm.with_structured_output(CritiqueSchema)

    # Prompt to guide the reflection
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a meticulous editorial reviewer. Your sole task is to analyze the 'Draft Answer' against the 'Original Question' and the 'Research Context'."
             "If the answer is factually correct, directly addresses the question, and is complete, set 'is_acceptable' to True and set 'reflection' to 'Answer accepted, finalizing.'."
             "If the answer is vague, misses key facts, or needs more context, set 'is_acceptable' to False and provide detailed instructions in the 'reflection' for what needs to be fixed or researched further."),
            ("user", 
             "Original Question: {question}\n\n"
             "Research Context:\n{research_data}\n\n"
             "Draft Answer:\n{draft_answer}")
        ]
    )
    
    # Chain the prompt and the structured model
    chain = critique_prompt | structured_llm

    # Invoke the chain
    critique: CritiqueSchema = chain.invoke({
        "question": state["question"],
        "research_data": state["research_data"],
        "draft_answer": state["draft_answer"]
    })

    # If acceptable, set the final answer to the current draft
    if critique.is_acceptable:
        return {
            "final_answer": state["draft_answer"],
            "reflection": critique.reflection
        }
    else:
        # Otherwise, return the critique and reflection to trigger the loop
        return {
            "reflection": critique.reflection,
            "draft_answer": f"Refinement needed based on critique: {critique.reflection}\n\nPrevious Draft:\n{state['draft_answer']}"
        }


def route_workflow(state: QAState) -> str:
    """Conditional router based on reflection and retry count. (Task B - Critique b)"""
    # Check if the critique node returned an accepted answer
    if state.get("final_answer"):
        print(f"--- ✅ ROUTER: Answer accepted. Proceeding to END. ---")
        return END

    # Check for maximum retries to prevent infinite loop
    max_retries = 3
    current_retry = state.get("retry_count", 0)

    if current_retry >= max_retries:
        print(f"--- 🛑 ROUTER: Max retries ({max_retries}) reached. Proceeding to END. ---")
        # Finalize the current best draft even if unacceptable
        return END
    
    # Otherwise, loop back for more research/refinement
    print(f"--- 🔄 ROUTER: Retrying. Cycle {current_retry}/{max_retries}. Going back to research. ---")
    return "research"


# --- 4. Build LangGraph Workflow ---
def build_workflow():
    """Defines the graph structure with nodes and conditional edges."""
    graph = StateGraph(QAState)
    
    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("critique", critique_answer_node)

    # Set up the loop and edges
    graph.add_edge(START, "research")
    graph.add_edge("research", "generate")
    graph.add_edge("generate", "critique")

    # Conditional loop logic: critique -> route_workflow (Task B)
    graph.add_conditional_edges(
        "critique",
        route_workflow,
        {
            "research": "research", # Loop back to research node
            END: END             # Finish the workflow
        }
    )

    # Note: We return the graph builder object uncompiled here, as compilation 
    # with the checkpointer happens in main.py (Task D).
    return graph

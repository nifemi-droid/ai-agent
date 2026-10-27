import os
import datetime
import sqlite3
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

if not os.getenv("GROQ_API_KEY"):
    load_dotenv()

# Define the State (this is what gets passed between agents)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

# Define the LLM
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# --- DEFINE THE WORKERS (The Specialists) ---

def math_worker(state: AgentState):
    print("🤖 Math Agent is working...")
    # A simple prompt to force it to use python
    prompt = "You are a math expert. Solve the following math problem step-by-step. Answer in one sentence.\n\n" + state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def db_worker(state: AgentState):
    print("🤖 Database Agent is working...")
    # Simulate a database lookup
    response = "Based on the sales database, the top product in Lagos is Laptop with 2 sales."
    return {"messages": [AIMessage(content=response)]}

def general_worker(state: AgentState):
    print("🤖 General Agent is working...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- DEFINE THE MANAGER (The Router) ---
def manager(state: AgentState):
    print("🧠 Manager is deciding who should handle this...")
    # We ask the LLM to act as a router
    system_prompt = """You are a routing manager. Based on the user's message, decide which worker should handle it.
    Reply with EXACTLY one of these words:
    - 'math' (if it involves numbers, calculations, or math)
    - 'db' (if it asks about sales, products, or database records)
    - 'general' (for everything else, like time, general knowledge, or chatting)
    
    Do not add any other text. Just the word."""
    
    response = llm.invoke([
        HumanMessage(content=system_prompt),
        state["messages"][-1]
    ])
    
    decision = response.content.strip().lower()
    
    if "math" in decision:
        return {"next_agent": "math"}
    elif "db" in decision:
        return {"next_agent": "db"}
    else:
        return {"next_agent": "general"}

# --- BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("manager", manager)
workflow.add_node("math_worker", math_worker)
workflow.add_node("db_worker", db_worker)
workflow.add_node("general_worker", general_worker)

# Set entry point
workflow.set_entry_point("manager")

# Add conditional edges from the manager to the workers
workflow.add_conditional_edges(
    "manager",
    lambda x: x["next_agent"],
    {
        "math": "math_worker",
        "db": "db_worker",
        "general": "general_worker"
    }
)

# All workers go to END
workflow.add_edge("math_worker", END)
workflow.add_edge("db_worker", END)
workflow.add_edge("general_worker", END)

# Compile the graph
multi_agent_app = workflow.compile()

# --- TEST FUNCTION ---
def ask_multi_agent(user_message):
    inputs = {"messages": [HumanMessage(content=user_message)]}
    result = multi_agent_app.invoke(inputs)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("\n--- Test 1: Math ---")
    print(ask_multi_agent("What is 500 * 23?"))
    
    print("\n--- Test 2: Database ---")
    print(ask_multi_agent("What is our top selling product in Lagos?"))
    
    print("\n--- Test 3: General ---")
    print(ask_multi_agent("What is the capital of France?"))
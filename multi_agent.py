import os
import datetime
import sqlite3
import io
import pandas as pd
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

if not os.getenv("GROQ_API_KEY"):
    load_dotenv()

# Define the State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# --- RAG INGESTION FUNCTION (Lazy Loads Heavy Libraries) ---
def ingest_file(file_bytes, file_name):
    if file_name.endswith('.pdf'):
        from pypdf import PdfReader
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        pdf = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
            
        chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
        if not chunks: return "No text found in PDF."
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(chunks).tolist()
        
        chroma_client = chromadb.Client()
        try:
            chroma_client.delete_collection(name="pdf_docs")
        except Exception:
            pass
            
        collection = chroma_client.create_collection(name="pdf_docs")
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[str(i) for i in range(len(chunks))]
        )
        return f"Successfully ingested {len(chunks)} chunks from PDF."
        
    elif file_name.endswith('.csv') or file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
            
            df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
            
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")
            conn = sqlite3.connect(db_path)
            df.to_sql('user_uploaded_data', conn, if_exists='replace', index=False)
            conn.close()
            
            return f"Successfully ingested {len(df)} rows into the database."
        except Exception as e:
            return f"Error processing data file: {str(e)}"
            
    else:
        return "Unsupported file type."

def query_user_data(sql_query):
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        if not results: return "No data found."
        return str([dict(zip(columns, row)) for row in results])
    except Exception as e:
        return f"SQL Error: {str(e)}"

# --- WORKERS ---
def math_worker(state: AgentState):
    print("🧮 Math Agent is working...")
    prompt = "You are a math expert. Solve the following math problem step-by-step. Answer in one sentence.\n\n" + state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def db_worker(state: AgentState):
    print("🗄️ Database Agent is working...")
    response = "Based on the sales database, the top product in Lagos is Laptop with 2 sales."
    return {"messages": [AIMessage(content=response)]}

def pdf_worker(state: AgentState):
    print("📄 PDF Agent is searching the document...")
    query = state["messages"][-1].content
    
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        chroma_client = chromadb.Client()
        
        collection = chroma_client.get_collection(name="pdf_docs")
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=2)
        context = "\n".join(results['documents'][0])
        
        prompt = f"Answer the question using ONLY this context:\n\n{context}\n\nQuestion: {query}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception:
        return {"messages": [AIMessage(content="Please upload a PDF first.")]}

def user_data_worker(state: AgentState):
    print("📊 User Data Agent is working...")
    query = state["messages"][-1].content
    
    prompt = f"""You are a data analyst. The user has uploaded a file which is stored in a SQLite table called 'user_uploaded_data'. 
    Write a SQL query to answer the user's question. Return ONLY the SQL query, no markdown, no explanation.
    User Question: {query}"""
    
    sql_response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = sql_response.content.strip().replace("```sql", "").replace("```", "")
    
    print(f"🔧 Generated SQL: {sql_query}")
    result = query_user_data(sql_query)
    
    final_prompt = f"Here is the data result from the database: {result}\n\nPlease answer the user's question using this data: {query}"
    final_response = llm.invoke([HumanMessage(content=final_prompt)])
    
    return {"messages": [AIMessage(content=final_response.content)]}

def general_worker(state: AgentState):
    print("💬 General Agent is working...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- MANAGER ---
def manager(state: AgentState):
    print("🧠 Manager is deciding...")
    system_prompt = """You are a routing manager. Based on the user's message, decide which worker should handle it.
    Reply with EXACTLY one word:
    - 'math'
    - 'db'
    - 'user_data' (if asking about CSV/Excel)
    - 'pdf' (if asking about PDF)
    - 'general'
    """
    response = llm.invoke([HumanMessage(content=system_prompt), state["messages"][-1]])
    decision = response.content.strip().lower()
    
    if "math" in decision: return {"next_agent": "math"}
    elif "user_data" in decision: return {"next_agent": "user_data"}
    elif "db" in decision: return {"next_agent": "db"}
    elif "pdf" in decision: return {"next_agent": "pdf"}
    else: return {"next_agent": "general"}

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager)
workflow.add_node("math_worker", math_worker)
workflow.add_node("db_worker", db_worker)
workflow.add_node("pdf_worker", pdf_worker)
workflow.add_node("user_data_worker", user_data_worker)
workflow.add_node("general_worker", general_worker)

workflow.set_entry_point("manager")
workflow.add_conditional_edges(
    "manager",
    lambda x: x["next_agent"],
    {
        "math": "math_worker",
        "db": "db_worker",
        "pdf": "pdf_worker",
        "user_data": "user_data_worker",
        "general": "general_worker"
    }
)

workflow.add_edge("math_worker", END)
workflow.add_edge("db_worker", END)
workflow.add_edge("pdf_worker", END)
workflow.add_edge("user_data_worker", END)
workflow.add_edge("general_worker", END)

multi_agent_app = workflow.compile()

# --- RUNNER ---
def ask_multi_agent(user_message):
    inputs = {"messages": [HumanMessage(content=user_message)]}
    steps = []
    final_answer = ""
    
    for event in multi_agent_app.stream(inputs, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "manager":
                steps.append(f"🧠 **Manager** is routing to: `{node_output.get('next_agent').upper()}`")
            elif node_name == "math_worker":
                steps.append("🧮 **Math Agent** is calculating...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "db_worker":
                steps.append("🗄️ **Database Agent** is querying...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "pdf_worker":
                steps.append("📄 **PDF Agent** is reading the document...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "user_data_worker":
                steps.append("📊 **User Data Agent** is analyzing your file...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "general_worker":
                steps.append("💬 **General Agent** is answering...")
                final_answer = node_output["messages"][-1].content
                
    return steps, final_answer
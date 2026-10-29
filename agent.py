import os
import json
import datetime
import sqlite3
import time
import ast
import operator
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = None

# --- DEFINE THE TOOLS ---
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression):
    allowed = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    def evaluate(node):
        if isinstance(node, ast.Expression): return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            left, right = evaluate(node.left), evaluate(node.right)
            if abs(left) > 1e12 or abs(right) > 1e12: raise ValueError("Number is too large")
            return allowed[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("Only numeric arithmetic is allowed")
    try:
        if len(expression) > 500: raise ValueError("Expression is too long")
        result = evaluate(ast.parse(expression, mode="eval"))
        return str(result)
    except Exception as e:
        return "Error: invalid arithmetic expression"

def web_search(query):
    return "Web search is disabled to prevent third-party data leakage."

def send_email(recipient, subject, body):
    return f"✅ Email successfully sent to {recipient} with subject '{subject}'."

def query_sales_database(sql_query):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "sales.db")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        if not results: return "No data found."
        formatted_results = [dict(zip(columns, row)) for row in results]
        return str(formatted_results)
    except Exception as e:
        return f"SQL Error: {str(e)}"

available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "web_search": web_search,
    "send_email": send_email,
    "query_sales_database": query_sales_database
}

tools_schema = [
    {"type": "function", "function": {"name": "get_current_time", "description": "Get the current date and time", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "calculate", "description": "Perform a math calculation", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the internet for current events or facts.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email to someone. ONLY use this if the user explicitly asks to send an email.", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["recipient", "subject", "body"]}}},
    {"type": "function", "function": {"name": "query_sales_database", "description": "Query the company sales database. Table 'sales' has columns: id, product, city, amount, date.", "parameters": {"type": "object", "properties": {"sql_query": {"type": "string"}}, "required": ["sql_query"]}}}
]

# --- THE AGENT LOOP (Now with Streaming) ---
def ask_agent(user_message, history=None):
    history = history or []
    start_time = time.time()
    used_tool = "None"
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. You have access to tools for math, time, web search, sending emails, and querying the sales database."},
        *history, 
        {"role": "user", "content": user_message}
    ]

    # First call to see if we need a tool
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools_schema,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": response_message.tool_calls
        })
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            used_tool = function_name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "send_email":
                yield {"requires_approval": True, "tool_call_id": tool_call.id, "tool_name": function_name, "tool_args": function_args, "message": f"I need permission to send an email to {function_args.get('recipient')}.", "tool_used": used_tool, "latency": 0}
                return

            if function_name not in available_functions:
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": "Error: Tool does not exist."})
                continue

            function_to_call = available_functions[function_name]
            try:
                if function_name == "calculate":
                    function_response = function_to_call(function_args.get("expression"))
                elif function_name == "web_search":
                    function_response = function_to_call(function_args.get("query"))
                elif function_name == "query_sales_database":
                    function_response = function_to_call(function_args.get("sql_query"))
                else:
                    function_response = function_to_call()
            except Exception as e:
                function_response = f"Tool error: {str(e)}"
                
            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})
            
        # Second call: STREAM the final answer
        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tool_choice="none",
            stream=True  # <--- THIS IS THE MAGIC
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "content", "text": chunk.choices[0].delta.content, "tool_used": used_tool, "latency": time.time() - start_time}
                
    else:
        # No tool needed, just stream the answer directly
        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "content", "text": chunk.choices[0].delta.content, "tool_used": "None", "latency": time.time() - start_time}

def execute_approved_tool(tool_name, tool_args):
    if tool_name == "send_email":
        return send_email(tool_args.get("recipient"), tool_args.get("subject"), tool_args.get("body"))
    return "Error: Unknown tool."
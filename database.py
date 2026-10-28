import sqlite3
import os
import uuid

DB_NAME = "chat_history.db"
SALES_DB = "sales.db"

def ensure_dbs_exist():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Sessions Table (One row per conversation)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Messages Table (Linked to a session)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)
    
    # Analytics Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            tool_used TEXT,
            latency_seconds REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

    # Ensure sales DB exists
    if not os.path.exists(SALES_DB):
        conn = sqlite3.connect(SALES_DB)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                product TEXT,
                city TEXT,
                amount REAL,
                date TEXT
            )
        """)
        data = [
            ("Laptop", "Lagos", 1200.00, "2026-01-15"),
            ("Phone", "Abuja", 800.00, "2026-01-16"),
            ("Laptop", "Lagos", 1200.00, "2026-02-10"),
            ("Headphones", "Lagos", 150.00, "2026-02-11"),
            ("Phone", "Lagos", 800.00, "2026-03-05"),
            ("Monitor", "Abuja", 300.00, "2026-03-15"),
            ("Laptop", "Port Harcourt", 1200.00, "2026-04-01")
        ]
        cursor.executemany("INSERT INTO sales (product, city, amount, date) VALUES (?, ?, ?, ?)", data)
        conn.commit()
        conn.close()

# --- SESSION MANAGEMENT ---
def create_session(title="New Chat"):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (session_id, title))
    conn.commit()
    conn.close()
    return session_id

def get_sessions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM sessions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_session(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

# --- MESSAGE MANAGEMENT ---
def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Update session title if it's the first user message
    if role == "user":
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        if cursor.fetchone()[0] == 0:
            # Use first 30 chars of message as title
            new_title = content[:30] + "..." if len(content) > 30 else content
            cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
            
    cursor.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

# --- ANALYTICS ---
def log_analytics(user_query, tool_used, latency):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO analytics (user_query, tool_used, latency_seconds) VALUES (?, ?, ?)", (user_query, tool_used, latency))
    conn.commit()
    conn.close()

def get_analytics():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_query, tool_used, latency_seconds, timestamp FROM analytics ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
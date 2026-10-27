import sqlite3
import os

DB_NAME = "chat_history.db"
SALES_DB = "sales.db"

def ensure_dbs_exist():
    # 1. Ensure chat history DB exists
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
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

    # 2. Ensure sales DB exists (and has sample data)
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

def save_message(role, content):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def load_messages():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

def log_analytics(user_query, tool_used, latency):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analytics (user_query, tool_used, latency_seconds) VALUES (?, ?, ?)",
        (user_query, tool_used, latency)
    )
    conn.commit()
    conn.close()

def get_analytics():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_query, tool_used, latency_seconds, timestamp FROM analytics ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
import sqlite3
import os
import uuid
import hashlib

DB_NAME = "chat_history.db"
SALES_DB = "sales.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def ensure_dbs_exist():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    
    # 2. Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    # 3. Messages Table
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
    
    # 4. Analytics Table
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

    # 5. Ensure sales DB exists
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

# --- AUTHENTICATION ---
def create_user(email, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    user_id = str(uuid.uuid4())
    try:
        cursor.execute("INSERT INTO users (id, email, password) VALUES (?, ?, ?)", (user_id, email, hash_password(password)))
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ? AND password = ?", (email, hash_password(password)))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# --- SESSIONS ---
def create_session(user_id, title="New Chat"):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)", (session_id, user_id, title))
    conn.commit()
    conn.close()
    return session_id

def get_sessions(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM sessions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
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

# --- MESSAGES ---
def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if role == "user":
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        if cursor.fetchone()[0] == 0:
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

def delete_last_message(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1", (session_id,))
    row = cursor.fetchone()
    if row:
        cursor.execute("DELETE FROM messages WHERE id = ?", (row[0],))
        conn.commit()
    conn.close()

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

def get_analytics_by_route():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT tool_used, COUNT(*) FROM analytics GROUP BY tool_used")
    rows = cursor.fetchall()
    conn.close()
    return dict(rows)

# --- PROFILE HELPERS ---
def get_user_email(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "Unknown"

def update_user_password(user_id, new_password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()

def is_admin(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] == 'admin' if row else False

def get_user_role(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "user"

def get_all_sessions_for_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, timestamp FROM sessions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows
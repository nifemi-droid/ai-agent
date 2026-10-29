import sqlite3
import os
import uuid
import hashlib
import datetime
import secrets
import base64

DB_NAME = "chat_history.db"
SALES_DB = "sales.db"

PBKDF2_ITERATIONS = 600_000

def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"

def verify_password(password, stored_hash):
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations)
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return secrets.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return secrets.compare_digest(hashlib.sha256(password.encode("utf-8")).hexdigest(), stored_hash)

def ensure_dbs_exist():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
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
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            tool_used TEXT,
            latency_seconds REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            user_id TEXT PRIMARY KEY,
            tokens INTEGER DEFAULT 10,
            last_reset DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

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

def create_user(email, password):
    email = email.strip().lower()
    if len(email) > 254 or "@" not in email or len(password) < 12:
        return None
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
    email = email.strip().lower()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    if not row or not verify_password(password, row[1]):
        conn.close()
        return None
    if not row[1].startswith("pbkdf2_sha256$"):
        cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(password), row[0]))
        conn.commit()
    conn.close()
    return row[0]

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

def delete_session(session_id, user_id):
    """IDOR FIX: Verify the session belongs to the user before deleting."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row or row[0] != user_id:
        conn.close()
        return False
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
    conn.commit()
    conn.close()
    return True

def save_message(session_id, role, content, user_id=None):
    if role not in ("user", "assistant") or len(content) > 100_000:
        return False
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if user_id:
        cursor.execute("SELECT 1 FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        if not cursor.fetchone():
            conn.close()
            return False
    if role == "user":
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        if cursor.fetchone()[0] == 0:
            new_title = content[:30] + "..." if len(content) > 30 else content
            cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
    cursor.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()
    return True

def load_messages(session_id, user_id):
    """IDOR FIX: Only load messages if the session belongs to the user."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.role, m.content 
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE m.session_id = ? AND s.user_id = ?
        ORDER BY m.timestamp ASC
    """, (session_id, user_id))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def delete_last_message(session_id, user_id):
    """IDOR FIX: Only delete if the session belongs to the user."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row or row[0] != user_id:
        conn.close()
        return False
    cursor.execute("SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1", (session_id,))
    row = cursor.fetchone()
    if row:
        cursor.execute("DELETE FROM messages WHERE id = ?", (row[0],))
        conn.commit()
    conn.close()
    return True

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

def get_user_email(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "Unknown"

def update_user_password(user_id, new_password):
    if len(new_password) < 12:
        return False
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()
    return True

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

def check_rate_limit(user_id, max_tokens=10, refill_seconds=60):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT tokens, last_reset FROM rate_limits WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    now = datetime.datetime.now()
    if not row:
        cursor.execute("INSERT INTO rate_limits (user_id, tokens, last_reset) VALUES (?, ?, ?)", (user_id, max_tokens - 1, now))
        conn.commit()
        conn.close()
        return True, max_tokens - 1
    tokens, last_reset = row
    last_reset_dt = datetime.datetime.fromisoformat(last_reset)
    if (now - last_reset_dt).total_seconds() > refill_seconds:
        tokens = max_tokens
        cursor.execute("UPDATE rate_limits SET tokens = ?, last_reset = ? WHERE user_id = ?", (tokens, now, user_id))
    if tokens <= 0:
        conn.commit()
        conn.close()
        return False, 0
    tokens -= 1
    cursor.execute("UPDATE rate_limits SET tokens = ? WHERE user_id = ?", (tokens, user_id))
    conn.commit()
    conn.close()
    return True, tokens

def get_rate_limit_status(user_id, max_tokens=10, refill_seconds=60):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT tokens, last_reset FROM rate_limits WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return max_tokens
    tokens, last_reset = row
    last_reset_dt = datetime.datetime.fromisoformat(last_reset)
    now = datetime.datetime.now()
    if (now - last_reset_dt).total_seconds() > refill_seconds:
        conn.close()
        return max_tokens
    conn.close()
    return max(tokens, 0)
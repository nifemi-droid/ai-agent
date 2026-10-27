import sqlite3

conn = sqlite3.connect("sales.db")
cursor = conn.cursor()

# Create a table
cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product TEXT,
    city TEXT,
    amount REAL,
    date TEXT
)
""")

# Insert fake data
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
print("Database 'sales.db' created successfully!")
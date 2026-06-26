import sqlite3
import os

db_path = 'data/processed/doc_store.db'
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables:", tables)

# Get schemas
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    schema = cursor.fetchall()
    print(f"\nTable '{table}' schema:")
    for col in schema:
        print(f"  {col[1]} ({col[2]})")

# Get counts
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"\nTable '{table}' count: {count:,}")

conn.close()

# scratch/check_db.py
import sqlite3
import os

db_path = os.path.expanduser('~/.ir_system_cache/doc_store.db')
print(f"Checking DB at: {db_path}")
print(f"File exists: {os.path.exists(db_path)}")
if os.path.exists(db_path):
    print(f"File size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        print("Connected. Querying COUNT...")
        cursor.execute("SELECT COUNT(*) FROM documents")
        print(f"Count: {cursor.fetchone()[0]}")
        print("Querying AVG length...")
        cursor.execute("SELECT AVG(length) FROM documents")
        print(f"Avg length: {cursor.fetchone()[0]}")
        conn.close()
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

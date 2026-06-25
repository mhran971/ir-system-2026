# scratch/add_index.py
import sqlite3
import os
import time

db_path = os.path.expanduser('~/.ir_system_cache/doc_store.db')
print(f"Connecting to {db_path}...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Creating covering index on (doc_id, length)...")
t0 = time.time()
cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_lengths ON documents (doc_id, length)")
conn.commit()
t1 = time.time()
print(f"Index created in {t1 - t0:.4f} seconds.")

print("Benchmarking SELECT doc_id, length...")
t0 = time.time()
cursor.execute("SELECT doc_id, length FROM documents")
rows = cursor.fetchall()
t1 = time.time()
print(f"Fetched {len(rows)} rows in {t1 - t0:.4f} seconds.")

conn.close()

# scratch/debug_ds.py
import time
import os
import sqlite3
import shutil

print("1. Checking DB file...", flush=True)
db_path = os.path.expanduser('~/.ir_system_cache/doc_store.db')
print(f"Exists: {os.path.exists(db_path)}", flush=True)

print("2. Connecting...", flush=True)
t0 = time.time()
conn = sqlite3.connect(db_path)
print(f"Connected in {time.time() - t0:.4f}s", flush=True)

print("3. Querying COUNT...", flush=True)
t0 = time.time()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM documents")
count = cursor.fetchone()[0]
print(f"Count: {count} (query took {time.time() - t0:.4f}s)", flush=True)

print("4. Fetching rows...", flush=True)
t0 = time.time()
cursor.execute("SELECT doc_id, length FROM documents")
rows = cursor.fetchall()
print(f"Fetched {len(rows)} rows in {time.time() - t0:.4f}s", flush=True)

print("5. Building dict...", flush=True)
t0 = time.time()
doc_lengths = {str(row[0]): int(row[1]) for row in rows}
print(f"Dict built in {time.time() - t0:.4f}s", flush=True)

print("6. Querying AVG...", flush=True)
t0 = time.time()
cursor.execute("SELECT AVG(length) FROM documents")
avg_row = cursor.fetchone()
avg_val = float(avg_row[0]) if avg_row and avg_row[0] is not None else 0.0
print(f"Avg: {avg_val} (query took {time.time() - t0:.4f}s)", flush=True)

conn.close()
print("Done!", flush=True)

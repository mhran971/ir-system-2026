# services/indexing/document_store.py
import os
import pickle
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

class DocumentStore:
    """
    Shared DocumentStore component.
    Loads and caches document lengths, and queries texts from SQLite on-demand
    to avoid multiple disk reads and prevent high memory usage.
    """
    _instance = None
    _doc_lengths: Dict[str, int] = {}
    _avg_doc_length: float = 0.0
    _is_loaded = False
    _db_path = 'data/processed/doc_store.db'

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DocumentStore, cls).__new__(cls)
            cls._instance._db_path = cls._instance._get_db_path()
        return cls._instance

    def __init__(self):
        # Already initialized via singleton
        pass

    def _get_db_path(self) -> str:
        default_path = 'data/processed/doc_store.db'
        try:
            abs_path = os.path.abspath(default_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            drive = os.path.splitdrive(abs_path)[0]
            usage = shutil.disk_usage(drive)
            # If free space is less than 1.5 GB, use C: drive fallback
            if usage.free < 1.5 * 1024 * 1024 * 1024:
                fallback_dir = os.path.expanduser('~/.ir_system_cache')
                os.makedirs(fallback_dir, exist_ok=True)
                path = os.path.join(fallback_dir, 'doc_store.db')
                print(f"ℹ️ [DocumentStore] Target drive space is low. Using C: drive fallback: {path}")
                return path
        except Exception as e:
            print(f"⚠️ [DocumentStore] Disk space check failed: {e}")
        return default_path

    def load(self, file_paths: List[str]) -> int:
        """
        Loads documents metadata (lengths) from SQLite database.
        If the SQLite database doesn't exist, it migrates from the first available pickle file.
        """
        if self._is_loaded:
            return len(self._doc_lengths)
        
        # 1. Try to load from SQLite database if it exists
        if os.path.exists(self._db_path):
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT doc_id, length FROM documents")
                rows = cursor.fetchall()
                if rows:
                    self._doc_lengths = {str(row[0]): int(row[1]) for row in rows}
                    
                    cursor.execute("SELECT AVG(length) FROM documents")
                    avg_row = cursor.fetchone()
                    self._avg_doc_length = float(avg_row[0]) if avg_row and avg_row[0] is not None else 0.0
                    
                    conn.close()
                    self._is_loaded = True
                    print(f"📂 [DocumentStore] Loaded metadata for {len(self._doc_lengths)} documents from SQLite DB ({self._db_path})")
                    return len(self._doc_lengths)
                conn.close()
            except Exception as e:
                print(f"⚠️ [DocumentStore] Error loading from SQLite: {e}. Falling back to pickle...")

        # 2. Fallback: migrate from pickle file to SQLite
        for path in file_paths:
            if Path(path).exists():
                print(f"📂 [DocumentStore] SQLite DB not found. Migrating {path} to SQLite (this runs ONCE)...")
                try:
                    with open(path, 'rb') as f:
                        docs = pickle.load(f)
                    
                    os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
                    
                    try:
                        conn = sqlite3.connect(self._db_path)
                        cursor = conn.cursor()
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS documents (
                                doc_id TEXT PRIMARY KEY,
                                text TEXT,
                                length INTEGER
                            )
                        """)
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_lengths ON documents (doc_id, length)")
                        
                        total_len = 0
                        batch = []
                        self._doc_lengths = {}
                        
                        for doc in docs:
                            doc_id = str(doc.get('doc_id', ''))
                            if doc_id:
                                text = doc.get('original', doc.get('text', ''))
                                tokens = doc.get('tokens', [])
                                length = len(tokens)
                                self._doc_lengths[doc_id] = length
                                total_len += length
                                batch.append((doc_id, text, length))
                                
                                if len(batch) >= 10000:
                                    cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                                    conn.commit()
                                    batch = []
                                    
                        if batch:
                            cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                            conn.commit()
                        
                        if len(self._doc_lengths) > 0:
                            self._avg_doc_length = total_len / len(self._doc_lengths)
                        
                        conn.close()
                        self._is_loaded = True
                        print(f"✅ [DocumentStore] Successfully migrated {len(self._doc_lengths)} documents to SQLite DB ({self._db_path})!")
                        return len(self._doc_lengths)
                        
                    except Exception as migration_error:
                        print(f"⚠️ [DocumentStore] Migration to {self._db_path} failed: {migration_error}")
                        # Fallback to C: drive path if error looks like disk full/quota or permission issue
                        fallback_dir = os.path.expanduser('~/.ir_system_cache')
                        os.makedirs(fallback_dir, exist_ok=True)
                        fallback_path = os.path.join(fallback_dir, 'doc_store.db')
                        print(f"ℹ️ [DocumentStore] Attempting fallback to C: drive: {fallback_path}")
                        
                        # Clean up failed database file if possible
                        if os.path.exists(self._db_path):
                            try:
                                os.remove(self._db_path)
                            except Exception:
                                pass
                                
                        self._db_path = fallback_path
                        
                        # Retry migration in C: drive
                        conn = sqlite3.connect(self._db_path)
                        cursor = conn.cursor()
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS documents (
                                doc_id TEXT PRIMARY KEY,
                                text TEXT,
                                length INTEGER
                            )
                        """)
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_lengths ON documents (doc_id, length)")
                        
                        total_len = 0
                        batch = []
                        self._doc_lengths = {}
                        
                        for doc in docs:
                            doc_id = str(doc.get('doc_id', ''))
                            if doc_id:
                                text = doc.get('original', doc.get('text', ''))
                                tokens = doc.get('tokens', [])
                                length = len(tokens)
                                self._doc_lengths[doc_id] = length
                                total_len += length
                                batch.append((doc_id, text, length))
                                
                                if len(batch) >= 10000:
                                    cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                                    conn.commit()
                                    batch = []
                                    
                        if batch:
                            cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", batch)
                            conn.commit()
                        
                        if len(self._doc_lengths) > 0:
                            self._avg_doc_length = total_len / len(self._doc_lengths)
                        
                        conn.close()
                        self._is_loaded = True
                        print(f"✅ [DocumentStore] Successfully migrated {len(self._doc_lengths)} documents to fallback SQLite DB!")
                        return len(self._doc_lengths)
                        
                except Exception as e:
                    print(f"❌ [DocumentStore] Migration failed: {e}")
                    raise e
        
        return 0

    def get_doc(self, doc_id: str) -> Optional[dict]:
        """Returns the document dictionary containing doc_id, text, and empty tokens list."""
        doc_id = str(doc_id)
        if not os.path.exists(self._db_path):
            return None
            
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT text, length FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'doc_id': doc_id,
                    'text': row[0],
                    'tokens': [] # Empty tokens list to save memory
                }
        except Exception as e:
            print(f"⚠️ [DocumentStore] Error querying SQLite: {e}")
            
        return None

    def get_length(self, doc_id: str) -> int:
        """Returns the token length of a document."""
        return self._doc_lengths.get(str(doc_id), 0)

    @property
    def avg_doc_length(self) -> float:
        """Returns the average document length across the loaded collection."""
        return self._avg_doc_length

    @property
    def total_docs(self) -> int:
        """Returns the total number of documents in the store."""
        return len(self._doc_lengths)

    def get_all_documents(self) -> Dict[str, dict]:
        """
        Returns a light dictionary of doc_id mapped to stub dicts
        to support get_all_documents().keys() calls without heavy memory footprint.
        """
        return {
            doc_id: {
                'doc_id': doc_id,
                'text': '',
                'tokens': []
            }
            for doc_id in self._doc_lengths.keys()
        }
import sqlite3
import secrets
import time
from pathlib import Path

class KeyManager:
    def __init__(self, db_path: str = "nexus_keys.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
            conn.commit()

    def generate_key(self, name: str) -> str:
        key = f"nexus_sk_{secrets.token_hex(24)}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO api_keys (key, name, created_at) VALUES (?, ?, ?)",
                (key, name, time.time())
            )
            conn.commit()
        return key

    def validate_key(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM api_keys WHERE key = ? AND status = 'active'",
                (key,)
            )
            return cursor.fetchone() is not None

    def revoke_key(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE key = ? AND status = 'active'",
                (key,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT key, name, created_at, status FROM api_keys")
            return [dict(row) for row in cursor.fetchall()]

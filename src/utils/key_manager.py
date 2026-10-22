import hashlib
import os
import secrets
import sqlite3
import time


class KeyManager:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("NEXUS_KEYS_DB", "nexus_keys.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
            # Preserve existing client keys while removing plaintext from storage.
            for (key,) in conn.execute("SELECT key FROM api_keys").fetchall():
                if not key.startswith("sha256:"):
                    conn.execute(
                        "UPDATE api_keys SET key = ? WHERE key = ?", (self._digest(key), key)
                    )
            conn.commit()

    @staticmethod
    def _digest(key: str) -> str:
        return "sha256:" + hashlib.sha256(key.encode()).hexdigest()

    def generate_key(self, name: str) -> str:
        key = f"nexus_sk_{secrets.token_hex(24)}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO api_keys (key, name, created_at) VALUES (?, ?, ?)",
                (self._digest(key), name, time.time()),
            )
            conn.commit()
        return key

    def validate_key(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM api_keys WHERE key = ? AND status = 'active'", (self._digest(key),)
            )
            return cursor.fetchone() is not None

    def revoke_key(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE key = ? AND status = 'active'",
                (self._digest(key),),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT key AS id, name, created_at, status FROM api_keys")
            return [dict(row) for row in cursor.fetchall()]

    def revoke_by_id(self, key_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE key = ? AND status = 'active'",
                (key_id,),
            )
            return cursor.rowcount > 0

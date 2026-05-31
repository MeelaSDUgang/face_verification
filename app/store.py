import sqlite3
import io
import numpy as np
from pathlib import Path
from typing import Optional


class EmbeddingStore:
    def __init__(self, db_path: str = "embeddings.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                user_id   TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def upsert(self, user_id: str, embedding: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, embedding)
        self.conn.execute(
            """
            INSERT INTO embeddings (user_id, embedding, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                embedding  = excluded.embedding,
                updated_at = excluded.updated_at
            """,
            (user_id, buf.getvalue()),
        )
        self.conn.commit()

    def get(self, user_id: str) -> Optional[np.ndarray]:
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return None
        return np.load(io.BytesIO(row[0]))

    def delete(self, user_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM embeddings WHERE user_id = ?", (user_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def list_users(self) -> list[str]:
        rows = self.conn.execute("SELECT user_id FROM embeddings ORDER BY updated_at DESC").fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def close(self):
        self.conn.close()

"""The memory store: a local SQLite file holding every MemoryRecord, with a
sqlite-vec virtual table for vector similarity search.

Two tables:
  - memories:     full record as JSON + a few columns we filter on
  - vec_memories: (rowid, embedding) for KNN search, joined back by rowid

The store also owns the LOOP-2 write-back: record_access() bumps hit_count +
last_accessed, then rescores/retiers, so genuinely useful memories climb.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import sqlite_vec
from sqlite_vec import serialize_float32

from momento.models.base import LLMBackend
from momento.memory.schema import MemoryRecord, FactType
from momento.memory import scoring
from momento.config import EMBED_DIM, DB_PATH


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    def __init__(self, backend: LLMBackend, db_path: str | None = None,
                 embed_dim: int | None = None):
        self.backend = backend
        self.embed_dim = embed_dim or EMBED_DIM
        path = db_path or DB_PATH
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
                id            TEXT UNIQUE NOT NULL,
                user_id       TEXT NOT NULL,
                subject       TEXT,
                fact_type     TEXT NOT NULL,
                kind          TEXT NOT NULL,
                tier          TEXT NOT NULL,
                score         REAL NOT NULL,
                superseded_by TEXT,
                data          TEXT NOT NULL
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_mem_subject ON memories(subject)")
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories "
            f"USING vec0(embedding float[{self.embed_dim}])"
        )
        self.db.commit()

    @staticmethod
    def _blob(record: MemoryRecord) -> str:
        """Serialize a record to JSON, dropping the embedding (the vector lives
        in vec_memories — no need to duplicate it here)."""
        d = record.to_dict()
        d["embedding"] = None
        return json.dumps(d)

    # --- write ---------------------------------------------------------
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Embed (if needed), score, and persist a new record."""
        if record.embedding is None:
            record.embedding = self.backend.embed([record.text])[0]
        scoring.rescore(record)

        cur = self.db.execute(
            """INSERT INTO memories
               (id, user_id, subject, fact_type, kind, tier, score, superseded_by, data)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (record.id, record.user_id, record.subject, record.fact_type.value,
             record.kind.value, record.tier.value, record.score,
             record.superseded_by, self._blob(record)),
        )
        self.db.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, serialize_float32(record.embedding)),
        )
        self.db.commit()
        return record

    def update(self, record: MemoryRecord) -> None:
        """Persist changes to an existing record (no re-embedding)."""
        self.db.execute(
            """UPDATE memories SET subject=?, fact_type=?, kind=?, tier=?, score=?,
               superseded_by=?, data=? WHERE id=?""",
            (record.subject, record.fact_type.value, record.kind.value,
             record.tier.value, record.score, record.superseded_by,
             self._blob(record), record.id),
        )
        self.db.commit()

    # --- read ----------------------------------------------------------
    def get(self, record_id: str) -> MemoryRecord | None:
        row = self.db.execute(
            "SELECT data FROM memories WHERE id=?", (record_id,)
        ).fetchone()
        return MemoryRecord.from_dict(json.loads(row["data"])) if row else None

    def all(self, *, include_inactive: bool = False) -> list[MemoryRecord]:
        rows = self.db.execute("SELECT data FROM memories").fetchall()
        recs = [MemoryRecord.from_dict(json.loads(r["data"])) for r in rows]
        return recs if include_inactive else [r for r in recs if r.is_active]

    def search(self, query: str, *, k: int = 5, user_id: str = "default",
               subject: str | None = None, fact_type: FactType | None = None,
               intent: str | None = None,
               include_inactive: bool = False) -> list[tuple[MemoryRecord, float]]:
        """Vector KNN, returned (record, distance) best-first. We over-fetch
        from the index then apply metadata filters in Python (sqlite-vec KNN +
        arbitrary SQL filters don't mix cleanly — fine at hackathon scale)."""
        q_vec = self.backend.embed([query])[0]
        fetch = max(k * 5, 25)
        sql = f"""
            WITH matches AS (
                SELECT rowid, distance FROM vec_memories
                WHERE embedding MATCH ? AND k = {fetch}
            )
            SELECT m.data AS data, x.distance AS distance
            FROM matches x JOIN memories m ON m.rowid = x.rowid
            ORDER BY x.distance
        """
        rows = self.db.execute(sql, (serialize_float32(q_vec),)).fetchall()

        out: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            rec = MemoryRecord.from_dict(json.loads(row["data"]))
            if rec.user_id != user_id:
                continue
            if not include_inactive and not rec.is_active:
                continue
            if subject and rec.subject != subject:
                continue
            if fact_type and rec.fact_type != fact_type:
                continue
            if intent and intent not in rec.intents:
                continue
            out.append((rec, float(row["distance"])))
            if len(out) >= k:
                break
        return out

    # --- LOOP 2: use promotes -----------------------------------------
    def record_access(self, records: list[MemoryRecord]) -> None:
        """Called when memories are actually used in a turn: bump usage,
        rescore, retier, persist. This is what promotes useful memories
        toward HOT over time."""
        now = _now()
        for rec in records:
            rec.hit_count += 1
            rec.last_accessed = now
            scoring.rescore(rec, now)
            self.update(rec)

    def close(self):
        self.db.close()

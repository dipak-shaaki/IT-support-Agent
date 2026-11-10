import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()

# The system prompt requires this phrase when the private KB has no answer.
UNDOCUMENTED_MARKERS = (
    "not documented in the company knowledge base",
    "not in the company knowledge base",
    "not found in the company knowledge base",
    "not documented in the knowledge base",
)


def _is_knowledge_gap(answer: str, source_used: str) -> bool:
    if source_used == "web_search":
        return True
    normalized = " ".join(answer.casefold().split())
    return any(marker in normalized for marker in UNDOCUMENTED_MARKERS)


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(settings.audit_db_path, timeout=5)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def init_db() -> None:
    Path(settings.audit_db_path).parent.mkdir(parents=True, exist_ok=True)
    con = _connect()
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL,
            knowledge_gap INTEGER NOT NULL DEFAULT 0
        )"""
    )
    columns = {row[1] for row in con.execute("PRAGMA table_info(query_audit)")}
    if "knowledge_gap" not in columns:
        con.execute(
            "ALTER TABLE query_audit ADD COLUMN knowledge_gap INTEGER NOT NULL DEFAULT 0"
        )
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str], answer: str = "") -> None:
    con = _connect()
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json, knowledge_gap) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            source_used,
            json.dumps(trace),
            int(_is_knowledge_gap(answer, source_used)),
        ),
    )
    con.commit()
    con.close()


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    con = _connect()
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(sql, params).fetchall()]
    con.close()
    return rows


def source_breakdown() -> list[dict]:
    return _rows(
        "SELECT source_used AS source, COUNT(*) AS count FROM query_audit "
        "GROUP BY source_used ORDER BY count DESC"
    )


def kb_gaps(limit: int = 20) -> list[dict]:
    return _rows(
        "SELECT question, COUNT(*) AS count, MAX(created_at) AS last_seen FROM query_audit "
        "WHERE source_used IN ('web_search', 'insufficient_evidence') OR knowledge_gap = 1 "
        "GROUP BY question ORDER BY count DESC, last_seen DESC LIMIT ?",
        (limit,),
    )


def gap_total() -> int:
    return _rows(
        "SELECT COUNT(*) AS count FROM query_audit "
        "WHERE source_used IN ('web_search', 'insufficient_evidence') OR knowledge_gap = 1"
    )[0]["count"]


def daily_counts(limit: int = 14) -> list[dict]:
    return _rows(
        "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count FROM query_audit "
        "GROUP BY day ORDER BY day DESC LIMIT ?",
        (limit,),
    )


def recent_queries(limit: int = 20) -> list[dict]:
    return _rows(
        "SELECT created_at, question, source_used AS source FROM query_audit "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )

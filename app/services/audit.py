import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()

def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )
    con.commit()
    con.close()


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    con = sqlite3.connect(settings.audit_db_path)
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
        "WHERE source_used IN ('web_search', 'insufficient_evidence') "
        "GROUP BY question ORDER BY count DESC, last_seen DESC LIMIT ?",
        (limit,),
    )


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

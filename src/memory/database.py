import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "./research_history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            research_brief TEXT,
            final_report TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_research(user_id: str, query: str, result: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO research_history (user_id, query, research_brief, final_report, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        query,
        result.get("research_brief", ""),
        result.get("final_report", ""),
        json.dumps(result.get("notes", [])),
        datetime.now().isoformat(),
    ))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_history(user_id: str, limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, final_report, created_at
        FROM research_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "query": r[1], "final_report": r[2][:200], "created_at": r[3]} for r in rows]

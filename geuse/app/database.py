"""
app/database.py — SQLite persistence layer.

Tables
------
  user           — single-row user profile
  assessment     — ROM / gesture assessment results per session
  rehab_plan     — generated exercise plans (history kept)
  session        — completed exercise session logs

All public functions accept/return plain Python dicts or primitives.
JSON columns store arbitrary nested data without a fixed schema.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Optional

DB_PATH = pathlib.Path(__file__).parent.parent / "geuse.db"


# --------------------------------------------------------------------------- #
# Connection helper
# --------------------------------------------------------------------------- #

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def init_db() -> None:
    """Create all tables if they do not exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.executescript("""
            -- --------------------------------------------------------
            -- user  (always row id=1, upserted)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS user (
                id            INTEGER PRIMARY KEY,
                name          TEXT    NOT NULL DEFAULT '',
                age           INTEGER,
                affected_hand TEXT    NOT NULL DEFAULT 'right',
                condition     TEXT    NOT NULL DEFAULT '',
                goals         TEXT    NOT NULL DEFAULT '[]',   -- JSON array
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- --------------------------------------------------------
            -- assessment  (one row per assessment run)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS assessment (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL DEFAULT 1,
                results       TEXT    NOT NULL DEFAULT '{}',   -- JSON: labels, closures, etc.
                score         REAL,                            -- 0–100 ROM score
                notes         TEXT    NOT NULL DEFAULT '',
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES user(id)
            );

            -- --------------------------------------------------------
            -- rehab_plan  (versioned; latest is highest id)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS rehab_plan (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id              INTEGER NOT NULL DEFAULT 1,
                exercises            TEXT    NOT NULL DEFAULT '[]',  -- JSON array
                sessions_per_week    INTEGER NOT NULL DEFAULT 3,
                notes                TEXT    NOT NULL DEFAULT '',
                source_assessment_id INTEGER,
                created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES user(id),
                FOREIGN KEY (source_assessment_id) REFERENCES assessment(id)
            );

            -- --------------------------------------------------------
            -- session  (completed workout log)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS session (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL DEFAULT 1,
                plan_id      INTEGER,
                exercises    TEXT    NOT NULL DEFAULT '[]',  -- JSON: [{exercise, sets, reps_done, …}]
                pain_before  INTEGER,
                pain_after   INTEGER,
                duration_s   INTEGER,
                completed_at TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES user(id),
                FOREIGN KEY (plan_id) REFERENCES rehab_plan(id)
            );
        """)


# --------------------------------------------------------------------------- #
# user
# --------------------------------------------------------------------------- #

def save_user(data: dict) -> None:
    """Upsert the single user row (id=1)."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO user (id, name, age, affected_hand, condition, goals, updated_at)
            VALUES (1, :name, :age, :affected_hand, :condition, :goals, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                name          = excluded.name,
                age           = excluded.age,
                affected_hand = excluded.affected_hand,
                condition     = excluded.condition,
                goals         = excluded.goals,
                updated_at    = excluded.updated_at
        """, {
            "name":          data.get("name", ""),
            "age":           data.get("age"),
            "affected_hand": data.get("affected_hand", "right"),
            "condition":     data.get("condition", ""),
            "goals":         json.dumps(data.get("goals", [])),
        })


def get_user() -> Optional[dict]:
    """Return the user profile dict, or None if not yet created."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM user WHERE id=1").fetchone()
    if row is None:
        return None
    d = dict(row)
    d["goals"] = json.loads(d["goals"])
    return d


# --------------------------------------------------------------------------- #
# assessment
# --------------------------------------------------------------------------- #

def save_assessment(results: dict, score: float = 0.0, notes: str = "") -> int:
    """Insert an assessment record; returns the new row id."""
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO assessment (user_id, results, score, notes)
            VALUES (1, ?, ?, ?)
        """, (json.dumps(results), score, notes))
        return cur.lastrowid


def get_latest_assessment() -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM assessment WHERE user_id=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["results"] = json.loads(d["results"])
    return d


def get_all_assessments() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM assessment WHERE user_id=1 ORDER BY id DESC"
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["results"] = json.loads(d["results"])
        result.append(d)
    return result


# --------------------------------------------------------------------------- #
# rehab_plan
# --------------------------------------------------------------------------- #

def save_plan(
    exercises: list,
    sessions_per_week: int = 3,
    notes: str = "",
    source_assessment_id: Optional[int] = None,
) -> int:
    """Insert a new plan version; returns the new row id."""
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO rehab_plan (user_id, exercises, sessions_per_week, notes, source_assessment_id)
            VALUES (1, ?, ?, ?, ?)
        """, (json.dumps(exercises), sessions_per_week, notes, source_assessment_id))
        return cur.lastrowid


def get_latest_plan() -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM rehab_plan WHERE user_id=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["exercises"] = json.loads(d["exercises"])
    return d


def get_all_plans() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM rehab_plan WHERE user_id=1 ORDER BY id DESC"
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        result.append(d)
    return result


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #

def save_session(
    exercises: list,
    plan_id: Optional[int] = None,
    pain_before: Optional[int] = None,
    pain_after: Optional[int] = None,
    duration_s: Optional[int] = None,
) -> int:
    """Insert a completed session record; returns the new row id."""
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO session (user_id, plan_id, exercises, pain_before, pain_after, duration_s)
            VALUES (1, ?, ?, ?, ?, ?)
        """, (plan_id, json.dumps(exercises), pain_before, pain_after, duration_s))
        return cur.lastrowid


def get_latest_session() -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session WHERE user_id=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["exercises"] = json.loads(d["exercises"])
    return d


def get_all_sessions() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session WHERE user_id=1 ORDER BY id DESC"
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        result.append(d)
    return result

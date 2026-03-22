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
import os
import pathlib
import sqlite3
import sys
from typing import Optional


def _resolve_db_path() -> pathlib.Path:
    """Return the database path.

    In a PyInstaller frozen build the exe directory is writable and is used
    so that user data persists across runs. In development mode the db lives
    alongside main.py in geuse/.
    """
    if getattr(sys, 'frozen', False):
        return pathlib.Path(os.path.dirname(sys.executable)) / "geuse.db"
    return pathlib.Path(__file__).parent.parent / "geuse.db"


DB_PATH = _resolve_db_path()


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
            -- self_report  (daily pre-session check-in)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS self_report (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL DEFAULT 1,
                pain_level   INTEGER NOT NULL DEFAULT 0,
                limitations  TEXT    NOT NULL DEFAULT '',
                goal         TEXT    NOT NULL DEFAULT '',
                created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES user(id)
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
# self_report
# --------------------------------------------------------------------------- #

def save_self_report(pain_level: int, limitations: str = "", goal: str = "") -> int:
    """Insert a daily check-in record; returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO self_report (user_id, pain_level, limitations, goal) VALUES (1, ?, ?, ?)",
            (pain_level, limitations, goal),
        )
        return cur.lastrowid


def get_latest_self_report() -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM self_report WHERE user_id=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


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


def _compute_streak(sessions: list) -> int:
    """Count consecutive days ending today (or yesterday) with at least one session."""
    import datetime
    dates: set = set()
    for s in sessions:
        try:
            dates.add(datetime.datetime.strptime(s["completed_at"][:19], "%Y-%m-%d %H:%M:%S").date())
        except Exception:
            pass
    if not dates:
        return 0
    today = datetime.date.today()
    check = today if today in dates else today - datetime.timedelta(days=1)
    streak = 0
    while check in dates:
        streak += 1
        check -= datetime.timedelta(days=1)
    return streak


def get_session_history() -> dict:
    """Return aggregated session data for the dashboard."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session WHERE user_id=1 ORDER BY id DESC LIMIT 30"
        ).fetchall()
        pain_rows = conn.execute(
            "SELECT pain_level FROM self_report WHERE user_id=1 ORDER BY id DESC LIMIT 5"
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM session WHERE user_id=1"
        ).fetchone()[0]

    sessions = []
    for row in rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        sessions.append(d)

    pain_history = [r["pain_level"] for r in reversed(pain_rows)]
    total_exercises = sum(len(s["exercises"]) for s in sessions)

    return {
        "sessions": sessions,
        "pain_history": pain_history,
        "streak": _compute_streak(sessions),
        "total_sessions": total,
        "total_exercises": total_exercises,
    }


def reset_db() -> None:
    """Drop all tables and recreate the schema. Wipes all user data."""
    with _connect() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS session;
            DROP TABLE IF EXISTS rehab_plan;
            DROP TABLE IF EXISTS self_report;
            DROP TABLE IF EXISTS assessment;
            DROP TABLE IF EXISTS user;
        """)
    init_db()


def get_progress_data() -> dict:
    """Return all data needed for the progress page in one call."""
    with _connect() as conn:
        session_rows = conn.execute(
            "SELECT * FROM session WHERE user_id=1 ORDER BY id DESC LIMIT 20"
        ).fetchall()
        assessment_rows = conn.execute(
            "SELECT * FROM assessment WHERE user_id=1 ORDER BY id ASC"
        ).fetchall()
        pain_rows = conn.execute(
            "SELECT pain_level, created_at FROM self_report "
            "WHERE user_id=1 ORDER BY id DESC LIMIT 30"
        ).fetchall()
        total_sessions = conn.execute(
            "SELECT COUNT(*) FROM session WHERE user_id=1"
        ).fetchone()[0]

    # Session list
    session_list = []
    for row in session_rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        session_list.append(d)

    # Pain history oldest-first for chart; most-recent-first slice for avg
    pain_history = [
        {"value": r["pain_level"], "date": r["created_at"]}
        for r in reversed(pain_rows)
    ]
    recent_pain = [r["pain_level"] for r in pain_rows[:7]]
    avg_pain = round(sum(recent_pain) / len(recent_pain), 1) if recent_pain else 0.0

    # Streak
    streak = _compute_streak(session_list)

    # Process assessments: closure chart + best hold
    best_hold = 0.0
    closure_chart = []

    for i, row in enumerate(assessment_rows):
        d = dict(row)
        try:
            results = json.loads(d["results"])
        except Exception:
            continue

        ex_list = results.get("exercises", [])
        point = {"idx": i + 1, "date": d["created_at"]}

        for ex in ex_list:
            ex_type = ex.get("exercise", "")
            attempts = ex.get("attempts", [])

            closures = []
            for a in attempts:
                v = a.get("closure")
                h = a.get("hold_s", 0) or 0
                try:
                    if float(h) > best_hold:
                        best_hold = float(h)
                except Exception:
                    pass
                if v is not None:
                    try:
                        closures.append(float(v))
                    except Exception:
                        pass

            avg_c = round(sum(closures) / len(closures), 3) if closures else None

            if ex_type == "open_palm":
                point["palm"] = avg_c
            elif ex_type == "mid_flexion":
                point["mid_flex"] = avg_c
            elif ex_type == "full_fist":
                point["fist"] = avg_c

        if any(k in point for k in ("palm", "mid_flex", "fist")):
            closure_chart.append(point)

    return {
        "total_sessions": total_sessions,
        "streak": streak,
        "best_hold": round(best_hold, 1),
        "avg_pain": avg_pain,
        "closure_chart": closure_chart,
        "session_history": session_list,
        "pain_history": pain_history,
    }


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

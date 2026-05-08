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
# Exercise display-name lookup
# --------------------------------------------------------------------------- #

_EX_SHORT_NAMES: dict[str, str] = {
    # Plan exercise IDs (stored by session.html via the plan)
    "open_palm_hold":    "Palm",
    "mid_flexion_hold":  "Mid flex",
    "full_fist_close":   "Fist",
    "thumb_index_pinch": "Pinch",
    # Assessment-style keys (stored by seed_demo_data / assessment.html)
    "open_palm":  "Palm",
    "mid_flexion": "Mid flex",
    "full_fist":  "Fist",
}


def _ex_display_name(ex: dict) -> str:
    """Return a short patient-facing label for a stored exercise dict."""
    key = ex.get("name") or ex.get("exercise", "")
    return _EX_SHORT_NAMES.get(key, key.replace("_", " ").title())


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
            -- session  (completed or in-progress workout log)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS session (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                  INTEGER NOT NULL DEFAULT 1,
                plan_id                  INTEGER,
                exercises                TEXT    NOT NULL DEFAULT '[]',  -- JSON: [{exercise, sets, reps_done, …}]
                pain_before              INTEGER,
                pain_after               INTEGER,
                duration_s               INTEGER,
                status                   TEXT    NOT NULL DEFAULT 'completed',
                last_exercise_index      INTEGER NOT NULL DEFAULT 0,
                exercises_completed_json TEXT,
                completed_at             TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES user(id),
                FOREIGN KEY (plan_id) REFERENCES rehab_plan(id)
            );

            -- --------------------------------------------------------
            -- progression_log  (adaptive plan change history)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS progression_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL DEFAULT 1,
                exercise        TEXT    NOT NULL,
                old_value       TEXT,
                new_value       TEXT,
                description     TEXT,
                acknowledged_at TIMESTAMP NULL,
                progressed_at   TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES user(id)
            );

            -- --------------------------------------------------------
            -- flags_log  (cross-validation health flags)
            -- --------------------------------------------------------
            CREATE TABLE IF NOT EXISTS flags_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL DEFAULT 1,
                flag_type    TEXT    NOT NULL,
                detected_at  TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                dismissed    INTEGER  NOT NULL DEFAULT 0,
                dismissed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user(id)
            );
        """)

    # Migration: add new session columns to databases created before this version.
    # SQLite ALTER TABLE defaults existing rows to the column default, so old sessions
    # get status='completed' automatically.
    with _connect() as conn:
        existing_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(session)").fetchall()
        }
        for col, definition in [
            ("status",                   "TEXT    NOT NULL DEFAULT 'completed'"),
            ("last_exercise_index",      "INTEGER NOT NULL DEFAULT 0"),
            ("exercises_completed_json", "TEXT"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE session ADD COLUMN {col} {definition}")



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
    d["exercises"]        = json.loads(d["exercises"])
    d["referral_required"] = d.get("notes", "").startswith("__REFERRAL__")
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


def save_session_checkpoint(
    plan_id: Optional[int],
    exercise_index: int,
    exercises_completed_json: str,
    session_id: Optional[int] = None,
) -> int:
    """Upsert an in-progress session checkpoint. Returns the session row id."""
    with _connect() as conn:
        if session_id:
            conn.execute("""
                UPDATE session
                SET last_exercise_index=?, exercises_completed_json=?,
                    completed_at=datetime('now')
                WHERE id=? AND user_id=1 AND status='in_progress'
            """, (exercise_index, exercises_completed_json, session_id))
            return session_id
        # No known id — insert a fresh in-progress row
        cur = conn.execute("""
            INSERT INTO session
                (user_id, plan_id, exercises, status, last_exercise_index, exercises_completed_json)
            VALUES (1, ?, '[]', 'in_progress', ?, ?)
        """, (plan_id, exercise_index, exercises_completed_json))
        return cur.lastrowid


def mark_session_completed(
    session_id: int,
    exercises: list,
    pain_before: Optional[int] = None,
    pain_after: Optional[int] = None,
    duration_s: Optional[int] = None,
) -> None:
    """Finalise a session: mark it completed and store all result data."""
    with _connect() as conn:
        conn.execute("""
            UPDATE session
            SET status='completed', exercises=?, pain_before=?, pain_after=?,
                duration_s=?, completed_at=datetime('now')
            WHERE id=? AND user_id=1
        """, (json.dumps(exercises), pain_before, pain_after, duration_s, session_id))


def get_incomplete_session() -> Optional[dict]:
    """Return the most recent in-progress session started within the last 24 hours, or None."""
    with _connect() as conn:
        row = conn.execute("""
            SELECT * FROM session
            WHERE user_id=1 AND status='in_progress'
              AND completed_at >= datetime('now', '-24 hours')
            ORDER BY id DESC LIMIT 1
        """).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["exercises_completed"] = (
        json.loads(d["exercises_completed_json"])
        if d.get("exercises_completed_json")
        else []
    )
    return d


def discard_incomplete_session(session_id: int) -> None:
    """Delete an abandoned in-progress session row."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM session WHERE id=? AND user_id=1 AND status='in_progress'",
            (session_id,),
        )


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
    """Return aggregated session data for the dashboard (completed sessions only)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session WHERE user_id=1 AND status='completed' ORDER BY id DESC LIMIT 30"
        ).fetchall()
        pain_rows = conn.execute(
            "SELECT pain_level FROM self_report WHERE user_id=1 ORDER BY id DESC LIMIT 5"
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM session WHERE user_id=1 AND status='completed'"
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
            DROP TABLE IF EXISTS flags_log;
            DROP TABLE IF EXISTS progression_log;
            DROP TABLE IF EXISTS session;
            DROP TABLE IF EXISTS rehab_plan;
            DROP TABLE IF EXISTS self_report;
            DROP TABLE IF EXISTS assessment;
            DROP TABLE IF EXISTS user;
        """)
    init_db()


def seed_demo_data() -> None:
    """Wipe all data and insert a realistic demo account for showcasing."""
    import datetime
    from app.plan import generate_plan

    reset_db()

    exercise_pool = ["open_palm_hold", "mid_flexion_hold", "full_fist_close", "thumb_index_pinch"]

    # ── Session specs: (year, month, day, pain_before, pain_after, duration_s, ex_count)
    # March: 12 sessions ~every 2–3 days, gradual improvement
    # April: 14 sessions, more consistent, pain dropping from 3–4 to 1–2
    # May:   sessions up to May 5 (today is May 8)
    session_specs = [
        (2026, 3,  1, 4, 4, 720,  3),
        (2026, 3,  4, 4, 4, 780,  3),
        (2026, 3,  8, 3, 4, 720,  3),
        (2026, 3, 11, 4, 3, 780,  3),
        (2026, 3, 14, 3, 4, 840,  3),
        (2026, 3, 17, 3, 3, 780,  3),
        (2026, 3, 20, 4, 3, 840,  4),
        (2026, 3, 22, 3, 3, 720,  3),
        (2026, 3, 25, 3, 3, 780,  4),
        (2026, 3, 27, 3, 2, 840,  4),
        (2026, 3, 29, 2, 3, 780,  3),
        (2026, 3, 31, 3, 2, 840,  4),
        (2026, 4,  1, 3, 3, 780,  3),
        (2026, 4,  3, 3, 3, 840,  4),
        (2026, 4,  5, 2, 3, 780,  3),
        (2026, 4,  7, 3, 2, 840,  4),
        (2026, 4, 10, 2, 3, 900,  4),
        (2026, 4, 12, 2, 2, 840,  3),
        (2026, 4, 15, 3, 2, 900,  4),
        (2026, 4, 17, 2, 2, 840,  4),
        (2026, 4, 19, 2, 2, 780,  3),
        (2026, 4, 22, 2, 1, 900,  4),
        (2026, 4, 24, 1, 2, 840,  4),
        (2026, 4, 26, 2, 1, 900,  4),
        (2026, 4, 28, 1, 2, 840,  3),
        (2026, 4, 30, 2, 1, 960,  4),
        (2026, 5,  1, 2, 1, 840,  4),
        (2026, 5,  3, 1, 2, 900,  4),
        (2026, 5,  5, 2, 1, 840,  4),
    ]
    n_sessions = len(session_specs)  # 29

    # ── Daily check-in pain history: March 1 – May 8 (69 days)
    march_pain = [4,3,4,4,3,4,3,4,4,3,3,4,3,3,4,3,4,3,3,4,3,3,3,2,3,3,2,3,3,2,3]
    april_pain = [3,2,3,3,2,3,2,3,2,2,3,2,2,3,2,2,1,2,2,1,2,2,1,2,1,2,1,2,1,2]
    may_pain   = [2,1,2,1,2,1,1,2]   # May 1–8

    # ── Assessment baseline (done Feb 28 before the exercise period)
    assessment_results = {
        "pain_level": 4,
        "exercises": [
            {
                "exercise":   "open_palm",
                "attempts":   [
                    {"closure": 0.11, "hold_s": 7.8},
                    {"closure": 0.12, "hold_s": 8.2},
                    {"closure": 0.10, "hold_s": 7.5},
                ],
                "pain_after":  1,
                "skipped":     False,
                "skip_reason": "",
            },
            {
                "exercise":   "mid_flexion",
                "attempts":   [
                    {"closure": 0.47, "hold_s": 5.1},
                    {"closure": 0.49, "hold_s": 5.4},
                    {"closure": 0.46, "hold_s": 4.9},
                ],
                "pain_after":  2,
                "skipped":     False,
                "skip_reason": "",
            },
            {
                "exercise":   "full_fist",
                "attempts":   [
                    {"closure": 0.70, "hold_s": 3.0},
                    {"closure": 0.72, "hold_s": 3.2},
                    {"closure": 0.69, "hold_s": 2.9},
                ],
                "pain_after":  4,
                "skipped":     False,
                "skip_reason": "",
            },
        ],
    }
    score = round((0.11 + 0.47 + 0.70) / 3 * 100, 1)

    user_row_for_plan = {
        "name": "Alex Johnson", "age": None, "affected_hand": "Right",
        "condition": "stroke",
        "goals": ["Regain grip strength and improve daily function"],
    }
    plan_data = generate_plan(
        user=user_row_for_plan,
        assessment={"results": assessment_results, "score": score},
    )

    with _connect() as conn:
        # ── user (created March 1 — day 1 of the 3-month window)
        conn.execute("""
            INSERT INTO user (id, name, age, affected_hand, condition, goals, created_at, updated_at)
            VALUES (1, 'Alex Johnson', NULL, 'Right', 'stroke',
                    '["Regain grip strength and improve daily function"]',
                    '2026-03-01 08:00:00', '2026-03-01 08:00:00')
        """)

        # ── assessment (Feb 28)
        cur = conn.execute("""
            INSERT INTO assessment (user_id, results, score, notes, created_at)
            VALUES (1, ?, ?, '', '2026-02-28 09:30:00')
        """, (json.dumps(assessment_results), score))
        assessment_id = cur.lastrowid

        # ── rehab_plan
        cur = conn.execute("""
            INSERT INTO rehab_plan
                (user_id, exercises, sessions_per_week, notes, source_assessment_id, created_at)
            VALUES (1, ?, ?, ?, ?, '2026-02-28 09:35:00')
        """, (json.dumps(plan_data["exercises"]), plan_data["sessions_per_week"],
               plan_data["notes"], assessment_id))
        plan_id = cur.lastrowid

        # ── session
        for i, (yr, mo, day, pb, pa, dur, ex_cnt) in enumerate(session_specs):
            progress     = i / (n_sessions - 1)
            session_date = datetime.date(yr, mo, day)
            session_ts   = f"{session_date} 10:15:00"

            exercises = []
            for ex_name in exercise_pool[:ex_cnt]:
                if i >= 26:
                    # Last 3 sessions: declining closures despite low pain —
                    # this is what triggers inconsistent_pain_closure in trend analysis.
                    step = i - 26
                    if ex_name == "open_palm_hold":
                        closure = round(0.15 - step * 0.05, 3)
                    elif ex_name == "mid_flexion_hold":
                        closure = round(0.52 - step * 0.05, 3)
                    elif ex_name == "full_fist_close":
                        closure = round(0.72 - step * 0.10, 3)
                    else:
                        closure = round(0.38 - step * 0.05, 3)
                else:
                    if ex_name == "open_palm_hold":
                        closure = round(0.10 + progress * 0.08, 3)
                    elif ex_name == "mid_flexion_hold":
                        closure = round(0.46 + progress * 0.12, 3)
                    elif ex_name == "full_fist_close":
                        closure = round(0.68 + progress * 0.14, 3)
                    else:
                        closure = round(0.28 + progress * 0.18, 3)

                avg_hold = round(3.5 + progress * 4.0, 1)
                exercises.append({
                    "exercise":    ex_name,
                    "sets_done":   3,
                    "reps_done":   5,
                    "avg_closure": closure,
                    "avg_hold_s":  avg_hold,
                })

            exercises_json = json.dumps(exercises)
            conn.execute("""
                INSERT INTO session
                    (user_id, plan_id, exercises, pain_before, pain_after, duration_s,
                     status, last_exercise_index, exercises_completed_json, completed_at)
                VALUES (1, ?, ?, ?, ?, ?, 'completed', ?, ?, ?)
            """, (plan_id, exercises_json, pb, pa, dur,
                  ex_cnt - 1, exercises_json, session_ts))

        # ── self_report (daily check-ins March 1 – May 8)
        all_pain   = march_pain + april_pain + may_pain
        start_date = datetime.date(2026, 3, 1)
        for i, pain_val in enumerate(all_pain):
            checkin_date = start_date + datetime.timedelta(days=i)
            conn.execute("""
                INSERT INTO self_report (user_id, pain_level, limitations, goal, created_at)
                VALUES (1, ?, '', '', ?)
            """, (pain_val, f"{checkin_date} 08:30:00"))

        # ── progression_log (3 events showing the plan got harder over time)
        progression_events = [
            {
                "exercise":       "open_palm_hold",
                "old_value":      "hold_s=5",
                "new_value":      "hold_s=7",
                "description":    "Open Palm Hold hold increased from 5s to 7s",
                "progressed_at":  "2026-03-25 10:30:00",
                "acknowledged_at": "2026-03-25 10:35:00",
            },
            {
                "exercise":       "mid_flexion_hold",
                "old_value":      "reps=5",
                "new_value":      "reps=7",
                "description":    "Mid Flexion Hold reps increased from 5 to 7",
                "progressed_at":  "2026-04-12 10:30:00",
                "acknowledged_at": "2026-04-12 10:35:00",
            },
            {
                "exercise":       "full_fist_close",
                "old_value":      "reps=8",
                "new_value":      "reps=10",
                "description":    "Full Fist Close reps increased from 8 to 10",
                "progressed_at":  "2026-04-28 10:30:00",
                "acknowledged_at": "2026-04-28 10:35:00",
            },
        ]
        conn.executemany("""
            INSERT INTO progression_log
                (user_id, exercise, old_value, new_value, description,
                 acknowledged_at, progressed_at)
            VALUES (1, :exercise, :old_value, :new_value, :description,
                    :acknowledged_at, :progressed_at)
        """, progression_events)

        # ── flags_log (inconsistent_pain_closure active — shows on check-in screen)
        conn.execute("""
            INSERT INTO flags_log (user_id, flag_type, detected_at, dismissed)
            VALUES (1, 'inconsistent_pain_closure', '2026-05-05 10:20:00', 0)
        """)


# --------------------------------------------------------------------------- #
# flags_log
# --------------------------------------------------------------------------- #

def save_flag(user_id: int, flag_type: str) -> int:
    """Insert a flag unless the same type was already logged (and not dismissed) within 7 days."""
    with _connect() as conn:
        existing = conn.execute("""
            SELECT id FROM flags_log
            WHERE user_id=? AND flag_type=? AND dismissed=0
              AND detected_at >= datetime('now', '-7 days')
        """, (user_id, flag_type)).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            "INSERT INTO flags_log (user_id, flag_type) VALUES (?, ?)",
            (user_id, flag_type),
        )
        return cur.lastrowid


def dismiss_flag(flag_id: int) -> None:
    with _connect() as conn:
        conn.execute("""
            UPDATE flags_log SET dismissed=1, dismissed_at=datetime('now') WHERE id=?
        """, (flag_id,))


def get_active_flags(user_id: int = 1) -> list:
    """Return all non-dismissed flags, newest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM flags_log WHERE user_id=? AND dismissed=0 ORDER BY detected_at DESC
        """, (user_id,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Trend analysis
# --------------------------------------------------------------------------- #

def get_trend_analysis(user_id: int = 1) -> dict:
    """
    Analyse the last 7 completed sessions for pain / closure cross-validation.
    Returns flags (list of flag-type strings) and supporting data.
    """
    with _connect() as conn:
        rows = conn.execute("""
            SELECT exercises, pain_before, pain_after
            FROM session
            WHERE user_id=? AND status='completed'
            ORDER BY id DESC LIMIT 7
        """, (user_id,)).fetchall()

    if len(rows) < 3:
        return {"flags": [], "session_count": len(rows)}

    sessions = list(reversed(rows))   # oldest first
    n = len(sessions)

    # Pain values — prefer pain_after, fall back to pain_before
    pain_vals: list = []
    for s in sessions:
        p = s["pain_after"] if s["pain_after"] is not None else s["pain_before"]
        pain_vals.append(int(p) if p is not None else 0)

    # Closure values per exercise name and per-session average
    ex_closures: dict = {}          # name -> [float, ...]  (one entry per session that has it)
    session_avg_closures: list = []

    for s in sessions:
        try:
            exs = json.loads(s["exercises"])
        except Exception:
            exs = []
        sess_cl: list = []
        for ex in exs:
            name = ex.get("exercise") or ex.get("name", "")
            c = ex.get("avg_closure")
            if c is None:
                continue
            try:
                c = float(c)
            except (TypeError, ValueError):
                continue
            sess_cl.append(c)
            ex_closures.setdefault(name, []).append(c)
        if sess_cl:
            session_avg_closures.append(sum(sess_cl) / len(sess_cl))

    flags: list = []

    # ── Pain trend ──────────────────────────────────────────────────────────
    if n >= 6:
        if sum(pain_vals[-3:]) / 3 - sum(pain_vals[:3]) / 3 > 1.5:
            flags.append("pain_increasing")

    # ── Closure trend per exercise ──────────────────────────────────────────
    closure_declining: list = []
    for ex_name, cl_list in ex_closures.items():
        if len(cl_list) >= 6:
            if sum(cl_list[:3]) / 3 - sum(cl_list[-3:]) / 3 > 0.08:
                closure_declining.append(ex_name)

    if closure_declining:
        flags.append("closure_declining")

    # ── Cross-validation ────────────────────────────────────────────────────
    if closure_declining and n >= 3 and all(p <= 2 for p in pain_vals[-3:]):
        flags.append("inconsistent_pain_closure")

    if n >= 3 and all(p >= 6 for p in pain_vals[-3:]) and len(session_avg_closures) >= 6:
        if sum(session_avg_closures[-3:]) / 3 - sum(session_avg_closures[:3]) / 3 > 0.05:
            flags.append("high_pain_good_closure")

    if len(session_avg_closures) >= 5:
        last5 = session_avg_closures[-5:]
        if max(last5) - min(last5) < 0.05:
            flags.append("plateau")

    if len(session_avg_closures) >= 2:
        if session_avg_closures[-2] - session_avg_closures[-1] > 0.15:
            flags.append("rapid_decline")

    return {
        "flags": flags,
        "session_count": n,
        "closure_declining_exercises": closure_declining,
        "pain_trend": pain_vals,
    }


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
            "SELECT COUNT(*) FROM session WHERE user_id=1 AND status='completed'"
        ).fetchone()[0]

    # Session list — annotate exercises with a short display name.
    # Include both completed and in-progress sessions so the progress page can
    # show incomplete rows; callers must check s["status"] to distinguish them.
    session_list = []
    for row in session_rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        for ex in d["exercises"]:
            ex["display_name"] = _ex_display_name(ex)
        # For in-progress sessions, also expose the partially-completed exercise list
        if d.get("exercises_completed_json"):
            d["exercises_completed"] = json.loads(d["exercises_completed_json"])
            for ex in d["exercises_completed"]:
                ex["display_name"] = _ex_display_name(ex)
        else:
            d["exercises_completed"] = d["exercises"]
        session_list.append(d)

    # Pain history oldest-first for chart; most-recent-first slice for avg
    pain_history = [
        {"value": r["pain_level"], "date": r["created_at"]}
        for r in reversed(pain_rows)
    ]
    recent_pain = [r["pain_level"] for r in pain_rows[:7]]
    avg_pain = round(sum(recent_pain) / len(recent_pain), 1) if recent_pain else 0.0

    # Streak — completed sessions only
    completed = [s for s in session_list if s.get("status", "completed") == "completed"]
    streak = _compute_streak(completed)

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


def get_last_n_sessions(n: int) -> list[dict]:
    """Return the last n completed sessions, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session WHERE user_id=1 ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        result.append(d)
    return result


# --------------------------------------------------------------------------- #
# progression_log
# --------------------------------------------------------------------------- #

def log_progression(entries: list[dict]) -> None:
    """Insert one row per progression event. Each entry needs exercise, old_value,
    new_value, description."""
    with _connect() as conn:
        conn.executemany(
            """INSERT INTO progression_log
               (user_id, exercise, old_value, new_value, description)
               VALUES (1, :exercise, :old_value, :new_value, :description)""",
            entries,
        )


def get_unacknowledged_progression() -> list[dict]:
    """Return progression_log rows that the user has not yet dismissed."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM progression_log "
            "WHERE user_id=1 AND acknowledged_at IS NULL "
            "ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def acknowledge_progression() -> None:
    """Mark all pending progression events as seen."""
    with _connect() as conn:
        conn.execute(
            "UPDATE progression_log "
            "SET acknowledged_at = datetime('now') "
            "WHERE user_id=1 AND acknowledged_at IS NULL"
        )


def get_progression_log() -> list[dict]:
    """Return the full progression history, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM progression_log WHERE user_id=1 ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_monthly_history() -> list:
    """Return all sessions grouped by month, newest first, for the monthly history panel."""
    import datetime
    from collections import defaultdict

    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session WHERE user_id=1 ORDER BY completed_at DESC"
        ).fetchall()

    months: dict = defaultdict(list)
    for row in rows:
        d = dict(row)
        d["exercises"] = json.loads(d["exercises"])
        for ex in d["exercises"]:
            ex["display_name"] = _ex_display_name(ex)
        # Expose partially-completed exercises for in-progress sessions
        if d.get("exercises_completed_json"):
            d["exercises_completed"] = json.loads(d["exercises_completed_json"])
            for ex in d["exercises_completed"]:
                ex["display_name"] = _ex_display_name(ex)
        else:
            d["exercises_completed"] = d["exercises"]
        months[d["completed_at"][:7]].append(d)

    result = []
    for month_key in sorted(months.keys(), reverse=True):
        sessions = months[month_key]
        year, month = int(month_key[:4]), int(month_key[5:7])
        # Counts and stats only from completed sessions
        completed = [s for s in sessions if s.get("status", "completed") == "completed"]
        pain_vals = [s["pain_after"] for s in completed if s.get("pain_after") is not None]
        avg_pain  = round(sum(pain_vals) / len(pain_vals), 1) if pain_vals else None
        best_closure: Optional[float] = None
        for s in completed:
            for ex in s["exercises"]:
                c = ex.get("avg_closure")
                if c is not None and (best_closure is None or c > best_closure):
                    best_closure = c
        result.append({
            "month_key":      month_key,
            "month_name":     datetime.date(year, month, 1).strftime("%B %Y"),
            "sessions_count": len(completed),
            "avg_pain":       avg_pain,
            "best_closure":   round(best_closure, 3) if best_closure is not None else None,
            "sessions":       sessions,
        })
    return result


def get_insights() -> dict:
    """Compute insight data for the dashboard and progress pages.

    Returns
    -------
    activity_grid        : list[bool]   — 7 bools, index 0 = 6 days ago, index 6 = today
    sessions_last_7d     : int
    avg_pain_this_week   : float | None — average pain_level from self_reports in last 7 days
    avg_pain_last_week   : float | None — average pain_level from self_reports 7–14 days ago
    palm_improvement     : float | None — % change in avg open_palm closure (first vs latest assessment)
    mid_flex_improvement : float | None
    fist_improvement     : float | None
    best_hold_s          : float        — highest hold_s seen across all assessments
    best_hold_date       : str          — ISO date of that assessment
    best_hold_type       : str          — exercise type (e.g. "full_fist")
    best_closures        : dict         — {"palm": float|None, "mid_flex": float|None, "fist": float|None}
    """
    import datetime

    with _connect() as conn:
        assessment_rows = conn.execute(
            "SELECT * FROM assessment WHERE user_id=1 ORDER BY id ASC"
        ).fetchall()
        session_rows = conn.execute(
            "SELECT completed_at FROM session WHERE user_id=1 AND status='completed'"
        ).fetchall()
        pain_rows = conn.execute(
            "SELECT pain_level, created_at FROM self_report "
            "WHERE user_id=1 ORDER BY id DESC LIMIT 60"
        ).fetchall()

    today = datetime.date.today()

    # ── Activity grid (last 7 days) ──────────────────────────────────
    session_dates: set = set()
    for row in session_rows:
        try:
            d = datetime.datetime.strptime(row["completed_at"][:10], "%Y-%m-%d").date()
            session_dates.add(d)
        except Exception:
            pass

    activity_grid = [
        (today - datetime.timedelta(days=i)) in session_dates
        for i in range(6, -1, -1)
    ]
    sessions_last_7d = sum(1 for d in session_dates if (today - d).days < 7)

    # ── Pain averages ─────────────────────────────────────────────────
    this_week_pain: list = []
    last_week_pain: list = []
    for row in pain_rows:
        try:
            d = datetime.datetime.strptime(row["created_at"][:10], "%Y-%m-%d").date()
            days_ago = (today - d).days
            if days_ago < 7:
                this_week_pain.append(row["pain_level"])
            elif days_ago < 14:
                last_week_pain.append(row["pain_level"])
        except Exception:
            pass

    avg_pain_this_week = (
        round(sum(this_week_pain) / len(this_week_pain), 1) if this_week_pain else None
    )
    avg_pain_last_week = (
        round(sum(last_week_pain) / len(last_week_pain), 1) if last_week_pain else None
    )

    # ── Closure data from assessments ────────────────────────────────
    palm_avgs: list = []
    mid_avgs:  list = []
    fist_avgs: list = []
    best_hold_s    = 0.0
    best_hold_date = ""
    best_hold_type = ""
    best_closures  = {"palm": None, "mid_flex": None, "fist": None}

    for row in assessment_rows:
        try:
            results = json.loads(row["results"])
        except Exception:
            continue

        date_str = row["created_at"][:10]
        for ex in results.get("exercises", []):
            ex_type  = ex.get("exercise", "")
            attempts = ex.get("attempts", [])

            closures: list = []
            for a in attempts:
                h = a.get("hold_s") or 0
                try:
                    if float(h) > best_hold_s:
                        best_hold_s    = float(h)
                        best_hold_date = date_str
                        best_hold_type = ex_type
                except Exception:
                    pass

                v = a.get("closure")
                if v is not None:
                    try:
                        closures.append(float(v))
                    except Exception:
                        pass

            if not closures:
                continue

            avg_c = round(sum(closures) / len(closures), 3)

            if ex_type == "open_palm":
                palm_avgs.append(avg_c)
                if best_closures["palm"] is None or avg_c > best_closures["palm"]:
                    best_closures["palm"] = avg_c
            elif ex_type == "mid_flexion":
                mid_avgs.append(avg_c)
                if best_closures["mid_flex"] is None or avg_c > best_closures["mid_flex"]:
                    best_closures["mid_flex"] = avg_c
            elif ex_type == "full_fist":
                fist_avgs.append(avg_c)
                if best_closures["fist"] is None or avg_c > best_closures["fist"]:
                    best_closures["fist"] = avg_c

    def _improvement(avgs: list) -> Optional[float]:
        if len(avgs) < 2 or avgs[0] == 0:
            return None
        return round((avgs[-1] - avgs[0]) / avgs[0] * 100, 1)

    return {
        "activity_grid":        activity_grid,
        "sessions_last_7d":     sessions_last_7d,
        "avg_pain_this_week":   avg_pain_this_week,
        "avg_pain_last_week":   avg_pain_last_week,
        "palm_improvement":     _improvement(palm_avgs),
        "mid_flex_improvement": _improvement(mid_avgs),
        "fist_improvement":     _improvement(fist_avgs),
        "best_hold_s":          round(best_hold_s, 1),
        "best_hold_date":       best_hold_date,
        "best_hold_type":       best_hold_type,
        "best_closures":        best_closures,
    }


def get_recovery_day() -> int:
    """Return the number of days since the user profile was created (1 = day of creation)."""
    import datetime
    with _connect() as conn:
        row = conn.execute("SELECT created_at FROM user WHERE id=1").fetchone()
    if row is None:
        return 1
    try:
        created = datetime.datetime.strptime(row["created_at"][:10], "%Y-%m-%d").date()
        return max(1, (datetime.date.today() - created).days + 1)
    except Exception:
        return 1

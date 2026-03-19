"""
app/plan.py — Rule-based rehabilitation plan generator.

Entry point
-----------
  generate_plan(user, assessment) -> dict

The returned dict matches the rehab_plan table schema:
  {
    "exercises":         [ {id, name, sets, reps, rest_s, target_gesture}, … ],
    "sessions_per_week": int,
    "notes":             str,
  }
"""

from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------- #
# Exercise catalogue
# --------------------------------------------------------------------------- #
# Each entry may specify a `target_gesture` that maps to the model's LABELS,
# allowing the session page to auto-detect when a rep is complete.

_CATALOGUE: dict[str, list[dict]] = {
    "grip_strength": [
        {
            "id": "power_fist",
            "name": "Power Fist",
            "sets": 3, "reps": 10, "rest_s": 30,
            "target_gesture": "fist",
            "description": "Close hand into a full fist, hold 2 s, then fully open.",
        },
        {
            "id": "finger_extension_band",
            "name": "Finger Extension Band",
            "sets": 3, "reps": 12, "rest_s": 30,
            "target_gesture": "palm",
            "description": "Place rubber band around fingers; spread outward against resistance.",
        },
    ],
    "rom": [
        {
            "id": "tendon_glide",
            "name": "Tendon Glide",
            "sets": 2, "reps": 10, "rest_s": 20,
            "target_gesture": "grabbing",
            "description": "Cycle through straight → hook → full fist → tabletop → straight.",
        },
        {
            "id": "composite_flexion",
            "name": "Composite Flexion",
            "sets": 2, "reps": 10, "rest_s": 20,
            "target_gesture": "fist",
            "description": "Slowly close all fingers into the tightest fist you can manage.",
        },
    ],
    "dexterity": [
        {
            "id": "thumb_opposition",
            "name": "Thumb Opposition",
            "sets": 3, "reps": 10, "rest_s": 20,
            "target_gesture": "thumb_index",
            "description": "Touch thumb to each fingertip in sequence, both directions.",
        },
        {
            "id": "pinch_release",
            "name": "Pinch & Release",
            "sets": 3, "reps": 15, "rest_s": 20,
            "target_gesture": "thumb_index",
            "description": "Pinch a soft object between thumb and index finger; release fully.",
        },
    ],
    "general": [
        {
            "id": "wrist_circles",
            "name": "Wrist Circles",
            "sets": 2, "reps": 10, "rest_s": 15,
            "target_gesture": None,
            "description": "Rotate wrist clockwise 10 × then counter-clockwise 10 ×.",
        },
        {
            "id": "finger_spread",
            "name": "Finger Spread & Close",
            "sets": 2, "reps": 10, "rest_s": 15,
            "target_gesture": "palm",
            "description": "Spread all fingers as wide as possible, then close gently.",
        },
    ],
}

# gesture label → implied ROM score band (used to select exercises from assessment)
_GESTURE_ROM: dict[str, float] = {
    "neutral":     0.5,
    "palm":        0.3,   # limited — can open but not much more
    "grabbing":    0.6,
    "fist":        1.0,   # full closure achieved
    "thumb_index": 0.7,
}


# --------------------------------------------------------------------------- #
# Plan generator
# --------------------------------------------------------------------------- #

def generate_plan(
    user: Optional[dict] = None,
    assessment: Optional[dict] = None,
) -> dict:
    """
    Build a personalised exercise plan.

    Parameters
    ----------
    user       : dict from database.get_user()
    assessment : dict from database.get_latest_assessment()

    Returns
    -------
    dict with keys: exercises, sessions_per_week, notes
    """
    user       = user or {}
    assessment = assessment or {}

    goals  = user.get("goals", [])
    pain   = int(assessment.get("results", {}).get("pain_level", 0))
    score  = float(assessment.get("score", 50.0))    # 0–100 ROM score

    exercises: list[dict] = []

    # Always include general warm-up
    exercises.extend(_CATALOGUE["general"])

    # Add goal-specific exercises
    for goal in goals:
        if goal in _CATALOGUE:
            exercises.extend(_CATALOGUE[goal])

    # If no specific goals, fall back to ROM exercises
    if not goals:
        exercises.extend(_CATALOGUE["rom"])

    # Scale volume based on pain and ROM score
    exercises = [_scale_exercise(ex, pain, score) for ex in exercises]

    sessions_per_week = _sessions_per_week(pain, score)
    notes = _build_notes(pain, score, goals)

    return {
        "exercises":         exercises,
        "sessions_per_week": sessions_per_week,
        "notes":             notes,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _scale_exercise(ex: dict, pain: int, score: float) -> dict:
    """Return a copy of the exercise with volume adapted to pain/ROM."""
    ex = dict(ex)  # shallow copy — don't mutate catalogue

    if pain >= 7:
        ex["sets"] = max(1, ex["sets"] - 1)
        ex["reps"] = max(4, ex["reps"] - 4)
    elif pain >= 4:
        ex["reps"] = max(5, ex["reps"] - 2)

    # Low ROM — reduce reps further, add a note
    if score < 40:
        ex["reps"] = max(4, ex["reps"] - 2)
        ex["description"] = ex["description"] + " (gentle — work within your pain-free range)"

    return ex


def _sessions_per_week(pain: int, score: float) -> int:
    if pain >= 7:
        return 2
    if pain >= 4 or score < 40:
        return 3
    return 4


def _build_notes(pain: int, score: float, goals: list[str]) -> str:
    lines: list[str] = []

    if pain == 0:
        lines.append("No pain reported — full programme at prescribed volume.")
    elif pain <= 3:
        lines.append("Mild discomfort — monitor throughout; stop if pain increases beyond 4/10.")
    elif pain <= 6:
        lines.append("Moderate pain — volume reduced; prioritise gentle ROM over strength.")
    else:
        lines.append("High pain — very gentle movements only. Consult your therapist before continuing.")

    if score < 40:
        lines.append("Limited ROM detected — focus on range before adding resistance.")
    elif score >= 80:
        lines.append("Good ROM — you can begin progressing toward strengthening exercises.")

    if not goals:
        lines.append("No specific goals set — complete your profile to get a targeted programme.")

    return " ".join(lines)

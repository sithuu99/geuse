"""
app/plan.py — Rule-based rehabilitation plan generator.

Only includes exercises the live model can actually track:
  - Closure value (0–1 continuous) for hold and flexion targets
  - thumb_index class detection for pinch reps

Entry point
-----------
  generate_plan(user, assessment) -> dict

Each exercise in the returned plan includes:
  id, name, sets, reps, hold_s, rest_s,
  tracking_type   : "hold" | "reps"
  target_closure_min : float | None   (inclusive lower bound)
  target_closure_max : float | None   (inclusive upper bound)
  target_class       : str  | None    (model class name, used when closure bounds are None)
  target_gesture, description
"""

from __future__ import annotations

from typing import Optional


# --------------------------------------------------------------------------- #
# Exercise catalogue — only movements the model can measure in real time
# --------------------------------------------------------------------------- #

_EXERCISES: dict[str, dict] = {
    # ── Hold-type: user holds a position; rep = one completed hold ──────────
    "open_palm_hold": {
        "id":                  "open_palm_hold",
        "name":                "Open Palm Hold",
        "sets":                3,
        "reps":                5,        # number of holds per set
        "hold_s":              5,        # target hold duration in seconds
        "rest_s":              20,
        "tracking_type":       "hold",
        "target_closure_min":  0.0,
        "target_closure_max":  0.2,
        "target_class":        "palm",
        "target_gesture":      "palm",
        "description": (
            "Spread your hand as wide as you can. "
            "Hold the open position for the target duration, then relax."
        ),
    },
    "mid_flexion_hold": {
        "id":                  "mid_flexion_hold",
        "name":                "Mid Flexion Hold",
        "sets":                3,
        "reps":                5,
        "hold_s":              5,
        "rest_s":              20,
        "tracking_type":       "hold",
        "target_closure_min":  0.4,
        "target_closure_max":  0.6,
        "target_class":        "grabbing",
        "target_gesture":      "grabbing",
        "description": (
            "Curl your fingers halfway into a relaxed half-grip. "
            "Hold steady at the midpoint for the target duration."
        ),
    },

    # ── Rep-type: user repeats a movement; rep counted on each detection ────
    "full_fist_close": {
        "id":                  "full_fist_close",
        "name":                "Full Fist Close",
        "sets":                3,
        "reps":                10,
        "hold_s":              None,
        "rest_s":              30,
        "tracking_type":       "reps",
        "target_closure_min":  0.8,
        "target_closure_max":  1.0,
        "target_class":        "fist",
        "target_gesture":      "fist",
        "description": (
            "Close your hand into the tightest fist you can manage, "
            "then fully open. Each complete close counts as one rep."
        ),
    },
    "thumb_index_pinch": {
        "id":                  "thumb_index_pinch",
        "name":                "Thumb-Index Pinch",
        "sets":                3,
        "reps":                10,
        "hold_s":              None,
        "rest_s":              20,
        "tracking_type":       "reps",
        "target_closure_min":  None,    # detection is class-based, not closure-based
        "target_closure_max":  None,
        "target_class":        "thumb_index",
        "target_gesture":      "thumb_index",
        "description": (
            "Bring your thumb and index finger together into a firm pinch, "
            "then release fully. Each pinch counts as one rep."
        ),
    },
}


# --------------------------------------------------------------------------- #
# Goal → exercise selection map
# Covers both profile goals (grip_strength / rom / dexterity)
# and self-report goals (regain_strength / improve_flexibility / etc.)
# --------------------------------------------------------------------------- #

_GOAL_EXERCISES: dict[str, list[str]] = {
    # Profile goals
    "grip_strength":       ["full_fist_close", "open_palm_hold"],
    "rom":                 ["open_palm_hold", "mid_flexion_hold", "full_fist_close"],
    "dexterity":           ["thumb_index_pinch", "mid_flexion_hold"],
    # Self-report goals
    "regain_strength":     ["full_fist_close", "open_palm_hold"],
    "improve_flexibility": ["open_palm_hold", "mid_flexion_hold"],
    "reduce_stiffness":    ["open_palm_hold", "mid_flexion_hold", "full_fist_close"],
    "recover_surgery":     ["open_palm_hold", "mid_flexion_hold"],
    "general_maintenance": [
        "open_palm_hold", "mid_flexion_hold",
        "full_fist_close", "thumb_index_pinch",
    ],
}

_DEFAULT_EXERCISES: list[str] = [
    "open_palm_hold", "mid_flexion_hold", "full_fist_close",
]

# Maps assessment exercise IDs to their plan exercise IDs
_ASSESSMENT_TO_PLAN: dict[str, str] = {
    "open_palm":   "open_palm_hold",
    "mid_flexion": "mid_flexion_hold",
    "full_fist":   "full_fist_close",
}


# --------------------------------------------------------------------------- #
# Modified exercises for range-skipped assessments
# --------------------------------------------------------------------------- #

_MODIFIED_EXERCISES: dict[str, dict] = {
    "full_fist": {
        "id":                  "partial_fist_attempt",
        "name":                "Partial Fist Attempt",
        "sets":                2,
        "reps":                5,
        "hold_s":              1,
        "rest_s":              20,
        "tracking_type":       "hold",
        "target_closure_min":  0.2,
        "target_closure_max":  0.4,
        "target_class":        "grabbing",
        "target_gesture":      "grabbing",
        "description": (
            "Gently close your hand as far as is comfortable — aim for a partial fist. "
            "Work only within your pain-free range; there is no need to force the movement."
        ),
        "modified_from_skip":  True,
    },
    "mid_flexion": {
        "id":                  "gentle_mid_flexion",
        "name":                "Gentle Mid Flexion",
        "sets":                2,
        "reps":                5,
        "hold_s":              1,
        "rest_s":              20,
        "tracking_type":       "hold",
        "target_closure_min":  0.15,
        "target_closure_max":  0.35,
        "target_class":        "grabbing",
        "target_gesture":      "grabbing",
        "description": (
            "Begin a gentle curl of your fingers, stopping before any discomfort. "
            "Work only within your available range."
        ),
        "modified_from_skip":  True,
    },
    "open_palm": {
        "id":                  "gentle_extension",
        "name":                "Gentle Hand Opening",
        "sets":                2,
        "reps":                5,
        "hold_s":              1,
        "rest_s":              20,
        "tracking_type":       "hold",
        "target_closure_min":  0.0,
        "target_closure_max":  0.5,
        "target_class":        "palm",
        "target_gesture":      "palm",
        "description": (
            "Gently open your hand as far as is comfortable. "
            "Work only within your pain-free range."
        ),
        "modified_from_skip":  True,
    },
}


# --------------------------------------------------------------------------- #
# Plan generator
# --------------------------------------------------------------------------- #

def generate_plan(
    user: Optional[dict] = None,
    assessment: Optional[dict] = None,
) -> dict:
    """
    Build a personalised exercise plan using only model-trackable movements.

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

    goals = user.get("goals", [])
    pain  = int(assessment.get("results", {}).get("pain_level", 0))
    score = float(assessment.get("score", 50.0))   # 0–100 ROM score

    # Goal-directed exercise selection — supported by evidence that patient-reported
    # outcome goals improve adherence and functional recovery when used to guide
    # exercise prescription (Levack et al., 2015, Cochrane Review; Cup et al., 2003,
    # Clin Rehabil). Each goal maps to the subset of trackable exercises that most
    # directly addresses the stated functional deficit.
    seen: set[str] = set()
    exercise_ids: list[str] = []
    for goal in goals:
        for ex_id in _GOAL_EXERCISES.get(goal, []):
            if ex_id not in seen:
                seen.add(ex_id)
                exercise_ids.append(ex_id)

    # Fallback to a balanced open–mid–close sequence when no goal is set.
    # This three-position arc (full extension → mid-range → full flexion) is the
    # standard baseline ROM assessment and exercise sequence in hand therapy
    # (Fess, 2011, Rehabilitation of the Hand and Upper Extremity, 6th ed.;
    # ASHT Clinical Assessment Recommendations, 2015).
    if not exercise_ids:
        exercise_ids = list(_DEFAULT_EXERCISES)

    # Identify skipped exercises from the assessment
    raw_exercises = assessment.get("results", {}).get("exercises", [])

    # If every assessed exercise was pain-skipped → referral required, no plan possible
    if raw_exercises and all(
        raw.get("skipped") and raw.get("skip_reason") == "pain"
        for raw in raw_exercises
    ):
        return {
            "exercises":         [],
            "sessions_per_week": 0,
            "referral_required": True,
            "notes": (
                "__REFERRAL__ All exercises were too painful to attempt during assessment. "
                "Please consult a qualified physiotherapist or hand therapist before "
                "continuing with self-guided rehabilitation."
            ),
        }

    pain_skipped_plan_ids: set[str] = set()
    range_skipped_assessment_ids: list[str] = []
    skip_notes: list[str] = []

    for raw in raw_exercises:
        if not raw.get("skipped"):
            continue
        assessment_ex_id = raw.get("exercise", "")
        reason = raw.get("skip_reason", "")
        plan_ex_id = _ASSESSMENT_TO_PLAN.get(assessment_ex_id)
        meta = _EX_META.get(assessment_ex_id, {"name": assessment_ex_id.replace("_", " ").title()})
        name = meta["name"]
        if reason == "pain":
            if plan_ex_id:
                pain_skipped_plan_ids.add(plan_ex_id)
            skip_notes.append(
                f"{name} skipped due to pain — will be introduced gradually as recovery progresses."
            )
        elif reason == "range":
            range_skipped_assessment_ids.append(assessment_ex_id)
            skip_notes.append(
                f"{name} not yet reachable — a modified version has been added as a starting point."
            )

    # Exclude pain-skipped exercises entirely
    exercise_ids = [eid for eid in exercise_ids if eid not in pain_skipped_plan_ids]

    # Resolve to exercise dicts and apply pain- and ROM-based volume scaling
    exercises = [
        _scale_exercise(_EXERCISES[ex_id], pain, score)
        for ex_id in exercise_ids
    ]

    # Add modified (reduced-target) versions for range-skipped exercises
    for assessment_ex_id in range_skipped_assessment_ids:
        modified = _MODIFIED_EXERCISES.get(assessment_ex_id)
        if modified:
            exercises.append(dict(modified))

    notes = _build_notes(pain, score, goals)
    if skip_notes:
        notes = notes + " " + " ".join(skip_notes) if notes else " ".join(skip_notes)

    return {
        "exercises":         exercises,
        "sessions_per_week": _sessions_per_week(pain, score),
        "notes":             notes,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _scale_exercise(ex: dict, pain: int, score: float) -> dict:
    """Return a copy of the exercise with volume adapted to pain level and ROM."""
    ex = dict(ex)   # shallow copy — never mutate the catalogue

    # Pain-directed volume reduction using NRS thresholds —————————————————————
    # NRS ≥ 7 ("severe") warrants significant load reduction or exercise
    # cessation per pain-monitoring models in musculoskeletal rehabilitation
    # (Fairbank & Pynsent, 2000; Thomeé, 1997, Sports Medicine).
    # Reducing sets by 1 and reps by 4 at this level reflects the conservative
    # management principle: maintain movement without provoking nociceptive input
    # beyond tolerable limits (Lewis, 2009, Man Ther; Nery et al., 2021, J Hand Surg).
    if pain >= 7:
        ex["sets"]  = max(1, ex["sets"] - 1)
        ex["reps"]  = max(3, ex["reps"] - 4)
        # For isometric hold exercises, hold duration is reduced proportionally.
        # Prolonged submaximal isometric contractions at high pain levels risk
        # sensitisation of peripheral nociceptors (Graven-Nielsen & Arendt-Nielsen,
        # 2010, Eur J Pain); a minimum of 2 s preserves neuromuscular activation
        # without accumulating noxious stimulus duration.
        if ex["hold_s"] is not None:
            ex["hold_s"] = max(2, ex["hold_s"] - 2)

    # NRS 4–6 ("moderate pain") — partial volume reduction ———————————————————
    # Moderate pain permits continued exercise with reduced repetition volume.
    # This threshold is consistent with the pain-monitoring model recommendation
    # that exercise should be modified but not discontinued when NRS < 7
    # (Zusman, 2010, Physiother Theory Pract; JOSPT Systematic Review, 2024).
    # Rep reduction of 2 preserves training stimulus while limiting cumulative load.
    # Hold duration reduced by 1 s (min 3 s) to maintain meaningful isometric
    # stimulus (Schoenfeld & Grgic, 2019, Strength Cond J) without exacerbating pain.
    elif pain >= 4:
        ex["reps"]  = max(4, ex["reps"] - 2)
        if ex["hold_s"] is not None:
            ex["hold_s"] = max(3, ex["hold_s"] - 1)

    # ROM-based load reduction ————————————————————————————————————————————————
    # A ROM score < 40 (out of 100) indicates clinically meaningful restriction,
    # analogous to a Disabilities of the Arm, Shoulder and Hand (DASH) score
    # indicating moderate-to-severe functional impairment (Beaton et al., 2001,
    # J Hand Surg). At this level, volume is further reduced and a "pain-free
    # range" cue is appended — consistent with the Saint John Hand Therapy
    # Protocol (2016) instruction to work within available range before
    # progressing load, and with ROM-first progression models advocated by
    # Lapresa et al. (2023, J Rehabil Med) and Nery et al. (2021, J Hand Surg).
    if score < 40:
        ex["reps"] = max(3, ex["reps"] - 2)
        ex["description"] = ex["description"] + " Work only within your pain-free range."

    return ex


def _sessions_per_week(pain: int, score: float) -> int:
    # Session frequency by pain severity and ROM status ————————————————————————
    #
    # NRS ≥ 7 → 2 sessions/week
    #   High pain warrants reduced frequency to allow tissue recovery and avoid
    #   pain sensitisation. 2×/week is the conservative lower bound for
    #   maintaining neuromuscular adaptations in acute or highly painful conditions
    #   (Woldag & Hummelsheim, 2002, J Neurol; NICE Chronic Pain Guidelines, 2021).
    #
    # NRS 4–6 or ROM score < 40 → 3 sessions/week
    #   Moderate pain or restricted ROM: 3×/week aligns with the Saint John Hand
    #   Therapy Protocol (2016) recommendation for post-fracture and post-tendon
    #   repair rehabilitation, and with systematic review evidence for hand
    #   osteoarthritis (Østerås et al., 2017, Cochrane; JOSPT OA Review, 2024).
    #   3×/week also satisfies the minimum frequency for strength retention per
    #   ACSM resistance training guidelines (Garber et al., 2011, Med Sci Sports).
    #
    # NRS ≤ 3 and ROM ≥ 40 → 4 sessions/week
    #   Low pain and adequate ROM permit higher frequency. 4×/week is consistent
    #   with evidence for neurological hand rehabilitation (Kwakkel et al., 2015,
    #   Neurorehabil Neural Repair) and with progressive overload principles for
    #   sub-acute and chronic-phase hand therapy (Lapresa et al., 2023, J Rehabil Med).
    if pain >= 7:
        return 2
    if pain >= 4 or score < 40:
        return 3
    return 4


def _build_notes(pain: int, score: float, goals: list[str]) -> str:
    lines: list[str] = []

    # NRS pain band guidance ——————————————————————————————————————————————————
    # Patient-facing notes are stratified by NRS band, reflecting the widely adopted
    # three-tier pain classification: mild (1–3), moderate (4–6), severe (7–10)
    # (Jensen et al., 2003, Pain; Hawker et al., 2011, Arthritis Care Res).
    # The 4/10 "stop" threshold for mild pain is consistent with pain-monitoring
    # models in exercise-based rehabilitation (Zusman, 2010; Lewis, 2009).
    if pain == 0:
        lines.append("No pain reported — full programme at prescribed volume.")
    elif pain <= 3:
        lines.append("Mild discomfort — monitor throughout; stop if pain exceeds 4/10.")
    elif pain <= 6:
        lines.append("Moderate pain — volume reduced; prioritise range over effort.")
    else:
        lines.append("High pain — very gentle movements only. Consult your therapist before continuing.")

    # ROM score thresholds ————————————————————————————————————————————————————
    # Score < 40: clinically significant restriction requiring range-first progression
    #   before any resistance or endurance load is introduced (Fess, 2011;
    #   Saint John Protocol, 2016; Lapresa et al., 2023).
    # Score ≥ 80: indicates near-normal ROM; progressive overload via hold duration
    #   and rep count is appropriate (Schoenfeld & Grgic, 2019; Nery et al., 2021).
    if score < 40:
        lines.append("Limited ROM detected — focus on completing the range before adding holds.")
    elif score >= 80:
        lines.append("Good ROM — begin progressing hold duration and rep count each session.")

    if not goals:
        lines.append("No goals set — complete your profile for a more targeted programme.")

    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Adaptive progression
# --------------------------------------------------------------------------- #

_MAX_HOLD_S = 15   # absolute cap for hold duration (seconds)
_MAX_REPS   = 15   # absolute cap for reps per set
_MAX_SETS   = 4    # absolute cap for sets


def evaluate_progression(user_id: int = 1) -> dict:
    """
    Inspect the last 3 completed sessions against the current plan targets and
    decide which exercises are ready to be progressed.

    Parameters
    ----------
    user_id : always 1 in this single-user app

    Returns
    -------
    dict mapping exercise_id -> bool  (True = consistently exceeding target)
    """
    from app.database import get_last_n_sessions, get_latest_plan

    plan = get_latest_plan()
    if not plan:
        return {}

    sessions = get_last_n_sessions(3)
    if len(sessions) < 3:
        return {}

    plan_exercises = {ex["id"]: ex for ex in plan.get("exercises", [])}
    results: dict = {}

    for ex_id, plan_ex in plan_exercises.items():
        tracking     = plan_ex.get("tracking_type", "reps")
        target_hold  = plan_ex.get("hold_s")
        target_reps  = int(plan_ex.get("reps", 10))

        # Collect the matching exercise record from each of the 3 sessions
        perf_records = []
        for sess in sessions:
            for sess_ex in sess.get("exercises", []):
                if sess_ex.get("exercise") == ex_id:
                    perf_records.append(sess_ex)
                    break

        if len(perf_records) < 3:
            results[ex_id] = False
            continue

        if tracking == "hold" and target_hold is not None:
            hold_vals = [
                float(p.get("avg_hold_s") or p.get("hold_s") or 0)
                for p in perf_records
            ]
            # All three sessions must have timing data and exceed target × 1.2
            results[ex_id] = (
                all(h > 0 for h in hold_vals)
                and all(h >= target_hold * 1.2 for h in hold_vals)
            )
        else:
            # Rep exercise: user completed all target reps in every session
            results[ex_id] = all(
                int(p.get("reps_done") or 0) >= target_reps
                for p in perf_records
            )

    return results


def apply_progression(user_id: int = 1, plan: Optional[dict] = None) -> dict:
    """
    Evaluate the current plan against recent session performance and, where
    exercises are ready, produce an updated plan saved to the database.

    Parameters
    ----------
    user_id : always 1 in this single-user app
    plan    : the current plan dict; fetched from the DB if not supplied

    Returns
    -------
    dict with keys:
      progressed  : bool
      changes     : list[str] — human-readable description of each change
      log_entries : list[dict] — raw {exercise, old_value, new_value, description}
    """
    import datetime
    from app.database import get_latest_plan, log_progression, save_plan

    if plan is None:
        plan = get_latest_plan()
    if not plan:
        return {"progressed": False, "changes": [], "log_entries": []}

    ready = evaluate_progression(user_id)
    if not any(ready.values()):
        return {"progressed": False, "changes": [], "log_entries": []}

    today       = datetime.date.today().isoformat()
    updated     : list = []
    changes     : list[str] = []
    log_entries : list[dict] = []

    for ex in plan.get("exercises", []):
        ex_id = ex["id"]
        ex    = dict(ex)   # shallow copy — never mutate the shared dict

        if ready.get(ex_id):
            tracking = ex.get("tracking_type", "reps")
            name     = ex.get("name", ex_id.replace("_", " ").title())

            if tracking == "hold" and ex.get("hold_s") is not None:
                old_hold = int(ex["hold_s"])
                increment = 2 if old_hold <= 5 else 1
                new_hold  = min(_MAX_HOLD_S, old_hold + increment)
                if new_hold != old_hold:
                    desc = f"{name} hold increased from {old_hold}s to {new_hold}s"
                    ex["hold_s"]       = new_hold
                    ex["progressed_on"] = today
                    changes.append(desc)
                    log_entries.append({
                        "exercise":    ex_id,
                        "old_value":   f"hold_s={old_hold}",
                        "new_value":   f"hold_s={new_hold}",
                        "description": desc,
                    })
            else:
                old_reps = int(ex["reps"])
                new_reps = min(_MAX_REPS, old_reps + 2)
                if new_reps != old_reps:
                    desc = f"{name} reps increased from {old_reps} to {new_reps}"
                    ex["reps"]         = new_reps
                    ex["progressed_on"] = today
                    changes.append(desc)
                    log_entries.append({
                        "exercise":    ex_id,
                        "old_value":   f"reps={old_reps}",
                        "new_value":   f"reps={new_reps}",
                        "description": desc,
                    })

            # Bump sets: 2 → 3 when an exercise first progresses (cap 4)
            old_sets = int(ex.get("sets", 3))
            if old_sets == 2:
                new_sets = 3
                desc = f"{name} sets increased from 2 to 3"
                ex["sets"] = new_sets
                changes.append(desc)
                log_entries.append({
                    "exercise":    ex_id,
                    "old_value":   "sets=2",
                    "new_value":   "sets=3",
                    "description": desc,
                })

        updated.append(ex)

    if not changes:
        return {"progressed": False, "changes": [], "log_entries": []}

    save_plan(
        exercises=updated,
        sessions_per_week=plan.get("sessions_per_week", 3),
        notes=plan.get("notes", ""),
    )
    log_progression(log_entries)

    return {
        "progressed":  True,
        "changes":     changes,
        "log_entries": log_entries,
    }


# --------------------------------------------------------------------------- #
# Assessment summary and milestone goals
# --------------------------------------------------------------------------- #

_EX_META: dict[str, dict] = {
    "open_palm":   {"name": "Open Palm",   "emoji": "✋"},
    "mid_flexion": {"name": "Mid Flexion", "emoji": "\U0001f91a"},
    "full_fist":   {"name": "Full Fist",   "emoji": "✊"},
}


def _best_closure_for_display(ex_id: str, closures: list) -> Optional[float]:
    if not closures:
        return None
    if ex_id == "open_palm":
        return min(closures)
    if ex_id == "full_fist":
        return max(closures)
    return min(closures, key=lambda v: abs(v - 0.50))


def _rate_exercise(ex_id: str, best: Optional[float]) -> tuple:
    """Return (rating: 'good'|'moderate'|'limited', interpretation: str)."""
    if best is None:
        return "limited", "No attempts recorded"
    if ex_id == "open_palm":
        if best <= 0.25:
            return "good",     "Good range of motion"
        if best <= 0.45:
            return "moderate", "Moderate limitation"
        return     "limited",  "Significant limitation — we’ll start gently"
    if ex_id == "mid_flexion":
        if 0.35 <= best <= 0.65:
            return "good",     "Good range of motion"
        if 0.20 <= best <= 0.80:
            return "moderate", "Moderate limitation"
        return     "limited",  "Significant limitation — we’ll start gently"
    if ex_id == "full_fist":
        if best >= 0.75:
            return "good",     "Good range of motion"
        if best >= 0.55:
            return "moderate", "Moderate limitation"
        return     "limited",  "Significant limitation — we’ll start gently"
    return "moderate", "Assessment recorded"


def _summary_from_ratings(ratings: dict) -> str:
    open_r = ratings.get("open_palm",   "limited")
    mid_r  = ratings.get("mid_flexion", "limited")
    fist_r = ratings.get("full_fist",   "limited")

    good_count    = sum(1 for r in (open_r, mid_r, fist_r) if r == "good")
    limited_count = sum(1 for r in (open_r, mid_r, fist_r) if r == "limited")

    if good_count == 3:
        s1 = "Your hand shows good range of motion across all three positions."
    elif good_count == 2:
        if open_r != "good":
            s1 = ("Your hand shows good mid-range control and grip strength, "
                  "with some difficulty reaching full extension.")
        elif mid_r != "good":
            s1 = ("Your hand extends well and can form a strong fist, "
                  "with some restriction at mid-range.")
        else:
            s1 = ("Your hand shows good extension and mid-range control, "
                  "with limited full grip closure.")
    elif good_count == 1:
        if open_r == "good":
            s1 = ("Your hand opens well, with restriction through "
                  "the mid-range and grip positions.")
        elif mid_r == "good":
            s1 = ("Your hand reaches the mid-range position well, "
                  "with restriction at full extension and full grip.")
        else:
            s1 = ("Your hand shows good grip strength, "
                  "with restriction through the opening range.")
    else:
        s1 = ("Your assessment shows restriction across all positions — "
              "this is very common early in recovery.")

    if limited_count == 0:
        s2 = ("We’ll build on your strong baseline with progressive "
              "hold durations and repetitions.")
    elif fist_r == "limited" and open_r in ("good", "moderate"):
        s2 = ("We’ll focus on gradually improving your grip strength "
              "while keeping pain levels low.")
    elif open_r == "limited":
        s2 = ("We’ll start with gentle extension exercises to recover "
              "your opening range before progressing.")
    elif mid_r == "limited":
        s2 = "Building mid-range control will be the key focus of your early sessions."
    else:
        s2 = ("Your programme will begin gently, prioritising range of motion "
              "before adding resistance.")

    if limited_count >= 2:
        s3 = ("Every session is designed to meet you where you are — "
              "progress builds steadily over time.")
        return f"{s1} {s2} {s3}"

    return f"{s1} {s2}"


def generate_assessment_summary(assessment: dict) -> dict:
    """
    Process a raw assessment record into display-ready per-exercise results
    and a plain-English overall summary.

    Parameters
    ----------
    assessment : dict from database.get_latest_assessment()

    Returns
    -------
    dict with:
      exercises : list[dict]  — one entry per recorded exercise
      summary   : str         — 2-3 sentence interpretation
    """
    raw_exercises = (assessment.get("results") or {}).get("exercises", [])

    exercises: list = []
    ratings: dict   = {}

    for raw in raw_exercises:
        ex_id      = raw.get("exercise", "")
        attempts   = raw.get("attempts", [])
        is_skipped = bool(raw.get("skipped", False))
        skip_reason = raw.get("skip_reason", "")

        closures  = [float(a["closure"]) for a in attempts if a.get("closure") is not None]
        hold_vals = [float(a["hold_s"])  for a in attempts if a.get("hold_s")  is not None]

        bc        = _best_closure_for_display(ex_id, closures)
        best_hold = max(hold_vals) if hold_vals else 0.0
        pain      = int(raw.get("pain_after", 0))
        meta = _EX_META.get(ex_id, {"name": ex_id.replace("_", " ").title(), "emoji": ""})

        if is_skipped:
            rating = "limited"
            if skip_reason == "pain":
                interpretation = "Skipped — too painful to attempt"
            elif skip_reason == "range":
                interpretation = "Not yet reachable"
            else:
                interpretation = "Skipped"
        else:
            rating, interpretation = _rate_exercise(ex_id, bc)

        ratings[ex_id] = rating
        exercises.append({
            "exercise_id":    ex_id,
            "name":           meta["name"],
            "emoji":          meta["emoji"],
            "best_closure":   round(bc, 2) if bc is not None else None,
            "best_hold_s":    round(best_hold, 1),
            "pain_after":     pain,
            "rating":         rating,
            "interpretation": interpretation,
            "skipped":        is_skipped,
            "skip_reason":    skip_reason,
        })

    return {"exercises": exercises, "summary": _summary_from_ratings(ratings)}


def generate_goals(assessment: dict) -> list:
    """
    Generate 3 milestone goals (Week 2, Week 4, Week 8) personalised to the
    user's actual assessment numbers.

    Strategy
    --------
    Build a 3-step progression triple for each assessed exercise, then map
    those triples onto the three milestones based on how many exercises still
    need work:
      - All good        → chain all 3 steps of open-palm progression
      - 1 needs work    → chain all 3 steps of that exercise
      - 2 need work     → Week 2 from the easier one; Weeks 4+8 from the harder
      - 3 need work     → one milestone per exercise (Week 2 step of each)

    Returns
    -------
    list of {"week": int, "label": str, "goal": str}
    """
    raw_exercises = (assessment.get("results") or {}).get("exercises", [])

    # ── Collect per-exercise metrics ──────────────────────────────────────────
    metrics: dict = {}
    for raw in raw_exercises:
        ex_id    = raw.get("exercise", "")
        attempts = raw.get("attempts", [])
        closures  = [float(a["closure"]) for a in attempts if a.get("closure") is not None]
        hold_vals = [float(a["hold_s"])  for a in attempts if a.get("hold_s")  is not None]
        bc        = _best_closure_for_display(ex_id, closures)
        best_hold = max(hold_vals) if hold_vals else 0.0
        rating, _ = _rate_exercise(ex_id, bc)
        metrics[ex_id] = {
            "best_closure": round(bc, 2) if bc is not None else None,
            "best_hold":    round(best_hold, 1),
            "rating":       rating,
        }

    def get_m(ex_id: str) -> dict:
        return metrics.get(ex_id, {"best_closure": None, "best_hold": 0.0, "rating": "limited"})

    # ── Per-exercise goal triples (immediate → mid-term → stretch) ────────────
    def palm_triple(m: dict) -> list:
        hold, rating = m["best_hold"], m["rating"]
        if rating == "limited" or hold < 1.0:
            return [
                "Achieve open palm extension pain-free",
                "Hold open palm for 3 seconds",
                "Hold open palm for 8 seconds",
            ]
        w2 = int(round(hold + 2))
        w4 = w2 + 3
        w8 = w4 + 4
        return [
            f"Hold open palm for {w2} seconds",
            f"Hold open palm for {w4} seconds",
            f"Hold open palm for {w8} seconds comfortably",
        ]

    def mid_triple(m: dict) -> list:
        hold, rating = m["best_hold"], m["rating"]
        if rating == "limited" or hold < 1.0:
            return [
                "Reach mid flexion range consistently",
                "Hold mid flexion for 3 seconds",
                "Hold mid flexion for 5 seconds comfortably",
            ]
        w2 = int(round(hold + 2))
        w4 = w2 + 3
        return [
            f"Hold mid flexion for {w2} seconds",
            f"Hold mid flexion for {w4} seconds",
            "Hold mid flexion for 8 seconds comfortably",
        ]

    def fist_triple(m: dict) -> list:
        hold, rating, bc = m["best_hold"], m["rating"], m["best_closure"]
        if rating == "limited":
            if bc is not None and bc < 0.7:
                target = round(min(0.9, bc + 0.2), 1)
                return [
                    "Achieve partial fist closure",
                    f"Achieve partial fist closure of {target:.1f}",
                    "Complete a full fist hold",
                ]
            return [
                "Begin attempting fist closure",
                "Achieve partial fist closure",
                "Complete a full fist hold",
            ]
        if rating == "moderate":
            return [
                "Complete a full fist close consistently",
                "Hold full fist for 5 seconds",
                "Hold full fist for 10 seconds comfortably",
            ]
        w2 = int(round(hold + 2)) if hold >= 1.0 else 5
        w4 = w2 + 3
        return [
            f"Hold full fist for {w2} seconds",
            f"Hold full fist for {w4} seconds",
            f"Hold full fist for {w4 + 3} seconds comfortably",
        ]

    open_m = get_m("open_palm")
    mid_m  = get_m("mid_flexion")
    fist_m = get_m("full_fist")

    # ── Determine which exercises still need work (not yet at "good") ─────────
    candidates: list = []
    if open_m["rating"] != "good":
        candidates.append(palm_triple(open_m))
    if mid_m["rating"] != "good":
        candidates.append(mid_triple(mid_m))
    if fist_m["rating"] != "good":
        candidates.append(fist_triple(fist_m))

    MILESTONES = [(2, "Week 2"), (4, "Week 4"), (8, "Week 8")]

    def mk(week: int, label: str, goal: str) -> dict:
        return {"week": week, "label": label, "goal": goal}

    # All exercises already at "good" — chain open-palm hold progression
    if not candidates:
        triple = palm_triple(open_m)
        goals = [mk(w, lbl, triple[i]) for i, (w, lbl) in enumerate(MILESTONES)]

    # One exercise needs work — chain its full 3-step progression
    elif len(candidates) == 1:
        triple = candidates[0]
        goals = [mk(w, lbl, triple[i]) for i, (w, lbl) in enumerate(MILESTONES)]

    # Two exercises need work — easier gets Week 2; harder gets Weeks 4 and 8
    elif len(candidates) == 2:
        goals = [
            mk(2, "Week 2", candidates[0][0]),
            mk(4, "Week 4", candidates[1][1]),
            mk(8, "Week 8", candidates[1][2]),
        ]

    # Three exercises need work — one milestone per exercise
    else:
        goals = [
            mk(2, "Week 2", candidates[0][0]),
            mk(4, "Week 4", candidates[1][0]),
            mk(8, "Week 8", candidates[2][0]),
        ]

    # Append week-8 stretch goals for any skipped exercises
    _SKIP_GOAL_TEXT: dict[str, str] = {
        "full_fist":   "Work toward full fist closure",
        "mid_flexion": "Work toward full mid-flexion range",
        "open_palm":   "Work toward full open-palm extension",
    }
    for raw in raw_exercises:
        if raw.get("skipped"):
            assessment_ex_id = raw.get("exercise", "")
            goal_text = _SKIP_GOAL_TEXT.get(
                assessment_ex_id,
                f"Work toward {assessment_ex_id.replace('_', ' ')}",
            )
            goals.append(mk(8, "Week 8", goal_text))

    return goals

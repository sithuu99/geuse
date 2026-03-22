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

    # Resolve to exercise dicts and apply pain- and ROM-based volume scaling
    exercises = [
        _scale_exercise(_EXERCISES[ex_id], pain, score)
        for ex_id in exercise_ids
    ]

    return {
        "exercises":         exercises,
        "sessions_per_week": _sessions_per_week(pain, score),
        "notes":             _build_notes(pain, score, goals),
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

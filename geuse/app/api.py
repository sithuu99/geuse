"""
app/api.py — PyWebView JS bridge.

All public methods are callable from JavaScript as:
  window.pywebview.api.<method>(args)

Methods always return a plain dict (JSON-serialisable).
On error they return {"ok": False, "error": "<message>"} so the
frontend never receives an unhandled Python exception.
"""

from __future__ import annotations

import traceback

import webview

from app.camera import CameraStream
from app.database import (
    get_latest_assessment,
    get_latest_plan,
    get_progress_data,
    get_session_history,
    get_user,
    reset_db,
    save_assessment,
    save_plan,
    save_self_report,
    save_session,
    save_user,
)
from app.plan import generate_plan


def _ok(**kwargs) -> dict:
    return {"ok": True, **kwargs}


def _err(exc: Exception) -> dict:
    traceback.print_exc()
    return {"ok": False, "error": str(exc)}


class Api:
    """Exposed to the WebView as the js_api object."""

    def __init__(self) -> None:
        self._camera = CameraStream()

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def navigate(self, page: str) -> dict:
        """
        Load a page by name using an absolute file:// URL.
        e.g. api.navigate("profile")  →  file:///…/ui/pages/profile.html
        Only whitelisted page names are accepted.
        """
        import sys
        from pathlib import Path
        import threading
        BASE_DIR = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent.parent.resolve()
        safe_pages = ["welcome", "profile", "self_report", "assessment", "plan", "session", "daily_checkin", "dashboard", "settings", "progress"]
        if page not in safe_pages:
            return _err(ValueError(f"Unknown page: {page!r}"))
        try:
            url = (BASE_DIR / "ui" / "pages" / f"{page}.html").as_uri()
            # Defer load_url by 50 ms so pywebview can deliver this _ok()
            # return value to the JS callback before the page navigates away
            # and destroys the current JavaScript context.
            threading.Timer(0.05, webview.windows[0].load_url, args=[url]).start()
            return _ok()
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #

    def start_camera(self, camera_index: int = 0) -> dict:
        """Open the webcam and start the background capture thread."""
        try:
            self._camera.start(camera_index)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def stop_camera(self) -> dict:
        """Stop the capture thread and release the webcam."""
        try:
            self._camera.stop()
            return _ok()
        except Exception as exc:
            return _err(exc)

    def get_frame(self) -> dict:
        """
        Return the latest processed frame.

        Response keys
        -------------
        ok            : bool
        jpeg_b64      : str   — base64 JPEG; use as <img src="data:image/jpeg;base64,{jpeg_b64}">
        hand_detected : bool
        label         : str   — gesture label or "no hand"
        learned_closure : float  0–1
        raw_closure     : float  0–1
        final_class   : int   — 0–4 (−1 if no hand)
        raw_class     : int   — smoothed model prediction before overrides
        """
        try:
            fd = self._camera.get_frame()
            return _ok(
                jpeg_b64=fd.jpeg_b64,
                hand_detected=fd.hand_detected,
                label=fd.label,
                learned_closure=fd.learned_closure,
                raw_closure=fd.raw_closure,
                final_class=fd.final_class,
                raw_class=fd.raw_class,
            )
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # User profile
    # ------------------------------------------------------------------ #

    def save_profile(self, data: dict) -> dict:
        """
        Upsert the user profile.

        Expected keys: name, age, affected_hand, condition, goals (list[str])
        """
        try:
            save_user(data)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def get_profile(self) -> dict:
        """Return the stored user profile, or {} if none exists yet."""
        try:
            user = get_user()
            return _ok(profile=user or {})
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Self-report (daily check-in)
    # ------------------------------------------------------------------ #

    def save_self_report(
        self,
        pain_level: int = 0,
        limitations: str = "",
        goal: str = "",
    ) -> dict:
        """
        Persist a daily check-in record.

        Called from self_report.html with positional args:
          api.save_self_report(pain_level, limitations, goal)
        """
        try:
            row_id = save_self_report(int(pain_level), str(limitations), str(goal))
            return _ok(self_report_id=row_id)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Daily check-in (returning-user lightweight pain capture)
    # ------------------------------------------------------------------ #

    def save_daily_checkin(self, pain_level: int = 0) -> dict:
        """Persist a returning-user daily check-in (pain level only)."""
        try:
            row_id = save_self_report(int(pain_level), "", "")
            return _ok(checkin_id=row_id)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Assessment
    # ------------------------------------------------------------------ #

    def save_assessment_result(
        self,
        results: dict,
        score: float = 0.0,
        notes: str = "",
    ) -> dict:
        """
        Persist an assessment result.

        Parameters
        ----------
        results : arbitrary dict — gesture labels, closure values, frame counts, etc.
        score   : 0–100 ROM score derived from the assessment
        notes   : free-text therapist / user notes
        """
        try:
            row_id = save_assessment(results, score, notes)
            return _ok(assessment_id=row_id)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Plan
    # ------------------------------------------------------------------ #

    def generate_and_save_plan(self) -> dict:
        """
        Run the rule-based plan generator using the current user profile
        and latest assessment, persist the result, and return it.
        """
        try:
            user       = get_user() or {}
            assessment = get_latest_assessment() or {}

            plan = generate_plan(user=user, assessment=assessment)

            assessment_id = assessment.get("id")
            plan_id = save_plan(
                exercises=plan["exercises"],
                sessions_per_week=plan["sessions_per_week"],
                notes=plan["notes"],
                source_assessment_id=assessment_id,
            )

            return _ok(plan_id=plan_id, plan=plan)
        except Exception as exc:
            return _err(exc)

    def get_plan(self) -> dict:
        """Return the most recently saved plan, or {} if none exists."""
        try:
            plan = get_latest_plan()
            return _ok(plan=plan or {})
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Session
    # ------------------------------------------------------------------ #

    def reset_app(self) -> dict:
        """
        Wipe all user data and navigate to the welcome page.
        Drops and recreates all tables, then reloads welcome.html.
        """
        import sys
        import threading
        from pathlib import Path
        BASE_DIR = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent.parent.resolve()
        try:
            reset_db()
            url = (BASE_DIR / "ui" / "pages" / "welcome.html").as_uri()
            threading.Timer(0.1, webview.windows[0].load_url, args=[url]).start()
            return _ok()
        except Exception as exc:
            return _err(exc)

    def get_session_history(self) -> dict:
        """Return session history, pain trend, streak, and totals for the dashboard."""
        try:
            history = get_session_history()
            return _ok(**history)
        except Exception as exc:
            return _err(exc)

    def get_progress_data(self) -> dict:
        """Return all progress page data in one call."""
        try:
            data = get_progress_data()
            return _ok(**data)
        except Exception as exc:
            return _err(exc)

    def save_session_result(self, data: dict) -> dict:
        """
        Persist a completed session.

        Expected keys in data
        ---------------------
        exercises   : list[dict]  — per-exercise completion records
        plan_id     : int | None
        pain_before : int | None  (0–10)
        pain_after  : int | None  (0–10)
        duration_s  : int | None  — wall-clock seconds
        """
        try:
            row_id = save_session(
                exercises=data.get("exercises", []),
                plan_id=data.get("plan_id"),
                pain_before=data.get("pain_before"),
                pain_after=data.get("pain_after"),
                duration_s=data.get("duration_s"),
            )
            return _ok(session_id=row_id)
        except Exception as exc:
            return _err(exc)

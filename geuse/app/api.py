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
    acknowledge_progression as db_acknowledge_progression,
    discard_incomplete_session as db_discard_incomplete_session,
    dismiss_flag as db_dismiss_flag,
    get_active_flags as db_get_active_flags,
    get_hold_progress as db_get_hold_progress,
    get_incomplete_session as db_get_incomplete_session,
    get_insights as db_get_insights,
    get_latest_assessment,
    get_latest_plan,
    get_monthly_history as db_get_monthly_history,
    get_progress_data,
    get_progression_log as db_get_progression_log,
    get_recovery_day as db_get_recovery_day,
    get_session_history,
    get_trend_analysis as db_get_trend_analysis,
    get_unacknowledged_progression,
    get_user,
    mark_session_completed,
    reset_db,
    save_assessment,
    save_flag,
    save_plan,
    save_self_report,
    save_session,
    save_session_checkpoint,
    save_user,
    seed_demo_data,
)
from app.plan import (
    apply_progression,
    generate_assessment_summary,
    generate_goals,
    generate_plan,
)


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
        safe_pages = ["welcome", "profile", "self_report", "assessment", "assessment_results", "plan", "session", "daily_checkin", "dashboard", "settings", "progress", "edit_profile", "report"]
        if page not in safe_pages:
            return _err(ValueError(f"Unknown page: {page!r}"))
        try:
            url = (BASE_DIR / "ui" / "pages" / f"{page}.html").as_uri()
            # Defer load_url by 50 ms so pywebview can deliver this _ok()
            # return value to the JS callback before the page navigates away
            # and destroys the current JavaScript context.
            threading.Timer(0.05, webview.windows[0].load_url, args=[url]).start()
            # Trigger a repaint on the newly loaded page to clear WebView2 checkerboard
            def _repaint():
                import time
                time.sleep(0.5)
                try:
                    webview.windows[0].evaluate_js(
                        "window.dispatchEvent(new Event('resize'))"
                    )
                except Exception:
                    pass
            threading.Thread(target=_repaint, daemon=True).start()
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

    def get_cameras(self) -> dict:
        """Return list of available camera devices (indices 0–4)."""
        try:
            cameras = self._camera.get_cameras()
            return _ok(cameras=cameras)
        except Exception as exc:
            return _err(exc)

    def switch_camera(self, index: int = 0) -> dict:
        """Stop the current camera and restart with a different device index."""
        try:
            self._camera.switch_camera(int(index))
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

    def get_last_checkin(self) -> dict:
        """Return the most recent daily check-in record (pain_level + created_at), or None."""
        try:
            from app.database import get_latest_self_report
            record = get_latest_self_report()
            return _ok(checkin=record)
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

            plan["id"] = plan_id
            return _ok(plan_id=plan_id, plan=plan)
        except Exception as exc:
            return _err(exc)

    def get_assessment_summary(self) -> dict:
        """
        Derive per-exercise display data, an overall plain-English summary,
        and milestone goals from the latest saved assessment.
        """
        try:
            assessment = get_latest_assessment() or {}
            summary    = generate_assessment_summary(assessment)
            goals      = generate_goals(assessment)
            return _ok(
                exercises=summary["exercises"],
                summary=summary["summary"],
                goals=goals,
            )
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
    # Camera status
    # ------------------------------------------------------------------ #

    def get_camera_status(self) -> dict:
        """Return whether the camera is running, stalled, or stopped."""
        try:
            return _ok(**self._camera.get_camera_status())
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Session checkpoints
    # ------------------------------------------------------------------ #

    def save_checkpoint(
        self,
        exercise_index: int,
        exercises_completed,
        plan_id=None,
        session_id=None,
    ) -> dict:
        """Upsert an in-progress session checkpoint after each completed exercise."""
        try:
            import json as _json
            sid = save_session_checkpoint(
                plan_id=int(plan_id) if plan_id else None,
                exercise_index=int(exercise_index),
                exercises_completed_json=_json.dumps(exercises_completed),
                session_id=int(session_id) if session_id else None,
            )
            return _ok(session_id=sid)
        except Exception as exc:
            return _err(exc)

    def get_incomplete_session(self) -> dict:
        """Return an unfinished session from the last 24 hours, if one exists."""
        try:
            session = db_get_incomplete_session()
            return _ok(session=session)
        except Exception as exc:
            return _err(exc)

    def mark_completed(
        self,
        session_id,
        exercises,
        pain_before=None,
        pain_after=None,
        duration_s=None,
    ) -> dict:
        """Finalise a session: store all result data and set status to completed."""
        try:
            mark_session_completed(
                session_id=int(session_id),
                exercises=exercises,
                pain_before=int(pain_before) if pain_before is not None else None,
                pain_after=int(pain_after) if pain_after is not None else None,
                duration_s=int(duration_s) if duration_s is not None else None,
            )
            return _ok()
        except Exception as exc:
            return _err(exc)

    def discard_incomplete_session(self, session_id) -> dict:
        """Delete an abandoned in-progress session so it won't show as resumable."""
        try:
            db_discard_incomplete_session(int(session_id))
            return _ok()
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

    def load_demo(self) -> dict:
        """Seed the database with demo data for Alex Johnson."""
        try:
            seed_demo_data()
            return _ok()
        except Exception as exc:
            return _err(exc)

    def is_demo_mode(self) -> dict:
        """Return whether the current user is the demo account."""
        try:
            user = get_user()
            return _ok(is_demo=bool(user and user.get("name") == "Alex Johnson"))
        except Exception as exc:
            return _err(exc)

    def get_recovery_day(self) -> dict:
        """Return days since the user profile was created (1-indexed)."""
        try:
            return _ok(day=db_get_recovery_day())
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

    def get_insights(self) -> dict:
        """Return insight data for dashboard and progress pages."""
        try:
            data = db_get_insights()
            return _ok(**data)
        except Exception as exc:
            return _err(exc)

    def get_monthly_history(self) -> dict:
        """Return session data grouped by month, newest first."""
        try:
            months = db_get_monthly_history()
            return _ok(months=months)
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

    # ------------------------------------------------------------------ #
    # Adaptive progression
    # ------------------------------------------------------------------ #

    def check_progression(self) -> dict:
        """
        Evaluate recent session performance and, if all exercises consistently
        exceed their targets, update the plan and log the changes.

        If unacknowledged progression events already exist (the user has not yet
        dismissed the notification from a prior run) those are returned as-is
        without re-evaluating, so the plan is not progressed twice.

        Returns
        -------
        ok         : bool
        progressed : bool
        changes    : list[str] — human-readable descriptions of what changed
        """
        try:
            pending = get_unacknowledged_progression()
            if pending:
                changes = [p["description"] for p in pending if p.get("description")]
                return _ok(progressed=True, changes=changes)

            plan   = get_latest_plan()
            result = apply_progression(user_id=1, plan=plan)
            return _ok(progressed=result["progressed"], changes=result["changes"])
        except Exception as exc:
            return _err(exc)

    def acknowledge_progression(self) -> dict:
        """Mark all pending progression events as acknowledged (notification dismissed)."""
        try:
            db_acknowledge_progression()
            return _ok()
        except Exception as exc:
            return _err(exc)

    def get_progression_log(self) -> dict:
        """Return the full plan progression history for the progress page."""
        try:
            entries = db_get_progression_log()
            return _ok(entries=entries)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ #
    # Trend analysis / health flags
    # ------------------------------------------------------------------ #

    def get_trend_analysis(self) -> dict:
        """
        Compute pain/closure cross-validation flags from recent sessions.
        Persists each detected flag and returns active flags with DB ids.
        """
        try:
            analysis = db_get_trend_analysis()
            for flag_type in analysis["flags"]:
                save_flag(1, flag_type)
            active = db_get_active_flags()
            return _ok(
                flags=analysis["flags"],
                active_flags=active,
                analysis=analysis,
            )
        except Exception as exc:
            return _err(exc)

    def get_active_flags(self) -> dict:
        """Return all non-dismissed health flags."""
        try:
            return _ok(flags=db_get_active_flags())
        except Exception as exc:
            return _err(exc)

    def dismiss_flag(self, flag_id) -> dict:
        """Mark a health flag as dismissed (hides it for 7 days on next detection)."""
        try:
            db_dismiss_flag(int(flag_id))
            return _ok()
        except Exception as exc:
            return _err(exc)

    def get_hold_progress(self) -> dict:
        """Return per-session best hold durations by exercise type, plus plan targets."""
        try:
            data = db_get_hold_progress()
            return _ok(**data)
        except Exception as exc:
            return _err(exc)

    def generate_pdf(self) -> dict:
        """
        Save the report as PDF to ~/Downloads/geuse_report.pdf.

        Avoids all Qt thread-safety issues by never touching Qt widgets directly:
          1. evaluate_js() captures the already-rendered HTML, replacing each
             <canvas> with an <img> data-URL of its current pixels.
          2. A <base> tag is injected so CSS/font/asset relative paths resolve.
          3. The HTML is saved to a temp file.
          4. Edge headless --print-to-pdf converts it to PDF (applies @media print).
          5. Explorer opens with the PDF selected.
        Falls back to opening the HTML in the default browser if Edge is absent.
        """
        try:
            import os
            import pathlib
            import subprocess
            import sys
            import tempfile

            # ── 1. Capture rendered HTML with canvases as PNG data-URLs ──────
            html = webview.windows[0].evaluate_js("""
                (function () {
                    var root = document.documentElement.cloneNode(true);

                    // ── Replace canvases with captured PNG images ──────────────
                    document.querySelectorAll('canvas[id]').forEach(function (c) {
                        var t = root.querySelector('#' + c.id);
                        if (!t) return;
                        var img = document.createElement('img');
                        img.src = c.toDataURL('image/png');
                        var nw = c.offsetWidth || parseInt(c.style.width) || 600;
                        // Use setAttribute so we can embed !important inline —
                        // inline !important beats any stylesheet !important rule.
                        img.setAttribute('style',
                            'display:block !important;' +
                            'width:100% !important;' +
                            'max-width:' + nw + 'px !important;' +
                            'height:auto !important;');
                        t.parentNode.replaceChild(img, t);
                    });

                    // ── Fix body so content is not clipped at screen width ─────
                    var body = root.querySelector('body');
                    if (body) body.style.overflow = 'visible';

                    // ── Fix tables: prevent cells from overflowing A4 width ────
                    root.querySelectorAll('.rpt-table').forEach(function (tbl) {
                        tbl.style.tableLayout = 'fixed';
                        tbl.style.width = '100%';
                    });
                    root.querySelectorAll('.rpt-table td, .rpt-table th').forEach(function (cell) {
                        cell.style.whiteSpace = 'normal';
                        cell.style.wordBreak  = 'break-word';
                        cell.style.overflow   = 'hidden';
                    });

                    return root.outerHTML;
                })()
            """)

            if not html:
                return _err(Exception("evaluate_js returned empty — is the report page loaded?"))

            # ── 2. Inject <base> so relative paths resolve from the pages dir ─
            base_dir = (
                pathlib.Path(sys._MEIPASS)
                if getattr(sys, "frozen", False)
                else pathlib.Path(__file__).parent.parent.resolve()
            )
            pages_url = (base_dir / "ui" / "pages").as_uri() + "/"
            html = html.replace("<head>", f"<head><base href=\"{pages_url}\">", 1)

            # ── 3. Write to a temp HTML file ──────────────────────────────────
            tmp = pathlib.Path(tempfile.gettempdir()) / "geuse_report_print.html"
            tmp.write_text("<!DOCTYPE html>\n" + html, encoding="utf-8")

            # ── 4. Convert to PDF via Edge headless ───────────────────────────
            out_pdf = pathlib.Path.home() / "Downloads" / "geuse_report.pdf"
            out_pdf.parent.mkdir(parents=True, exist_ok=True)

            edge = None
            for candidate in [
                os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
                os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            ]:
                if os.path.isfile(candidate):
                    edge = candidate
                    break

            if edge:
                subprocess.run(
                    [
                        edge,
                        "--headless",
                        "--disable-gpu",
                        "--no-sandbox",
                        "--run-all-compositor-stages-before-draw",
                        "--no-pdf-header-footer",
                        # Wide viewport so tables lay out correctly before the
                        # print reflow shrinks them to A4 width (~681px usable).
                        "--window-size=1200,1700",
                        # Allow the temp HTML to load local file:// assets
                        # (logo PNG, CSS, fonts) without security errors.
                        "--allow-file-access-from-files",
                        f"--print-to-pdf={out_pdf}",
                        tmp.as_uri(),
                    ],
                    timeout=20,
                    check=False,
                    capture_output=True,
                )

            # ── 5. Open result ────────────────────────────────────────────────
            if out_pdf.exists():
                subprocess.Popen(f'explorer /select,"{out_pdf}"', shell=True)
                return _ok(path=str(out_pdf))

            # Edge not found or failed — open the HTML in the default browser
            os.startfile(str(tmp))
            return _ok(path=str(tmp), note="Opened HTML in browser — use Ctrl+P to save PDF")

        except Exception as exc:
            return _err(exc)

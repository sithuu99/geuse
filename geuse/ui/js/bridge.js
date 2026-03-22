/**
 * bridge.js — Promise wrappers around window.pywebview.api
 *
 * Usage (any page):
 *   <script src="../js/bridge.js"></script>
 *   const data = await api.getFrame();
 *
 * Every method returns a Promise that resolves with the backend's
 * response dict  { ok: true, … }  or rejects on JS-level error.
 * Python-side errors surface as  { ok: false, error: "…" }.
 */

const api = (() => {
  /* -------------------------------------------------------------- */
  /* Internal: wait for pywebview to inject its JS bridge           */
  /* -------------------------------------------------------------- */
  function ready() {
    return new Promise((resolve) => {
      if (window.pywebview?.api) {
        resolve(window.pywebview.api);
        return;
      }
      window.addEventListener("pywebviewready", () => resolve(window.pywebview.api), {
        once: true,
      });
    });
  }

  async function call(method, ...args) {
    const backend = await ready();
    if (typeof backend[method] !== "function") {
      throw new Error(`pywebview.api.${method} is not a function`);
    }
    return backend[method](...args);
  }

  /* -------------------------------------------------------------- */
  /* Navigation                                                      */
  /* -------------------------------------------------------------- */
  const navigate = (page) => call("navigate", page);

  /* -------------------------------------------------------------- */
  /* Camera                                                          */
  /* -------------------------------------------------------------- */
  /** Open webcam and start background capture thread. */
  const startCamera = (cameraIndex = 0) => call("start_camera", cameraIndex);

  /** Stop capture thread and release webcam. */
  const stopCamera = () => call("stop_camera");

  /**
   * Fetch the latest processed frame.
   * Resolves with:
   *   { ok, jpeg_b64, hand_detected, label,
   *     learned_closure, raw_closure, final_class, raw_class }
   */
  const getFrame = () => call("get_frame");

  /* -------------------------------------------------------------- */
  /* User profile                                                    */
  /* -------------------------------------------------------------- */
  /** Upsert profile. data: { name, age, affected_hand, condition, goals[] } */
  const saveProfile = (data) => call("save_profile", data);

  /** Returns { ok, profile: { … } } or { ok, profile: {} } */
  const getProfile = () => call("get_profile");

  /* -------------------------------------------------------------- */
  /* Self-report (onboarding check-in)                              */
  /* -------------------------------------------------------------- */
  /**
   * Persist a full onboarding check-in (pain + limitations + goal).
   * @param {number} painLevel    - 0–10
   * @param {string} limitations  - free text
   * @param {string} goal         - selected goal value
   */
  const saveSelfReport = (painLevel, limitations, goal) =>
    call("save_self_report", painLevel, limitations, goal);

  /* -------------------------------------------------------------- */
  /* Daily check-in (returning-user)                                */
  /* -------------------------------------------------------------- */
  /** Persist a returning-user daily check-in (pain level only). */
  const saveDailyCheckin = (painLevel) =>
    call("save_daily_checkin", painLevel);

  /* -------------------------------------------------------------- */
  /* Assessment                                                      */
  /* -------------------------------------------------------------- */
  /**
   * Persist an assessment result.
   * @param {object} results  - arbitrary gesture/closure data
   * @param {number} score    - 0–100 ROM score
   * @param {string} notes    - free text
   */
  const saveAssessmentResult = (results, score = 0, notes = "") =>
    call("save_assessment_result", results, score, notes);

  /* -------------------------------------------------------------- */
  /* Plan                                                            */
  /* -------------------------------------------------------------- */
  /** Generate plan from current profile + latest assessment, persist, return it. */
  const generateAndSavePlan = () => call("generate_and_save_plan");

  /** Returns { ok, plan: { exercises, sessions_per_week, notes } } */
  const getPlan = () => call("get_plan");

  /* -------------------------------------------------------------- */
  /* App reset                                                      */
  /* -------------------------------------------------------------- */
  /** Wipe all data and restart from welcome.html. */
  const resetApp = () => call("reset_app");

  /* -------------------------------------------------------------- */
  /* Session history (dashboard)                                    */
  /* -------------------------------------------------------------- */
  /** Returns { ok, sessions, pain_history, streak, total_sessions, total_exercises } */
  const getSessionHistory = () => call("get_session_history");

  /** Returns { ok, total_sessions, streak, best_hold, avg_pain, closure_chart, session_history, pain_history } */
  const getProgressData = () => call("get_progress_data");

  /* -------------------------------------------------------------- */
  /* Session                                                         */
  /* -------------------------------------------------------------- */
  /**
   * Save a completed session.
   * data: { exercises[], plan_id, pain_before, pain_after, duration_s }
   */
  const saveSessionResult = (data) => call("save_session_result", data);

  /* -------------------------------------------------------------- */
  /* Public surface                                                  */
  /* -------------------------------------------------------------- */
  return {
    navigate,
    startCamera, stopCamera, getFrame,
    saveProfile, getProfile,
    saveSelfReport,
    saveDailyCheckin,
    saveAssessmentResult,
    generateAndSavePlan, getPlan,
    getSessionHistory,
    getProgressData,
    saveSessionResult,
    resetApp,
  };
})();

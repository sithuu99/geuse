"""
app/camera.py — Background webcam thread with MediaPipe hand detection.

Public API
----------
CameraStream.start(camera_index)  — open webcam and begin capture loop
CameraStream.stop()               — stop loop, release resources
CameraStream.get_frame()          — returns FrameData (base64 JPEG + inference)
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class FrameData:
    jpeg_b64: str                    # base64-encoded JPEG, ready for <img src="data:…">
    hand_detected: bool
    label: str                       # gesture label or "no hand"
    learned_closure: float           # 0–1, from model
    raw_closure: float               # 0–1, geometry proxy
    final_class: int                 # 0–4
    raw_class: int


_NO_HAND_FRAME = FrameData(
    jpeg_b64="",
    hand_detected=False,
    label="no hand",
    learned_closure=0.0,
    raw_closure=0.0,
    final_class=-1,
    raw_class=-1,
)


# --------------------------------------------------------------------------- #
# MediaPipe hand detector (shared, lazy-init)
# --------------------------------------------------------------------------- #

def _make_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


# --------------------------------------------------------------------------- #
# CameraStream
# --------------------------------------------------------------------------- #

class CameraStream:
    """
    Manages a background capture thread.
    One instance per application — keep it on the Api class.
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: FrameData = _NO_HAND_FRAME
        self._lock = threading.Lock()
        self._model = None   # injected lazily when available

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self, camera_index: int = 0) -> None:
        if self._running:
            return

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # Try to load the model (silently skip if weights not present yet)
        try:
            from app.model import GeuseModel
            self._model = GeuseModel.load()
        except FileNotFoundError:
            self._model = None

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-thread")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = _NO_HAND_FRAME
        if self._model:
            self._model.reset_buffers()

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #

    def get_frame(self) -> FrameData:
        """Return the most recent processed frame. Thread-safe."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------ #
    # Background loop
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        mp_hands = mp.solutions.hands
        draw_utils = mp.solutions.drawing_utils
        hands = _make_hands()

        while self._running and self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            data = self._process(frame, res, mp_hands, draw_utils)

            with self._lock:
                self._latest = data

        hands.close()

    def _process(self, frame: np.ndarray, res, mp_hands, draw_utils) -> FrameData:
        """Run detection + optional inference on one frame; encode to JPEG."""
        hand_detected = bool(res.multi_hand_landmarks)
        label = "no hand"
        learned_closure = 0.0
        raw_closure = 0.0
        final_class = -1
        raw_class = -1

        if hand_detected:
            hl = res.multi_hand_landmarks[0]
            draw_utils.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            if self._model is not None:
                result = self._model.infer(hl)
                label           = result.label
                learned_closure = result.learned_closure
                raw_closure     = result.raw_closure
                final_class     = result.final_class
                raw_class       = result.raw_class
            else:
                label = "hand detected"

        # Overlay label text
        cv2.putText(
            frame, label,
            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2, cv2.LINE_AA,
        )

        # Encode to JPEG → base64
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_b64 = base64.b64encode(buf).decode("ascii")

        return FrameData(
            jpeg_b64=jpeg_b64,
            hand_detected=hand_detected,
            label=label,
            learned_closure=learned_closure,
            raw_closure=raw_closure,
            final_class=final_class,
            raw_class=raw_class,
        )

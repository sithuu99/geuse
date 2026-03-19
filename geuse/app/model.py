"""
app/model.py — GeuseModel wraps the GeuseMultiTask neural network.

Provides:
  GeuseModel.load(path)  — class-method, returns a ready-to-use instance
  GeuseModel.infer(lms)  — takes a MediaPipe hand-landmark result object,
                           returns an InferResult dataclass
"""

from __future__ import annotations

import pathlib
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Constants (mirror realtime_multitask_demo.py)
# --------------------------------------------------------------------------- #

LABELS: dict[int, str] = {
    0: "neutral",
    1: "palm",
    2: "grabbing",
    3: "fist",
    4: "thumb_index",
}

FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

DEFAULT_MODEL_PATH = pathlib.Path(__file__).parent.parent / "assets" / "models" / "geuse_multitask.pt"


# --------------------------------------------------------------------------- #
# Neural network architecture
# --------------------------------------------------------------------------- #

class GeuseMultiTask(nn.Module):
    def __init__(self, in_dim: int = 63, num_classes: int = 5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.cls_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.cls_head(h), self.reg_head(h)


# --------------------------------------------------------------------------- #
# Feature helpers
# --------------------------------------------------------------------------- #

def _landmarks_to_features(lms) -> tuple[np.ndarray, np.ndarray]:
    """Normalise MediaPipe hand landmarks → flat feature vector + (21,3) pts."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)
    pts -= pts[0]                               # translate to wrist origin
    scale = np.linalg.norm(pts[9]) + 1e-6      # normalise by middle-MCP distance
    pts /= scale
    return pts.flatten(), pts


def _openness_from_pts(pts: np.ndarray) -> float:
    palm_center = pts[PALM_POINTS].mean(axis=0)
    dists = [np.linalg.norm(pts[i] - palm_center) for i in FINGERTIPS]
    return float(np.mean(dists))


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class InferResult:
    label: str            # final gesture label string
    final_class: int      # final class index (after rule overrides)
    raw_class: int        # model's smoothed prediction before overrides
    learned_closure: float  # model regression output (0–1), smoothed
    raw_closure: float    # geometry-based closure proxy (0–1)
    hand_detected: bool


# --------------------------------------------------------------------------- #
# GeuseModel public API
# --------------------------------------------------------------------------- #

class GeuseModel:
    """Wraps GeuseMultiTask + smoothing buffers for real-time inference."""

    _SMOOTH_WIN = 10

    def __init__(
        self,
        net: GeuseMultiTask,
        open_ref: float,
        closed_ref: float,
    ):
        self._net = net
        self._open_ref = open_ref
        self._closed_ref = closed_ref
        self._cls_buf: deque[int]   = deque(maxlen=self._SMOOTH_WIN)
        self._clo_buf: deque[float] = deque(maxlen=self._SMOOTH_WIN)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | pathlib.Path = DEFAULT_MODEL_PATH) -> "GeuseModel":
        """Load weights from a checkpoint file and return a GeuseModel instance."""
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu")
        net = GeuseMultiTask()
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        return cls(
            net=net,
            open_ref=float(ckpt["open_ref"]),
            closed_ref=float(ckpt["closed_ref"]),
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def infer(self, lms) -> InferResult:
        """
        Run a single inference pass.

        Parameters
        ----------
        lms : mediapipe hand landmark result (multi_hand_landmarks[0])

        Returns
        -------
        InferResult
        """
        feats, pts = _landmarks_to_features(lms)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, clo_t = self._net(x)
            raw_pred = int(torch.argmax(logits, dim=1).item())
            clo_val  = float(clo_t.item())

        self._cls_buf.append(raw_pred)
        self._clo_buf.append(clo_val)

        pred_smoothed = max(set(self._cls_buf), key=list(self._cls_buf).count)
        clo_smoothed  = sum(self._clo_buf) / len(self._clo_buf)

        # Geometry-based closure (reliable ground truth for thresholding)
        raw_open = _openness_from_pts(pts)
        denom = (self._open_ref - self._closed_ref) if abs(self._open_ref - self._closed_ref) > 1e-6 else 1.0
        raw_closure = float(np.clip((self._open_ref - raw_open) / denom, 0.0, 1.0))

        # Rule-based overrides on top of smoothed prediction
        if raw_closure >= 0.99:
            final_class = 3   # fist
        elif raw_closure <= 0.40:
            final_class = 1   # palm
        else:
            final_class = 2   # grabbing

        if pred_smoothed == 4:
            final_class = 4   # preserve thumb_index

        return InferResult(
            label=LABELS[final_class],
            final_class=final_class,
            raw_class=pred_smoothed,
            learned_closure=round(clo_smoothed, 4),
            raw_closure=round(raw_closure, 4),
            hand_detected=True,
        )

    def reset_buffers(self) -> None:
        """Clear smoothing history (call between sessions/exercises)."""
        self._cls_buf.clear()
        self._clo_buf.clear()

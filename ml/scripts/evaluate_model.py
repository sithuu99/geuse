"""
evaluate_model.py — Offline evaluation of the GeuseMultiTask model.

Usage (run from the repo root or from ml/scripts/):
    python ml/scripts/evaluate_model.py

Produces:
  - Terminal output: accuracy, per-class metrics, confusion matrix, regression MAE/MSE
  - ml/scripts/confusion_matrix.png
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = pathlib.Path(__file__).parent.resolve()
ML_ROOT     = SCRIPT_DIR.parent
DATASET     = ML_ROOT / "data" / "hand_dataset.npz"
MODEL_PATH  = ML_ROOT / "models" / "geuse_multitask.pt"
CM_OUT      = SCRIPT_DIR / "confusion_matrix.png"

# ── Labels (must match training order) ─────────────────────────────────────
LABELS = ["neutral", "palm", "grabbing", "fist", "thumb_index"]
NUM_CLASSES = 5

FINGERTIPS  = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]


# ── Model definition (identical to train_multitask.py) ─────────────────────
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

    def forward(self, x):
        h = self.shared(x)
        return self.cls_head(h), self.reg_head(h)


# ── Closure target builder (identical to train_multitask.py) ───────────────
def compute_openness(x63: np.ndarray) -> float:
    pts = x63.reshape(21, 3)
    palm_center = pts[PALM_POINTS].mean(axis=0)
    return float(np.mean([np.linalg.norm(pts[i] - palm_center) for i in FINGERTIPS]))


def build_closure_targets(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float, float]:
    openness  = np.array([compute_openness(v) for v in X], dtype=np.float32)
    open_ref  = float(np.median(openness[y == 1]))   # palm   → open
    closed_ref = float(np.median(openness[y == 3]))  # fist   → closed
    denom = (open_ref - closed_ref) if abs(open_ref - closed_ref) > 1e-6 else 1.0
    closure = np.clip((open_ref - openness) / denom, 0.0, 1.0).astype(np.float32)
    return closure, open_ref, closed_ref


# ── Load dataset and reproduce the exact train/val/test split ──────────────
def load_test_split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not DATASET.exists():
        sys.exit(f"[ERROR] Dataset not found: {DATASET}")

    data = np.load(DATASET)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    closure, open_ref, closed_ref = build_closure_targets(X, y)
    print(f"  Dataset loaded:  {len(X):,} samples, {NUM_CLASSES} classes")
    print(f"  Calibration:     open_ref={open_ref:.4f}, closed_ref={closed_ref:.4f}")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    c_t = torch.tensor(closure, dtype=torch.float32).unsqueeze(1)

    # Identical split to train_multitask.py (same seed + ratios)
    _, X_tmp, _, y_tmp, _, c_tmp = train_test_split(
        X_t, y_t, c_t, test_size=0.30, random_state=42, stratify=y
    )
    _, X_test, _, y_test, _, c_test = train_test_split(
        X_tmp, y_tmp, c_tmp, test_size=0.50, random_state=42,
        stratify=y_tmp.numpy()
    )

    print(f"  Test split:      {len(X_test):,} samples "
          f"({len(X_test)/len(X)*100:.1f}% of dataset)")
    return X_test, y_test, c_test


# ── Load model ──────────────────────────────────────────────────────────────
def load_model() -> GeuseMultiTask:
    if not MODEL_PATH.exists():
        sys.exit(f"[ERROR] Model not found: {MODEL_PATH}")

    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    model = GeuseMultiTask(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── Run inference on the full test set ─────────────────────────────────────
@torch.no_grad()
def run_inference(
    model: GeuseMultiTask,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    c_test: torch.Tensor,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_preds, all_true, all_reg_pred, all_reg_true = [], [], [], []

    for start in range(0, len(X_test), batch_size):
        xb = X_test[start : start + batch_size]
        yb = y_test[start : start + batch_size]
        cb = c_test[start : start + batch_size]

        logits, clo = model(xb)
        preds = torch.argmax(logits, dim=1).numpy()
        all_preds.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())
        all_reg_pred.extend(clo.squeeze(1).numpy().tolist())
        all_reg_true.extend(cb.squeeze(1).numpy().tolist())

    return (
        np.array(all_true,     dtype=np.int64),
        np.array(all_preds,    dtype=np.int64),
        np.array(all_reg_true, dtype=np.float32),
        np.array(all_reg_pred, dtype=np.float32),
    )


# ── Pretty-print confusion matrix ──────────────────────────────────────────
def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    col_w = max(len(l) for l in labels) + 2
    num_w = max(5, col_w)

    header = " " * (col_w + 2) + "  ".join(f"{l:>{num_w}}" for l in labels)
    print(header)
    print(" " * (col_w + 2) + "  ".join("-" * num_w for _ in labels))

    for i, row_label in enumerate(labels):
        row = "  ".join(f"{cm[i, j]:>{num_w}d}" for j in range(len(labels)))
        print(f"  {row_label:<{col_w}}{row}")


# ── Save confusion matrix as PNG ────────────────────────────────────────────
def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=8)
    ax.set_ylabel("True label", fontsize=12, labelpad=8)
    ax.set_title("GeuseMultiTask — Confusion Matrix (test set)", fontsize=13, pad=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:d}",
                ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    sep = "-" * 60

    print(f"\n{sep}")
    print("  GeuseMultiTask — Model Evaluation")
    print(sep)

    print("\n[1/4] Loading dataset and reproducing test split …")
    X_test, y_test, c_test = load_test_split()

    print("\n[2/4] Loading model …")
    model = load_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded:    {total_params:,} parameters")

    print("\n[3/4] Running inference …")
    y_true, y_pred, c_true, c_pred = run_inference(model, X_test, y_test, c_test)

    # ── Classification metrics ─────────────────────────────────────────────
    overall_acc = float((y_true == y_pred).mean())

    print(f"\n{sep}")
    print("  CLASSIFICATION RESULTS")
    print(sep)
    print(f"\n  Overall accuracy:  {overall_acc * 100:.2f}%  "
          f"({int(overall_acc * len(y_true))}/{len(y_true)} correct)\n")

    # Per-class accuracy
    print("  Per-class accuracy:")
    for cls_idx, cls_name in enumerate(LABELS):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            print(f"    {cls_name:<15} N/A  (no samples in test set)")
            continue
        cls_acc = float((y_pred[mask] == cls_idx).mean())
        n = int(mask.sum())
        correct = int((y_pred[mask] == cls_idx).sum())
        print(f"    {cls_name:<15} {cls_acc * 100:6.2f}%  ({correct}/{n})")

    # Confusion matrix (text)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print(f"\n  Confusion matrix (rows = true, cols = predicted):\n")
    print_confusion_matrix(cm, LABELS)

    # sklearn classification report
    print(f"\n  Precision / Recall / F1 per class:\n")
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
    for line in report.splitlines():
        print(f"    {line}")

    # ── Regression metrics ─────────────────────────────────────────────────
    mae  = float(np.mean(np.abs(c_pred - c_true)))
    mse  = float(np.mean((c_pred - c_true) ** 2))
    rmse = float(np.sqrt(mse))

    print(f"\n{sep}")
    print("  REGRESSION RESULTS  (closure head)")
    print(sep)
    print(f"\n  MAE  (mean absolute error):  {mae:.4f}")
    print(f"  MSE  (mean squared error):   {mse:.4f}")
    print(f"  RMSE (root MSE):             {rmse:.4f}")
    print(f"\n  Ground-truth closure — min={c_true.min():.3f}  "
          f"max={c_true.max():.3f}  mean={c_true.mean():.3f}")
    print(f"  Predicted  closure — min={c_pred.min():.3f}  "
          f"max={c_pred.max():.3f}  mean={c_pred.mean():.3f}")

    # ── Save confusion matrix PNG ──────────────────────────────────────────
    print(f"\n[4/4] Saving confusion matrix image …")
    save_confusion_matrix_png(cm, LABELS, CM_OUT)
    print(f"  Saved: {CM_OUT}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()

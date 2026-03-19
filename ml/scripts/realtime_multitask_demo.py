import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque

MODEL_PATH = "geuse_multitask.pt"

LABELS = {
    0: "neutral",
    1: "palm",
    2: "grabbing",
    3: "fist",
    4: "thumb_index",
}

FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

def landmarks_to_features(lms):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten(), pts

def openness_from_pts(pts):
    palm_center = pts[PALM_POINTS].mean(axis=0)
    dists = [np.linalg.norm(pts[i] - palm_center) for i in FINGERTIPS]
    return float(np.mean(dists))

class GeuseMultiTask(nn.Module):
    def __init__(self, in_dim=63, num_classes=5):
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
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.shared(x)
        return self.cls_head(h), self.reg_head(h)

def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = GeuseMultiTask()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # These are used if you want to compute a raw closure too (optional)
    open_ref = float(ckpt["open_ref"])
    closed_ref = float(ckpt["closed_ref"])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    cls_buf = deque(maxlen=10)
    clo_buf = deque(maxlen=10)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        text = "no hand"
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            feats, pts = landmarks_to_features(hl)
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits, clo = model(x)
                pred = int(torch.argmax(logits, dim=1).item())
                clo = float(clo.item())

            cls_buf.append(pred)
            clo_buf.append(clo)

            # Smooth
            pred_smoothed = max(set(cls_buf), key=list(cls_buf).count)
            clo_smoothed = sum(clo_buf) / len(clo_buf)

            # Raw closure proxy (use THIS for thresholds)
            raw_open = openness_from_pts(pts)
            denom = (open_ref - closed_ref) if abs(open_ref - closed_ref) > 1e-6 else 1.0
            raw_closure = float(np.clip((open_ref - raw_open) / denom, 0.0, 1.0))

            # Override flexion states using raw_closure thresholds
            if raw_closure >= 0.99:
                final_class = 3  # fist
            elif raw_closure <= 0.40:
                final_class = 1  # palm
            else:
                final_class = 2  # grabbing

            # Preserve thumb_index if predicted (you can also require confidence later)
            if pred_smoothed == 4:
                final_class = 4

            label = LABELS[final_class]
            text = f"{label}  learned={clo_smoothed:.2f}  raw={raw_closure:.2f}"


        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Geuse Rehab Demo", frame)



        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


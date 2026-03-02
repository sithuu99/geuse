import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ===== CONFIG =====
DATASET_ROOT = "datasets"
CLASSES = {
    "neutral": 0,
    "palm": 1,
    "grabbing": 2,
    "fist": 3,
    "thumb_index": 4
}
OUTPUT_FILE = "hand_dataset.npz"

# ===== MEDIAPIPE SETUP =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks)

    # ---- Normalization ----
    landmarks = landmarks.reshape(21, 3)

    # Translate so wrist is origin
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale by hand size (distance wrist to middle finger MCP)
    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


X = []
y = []

for class_name, label in CLASSES.items():
    class_path = os.path.join(DATASET_ROOT, class_name)

    for root, dirs, files in os.walk(class_path):
        for file in tqdm(files, desc=f"Processing {class_name}"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                features = extract_landmarks(image)
                if features is not None:
                    X.append(features)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print("Final dataset shape:", X.shape)

np.savez(OUTPUT_FILE, X=X, y=y)
print(f"Saved to {OUTPUT_FILE}")

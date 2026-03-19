import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET = "hand_dataset.npz"
MODEL_OUT = "geuse_multitask.pt"

# Label mapping (keep consistent)
# 0 neutral, 1 palm, 2 grabbing, 3 fist, 4 thumb_index
NUM_CLASSES = 5

FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

def compute_openness_from_features(x63: np.ndarray) -> float:
    pts = x63.reshape(21, 3)
    palm_center = pts[PALM_POINTS].mean(axis=0)
    dists = [np.linalg.norm(pts[i] - palm_center) for i in FINGERTIPS]
    return float(np.mean(dists))

def build_closure_targets(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Openness for all samples
    openness = np.array([compute_openness_from_features(v) for v in X], dtype=np.float32)

    # Calibrate using PALM as open reference and FIST as closed reference
    open_ref = np.median(openness[y == 1])   # palm
    closed_ref = np.median(openness[y == 3]) # fist

    denom = (open_ref - closed_ref) if abs(open_ref - closed_ref) > 1e-6 else 1.0
    closure = (open_ref - openness) / denom
    closure = np.clip(closure, 0.0, 1.0).astype(np.float32)

    return closure, float(open_ref), float(closed_ref)

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
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.cls_head(h)
        closure = self.reg_head(h)
        return logits, closure

def main():
    data = np.load(DATASET)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    closure, open_ref, closed_ref = build_closure_targets(X, y)
    print(f"Calibration refs: open_ref={open_ref:.4f}, closed_ref={closed_ref:.4f}")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    c_t = torch.tensor(closure, dtype=torch.float32).unsqueeze(1)

    # stratified split
    X_train, X_tmp, y_train, y_tmp, c_train, c_tmp = train_test_split(
        X_t, y_t, c_t, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, c_val, c_test = train_test_split(
        X_tmp, y_tmp, c_tmp, test_size=0.50, random_state=42, stratify=y_tmp.numpy()
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train, c_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val, c_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test, c_test), batch_size=256, shuffle=False)

    # Class weights (neutral is smaller)
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    cls_weights = (counts.sum() / (counts + 1e-6))
    cls_weights = cls_weights / cls_weights.mean()

    loss_cls = nn.CrossEntropyLoss(weight=cls_weights)
    loss_reg = nn.MSELoss()

    model = GeuseMultiTask(num_classes=NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = -1.0
    best_state = None
    lambda_reg = 2.0

    for epoch in range(40):
        model.train()
        for xb, yb, cb in train_loader:
            opt.zero_grad()
            logits, cp = model(xb)
            l1 = loss_cls(logits, yb)
            l2 = loss_reg(cp, cb)
            loss = l1 + lambda_reg * l2
            loss.backward()
            opt.step()

        # validate on classification accuracy (primary)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb, cb in val_loader:
                logits, cp = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / total

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d}/40  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "open_ref": open_ref,
            "closed_ref": closed_ref,
        },
        MODEL_OUT
    )
    print(f"Saved multitask model to {MODEL_OUT} (best val_acc={best_val:.4f})")

    # Test classification + regression quality
    model.eval()
    all_preds, all_true = [], []
    reg_preds, reg_true = [], []
    with torch.no_grad():
        for xb, yb, cb in test_loader:
            logits, cp = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(pred.tolist())
            all_true.extend(yb.cpu().numpy().tolist())
            reg_preds.extend(cp.squeeze(1).cpu().numpy().tolist())
            reg_true.extend(cb.squeeze(1).cpu().numpy().tolist())

    print("\nConfusion matrix:")
    print(confusion_matrix(all_true, all_preds))
    print("\nClassification report:")
    print(classification_report(all_true, all_preds, digits=4))

    # Regression summary
    reg_preds = np.array(reg_preds, dtype=np.float32)
    reg_true = np.array(reg_true, dtype=np.float32)
    mae = float(np.mean(np.abs(reg_preds - reg_true)))
    rmse = float(np.sqrt(np.mean((reg_preds - reg_true) ** 2)))
    print(f"\nClosure regression: MAE={mae:.4f} RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()

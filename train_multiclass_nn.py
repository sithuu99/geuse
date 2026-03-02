import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET = "hand_dataset.npz"
MODEL_OUT = "geuse_multiclass.pt"
NUM_CLASSES = 5

class GeuseNet(nn.Module):
    def __init__(self, in_dim=63, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def main():
    data = np.load(DATASET)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    # stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y.numpy()
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp.numpy()
    )

    # class weights to handle imbalance (neutral is smaller)
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()  # normalize weights
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    device = torch.device("cpu")
    model = GeuseNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(40):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = torch.argmax(model(xb), dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d}/40  val_acc={val_acc:.4f}")

    # load best
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved best model to {MODEL_OUT} (val_acc={best_val_acc:.4f})")

    # test report
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

    print("\nConfusion matrix:")
    print(confusion_matrix(all_true, all_preds))
    print("\nClassification report:")
    print(classification_report(all_true, all_preds, digits=4))

if __name__ == "__main__":
    main()

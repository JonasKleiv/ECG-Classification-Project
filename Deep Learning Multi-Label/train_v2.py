import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# Configuration
torch.manual_seed(42)

data_dir = Path(r"E:\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

classes = ["NORM", "MI", "STTC", "CD", "HYP"]

epochs = 10
batch_size = 32
learning_rate = 0.001

# Read metadata
metadata = pd.read_csv(data_dir / "ptbxl_database.csv")
statements = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)


def get_labels(code_text):
    codes = ast.literal_eval(code_text)
    diagnoses = set()

    for code in codes:
        if code in statements.index:
            group = statements.at[code, "diagnostic_class"]

            if group in classes:
                diagnoses.add(group)

    return [int(name in diagnoses) for name in classes]


metadata["labels"] = metadata["scp_codes"].apply(get_labels)
metadata = metadata[metadata["labels"].apply(any)].copy()

# Patient-separated training and validation folds
train_rows = metadata[metadata["strat_fold"] <= 8].sample(n=4000, random_state=42)

val_rows = metadata[metadata["strat_fold"] == 9]


# Load ECG signals
def load_data(rows):
    signals = []
    labels = []

    for row in rows.itertuples():
        signal, _ = wfdb.rdsamp(
            str(data_dir / row.filename_lr)
        )

        if not np.isfinite(signal).all():
            continue

        signal = signal.astype(np.float32)
        signal -= signal.mean(axis=0)

        signals.append(signal.T.copy())
        labels.append(row.labels)

    x = torch.from_numpy(np.stack(signals))
    y = torch.tensor(np.array(labels), dtype=torch.float32)

    return x, y


x_train, y_train = load_data(train_rows)
x_val, y_val = load_data(val_rows)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(x_val, y_val),
    batch_size=64
)

print("Training:", x_train.shape)
print("Validation:", x_val.shape)


# CNN architecture
model = nn.Sequential(
    nn.Conv1d(12, 32, kernel_size=7, padding=3),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.Conv1d(32, 64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(64, 5)
)


# Handle class imbalance
positive_counts = y_train.sum(dim=0)
negative_counts = len(y_train) - positive_counts

pos_weight = negative_counts / positive_counts.clamp(min=1)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate
)

print("\nClass weights:")
for name, weight in zip(classes, pos_weight):
    print(f"{name}: {weight.item():.2f}")


# Validation
def evaluate():
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for signals, labels in val_loader:
            outputs = model(signals)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(signals)
            all_labels.append(labels.numpy())
            all_scores.append(torch.sigmoid(outputs).numpy())

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    predictions = (scores >= 0.5).astype(int)

    val_loss = total_loss / len(x_val)
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_auc = roc_auc_score(labels, scores, average="macro")

    return val_loss, micro_f1, macro_auc, labels, scores


# Training loop
best_val_loss = float("inf")
model_path = Path(__file__).parent / "ecg_cnn_v2.pt"

# Save training results after each epoch
history = []
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
history_file = results_dir / "train_v2_history.csv"

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for signals, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(signals)

    train_loss = total_loss / len(x_train)

    val_loss, micro_f1, macro_auc, _, _ = evaluate()

    # Record and save the actual results from this training run
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "micro_f1": micro_f1,
        "macro_auc": macro_auc
    })
    pd.DataFrame(history).to_csv(history_file, index=False)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Train: {train_loss:.4f} | "
        f"Val: {val_loss:.4f} | "
        f"F1: {micro_f1:.4f} | "
        f"AUROC: {macro_auc:.4f}"
    )


# Evaluate the best checkpoint
model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)

_, micro_f1, macro_auc, labels, scores = evaluate()
predictions = (scores >= 0.5).astype(int)

print("\nBest model - validation results")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Macro AUROC: {macro_auc:.4f}")

for i, name in enumerate(classes):
    class_f1 = f1_score(
        labels[:, i], predictions[:, i], zero_division=0
    )

    class_auc = roc_auc_score(labels[:, i], scores[:, i])

    print(
        f"{name}: F1={class_f1:.3f}, "
        f"AUROC={class_auc:.3f}"
    )

print("\nModel saved:", model_path)

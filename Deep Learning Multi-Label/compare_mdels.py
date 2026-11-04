
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score


data_dir = Path(
    r"E:\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

model_dir = Path(r"E:\EKG")
classes = ["NORM", "MI", "STTC", "CD", "HYP"]

results_dir = model_dir / "results"
results_dir.mkdir(exist_ok=True)


# Read ECG information
data = pd.read_csv(data_dir / "ptbxl_database.csv")
statements = pd.read_csv(
    data_dir / "scp_statements.csv", index_col=0
)


def get_labels(text):
    codes = ast.literal_eval(text)
    diagnoses = []

    for code in codes:
        if code in statements.index:
            group = statements.at[code, "diagnostic_class"]

            if group in classes:
                diagnoses.append(group)

    return [int(name in diagnoses) for name in classes]


data["labels"] = data["scp_codes"].apply(get_labels)

# Use the same validation recordings
data = data[data["labels"].apply(any)]
data = data[data["strat_fold"] == 9]


# Load ECG recordings
signals = []
labels = []
ecg_ids = []

for row in data.itertuples():
    signal, _ = wfdb.rdsamp(
        str(data_dir / row.filename_lr)
    )

    if not np.isfinite(signal).all():
        continue

    signal = signal.astype(np.float32)
    signal -= signal.mean(axis=0)

    signals.append(signal.T.copy())
    labels.append(row.labels)
    ecg_ids.append(row.ecg_id)

x = torch.from_numpy(np.stack(signals))
y = np.array(labels)

print("ECG recordings:", len(x))


# Same CNN used for V2 and V3
def create_model():
    return nn.Sequential(
        nn.Conv1d(12, 32, 7, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(32, 64, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 5)
    )


# Load model and calculate predictions
def predict(filename):
    model = create_model()

    weights = torch.load(
        model_dir / filename,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    results = []

    with torch.no_grad():
        for start in range(0, len(x), 64):
            batch = x[start:start + 64]
            output = model(batch)
            scores = torch.sigmoid(output)
            results.append(scores.numpy())

    return np.concatenate(results)


print("Comparing models...")

scores_v2 = predict("ecg_cnn_v2.pt")
scores_v3 = predict("ecg_cnn_v3.pt")

pred_v2 = (scores_v2 >= 0.5).astype(int)
pred_v3 = (scores_v3 >= 0.5).astype(int)


# Overall results
print("\nOverall results")

for name, scores, predictions in [
    ("V2", scores_v2, pred_v2),
    ("V3", scores_v3, pred_v3)
]:
    f1 = f1_score(y, predictions, average="micro")
    auc = roc_auc_score(y, scores, average="macro")

    print(name, "| F1:", round(f1, 4),
          "| AUROC:", round(auc, 4))


# Compare diagnoses and errors
print("\nResults by diagnosis")

f1_list_v2 = []
f1_list_v3 = []

for i, name in enumerate(classes):
    f1_v2 = f1_score(y[:, i], pred_v2[:, i])
    f1_v3 = f1_score(y[:, i], pred_v3[:, i])

    f1_list_v2.append(f1_v2)
    f1_list_v3.append(f1_v3)

    fixed = np.sum(
        (pred_v2[:, i] != y[:, i]) &
        (pred_v3[:, i] == y[:, i])
    )

    new_errors = np.sum(
        (pred_v2[:, i] == y[:, i]) &
        (pred_v3[:, i] != y[:, i])
    )

    print(name)
    print("  V2 F1:", round(f1_v2, 3))
    print("  V3 F1:", round(f1_v3, 3))
    print("  Errors fixed:", fixed)
    print("  New errors:", new_errors)


# Show examples where models disagree
print("\nExamples of different predictions")

different = np.where(
    np.any(pred_v2 != pred_v3, axis=1)
)[0]

for i in different[:5]:
    actual = [classes[j] for j in range(5) if y[i, j] == 1]
    v2 = [classes[j] for j in range(5) if pred_v2[i, j] == 1]
    v3 = [classes[j] for j in range(5) if pred_v3[i, j] == 1]

    print("\nECG:", ecg_ids[i])
    print("Actual:", actual)
    print("V2:", v2)
    print("V3:", v3)


# Plot F1 scores
plt.figure(figsize=(8, 5))

plt.plot(classes, f1_list_v2, marker="o", label="V2")
plt.plot(classes, f1_list_v3, marker="o", label="V3")

plt.title("F1 score by diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("F1 score")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(results_dir / "f1_comparison.png")


# Plot learning curves from the saved training history
for name in ["train_v2", "train_v3"]:
    history_file = results_dir / (name + "_history.csv")

    if not history_file.exists():
        print("Missing training history:", history_file)
        continue

    history = pd.read_csv(history_file)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Training")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    plt.title(name + " - Learning curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(history["epoch"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / (name + "_learning.png"))

plt.show()


import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from sklearn.metrics import precision_score, recall_score


# Data folders 
data_dir = Path(r"E:\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
project_dir = Path(r"E:\EKG")
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

# Classes of diagnosis
classes = ["NORM", "MI", "STTC", "CD", "HYP"]


# Read diagnosis information
data = pd.read_csv(data_dir / "ptbxl_database.csv")
statements = pd.read_csv(
    data_dir / "scp_statements.csv", index_col=0
)


def get_labels(text):
    codes = ast.literal_eval(text)
    found = []

    for code in codes:
        if code in statements.index:
            group = statements.at[code, "diagnostic_class"]

            if group in classes:
                found.append(group)

    return [int(name in found) for name in classes]


data["labels"] = data["scp_codes"].apply(get_labels)

# Use test recordings only
data = data[data["strat_fold"] == 10]
data = data[data["labels"].apply(any)]


# Load ECG recordings
signals = []
labels = []
ecg_ids = []

for i, row in enumerate(data.itertuples(), start=1):
    if i == 1 or i % 500 == 0:
        print("Loading ECG:", i, "/", len(data), flush=True)

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

print("Test recordings:", len(x))


# Build the same CNN
model = nn.Sequential(
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


# Load trained V3 model
weights = torch.load(
    project_dir / "ecg_cnn_v3.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)
model.eval()


# Make predictions
all_scores = []

with torch.no_grad():
    for start in range(0, len(x), 64):
        batch = x[start:start + 64]
        output = model(batch)
        scores = torch.sigmoid(output)
        all_scores.append(scores.numpy())

scores = np.concatenate(all_scores)
predictions = (scores >= 0.5).astype(int)



# Overall results
micro_f1 = f1_score(y, predictions, average="micro")
macro_auc = roc_auc_score(y, scores, average="macro")
hamming = hamming_loss(y, predictions)
macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)



# Compare V2 and V3 on the same test recordings
from copy import deepcopy

# Load V2 without changing the V3 model
model_v2 = deepcopy(model)

weights_v2 = torch.load(
    project_dir / "ecg_cnn_v2.pt",
    map_location="cpu",
    weights_only=True
)

model_v2.load_state_dict(weights_v2)
model_v2.eval()

# Make V2 predictions
v2_scores_list = []

with torch.no_grad():
    for start in range(0, len(x), 64):
        batch = x[start:start + 64]
        outputs = model_v2(batch)
        v2_scores_list.append(torch.sigmoid(outputs).numpy())

v2_scores = np.concatenate(v2_scores_list)
v2_predictions = (v2_scores >= 0.5).astype(int)

# Calculate V2 results
v2_hamming = hamming_loss(y, v2_predictions)
v2_macro_f1 = f1_score( y, v2_predictions, average="macro", zero_division=0)

# Get feature information from ECG data
features = f"Raw data ({x.shape[1]} x {x.shape[2]})"

# Get model names from model files
v2_name = Path("ecg_cnn_v2.pt").stem
v3_name = Path("ecg_cnn_v3.pt").stem

# Create comparison table
summary = pd.DataFrame([
    {
        "Model": v2_name,
        "Features": features,
        "Hamming Loss": f"{v2_hamming:.4f}",
        "Macro F1": f"{v2_macro_f1:.4f}"
    },
    {
        "Model": v3_name,
        "Features": features,
        "Hamming Loss": f"{hamming:.4f}",
        "Macro F1": f"{macro_f1:.4f}"
    }
])

print("\nModel results")
print(summary.to_string(index=False))

summary.to_csv(results_dir / "model_summary.csv", index=False)

# Results for each diagnosis
f1_values = []
false_positive = []
false_negative = []
results = []

for i, name in enumerate(classes):
    actual = y[:, i]
    predicted = predictions[:, i]

    f1 = f1_score(actual, predicted, zero_division=0)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    auc = roc_auc_score(actual, scores[:, i])

    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    f1_values.append(f1)
    false_positive.append(fp)
    false_negative.append(fn)

    results.append({
        "Diagnosis": name,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUROC": auc,
        "False positive": fp,
        "False negative": fn
    })

    print(
        name,
        "| F1:", round(f1, 3),
        "| Precision:", round(precision, 3),
        "| Recall:", round(recall, 3),
        "| AUROC:", round(auc, 3)
    )

# Save results
pd.DataFrame(results).to_csv(
    results_dir / "test_v3_results.csv",
    index=False
)

# Save predictions for each ECG
prediction_data = pd.DataFrame({"ecg_id": ecg_ids})

for i, name in enumerate(classes):
    prediction_data[name + "_actual"] = y[:, i]
    prediction_data[name + "_score"] = scores[:, i]
    prediction_data[name + "_predicted"] = predictions[:, i]

prediction_data.to_csv(
    results_dir / "test_v3_predictions.csv",
    index=False
)


# Plot F1 scores
plt.figure(figsize=(8, 5))
plt.bar(classes, f1_values)
plt.title("V3 test - F1 by diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("F1 score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(results_dir / "test_v3_f1.png")


# Plot prediction errors
positions = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(
    positions - width / 2,
    false_positive,
    width,
    label="False positive"
)
plt.bar(
    positions + width / 2,
    false_negative,
    width,
    label="False negative"
)

plt.xticks(positions, classes)
plt.title("V3 test - Prediction errors")
plt.ylabel("Number of errors")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "test_v3_errors.png")

print("\nResults saved in:", results_dir)

plt.close("all")

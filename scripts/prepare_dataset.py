from datasets import load_dataset
from collections import defaultdict
import pandas as pd
import numpy as np
import random
import json
import os

random.seed(42)

print("Loading CIC-IDS2017...")

ds = load_dataset(
    "muzom/CIC-IDS2017",
    split="train",
    streaming=True
)

limits = {
    "normal": 20000,
    "port_scan": 20000,
    "brute_force": 10000,
    "dos": 20000,
    "web_attack": 10000
}

samples = defaultdict(list)
seen = defaultdict(int)

def classify(label):
    label = label.strip().lower()

    if label == "benign":
        return "normal"

    if label == "portscan":
        return "port_scan"

    if "patator" in label:
        return "brute_force"

    if label.startswith("dos") or label == "ddos":
        return "dos"

    if label.startswith("web attack"):
        return "web_attack"

    return None


print("Scanning and sampling records...")

for row in ds:
    label = classify(row[" Label"])

    if label is None:
        continue

    seen[label] += 1
    limit = limits[label]

    if len(samples[label]) < limit:
        samples[label].append(row)
    else:
        index = random.randrange(seen[label])
        if index < limit:
            samples[label][index] = row

print("\nSample collection complete.")

frames = []

for label, records in samples.items():
    df = pd.DataFrame(records)
    df["label"] = label
    frames.append(df)
    print(f"{label}: {len(df)} records")

data = pd.concat(frames, ignore_index=True)

# Standardise column names
data.columns = data.columns.str.strip()

# Remove the original target column
data = data.drop(columns=["Label"], errors="ignore")

# Convert features to numeric
for col in data.columns:
    if col != "label":
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

# Replace infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove columns that contain no usable data
data.dropna(axis=1, how="all", inplace=True)

# Median imputation
for col in data.columns:
    if col != "label":
        data[col] = data[col].fillna(data[col].median())

# Remove any remaining invalid rows
data.dropna(inplace=True)

# Shuffle records
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("dataset", exist_ok=True)

# Export
data.to_csv("dataset/dataset.csv", index=False)

# Save class distribution
distribution = data["label"].value_counts().to_dict()

with open("dataset/class_distribution.json", "w") as f:
    json.dump(distribution, f, indent=4)

# Save feature dictionary
with open("dataset/feature_dictionary.txt", "w") as f:
    f.write("CIC-IDS2017 FEATURE DICTIONARY\n")
    f.write("Each row represents a network flow.\n\n")

    for col in data.columns:
        if col == "label":
            f.write("label: Target classification category\n")
        else:
            f.write(f"{col}: Numerical network-flow feature\n")

print("\nDATASET PREPARATION COMPLETE")
print("Total records:", len(data))
print("Total features:", len(data.columns) - 1)
print("\nFinal distribution:")
print(data["label"].value_counts())
print("\nSaved to dataset/dataset.csv")

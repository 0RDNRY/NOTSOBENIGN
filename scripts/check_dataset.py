from datasets import load_dataset
from collections import Counter

print("Loading CIC-IDS2017...")

ds = load_dataset(
    "muzom/CIC-IDS2017",
    split="train",
    streaming=True
)

counts = Counter()

print("Scanning COMPLETE dataset...")

for row in ds:
    counts[row[" Label"].strip()] += 1

print("\nComplete dataset labels:")
for label, count in counts.most_common():
    print(f"{label}: {count:,}")

print("\nTotal records:", sum(counts.values()))

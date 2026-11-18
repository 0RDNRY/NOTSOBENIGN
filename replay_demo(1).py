"""
Replay demo: streams network flows from the dataset one at a time through the model,
as if it were watching a live network. No lab or packet capture needed.

    python replay_demo.py                 # full scenario, ~2 minutes
    python replay_demo.py --speed 3       # 3x faster
    python replay_demo.py --stress flags  # simulate a broken collector (see below)
    python replay_demo.py --hide-truth    # do not show the true label next to the prediction
    python replay_demo.py --errors        # also replay the model's real mistakes (about 18 flows)

--stress options: noise (5% random noise), flags (TCP flag counts missing),
                  port (destination port shuffled), units (time in seconds not microseconds)
Press Ctrl+C to stop; a summary is still printed.
"""
import argparse, json, os, sys, time, warnings
import joblib, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=".")
ap.add_argument("--speed", type=float, default=1.0, help="1 = normal pace, 3 = three times faster, 0 = no delay")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--threshold", type=float, default=0.90)
ap.add_argument("--hide-truth", action="store_true")
ap.add_argument("--errors", action="store_true", help="add a final phase replaying the model's real mistakes from the dataset")
ap.add_argument("--stress", choices=["noise", "flags", "port", "units"])
a = ap.parse_args()

def find(name):
    for sub in ("", "models", "dataset", "dataset_handover/dataset", "../models", "../dataset"):
        p = os.path.join(a.dir, sub, name)
        if os.path.isfile(p):
            return p
    sys.exit(f"Could not find {name}. Run from the project folder or pass --dir <path>.")

model = joblib.load(find("final_classifier.joblib"))
enc = joblib.load(find("label_encoder.joblib"))
feats = json.load(open(find("feature_columns.json")))
data = pd.read_csv(find("dataset.csv"))

rng = np.random.default_rng(a.seed)
USE_COLOR = sys.stdout.isatty()
def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text
COL = {"normal": "32", "port_scan": "33", "brute_force": "35", "dos": "31", "web_attack": "36"}

# Scenario: (title, true class, number of flows, pace in seconds per flow)
scenario = [
    ("Normal browsing and connections", "normal", 12, 0.5),
    ("Reconnaissance: a port scan starts", "port_scan", 12, 0.25),
    ("Normal traffic resumes", "normal", 6, 0.5),
    ("Login guessing against the server", "brute_force", 10, 0.4),
    ("A flood of requests hits the web server", "dos", 12, 0.2),
    ("Normal traffic resumes", "normal", 5, 0.5),
    ("Malicious web requests (injection attempts)", "web_attack", 10, 0.4),
    ("Normal traffic resumes", "normal", 5, 0.5),
]

if a.errors:
    allp = model.predict_proba(data[feats])
    wrong_idx = np.where(enc.classes_[allp.argmax(1)] != data["label"].values)[0]
    scenario.append((f"The model's real mistakes: all {len(wrong_idx)} errors in the dataset", None, len(wrong_idx), 0.6))

def sleep(s):
    if a.speed > 0:
        time.sleep(s / a.speed)

def apply_stress(X):
    X = X.copy()
    if a.stress == "noise":
        X = X * (1 + rng.normal(0, 0.05, X.shape))
    elif a.stress == "flags":
        X[[f for f in feats if "Flag" in f]] = 0
    elif a.stress == "port":
        X["Destination Port"] = rng.integers(1, 65535, len(X))
    elif a.stress == "units":
        X[[f for f in feats if any(k in f for k in ("IAT", "Duration", "Active", "Idle"))]] /= 1e6
    return X

print(c("\n  NETWORK ATTACK DETECTION - live monitor (replaying recorded flows)", "1"))
if a.stress:
    print(c(f"  STRESS MODE: {a.stress}", "1;31"))
print(f"  Model: XGBoost | {len(feats)} flow features | classes: {', '.join(enc.classes_)}")
print(f"  Flows below {a.threshold:.0%} confidence are marked REVIEW\n")
header = f"  {'#':>3}  {'Dst port':>8}  {'Pkts':>5}  {'Duration':>10}  {'Prediction':<12} {'Conf':>6}"
if not a.hide_truth:
    header += f"  {'Actual':<12}"
print(c(header, "2"))

rows, n = [], 0
try:
    for title, cls, count, pace in scenario:
        print(c(f"\n  >>> {title}", "1;37"))
        sleep(1.2)
        if cls is None:
            pick = wrong_idx
        else:
            pool = np.where(data["label"].values == cls)[0]
            pick = rng.choice(pool, count, replace=False)
        for i in pick:
            flow = data.iloc[[i]]
            cls_true = flow["label"].iloc[0]
            X = apply_stress(flow[feats])
            proba = model.predict_proba(X)[0]
            pred = enc.classes_[proba.argmax()]
            conf = proba.max()
            n += 1
            rows.append((cls_true, pred, conf))
            r = flow.iloc[0]
            alert = "" if pred == "normal" else c("ALERT ", "1;31")
            line = (f"  {n:>3}  {int(r['Destination Port']):>8}  {int(r['Total Fwd Packets'] + r['Total Backward Packets']):>5}  "
                    f"{r['Flow Duration'] / 1e6:>9.3f}s  {c(f'{pred:<12}', COL[pred])} {conf:>6.1%}")
            if not a.hide_truth:
                line += f"  {cls_true:<12}" + ("  " + c("ok", "32") if pred == cls_true else "  " + c("WRONG", "1;31"))
            if conf < a.threshold:
                line += "  " + c("REVIEW", "1;33")
            print(line + ("  " + alert if alert else ""))
            sleep(pace)
except KeyboardInterrupt:
    print("\n  (stopped)")

# Summary
df = pd.DataFrame(rows, columns=["actual", "pred", "conf"])
if len(df):
    ok = (df.actual == df.pred).sum()
    print(c("\n  SUMMARY", "1"))
    print(f"  Flows processed: {len(df)}   Correct: {ok} ({ok / len(df):.0%})   Flagged for review: {(df.conf < a.threshold).sum()}")
    labels = list(enc.classes_)
    cm = pd.crosstab(df.actual, df.pred).reindex(index=labels, columns=labels, fill_value=0)
    print("\n  Actual (rows) vs predicted (columns):")
    print("  " + cm.to_string().replace("\n", "\n  "))
    attacks = df[df.actual != "normal"]
    if len(attacks):
        print(f"\n  Attack flows flagged as an attack: {(attacks.pred != 'normal').sum()}/{len(attacks)}")
        print(f"  Normal flows wrongly flagged: {((df.actual == 'normal') & (df.pred != 'normal')).sum()}/{(df.actual == 'normal').sum()}")
if a.errors:
    print("  Note: the summary includes the hand-picked errors, so it is NOT typical accuracy.")
    print("  Typical accuracy is 99.98% (18 errors in 72,180 flows).")
print()

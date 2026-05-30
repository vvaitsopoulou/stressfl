"""
WESAD Leakage Diagnostic
Run in project root to determine whether
perfect LOSOCV scores are genuine or caused by data leakage.

Three tests:
1. Permutation test — shuffle labels, retrain. If still perfect -> leak.
2. Subject ID prediction — can RF predict which subject from features alone?
3. Imputation audit — check if global imputation contaminates folds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

# ── Load ──
CHEST_FEATURES = "data/processed/wesad/features/chest_features.pkl"
META_COLS = ["sid", "label", "window_idx", "is_overlapped"]

df = pd.read_pickle(CHEST_FEATURES)
feat_cols = [c for c in df.columns if c not in META_COLS]
df["binary"] = df["label"].map({1.0: 0, 2.0: 1, 3.0: 0})

X = df[feat_cols].fillna(0).values
y = df["binary"].values
groups = df["sid"].values
is_indep = (~df["is_overlapped"]).values

print("LEAKAGE DIAGNOSTIC")
print("\n")

# ── 1. Label distribution ──
print("\n1. PER-SUBJECT VALIDATION SET SIZES (independent windows only)")
for sid in sorted(df["sid"].unique()):
    sub = df[df["sid"] == sid]
    indep = sub[~sub["is_overlapped"]]
    print(
        f"  S{int(sid):>2}: {len(indep):>4} independent windows "
        f"(stress={int((indep['binary']==1).sum())}, "
        f"non-stress={int((indep['binary']==0).sum())})"
    )

# ── 2. Permutation test ──
print("\n2. PERMUTATION TEST")
print("   Training RF with real vs shuffled labels (LOSO, independent val only)...")

rng = np.random.RandomState(42)
y_shuffled = y.copy()
for sid in np.unique(groups):
    mask = groups == sid
    y_shuffled[mask] = rng.permutation(y_shuffled[mask])

logo = LeaveOneGroupOut()
real_scores = []
shuffled_scores = []

for train_idx, test_idx in logo.split(X, y, groups):
    test_idx_indep = test_idx[is_indep[test_idx]]
    if len(test_idx_indep) == 0:
        continue

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X[train_idx], y[train_idx])
    real_scores.append(rf.score(X[test_idx_indep], y[test_idx_indep]))

    rf2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf2.fit(X[train_idx], y_shuffled[train_idx])
    shuffled_scores.append(rf2.score(X[test_idx_indep], y_shuffled[test_idx_indep]))

print(f"   Real labels:     {np.mean(real_scores):.4f} +/- {np.std(real_scores):.4f}")
print(
    f"   Shuffled labels: {np.mean(shuffled_scores):.4f} +/- {np.std(shuffled_scores):.4f}"
)

if np.mean(shuffled_scores) > 0.80:
    print("   *** LEAKAGE DETECTED: model predicts even shuffled labels ***")
elif np.mean(real_scores) > 0.99 and np.mean(shuffled_scores) < 0.60:
    print("   Task is genuinely easy — real >> shuffled. No structural leakage.")
else:
    print("   Results look plausible.")

# ── 3. Subject ID prediction ──
print("\n3. SUBJECT ID PREDICTION (features encode subject identity?)")
X_indep = X[is_indep]
y_sid = groups[is_indep]
sid_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_indep,
    y_sid,
    cv=5,
    scoring="accuracy",
)
print(f"   Subject ID accuracy: {sid_scores.mean():.4f} +/- {sid_scores.std():.4f}")
print(f"   Chance level: {1/len(np.unique(y_sid)):.4f}")
print("   (High accuracy expected after per-subject normalization)")

# ── 4. Imputation audit ──
print("\n4. IMPUTATION AUDIT")
n_nan = df[feat_cols].isna().sum().sum()
print(f"   NaN in loaded feature matrix: {n_nan}")
if n_nan == 0:
    print("   Imputation was applied GLOBALLY before LOSOCV.")
    print("   This is a minor leak: held-out subject's values influenced fill values.")
    print("   FIX: Move imputation inside the CV loop (use training-fold median only).")
else:
    print("   NaN values present — imputation not yet applied (good, do it inside CV).")

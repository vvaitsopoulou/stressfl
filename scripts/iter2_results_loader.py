# %% [markdown]
# # Iteration 2 — Results Loader Notebook
#
# This notebook loads all on-disk results from the Iteration 2 federated training
# runs and produces the chapter artefacts for §5.4 (Evaluation).
#
# **It does not retrain anything.** It only reads from disk and writes LaTeX
# tables and figures to `artefacts/iter2/`.
#
# ## Why this notebook exists
#
# Iteration 2 was trained on a vast.ai instance that was unstable, which led to
# multiple backup copies of the result folders. The actual authoritative results
# need to be sorted out before any aggregation is done. This notebook does that
# sorting **first**, then proceeds to aggregation.
#
# ## Expected result store layout (verified against project tree)
#
# - `production_downloads/fedavg/results/FedAvg_fold{2..17}.h5` (×15)
#       Each: HDF5 with FoldResult fields including `test_subject_f1` (the LOSO
#       fold metric under the held-out-with-calibration protocol).
#
# - `production_downloads/fedrep/results_coldstart_backup2/FedRep_fold{2..17}.h5`
#       Cold-start FedRep, pre-warm-start sweep.
#
# - `production_downloads/fedrep/results_warmstart/FedRep_fold{2..17}.h5`
#       Warm-start FedRep, after the calibration sweep was committed.
#       Plus `FedRep_warmstart_sweep*.json` files documenting the sweep.
#
# - `production_downloads/pefll/results/PeFLL_fold{2..17}.h5`
#       PeFLL hypernetwork-based PFL.
#
# - **AND** a top-level `stressfl/results/` folder which mixes FedAvg, FedRep
#       and warm-start sweep JSONs. This is likely a sync artefact from one of
#       the runs. We need to programmatically check whether it duplicates any
#       of the production_downloads copies or is something else.
#
# ## What this notebook does
#
# 1. **Discovery**: enumerate every .h5 and .json file in every candidate folder.
# 2. **Sanity inspection**: open every HDF5 and check which `test_subject_*` fields
#    are populated, when the file was written, and what algorithm tag it claims.
# 3. **Authority resolution**: between competing copies of the same fold,
#    apply a deterministic priority rule and produce a "manifest" of which file
#    is the authoritative source for each (algorithm, fold) pair.
# 4. **Diff alarm**: if two competing copies disagree on the LOSO F1 by more
#    than a tolerance, raise a warning so the user can investigate.
# 5. **Aggregation**: build per-fold tables and the 15-fold LOSO summary.
# 6. **Plots and LaTeX**: produce all the §5.4 figures and tables.

# %%
from datetime import datetime
import json
from pathlib import Path

# Project source — needed for FoldResult helpers
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns

sys.path.insert(0, str(Path(".").resolve()))
from src.utils.result_store import FoldResult, load_fold_result  # noqa: E402

# --- Paths -------------------------------------------------------------------
# Adjust if the notebook is run from a sub-directory.
PROJECT_ROOT = Path(".")

# Candidate folders containing .h5 results (in priority order per algorithm)
CANDIDATES = {
    "FedAvg": [
        PROJECT_ROOT / "production_downloads" / "fedavg" / "results",
        PROJECT_ROOT / "results",  # top-level — may or may not be a duplicate
    ],
    "FedRep_cold": [
        PROJECT_ROOT / "production_downloads" / "fedrep" / "results_coldstart_backup2",
    ],
    "FedRep_warm": [
        PROJECT_ROOT / "production_downloads" / "fedrep" / "results_warmstart",
        PROJECT_ROOT / "results",  # also contains warm-start sweep JSONs
    ],
    "PeFLL": [
        PROJECT_ROOT / "production_downloads" / "pefll" / "results",
    ],
}

# Iteration 1 baselines (read-only; not modified here)
ITER1_WRIST_LOSO = (
    PROJECT_ROOT
    / "production_downloads"
    / "fedrep"
    / "results"
    / "dl"
    / "wrist_baseline"
    / "loso_results.json"
)
ITER1_WRIST_PER_FOLD = (
    PROJECT_ROOT
    / "production_downloads"
    / "fedrep"
    / "results"
    / "dl"
    / "wrist_baseline"
    / "_fold_results"
)

# Output destination — chapter artefacts
ARTEFACTS = PROJECT_ROOT / "artefacts" / "iter2"
TABLES_DIR = ARTEFACTS / "tables"
FIGURES_DIR = ARTEFACTS / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Known subject IDs (matches Iteration 1 — WESAD skips S12)
ALL_SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Display options
pd.set_option("display.precision", 4)
pd.set_option("display.width", 200)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
sns.set_style("whitegrid")

print(f"PROJECT_ROOT : {PROJECT_ROOT.resolve()}")
print(f"ARTEFACTS    : {ARTEFACTS.resolve()}")
print()

# %% [markdown]
# ## Section 1: Discovery
#
# Walk every candidate folder, list every .h5 file present, and record its
# mtime + size. This is purely informational and does not open any HDF5 yet.


# %%
def discover(candidates: dict) -> pd.DataFrame:
    """Return a DataFrame with one row per (key, fold, file)."""
    rows = []
    for key, folders in candidates.items():
        for folder in folders:
            if not folder.exists():
                rows.append(
                    {
                        "key": key,
                        "folder": str(folder),
                        "fold": None,
                        "filename": None,
                        "exists": False,
                        "size_kb": None,
                        "mtime": None,
                    }
                )
                continue
            for f in sorted(folder.glob("*.h5")):
                # Extract fold number from filename
                stem = f.stem  # e.g. "FedAvg_fold10"
                try:
                    fold = int(stem.split("_fold")[1])
                except (IndexError, ValueError):
                    fold = None
                rows.append(
                    {
                        "key": key,
                        "folder": str(folder),
                        "fold": fold,
                        "filename": f.name,
                        "exists": True,
                        "size_kb": round(f.stat().st_size / 1024, 1),
                        "mtime": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                )
    return pd.DataFrame(rows)


discovery_df = discover(CANDIDATES)
print("=" * 100)
print("DISCOVERY: all candidate files")
print("=" * 100)
print(discovery_df.to_string(index=False))

# Per-key fold count summary
print("\n" + "=" * 60)
print("PER-KEY FOLD INVENTORY")
print("=" * 60)
for key in CANDIDATES:
    sub = discovery_df[(discovery_df["key"] == key) & discovery_df["exists"]]
    folders = sub["folder"].unique()
    for folder in folders:
        folds = sorted(
            sub[sub["folder"] == folder]["fold"].dropna().astype(int).tolist()
        )
        missing = sorted(set(ALL_SUBJECT_IDS) - set(folds))
        print(f"  [{key:14s}] {folder}")
        print(f"    {len(folds):>2d} folds present, missing: {missing}")

# %% [markdown]
# ## Section 2: Sanity inspection
#
# Open every HDF5 file, inspect its metadata, and record:
# - Algorithm tag (from attrs)
# - Fold ID claimed
# - Test subject claimed (sanity check it matches the filename)
# - Whether `test_subject_f1` is populated (i.e. uses the new
#   held-out-with-calibration protocol — sentinel -1.0 means it does not)
# - Number of evaluation rounds (n_rounds = 21 expected for 100 rounds / eval_gap=5)
# - The actual LOSO macro-F1 (`test_subject_f1`)


# %%
def inspect_h5(path: Path) -> dict:
    """Open one HDF5 file and extract diagnostic info."""
    try:
        with h5py.File(path, "r") as hf:
            group_names = list(hf.keys())
            if not group_names:
                return {"error": "no groups"}
            grp = hf[group_names[0]]
            attrs = dict(grp.attrs)
            n_rounds = grp["rs_test_f1"].shape[0] if "rs_test_f1" in grp else 0

            # Use FoldResult-style sentinel check
            test_subj_f1 = (
                float(np.asarray(grp["test_subject_f1"]).item())
                if "test_subject_f1" in grp
                else -1.0
            )
            test_subj_acc = (
                float(np.asarray(grp["test_subject_acc"]).item())
                if "test_subject_acc" in grp
                else -1.0
            )
            test_subj_recall = (
                float(np.asarray(grp["test_subject_stress_recall"]).item())
                if "test_subject_stress_recall" in grp
                else -1.0
            )
            test_subj_n_samples = int(attrs.get("test_subject_n_samples", 0))

            return {
                "group": group_names[0],
                "algorithm_attr": str(attrs.get("algorithm", "?")),
                "fold_id_attr": int(attrs.get("fold_id", -1)),
                "test_subject_attr": int(attrs.get("test_subject", -1)),
                "val_subject_attr": int(attrs.get("val_subject", -1)),
                "n_rounds": int(n_rounds),
                "test_subject_f1": test_subj_f1,
                "test_subject_acc": test_subj_acc,
                "test_subject_stress_recall": test_subj_recall,
                "test_subject_n_samples": test_subj_n_samples,
                "has_loso_metric": (test_subj_n_samples > 0) and (test_subj_f1 >= 0.0),
                "error": None,
            }
    except Exception as e:
        return {"error": str(e)}


inspect_rows = []
for _, row in discovery_df[discovery_df["exists"]].iterrows():
    full_path = Path(row["folder"]) / row["filename"]
    info = inspect_h5(full_path)
    inspect_rows.append(
        {
            **row.to_dict(),
            **info,
        }
    )

inspect_df = pd.DataFrame(inspect_rows)

# Sanity flags
inspect_df["filename_fold_matches_attr"] = (
    inspect_df["fold"] == inspect_df["fold_id_attr"]
)

print("=" * 100)
print("HDF5 INSPECTION RESULTS")
print("=" * 100)
display_cols = [
    "key",
    "folder",
    "fold",
    "algorithm_attr",
    "n_rounds",
    "test_subject_f1",
    "has_loso_metric",
    "filename_fold_matches_attr",
    "mtime",
]
print(inspect_df[display_cols].to_string(index=False))

# Save the inspection table for audit
inspect_df.to_csv(ARTEFACTS / "inspection_audit.csv", index=False)
print(f"\nSaved inspection audit → {ARTEFACTS / 'inspection_audit.csv'}")

# %% [markdown]
# ## Section 3: Cross-check between competing copies
#
# For each (algorithm, fold) pair that exists in **two or more** folders,
# compare the `test_subject_f1`. If they disagree by more than `TOL`, raise
# a warning. This is the safety check against silently using stale results.

# %%
TOL = 1e-4  # F1 tolerance below which two values are considered identical

# Pivot: for each (key, fold), get list of (folder, F1) pairs
print("=" * 100)
print("CROSS-CHECK: competing copies for the same (key, fold)")
print("=" * 100)
conflict_count = 0
for key in CANDIDATES:
    sub = inspect_df[(inspect_df["key"] == key) & inspect_df["exists"]]
    for fold in sorted(sub["fold"].dropna().unique()):
        cf = sub[sub["fold"] == fold]
        if len(cf) <= 1:
            continue
        # Multiple copies
        vals = list(zip(cf["folder"], cf["test_subject_f1"], cf["has_loso_metric"]))
        f1s = [v[1] for v in vals if v[2]]  # only count copies with LOSO metric
        if len(f1s) < 2:
            continue
        if max(f1s) - min(f1s) > TOL:
            conflict_count += 1
            print(f"\n  CONFLICT [{key} fold {int(fold)}]:")
            for folder, f1, has_loso in vals:
                print(f"    {folder:80s}  F1={f1:.6f}  has_loso={has_loso}")

if conflict_count == 0:
    print("\nNo cross-folder conflicts — the duplicates agree on F1 within tolerance.")
else:
    print(f"\n{conflict_count} conflict(s) found. Audit before proceeding.")

# %% [markdown]
# ## Section 4: Authority resolution
#
# Apply a deterministic priority rule to pick the authoritative file for each
# (key, fold) pair. Priority is the order in which folders appear in the
# CANDIDATES dictionary. Within priority, we additionally require:
#
#   1. The file has the new-protocol LOSO metric (`has_loso_metric == True`).
#   2. The fold ID in the file's attributes matches the filename.
#   3. The number of rounds is reasonable (>= 21 expected).
#
# Files failing these criteria are demoted in favour of the next-priority copy.


# %%
def resolve_authority(inspect_df: pd.DataFrame, candidates: dict) -> pd.DataFrame:
    """Return a DataFrame with one row per (key, fold) — the authoritative pick."""
    rows = []
    for key, folder_list in candidates.items():
        folders_ordered = [str(f) for f in folder_list]
        for fold in ALL_SUBJECT_IDS:
            # All copies for this (key, fold)
            sub = inspect_df[
                (inspect_df["key"] == key)
                & (inspect_df["fold"] == fold)
                & inspect_df["exists"]
                & (inspect_df["error"].isna())
            ].copy()
            if sub.empty:
                rows.append(
                    {
                        "key": key,
                        "fold": fold,
                        "status": "MISSING",
                        "chosen_folder": None,
                        "chosen_path": None,
                        "test_subject_f1": None,
                        "reason": "no file in any candidate folder",
                    }
                )
                continue

            # Prefer copies that meet quality criteria, in priority order
            def priority_key(row, folders_ordered=folders_ordered):
                """Lower is better."""
                folder = row["folder"]
                try:
                    folder_priority = folders_ordered.index(folder)
                except ValueError:
                    folder_priority = 999
                meets_loso = 0 if row["has_loso_metric"] else 1
                meets_rounds = 0 if row["n_rounds"] >= 21 else 1
                meets_match = 0 if row["filename_fold_matches_attr"] else 1
                # Folder priority dominates, then LOSO metric, then rounds, then match
                return (folder_priority, meets_loso, meets_rounds, meets_match)

            sub = sub.assign(_priority=sub.apply(priority_key, axis=1))
            sub = sub.sort_values("_priority")
            chosen = sub.iloc[0]

            reason_parts = [
                f"folder priority {sub.iloc[0]['_priority'][0]}",
                f"has_loso={chosen['has_loso_metric']}",
                f"rounds={chosen['n_rounds']}",
            ]
            rows.append(
                {
                    "key": key,
                    "fold": int(fold),
                    "status": "OK" if chosen["has_loso_metric"] else "WARNING_NO_LOSO",
                    "chosen_folder": chosen["folder"],
                    "chosen_path": str(Path(chosen["folder"]) / chosen["filename"]),
                    "test_subject_f1": chosen["test_subject_f1"],
                    "test_subject_acc": chosen["test_subject_acc"],
                    "test_subject_stress_recall": chosen["test_subject_stress_recall"],
                    "n_rounds": chosen["n_rounds"],
                    "mtime": chosen["mtime"],
                    "reason": " | ".join(reason_parts),
                }
            )
    return pd.DataFrame(rows)


authority_df = resolve_authority(inspect_df, CANDIDATES)
print("=" * 100)
print("AUTHORITY RESOLUTION")
print("=" * 100)
display_cols = [
    "key",
    "fold",
    "status",
    "test_subject_f1",
    "test_subject_stress_recall",
    "chosen_folder",
    "mtime",
]
print(authority_df[display_cols].to_string(index=False))

# Save manifest
authority_df.to_csv(ARTEFACTS / "manifest.csv", index=False)
print(f"\nManifest saved → {ARTEFACTS / 'manifest.csv'}")

# Final completeness check per algorithm
print("\n" + "=" * 60)
print("FINAL INVENTORY (after authority resolution)")
print("=" * 60)
for key in CANDIDATES:
    sub = authority_df[authority_df["key"] == key]
    ok = sub[sub["status"] == "OK"]
    warn = sub[sub["status"].str.startswith("WARNING")]
    miss = sub[sub["status"] == "MISSING"]
    print(
        f"  [{key:14s}]  OK: {len(ok):>2d}/15  Warning: {len(warn):>2d}  Missing: {len(miss):>2d}"
    )
    if not warn.empty:
        for _, r in warn.iterrows():
            print(f"     warning: fold {r['fold']} — {r['reason']}")
    if not miss.empty:
        for _, r in miss.iterrows():
            print(f"     missing: fold {r['fold']}")

# %% [markdown]
# ## Section 5: Load all authoritative results into FoldResult dataclasses
#
# Using the manifest, load each chosen .h5 file via `load_fold_result` and
# store in a dict keyed by (algorithm_key, fold_id).

# %%
authoritative_results: dict[tuple[str, int], FoldResult] = {}

for _, row in authority_df.iterrows():
    if row["status"] != "OK":
        continue
    key = row["key"]
    fold = int(row["fold"])
    path = Path(row["chosen_path"])
    try:
        fr = load_fold_result(str(path))
        authoritative_results[(key, fold)] = fr
    except Exception as e:
        print(f"  FAILED to load {path}: {e}")

print(f"\nLoaded {len(authoritative_results)} authoritative FoldResults.")
print("  Folds per algorithm:")
for key in CANDIDATES:
    n = sum(1 for k in authoritative_results if k[0] == key)
    print(f"    {key:14s}  {n:>2d}/15")

# %% [markdown]
# ## Section 6: Per-fold dataframes
#
# Build a long-format DataFrame: one row per (algorithm_key, fold) with the
# LOSO macro-F1, accuracy, stress recall and stress-class F1.

# %%
per_fold_rows = []
for (key, fold), fr in authoritative_results.items():
    per_fold_rows.append(
        {
            "method": key,
            "fold": fold,
            "test_subject": int(fr.test_subject),
            "val_subject": int(fr.val_subject),
            "macro_f1": fr.test_subject_f1,
            "accuracy": fr.test_subject_acc,
            "stress_recall": fr.test_subject_stress_recall,
            "stress_f1": fr.test_subject_stress_f1,
            "n_samples": fr.test_subject_n_samples,
        }
    )

per_fold_df = pd.DataFrame(per_fold_rows).sort_values(["method", "fold"])
print("=" * 100)
print("PER-FOLD LOSO RESULTS (all methods)")
print("=" * 100)
print(per_fold_df.to_string(index=False))

per_fold_df.to_csv(TABLES_DIR / "per_fold_results.csv", index=False)
print(f"\nSaved → {TABLES_DIR / 'per_fold_results.csv'}")

# %% [markdown]
# ## Section 7: Aggregate summary across methods
#
# Mean ± std across 15 folds, plus worst-fold F1 (Mohri agnostic-FL fairness
# indicator).


# %%
def summarise(group: pd.DataFrame) -> dict:
    return {
        "n_folds": len(group),
        "macro_f1_mean": group["macro_f1"].mean(),
        "macro_f1_std": group["macro_f1"].std(ddof=1),
        "macro_f1_min": group["macro_f1"].min(),  # worst-fold F1
        "accuracy_mean": group["accuracy"].mean(),
        "accuracy_std": group["accuracy"].std(ddof=1),
        "stress_recall_mean": group["stress_recall"].mean(),
        "stress_recall_std": group["stress_recall"].std(ddof=1),
        "stress_f1_mean": group["stress_f1"].mean(),
        "stress_f1_std": group["stress_f1"].std(ddof=1),
        # Subjects S9, S13, S14 — the centralised failure folds
        "S9_f1": (
            group.loc[group["fold"] == 9, "macro_f1"].squeeze()
            if (group["fold"] == 9).any()
            else np.nan
        ),
        "S13_f1": (
            group.loc[group["fold"] == 13, "macro_f1"].squeeze()
            if (group["fold"] == 13).any()
            else np.nan
        ),
        "S14_f1": (
            group.loc[group["fold"] == 14, "macro_f1"].squeeze()
            if (group["fold"] == 14).any()
            else np.nan
        ),
    }


agg_rows = []
for key in CANDIDATES:
    sub = per_fold_df[per_fold_df["method"] == key]
    if sub.empty:
        continue
    agg_rows.append({"method": key, **summarise(sub)})

# Add the Iter 1 centralised reference (read from disk)
if ITER1_WRIST_LOSO.exists():
    iter1_data = json.loads(ITER1_WRIST_LOSO.read_text())
    iter1_agg = iter1_data.get("aggregate", {}) or iter1_data.get("summary", {})
    agg_rows.insert(
        0,
        {
            "method": "Centralised (Iter 1)",
            "n_folds": 15,
            "macro_f1_mean": iter1_agg.get("f1_macro_mean", 0.893),
            "macro_f1_std": iter1_agg.get("f1_macro_std", 0.069),
            "macro_f1_min": iter1_agg.get("f1_macro_min", np.nan),
            "accuracy_mean": iter1_agg.get("accuracy_mean", 0.908),
            "accuracy_std": iter1_agg.get("accuracy_std", 0.060),
            "stress_recall_mean": iter1_agg.get("recall_mean", np.nan),
            "stress_recall_std": iter1_agg.get("recall_std", np.nan),
            "stress_f1_mean": np.nan,
            "stress_f1_std": np.nan,
            "S9_f1": 0.80,  # from §4.4.4 (Iter 1 eval)
            "S13_f1": 0.702,  # likewise
            "S14_f1": 0.653,  # likewise
        },
    )
else:
    print(f"NOTE: Iter 1 LOSO file not found at {ITER1_WRIST_LOSO}.")
    print("  Hardcoding the values from the Iter 1 chapter for the reference row.")
    agg_rows.insert(
        0,
        {
            "method": "Centralised (Iter 1)",
            "n_folds": 15,
            "macro_f1_mean": 0.893,
            "macro_f1_std": 0.069,
            "macro_f1_min": 0.80,
            "accuracy_mean": 0.908,
            "accuracy_std": 0.060,
            "stress_recall_mean": np.nan,
            "stress_recall_std": np.nan,
            "stress_f1_mean": np.nan,
            "stress_f1_std": np.nan,
            "S9_f1": 0.80,
            "S13_f1": 0.702,
            "S14_f1": 0.653,
        },
    )

agg_df = pd.DataFrame(agg_rows)
print("=" * 100)
print("AGGREGATE SUMMARY")
print("=" * 100)
print(agg_df.to_string(index=False))
agg_df.to_csv(TABLES_DIR / "aggregate_summary.csv", index=False)


def fmt_meanstd(m, s, n=3):
    if pd.isna(m) or pd.isna(s):
        return "—"
    return f"{m:.{n}f} ± {s:.{n}f}"


# Build the LaTeX-ready presentation table
display_df = pd.DataFrame(
    {
        "Method": agg_df["method"],
        "Macro-F1": agg_df.apply(
            lambda r: fmt_meanstd(r["macro_f1_mean"], r["macro_f1_std"]), axis=1
        ),
        "Worst F1": agg_df["macro_f1_min"].apply(
            lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
        ),
        "Accuracy": agg_df.apply(
            lambda r: fmt_meanstd(r["accuracy_mean"], r["accuracy_std"]), axis=1
        ),
        "Stress recall": agg_df.apply(
            lambda r: fmt_meanstd(r["stress_recall_mean"], r["stress_recall_std"]),
            axis=1,
        ),
    }
)

print("\n" + "=" * 100)
print("CHAPTER TABLE: cross-method LOSO summary")
print("=" * 100)
print(display_df.to_string(index=False))

display_df.to_csv(TABLES_DIR / "cross_method_summary.csv", index=False)
display_df.to_latex(
    TABLES_DIR / "cross_method_summary.tex",
    index=False,
    escape=False,
    caption="Cross-method LOSO summary on 15-fold WESAD evaluation. "
    "Worst F1 is the minimum across the 15 folds "
    "(Mohri agnostic-FL fairness indicator).",
    label="tab:iter2-loso-summary",
)

# %% [markdown]
# ## Section 8: Per-subject grouped bar chart
#
# 15 subjects × N methods. Reference line at the Iter 1 centralised
# per-fold F1 mean.

# %%
fig, ax = plt.subplots(figsize=(14, 6))
methods_for_plot = [
    m
    for m in ["FedAvg", "FedRep_cold", "FedRep_warm", "PeFLL"]
    if m in per_fold_df["method"].values
]
n_methods = len(methods_for_plot)
bar_width = 0.8 / max(n_methods, 1)
colors = {
    "FedAvg": "#9E9E9E",
    "FedRep_cold": "#FF9800",
    "FedRep_warm": "#4CAF50",
    "PeFLL": "#2196F3",
}

x = np.arange(len(ALL_SUBJECT_IDS))
for i, method in enumerate(methods_for_plot):
    sub = (
        per_fold_df[per_fold_df["method"] == method]
        .set_index("fold")
        .reindex(ALL_SUBJECT_IDS)
    )
    ax.bar(
        x + i * bar_width,
        sub["macro_f1"],
        bar_width,
        label=method,
        color=colors.get(method, "#888"),
        edgecolor="black",
        linewidth=0.3,
    )

# Reference line — Iter 1 mean
ax.axhline(
    y=0.893,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.6,
    label="Iter 1 centralised mean (0.893)",
)
# Highlight S9, S13, S14
for sid in [9, 13, 14]:
    idx = ALL_SUBJECT_IDS.index(sid)
    ax.axvspan(
        idx - 0.2, idx + 0.2 + (n_methods - 1) * bar_width, alpha=0.1, color="red"
    )

ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
ax.set_xticklabels([f"S{s}" for s in ALL_SUBJECT_IDS])
ax.set_ylabel("Macro-F1 (LOSO held-out evaluation)")
ax.set_title(
    "Iteration 2: per-subject macro-F1 across methods\n"
    "(red highlight: S9 / S13 / S14 — Iter 1 centralised failure folds)"
)
ax.legend(loc="lower left", framealpha=0.95)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "per_subject_grouped.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "per_subject_grouped.png", bbox_inches="tight")
plt.show()
print(f"\nSaved → {FIGURES_DIR / 'per_subject_grouped.pdf'}")

# %% [markdown]
# ## Section 9: Convergence curves (development fold S2)
#
# Macro-F1 against federation round on the dev fold S2, one curve per method.

# %%
if any((m, 2) in authoritative_results for m in ["FedAvg", "FedRep_warm", "PeFLL"]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods_for_plot:
        if (method, 2) not in authoritative_results:
            continue
        fr = authoritative_results[(method, 2)]
        rounds = np.arange(len(fr.rs_test_f1)) * 5  # eval_gap = 5
        ax.plot(
            rounds,
            fr.rs_test_f1,
            marker="o",
            markersize=4,
            label=method,
            color=colors.get(method, "#888"),
            linewidth=1.4,
        )

    ax.axhline(
        y=0.893,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Iter 1 centralised mean",
    )
    ax.set_xlabel("Federation round")
    ax.set_ylabel("Macro-F1 on dev fold S2 (training-client mean)")
    ax.set_title("Iteration 2: convergence curves on the development fold (S2)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "convergence_S2.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "convergence_S2.png", bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {FIGURES_DIR / 'convergence_S2.pdf'}")

# %% [markdown]
# ## Section 10: Statistical testing — Wilcoxon signed-rank
#
# Paired comparisons between methods on the 15 LOSO folds:
#   - FedRep_warm vs FedAvg
#   - FedRep_warm vs FedRep_cold (warm-start contribution)
#   - PeFLL vs FedAvg
#   - PeFLL vs FedRep_warm

# %%


def paired_wilcoxon(a_key: str, b_key: str) -> dict:
    """Wilcoxon signed-rank test between two methods on paired folds."""
    a = per_fold_df[per_fold_df["method"] == a_key].set_index("fold")["macro_f1"]
    b = per_fold_df[per_fold_df["method"] == b_key].set_index("fold")["macro_f1"]
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return {"a": a_key, "b": b_key, "n": len(common), "stat": None, "p": None}
    a_vals = a.loc[common].values
    b_vals = b.loc[common].values
    try:
        stat, p = wilcoxon(
            a_vals, b_vals, zero_method="wilcox", alternative="two-sided"
        )
    except ValueError as e:
        return {
            "a": a_key,
            "b": b_key,
            "n": len(common),
            "stat": None,
            "p": None,
            "note": str(e),
        }
    diff = a_vals - b_vals
    return {
        "a": a_key,
        "b": b_key,
        "n": len(common),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "a_better_in": int(np.sum(diff > 0)),
        "b_better_in": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }


comparisons = [
    ("FedRep_warm", "FedAvg"),
    ("FedRep_warm", "FedRep_cold"),
    ("PeFLL", "FedAvg"),
    ("PeFLL", "FedRep_warm"),
]
wilcoxon_rows = []
for a, b in comparisons:
    if a in per_fold_df["method"].values and b in per_fold_df["method"].values:
        wilcoxon_rows.append(paired_wilcoxon(a, b))

wilcoxon_df = pd.DataFrame(wilcoxon_rows)
print("=" * 100)
print("WILCOXON SIGNED-RANK TESTS (paired, two-sided)")
print("=" * 100)
print(wilcoxon_df.to_string(index=False))
wilcoxon_df.to_csv(TABLES_DIR / "wilcoxon.csv", index=False)

# %% [markdown]
# ## Section 11: Warm-start vs cold-start calibration sweep (dev fold S2)
#
# Loads the sweep JSON files (multiple snapshots — pick the most authoritative).
# These contain the calibration hyperparameter sweep results that motivated
# the warm-start configuration committed in §5.2.5.

# %%
sweep_candidates = [
    PROJECT_ROOT / "results" / "FedRep_warmstart_sweep_final15.json",
    PROJECT_ROOT / "results" / "FedRep_warmstart_sweep.json",
    PROJECT_ROOT
    / "production_downloads"
    / "fedrep"
    / "results_warmstart"
    / "FedRep_warmstart_sweep_final15.json",
    PROJECT_ROOT
    / "production_downloads"
    / "fedrep"
    / "results_warmstart"
    / "FedRep_warmstart_sweep.json",
]
sweep_data = None
sweep_source = None
for p in sweep_candidates:
    if p.exists():
        try:
            sweep_data = json.loads(p.read_text())
            sweep_source = p
            print(f"Loaded sweep from: {p}")
            break
        except Exception as e:
            print(f"Could not load {p}: {e}")

if sweep_data is None:
    print("No warm-start sweep JSON found in any candidate location.")
else:
    # Sweep schemas vary; print a sample of the structure for inspection
    print(
        f"Top-level keys: {list(sweep_data.keys()) if isinstance(sweep_data, dict) else type(sweep_data)}"
    )
    # If it's a list of configurations:
    if isinstance(sweep_data, list):
        print(f"  List of {len(sweep_data)} configurations.")
        print(
            "  Sample config:", json.dumps(sweep_data[0], indent=2, default=str)[:500]
        )
    elif isinstance(sweep_data, dict):
        # Try common patterns
        for k in list(sweep_data.keys())[:3]:
            v = sweep_data[k]
            print(f"  {k}: {str(v)[:200]}")

# %% [markdown]
# ## Section 12: PeFLL unseen-client deployment (separate experiment)
#
# This experiment is mentioned in the chapter but lives in a separate run.
# Check whether the results file exists; if so, load and report.

# %%
pefll_unseen_candidates = [
    PROJECT_ROOT
    / "production_downloads"
    / "pefll"
    / "results"
    / "PeFLL_unseen_summary.json",
    PROJECT_ROOT / "results" / "PeFLL_unseen_summary.json",
]
pefll_unseen = None
for p in pefll_unseen_candidates:
    if p.exists():
        pefll_unseen = json.loads(p.read_text())
        print(f"Loaded PeFLL unseen-client results from: {p}")
        print(json.dumps(pefll_unseen, indent=2, default=str)[:1500])
        break

if pefll_unseen is None:
    print("PeFLL unseen-client deployment results not found.")
    print(
        "If this experiment was not run, the corresponding figure (Figure DD) "
        "in §5.4 should be either removed from the planning list or generated "
        "from the standard PeFLL run by selecting three held-out subjects post-hoc."
    )

# %% [markdown]
# ## Section 13: Manifest summary
#
# Final manifest of artefacts written by this notebook.

# %%
print("\n" + "=" * 80)
print("ARTEFACTS PRODUCED")
print("=" * 80)
for category, root in [
    ("Tables", TABLES_DIR),
    ("Figures", FIGURES_DIR),
    ("Audit", ARTEFACTS),
]:
    print(f"\n{category}:")
    if not root.exists():
        continue
    files = sorted([p for p in root.iterdir() if p.is_file()])
    for p in files:
        size = p.stat().st_size
        print(f"  {p.name:50s} {size:>10,} bytes")

print("\nDone. Inspect inspection_audit.csv and manifest.csv before trusting numbers.")

# %% [markdown]
# # Iteration 1 — Results Loader Notebook
#
# This notebook loads all on-disk results from the Iteration 1 training runs
# and produces the chapter artefacts for §4.4 (Evaluation).
#
# **It does not retrain anything.** It only reads from `results/` and writes
# LaTeX tables and PNG figures to `artefacts/iter1/`.
#
# ## Result store layout (verified against the demo notebook)
#
# - `results/binary/{chest,wrist}_fs_ablation/fs_{none,tree,kbest,pca}/`
#       Classical RF, BINARY label scheme, 4 feature-selection methods per device.
#       Files: `losocv_results.pkl`, `losocv_summary.csv`, `model_S*.pkl` per fold.
#
# - `results/binary/norm_ablation/{chest,wrist}/unnormalized/`
#       Classical RF, BINARY, no per-subject baseline normalisation, FS=tree.
#       (The normalised counterpart is `*_fs_ablation/fs_tree/` — same model, normalised.)
#
# - `results/binary/diagnostic_ablation/{chest,wrist}/unnorm_indep_only_{BINARY,3CLASS}/`
#       Classical RF, no normalisation AND independent windows only, FS=tree,
#       both binary and 3-class label schemes.
#
# - `results/dl/wrist_baseline/`
#       CNN-LSTM wrist (carry-forward), BINARY. Contains `loso_results.json`,
#       `evaluation_report.json` (full report with Kappa, MCC, variability),
#       and pre-generated figures.
#
# - `results/dl/ablation_{window,cnn_layers,dropout,filters,optimizer,batch_size}/`
#       CNN-LSTM ablations on the wrist configuration.
#
# **Chest baseline was not run to completion** (only fold_S2 exists in
# `results/dl/chest_baseline/`). The dissertation deliberately excludes it on
# the grounds that the literature consistently documents only a small
# chest-versus-wrist gap and the wrist is the carry-forward artefact.

# %%
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(".")
RESULTS_BASE = PROJECT_ROOT / "results"
ARTEFACTS = PROJECT_ROOT / "artefacts" / "iter1"
ARTEFACTS.mkdir(parents=True, exist_ok=True)

# Subdirectories for chapter artefacts
TABLES_DIR = ARTEFACTS / "tables"
FIGURES_DIR = ARTEFACTS / "figures"
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Reading from : {RESULTS_BASE.resolve()}")
print(f"Writing to   : {ARTEFACTS.resolve()}")

# --- Display options ---------------------------------------------------------
pd.set_option("display.precision", 4)
pd.set_option("display.width", 200)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
sns.set_style("whitegrid")


# %% [markdown]
# ## Section 1: Helpers
#
# Small utilities for loading the various result formats and detecting label scheme.


# %%
def load_pkl(path: Path) -> dict:
    """Load a classical-pipeline losocv_results.pkl.
    Expected schema: {'aggregate': {...}, 'fold_results': [...]}
    """
    with open(path, "rb") as f:
        data: dict = pickle.load(f)
    return data


def load_json(path: Path) -> dict:
    """Load a DL-pipeline JSON result file."""
    with open(path) as f:
        data: dict = json.load(f)
    return data


def detect_label_scheme(result: dict) -> str:
    """Detect 'binary' or '3class' from the first fold's confusion matrix shape."""
    fold_results = result.get("fold_results") or result.get("per_fold") or []
    if not fold_results:
        return "unknown"
    cm = np.array(fold_results[0]["confusion_matrix"])
    return (
        "binary"
        if cm.shape == (2, 2)
        else "3class" if cm.shape == (3, 3) else f"unknown_{cm.shape}"
    )


def per_fold_dataframe(result: dict, source_label: str = "") -> pd.DataFrame:
    """Flatten per-fold results into a tidy DataFrame."""
    fold_results = result.get("fold_results") or result.get("per_fold") or []
    rows = []
    for f in fold_results:
        rows.append(
            {
                "source": source_label,
                "subject": f.get("subject") or f.get("test_sid"),
                "accuracy": f.get("accuracy"),
                "f1_macro": f.get("f1_macro"),
                "f1_weighted": f.get("f1_weighted"),
                "precision": f.get("precision"),
                "recall": f.get("recall"),
            }
        )
    return pd.DataFrame(rows)


def fmt_meanstd(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# %% [markdown]
# ## Section 2: Classical RF — Feature-Selection Ablation
#
# Loads the 8 RF configurations (chest+wrist × {none, tree, kbest, pca}).
# All BINARY label scheme (verified from notebook line 634).

# %%
FS_METHODS = ["none", "tree", "kbest", "pca"]

fs_results = {}  # (device, method) -> result dict
for device in ["chest", "wrist"]:
    for method in FS_METHODS:
        path = (
            RESULTS_BASE
            / "binary"
            / f"{device}_fs_ablation"
            / f"fs_{method}"
            / "losocv_results.pkl"
        )
        if path.exists():
            res = load_pkl(path)
            scheme = detect_label_scheme(res)
            fs_results[(device, method)] = res
            print(
                f"  [{device:5s} fs_{method:5s}] scheme={scheme}  "
                f"F1_macro={res['aggregate']['f1_macro_mean']:.4f} ± "
                f"{res['aggregate']['f1_macro_std']:.4f}"
            )
        else:
            print(f"  [{device:5s} fs_{method:5s}] MISSING: {path}")


# %% [markdown]
# ### 2.1 FS-ablation summary table

# %%
fs_rows = []
for (device, method), res in fs_results.items():
    agg = res["aggregate"]
    fs_rows.append(
        {
            "Device": device.capitalize(),
            "FS method": method,
            "Accuracy": fmt_meanstd(agg["accuracy_mean"], agg["accuracy_std"]),
            "F1 weighted": fmt_meanstd(agg["f1_weighted_mean"], agg["f1_weighted_std"]),
            "F1 macro": fmt_meanstd(agg["f1_macro_mean"], agg["f1_macro_std"]),
            "_f1_macro_mean": agg["f1_macro_mean"],  # for sorting
        }
    )

fs_df = pd.DataFrame(fs_rows).sort_values(
    ["Device", "_f1_macro_mean"], ascending=[True, False]
)
display_cols = ["Device", "FS method", "Accuracy", "F1 weighted", "F1 macro"]
print("\nFEATURE-SELECTION ABLATION (BINARY, all chest+wrist features)")
print("=" * 80)
print(fs_df[display_cols].to_string(index=False))

# Save as CSV and LaTeX
fs_df[display_cols].to_csv(TABLES_DIR / "fs_ablation.csv", index=False)
fs_df[display_cols].to_latex(TABLES_DIR / "fs_ablation.tex", index=False, escape=False)
print(f"\n  → {TABLES_DIR / 'fs_ablation.tex'}")


# %% [markdown]
# ### 2.2 FS-ablation bar chart

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
colors = {"none": "#9E9E9E", "tree": "#4CAF50", "kbest": "#2196F3", "pca": "#FF9800"}

for ax, device in zip(axes, ["chest", "wrist"]):
    means = [fs_results[(device, m)]["aggregate"]["f1_macro_mean"] for m in FS_METHODS]
    stds = [fs_results[(device, m)]["aggregate"]["f1_macro_std"] for m in FS_METHODS]
    bars = ax.bar(
        FS_METHODS,
        means,
        yerr=stds,
        capsize=4,
        color=[colors[m] for m in FS_METHODS],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title(f"{device.capitalize()} — RF FS ablation (binary, LOSO)")
    ax.set_ylabel("F1 macro (mean ± std across 15 LOSO folds)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature-selection method")
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fs_ablation.png", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fs_ablation.pdf", bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Section 3: Classical RF — Normalisation Ablation
#
# Compares per-subject baseline normalisation (the original demo design;
# loaded from `*_fs_ablation/fs_tree/` since the normalised baseline IS that
# tree configuration) against the unnormalised counterpart.

# %%
norm_results = {}
for device in ["chest", "wrist"]:
    # Normalised baseline = the fs_tree result (chosen because tree is FS-best)
    norm_results[(device, "normalised")] = fs_results[(device, "tree")]
    # Unnormalised version
    path = (
        RESULTS_BASE
        / "binary"
        / "norm_ablation"
        / device
        / "unnormalized"
        / "losocv_results.pkl"
    )
    if path.exists():
        norm_results[(device, "unnormalised")] = load_pkl(path)
        scheme = detect_label_scheme(norm_results[(device, "unnormalised")])
        agg = norm_results[(device, "unnormalised")]["aggregate"]
        print(
            f"  [{device:5s} unnormalised] scheme={scheme}  "
            f"F1_macro={agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}"
        )
    else:
        print(f"  [{device:5s} unnormalised] MISSING: {path}")

# Build comparison table with delta
norm_rows = []
for device in ["chest", "wrist"]:
    n_agg = norm_results[(device, "normalised")]["aggregate"]
    u_agg = norm_results[(device, "unnormalised")]["aggregate"]
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("f1_weighted", "F1 weighted"),
        ("f1_macro", "F1 macro"),
    ]:
        delta = n_agg[f"{metric}_mean"] - u_agg[f"{metric}_mean"]
        norm_rows.append(
            {
                "Device": device.capitalize(),
                "Metric": label,
                "Normalised": fmt_meanstd(
                    n_agg[f"{metric}_mean"], n_agg[f"{metric}_std"]
                ),
                "Unnormalised": fmt_meanstd(
                    u_agg[f"{metric}_mean"], u_agg[f"{metric}_std"]
                ),
                "Δ": f"{delta:+.4f}",
            }
        )

norm_df = pd.DataFrame(norm_rows)
print("\nNORMALISATION ABLATION")
print("=" * 80)
print(norm_df.to_string(index=False))
norm_df.to_csv(TABLES_DIR / "norm_ablation.csv", index=False)
norm_df.to_latex(TABLES_DIR / "norm_ablation.tex", index=False, escape=False)


# %% [markdown]
# ## Section 4: Classical RF — Diagnostic Ablation
#
# Loads ALL diagnostic configurations found on disk. Each is identified by
# its label scheme (binary 2×2 or 3-class 3×3) from its confusion matrices,
# regardless of the directory name. The expected directories are:
#
# - `binary/diagnostic_ablation/{chest,wrist}/unnorm_indep_only_BINARY/`
# - `binary/diagnostic_ablation/{chest,wrist}/unnorm_indep_only_3CLASS/`
# - `binary/diagnostic_ablation/{chest,wrist}/unnorm_indep_only/` (legacy, if exists)

# %%
diag_root = RESULTS_BASE / "binary" / "diagnostic_ablation"
diag_results = {}  # (device, scheme) -> result

for device in ["chest", "wrist"]:
    device_dir = diag_root / device
    if not device_dir.exists():
        continue
    for subdir in sorted(device_dir.iterdir()):
        pkl = subdir / "losocv_results.pkl"
        if not pkl.exists():
            continue
        res = load_pkl(pkl)
        scheme = detect_label_scheme(res)
        key = (device, scheme)
        # Prefer suffixed _BINARY/_3CLASS dirs over the legacy unsuffixed one
        if key not in diag_results or subdir.name.endswith(("_BINARY", "_3CLASS")):
            diag_results[key] = res
            agg = res["aggregate"]
            print(
                f"  [{device:5s} {scheme:6s}] from {subdir.name:30s}  "
                f"F1_macro={agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}"
            )

if not diag_results:
    print("  No diagnostic-ablation results found.")


# %% [markdown]
# ## Section 5: The headline ablation narrative — 4-row table
#
# This is the table that tells the §4.4.1 leakage-investigation story:
#
# | Condition | Binary chest | Binary wrist |
# |---|---|---|
# | Normalised + overlapped (original RF) | ~1.0 | ~1.0 |
# | Unnormalised + overlapped | (norm_ablation) | (norm_ablation) |
# | Unnormalised + independent only (binary) | (diag binary) | (diag binary) |
# | Unnormalised + independent only (3-class) | (diag 3class) | (diag 3class) |

# %%
narrative_rows = []

# Row 1: Normalised + overlapped (= fs_tree binary)
narrative_rows.append(
    {
        "Condition": "(A) Normalised + overlapped (original)",
        "Task": "Binary",
        "Chest F1 macro": fmt_meanstd(
            fs_results[("chest", "tree")]["aggregate"]["f1_macro_mean"],
            fs_results[("chest", "tree")]["aggregate"]["f1_macro_std"],
        ),
        "Wrist F1 macro": fmt_meanstd(
            fs_results[("wrist", "tree")]["aggregate"]["f1_macro_mean"],
            fs_results[("wrist", "tree")]["aggregate"]["f1_macro_std"],
        ),
    }
)

# Row 2: Unnormalised + overlapped (= norm_ablation/unnormalized)
narrative_rows.append(
    {
        "Condition": "(B) Unnormalised + overlapped",
        "Task": "Binary",
        "Chest F1 macro": fmt_meanstd(
            norm_results[("chest", "unnormalised")]["aggregate"]["f1_macro_mean"],
            norm_results[("chest", "unnormalised")]["aggregate"]["f1_macro_std"],
        ),
        "Wrist F1 macro": fmt_meanstd(
            norm_results[("wrist", "unnormalised")]["aggregate"]["f1_macro_mean"],
            norm_results[("wrist", "unnormalised")]["aggregate"]["f1_macro_std"],
        ),
    }
)

# Row 3: Diagnostic binary (unnormalised + independent only, binary)
if ("chest", "binary") in diag_results and ("wrist", "binary") in diag_results:
    narrative_rows.append(
        {
            "Condition": "(C) Unnormalised + independent only",
            "Task": "Binary",
            "Chest F1 macro": fmt_meanstd(
                diag_results[("chest", "binary")]["aggregate"]["f1_macro_mean"],
                diag_results[("chest", "binary")]["aggregate"]["f1_macro_std"],
            ),
            "Wrist F1 macro": fmt_meanstd(
                diag_results[("wrist", "binary")]["aggregate"]["f1_macro_mean"],
                diag_results[("wrist", "binary")]["aggregate"]["f1_macro_std"],
            ),
        }
    )

# Row 4: Diagnostic 3-class
if ("chest", "3class") in diag_results and ("wrist", "3class") in diag_results:
    narrative_rows.append(
        {
            "Condition": "(D) Unnormalised + independent only",
            "Task": "3-class",
            "Chest F1 macro": fmt_meanstd(
                diag_results[("chest", "3class")]["aggregate"]["f1_macro_mean"],
                diag_results[("chest", "3class")]["aggregate"]["f1_macro_std"],
            ),
            "Wrist F1 macro": fmt_meanstd(
                diag_results[("wrist", "3class")]["aggregate"]["f1_macro_mean"],
                diag_results[("wrist", "3class")]["aggregate"]["f1_macro_std"],
            ),
        }
    )

narrative_df = pd.DataFrame(narrative_rows)
print("\nNARRATIVE ABLATION TABLE (§4.4.1 leakage investigation)")
print("=" * 100)
print(narrative_df.to_string(index=False))
narrative_df.to_csv(TABLES_DIR / "narrative_ablation.csv", index=False)
narrative_df.to_latex(TABLES_DIR / "narrative_ablation.tex", index=False, escape=False)


# %% [markdown]
# ## Section 6: Per-subject failure analysis
#
# Identifies which subjects fail on each device + task, and surfaces their
# confusion matrices. This is the FL-motivation evidence.


# %%
def failing_subjects(result: dict, threshold: float = 0.85) -> pd.DataFrame:
    """Return per-fold rows where F1 macro < threshold."""
    fold_results = result.get("fold_results") or []
    rows = []
    for f in fold_results:
        if f["f1_macro"] < threshold:
            rows.append(
                {
                    "subject": f.get("subject") or f.get("test_sid"),
                    "accuracy": f["accuracy"],
                    "f1_macro": f["f1_macro"],
                    "confusion_matrix": f["confusion_matrix"],
                }
            )
    return pd.DataFrame(rows)


print("\nFAILING SUBJECTS (F1 macro < 0.85)")
print("=" * 80)

failure_summary = []
for (device, scheme), res in diag_results.items():
    fails = failing_subjects(res, threshold=0.85)
    print(
        f"\n[{device.upper()} — {scheme}]  "
        f"({len(fails)} folds below threshold out of 15)"
    )
    for _, row in fails.iterrows():
        cm = np.array(row["confusion_matrix"])
        print(
            f"  S{int(row['subject'])}: F1={row['f1_macro']:.3f}  Acc={row['accuracy']:.3f}"
        )
        print(f"    CM: {cm.tolist()}")
        failure_summary.append(
            {
                "Device": device.capitalize(),
                "Task": scheme,
                "Subject": f"S{int(row['subject'])}",
                "F1 macro": f"{row['f1_macro']:.3f}",
                "Accuracy": f"{row['accuracy']:.3f}",
            }
        )

failure_df = pd.DataFrame(failure_summary)
failure_df.to_csv(TABLES_DIR / "failing_subjects.csv", index=False)
failure_df.to_latex(TABLES_DIR / "failing_subjects.tex", index=False, escape=False)


# %% [markdown]
# ### 6.1 Subject-device interaction (3-class)
#
# Shows whether failing subjects fail on chest, wrist, or both —
# evidence for the "different subjects fail on different devices" claim.

# %%
if ("chest", "3class") in diag_results and ("wrist", "3class") in diag_results:
    chest_f1 = {
        f["subject"]: f["f1_macro"]
        for f in diag_results[("chest", "3class")]["fold_results"]
    }
    wrist_f1 = {
        f["subject"]: f["f1_macro"]
        for f in diag_results[("wrist", "3class")]["fold_results"]
    }

    interaction_rows = []
    for subj in sorted(set(chest_f1) | set(wrist_f1)):
        cf = chest_f1.get(subj, np.nan)
        wf = wrist_f1.get(subj, np.nan)
        flag_chest = "FAIL" if cf < 0.85 else ""
        flag_wrist = "FAIL" if wf < 0.85 else ""
        interaction_rows.append(
            {
                "Subject": f"S{int(subj)}",
                "Chest F1 macro": f"{cf:.3f}",
                "Wrist F1 macro": f"{wf:.3f}",
                "Chest flag": flag_chest,
                "Wrist flag": flag_wrist,
            }
        )
    inter_df = pd.DataFrame(interaction_rows)
    print("\nSUBJECT-DEVICE INTERACTION (3-class, F1 macro)")
    print("=" * 80)
    print(inter_df.to_string(index=False))
    inter_df.to_csv(TABLES_DIR / "subject_device_interaction.csv", index=False)
    inter_df.to_latex(
        TABLES_DIR / "subject_device_interaction.tex", index=False, escape=False
    )


# %% [markdown]
# ### 6.2 Per-subject F1 bar chart (3-class, both devices)

# %%
if ("chest", "3class") in diag_results and ("wrist", "3class") in diag_results:
    subjects = sorted(chest_f1.keys())
    x = np.arange(len(subjects))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(
        x - width / 2,
        [chest_f1[s] for s in subjects],
        width,
        label="Chest",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        [wrist_f1[s] for s in subjects],
        width,
        label="Wrist",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        0.85,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Failure threshold (0.85)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s)}" for s in subjects])
    ax.set_ylabel("F1 macro")
    ax.set_xlabel("Subject")
    ax.set_title(
        "Per-subject F1 macro under the strictest condition "
        "(unnormalised + independent + 3-class)"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_subject_f1_3class.png", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "per_subject_f1_3class.pdf", bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## Section 7: CNN-LSTM wrist baseline
#
# Loads the pre-computed evaluation report, which already contains aggregate
# metrics, Cohen's κ, MCC, variability analysis, and global confusion matrix.

# %%
wrist_report_path = RESULTS_BASE / "dl" / "wrist_baseline" / "evaluation_report.json"
wrist_report: dict | None = None
if wrist_report_path.exists():
    wrist_report = load_json(wrist_report_path)
    agg = wrist_report["aggregate"]
    global_metrics = wrist_report["global_metrics"]
    variability = wrist_report["variability_analysis"]

    print("CNN-LSTM WRIST BASELINE (carry-forward configuration)")
    print("=" * 80)
    print(f"  Folds completed   : {wrist_report['n_folds']}")
    print(
        f"  Accuracy          : {fmt_meanstd(agg['accuracy_mean'], agg['accuracy_std'])}"
    )
    print(
        f"  F1 macro          : {fmt_meanstd(agg['f1_macro_mean'], agg['f1_macro_std'])}"
    )
    print(
        f"  F1 weighted       : {fmt_meanstd(agg['f1_weighted_mean'], agg['f1_weighted_std'])}"
    )
    print(
        f"  Precision         : {fmt_meanstd(agg['precision_mean'], agg['precision_std'])}"
    )
    print(f"  Recall            : {fmt_meanstd(agg['recall_mean'], agg['recall_std'])}")
    print(f"  Cohen's κ (global): {global_metrics['cohen_kappa']:.4f}")
    print(f"  MCC (global)      : {global_metrics['matthews_corrcoef']:.4f}")
    print("\n  Variability:")
    for k, v in variability.items():
        if isinstance(v, (int, float)):
            print(f"    {k:25s}: {v:.4f}")
        else:
            print(f"    {k:25s}: {v}")
else:
    print(f"  MISSING: {wrist_report_path}")


# %% [markdown]
# ### 7.1 CNN-LSTM wrist summary table

# %%
if wrist_report is not None:
    agg = wrist_report["aggregate"]
    gm = wrist_report["global_metrics"]
    dl_summary = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Value": fmt_meanstd(agg["accuracy_mean"], agg["accuracy_std"]),
            },
            {
                "Metric": "F1 macro",
                "Value": fmt_meanstd(agg["f1_macro_mean"], agg["f1_macro_std"]),
            },
            {
                "Metric": "F1 weighted",
                "Value": fmt_meanstd(agg["f1_weighted_mean"], agg["f1_weighted_std"]),
            },
            {
                "Metric": "Precision",
                "Value": fmt_meanstd(agg["precision_mean"], agg["precision_std"]),
            },
            {
                "Metric": "Recall",
                "Value": fmt_meanstd(agg["recall_mean"], agg["recall_std"]),
            },
            {"Metric": "Cohen's κ (global)", "Value": f"{gm['cohen_kappa']:.4f}"},
            {"Metric": "MCC (global)", "Value": f"{gm['matthews_corrcoef']:.4f}"},
        ]
    )
    dl_summary.to_csv(TABLES_DIR / "dl_wrist_baseline.csv", index=False)
    dl_summary.to_latex(TABLES_DIR / "dl_wrist_baseline.tex", index=False, escape=False)
    print(dl_summary.to_string(index=False))


# %% [markdown]
# ## Section 8: DL ablations
#
# Each ablation directory contains `loso_results.json` per configuration.
# We load the summary aggregate and produce a comparison table per ablation.

# %%
DL_ABL_ROOT = RESULTS_BASE / "dl"

dl_ablations = {
    "window": ["10s", "16s", "30s"],
    "cnn_layers": ["layers_1", "layers_2"],
    "dropout": ["p_0.3", "p_0.5"],
    "filters": ["f_32", "f_64", "f_128", "f_default"],
    "optimizer": ["adamw", "rmsprop"],
    "batch_size": ["bs_32", "bs_64", "bs_100"],
}

dl_abl_rows = []
for ablation_name, configs in dl_ablations.items():
    for cfg in configs:
        path = DL_ABL_ROOT / f"ablation_{ablation_name}" / cfg / "loso_results.json"
        if path.exists():
            r = load_json(path)
            s = r.get("summary", {})
            dl_abl_rows.append(
                {
                    "Ablation": ablation_name,
                    "Config": cfg,
                    "Folds": len(r.get("fold_results", [])),
                    "Accuracy": fmt_meanstd(
                        s.get("accuracy_mean", np.nan), s.get("accuracy_std", np.nan)
                    ),
                    "F1 macro": fmt_meanstd(
                        s.get("f1_macro_mean", np.nan), s.get("f1_macro_std", np.nan)
                    ),
                    "F1 weighted": fmt_meanstd(
                        s.get("f1_weighted_mean", np.nan),
                        s.get("f1_weighted_std", np.nan),
                    ),
                    "_f1_macro_mean": s.get("f1_macro_mean", np.nan),
                }
            )
        else:
            # cnn_layers does not save loso_results.json — only _fold_results/
            # Try aggregating fold JSONs manually
            fold_dir = DL_ABL_ROOT / f"ablation_{ablation_name}" / cfg / "_fold_results"
            if fold_dir.exists():
                fold_files = sorted(fold_dir.glob("fold_S*.json"))
                if fold_files:
                    folds = []
                    for p in fold_files:
                        f = load_json(p)
                        # Handle possible nesting: {'metrics': {...}} vs flat schema
                        if "accuracy" not in f and "metrics" in f:
                            f = f["metrics"]
                        if "accuracy" not in f and "test_metrics" in f:
                            f = f["test_metrics"]
                        folds.append(f)
                    # Filter folds that have what we need
                    folds = [f for f in folds if "accuracy" in f and "f1_macro" in f]
                    if not folds:
                        # Diagnose
                        sample = load_json(fold_files[0])
                        print(
                            f"  [SKIP] {ablation_name}/{cfg}: cannot find 'accuracy' key."
                        )
                        print(
                            f"         Top-level keys in fold file: {list(sample.keys())}"
                        )
                        continue
                    acc = np.array([f["accuracy"] for f in folds])
                    f1m = np.array([f["f1_macro"] for f in folds])
                    f1w = np.array([f.get("f1_weighted", np.nan) for f in folds])
                    dl_abl_rows.append(
                        {
                            "Ablation": ablation_name,
                            "Config": cfg,
                            "Folds": len(folds),
                            "Accuracy": fmt_meanstd(acc.mean(), acc.std(ddof=1)),
                            "F1 macro": fmt_meanstd(f1m.mean(), f1m.std(ddof=1)),
                            "F1 weighted": fmt_meanstd(f1w.mean(), f1w.std(ddof=1)),
                            "_f1_macro_mean": f1m.mean(),
                        }
                    )
                    continue
            print(f"  [MISSING] {path}")

dl_abl_df = pd.DataFrame(dl_abl_rows)
print("\nDL ABLATIONS (wrist CNN-LSTM, 15-fold LOSO)")
print("=" * 100)
display_cols = ["Ablation", "Config", "Folds", "Accuracy", "F1 macro", "F1 weighted"]
print(dl_abl_df[display_cols].to_string(index=False))
dl_abl_df[display_cols].to_csv(TABLES_DIR / "dl_ablations.csv", index=False)
dl_abl_df[display_cols].to_latex(
    TABLES_DIR / "dl_ablations.tex", index=False, escape=False
)


# %% [markdown]
# ### 8.1 DL-ablation grouped bar chart

# %%
if not dl_abl_df.empty:
    n_ablations = dl_abl_df["Ablation"].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, ablation in zip(axes, dl_abl_df["Ablation"].unique()):
        sub = dl_abl_df[dl_abl_df["Ablation"] == ablation]
        ax.bar(
            sub["Config"],
            sub["_f1_macro_mean"],
            color="#4CAF50",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(f"Ablation: {ablation}")
        ax.set_ylabel("F1 macro (mean)")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(
                i,
                row["_f1_macro_mean"] + 0.02,
                f"{row['_f1_macro_mean']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    # Hide extra subplots
    for ax in axes[n_ablations:]:
        ax.set_visible(False)

    plt.suptitle("CNN-LSTM ablations (wrist, 15-fold LOSO, F1 macro)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dl_ablations.png", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "dl_ablations.pdf", bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## Section 9: Literature comparison
#
# Compares this work against Schmidt et al. 2018 — the benchmark paper.
# Schmidt's numbers below are quoted directly from their Table 6 and the
# §3 reported chest+wrist binary configuration with their best classifier (LDA).
#
# **Note**: Schmidt et al. report accuracy and F1, not F1 macro specifically.
# We treat Schmidt's "F1" as F1 weighted for comparison, since the WESAD label
# distribution is unbalanced and weighted is what Schmidt's table represents.

# %%
schmidt_binary = {
    "Reference": "Schmidt et al. (2018), chest, LDA",
    "Task": "Binary",
    "Accuracy": "0.9312",
    "F1 (weighted)": "0.9147",
    "F1 macro": "—",
}
schmidt_3class = {
    "Reference": "Schmidt et al. (2018), chest, LDA",
    "Task": "3-class",
    "Accuracy": "0.8034",
    "F1 (weighted)": "0.7251",
    "F1 macro": "—",
}

# Best of our binary chest = best FS-tree config
our_binary_chest = fs_results[("chest", "tree")]["aggregate"]
our_binary_wrist = fs_results[("wrist", "tree")]["aggregate"]

lit_rows = [schmidt_binary, schmidt_3class]

lit_rows.append(
    {
        "Reference": "This work, RF, chest, fs=tree",
        "Task": "Binary",
        "Accuracy": fmt_meanstd(
            our_binary_chest["accuracy_mean"], our_binary_chest["accuracy_std"]
        ),
        "F1 (weighted)": fmt_meanstd(
            our_binary_chest["f1_weighted_mean"], our_binary_chest["f1_weighted_std"]
        ),
        "F1 macro": fmt_meanstd(
            our_binary_chest["f1_macro_mean"], our_binary_chest["f1_macro_std"]
        ),
    }
)
lit_rows.append(
    {
        "Reference": "This work, RF, wrist, fs=tree",
        "Task": "Binary",
        "Accuracy": fmt_meanstd(
            our_binary_wrist["accuracy_mean"], our_binary_wrist["accuracy_std"]
        ),
        "F1 (weighted)": fmt_meanstd(
            our_binary_wrist["f1_weighted_mean"], our_binary_wrist["f1_weighted_std"]
        ),
        "F1 macro": fmt_meanstd(
            our_binary_wrist["f1_macro_mean"], our_binary_wrist["f1_macro_std"]
        ),
    }
)

if ("chest", "3class") in diag_results:
    a = diag_results[("chest", "3class")]["aggregate"]
    lit_rows.append(
        {
            "Reference": "This work, RF, chest, unnorm + indep",
            "Task": "3-class",
            "Accuracy": fmt_meanstd(a["accuracy_mean"], a["accuracy_std"]),
            "F1 (weighted)": fmt_meanstd(a["f1_weighted_mean"], a["f1_weighted_std"]),
            "F1 macro": fmt_meanstd(a["f1_macro_mean"], a["f1_macro_std"]),
        }
    )

if ("wrist", "3class") in diag_results:
    a = diag_results[("wrist", "3class")]["aggregate"]
    lit_rows.append(
        {
            "Reference": "This work, RF, wrist, unnorm + indep",
            "Task": "3-class",
            "Accuracy": fmt_meanstd(a["accuracy_mean"], a["accuracy_std"]),
            "F1 (weighted)": fmt_meanstd(a["f1_weighted_mean"], a["f1_weighted_std"]),
            "F1 macro": fmt_meanstd(a["f1_macro_mean"], a["f1_macro_std"]),
        }
    )

if wrist_report is not None:
    agg = wrist_report["aggregate"]
    lit_rows.append(
        {
            "Reference": "This work, CNN-LSTM, wrist",
            "Task": "Binary",
            "Accuracy": fmt_meanstd(agg["accuracy_mean"], agg["accuracy_std"]),
            "F1 (weighted)": fmt_meanstd(
                agg["f1_weighted_mean"], agg["f1_weighted_std"]
            ),
            "F1 macro": fmt_meanstd(agg["f1_macro_mean"], agg["f1_macro_std"]),
        }
    )

lit_df = pd.DataFrame(lit_rows)
print("\nCOMPARISON WITH LITERATURE")
print("=" * 100)
print(lit_df.to_string(index=False))
lit_df.to_csv(TABLES_DIR / "literature_comparison.csv", index=False)
lit_df.to_latex(TABLES_DIR / "literature_comparison.tex", index=False, escape=False)


# %% [markdown]
# ## Section 10: Summary index of artefacts produced
#
# Final manifest of everything saved to `artefacts/iter1/`.

# %%
print("\nARTEFACTS PRODUCED")
print("=" * 80)
for category, root in [("Tables", TABLES_DIR), ("Figures", FIGURES_DIR)]:
    print(f"\n{category}:")
    for p in sorted(root.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:40s} {size:>8,} bytes")

print("\nDone.")

# """
# KD Evaluation: Four-Way Comparison and Ablation Framework

# Purpose:
# Post-training evaluation for the cross-modal KD pipeline:

#   * Four-way comparison table (teacher, no-KD, KD, KD+personalised)
#   * Statistical tests (Wilcoxon signed-rank with Bonferroni correction)
#   * Ablation runners for temperature, alpha, feature set, beta, K

# All results are structured as dicts/DataFrames for downstream
# visualisation (``visualizations.py``) and dissertation tables.


from __future__ import annotations

import json
import os
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    # accuracy_score,
    confusion_matrix,
    # f1_score,
    # precision_score,
    # recall_score,
    # roc_auc_score,
)

from kd.kd_config import (
    # ABLATION_ALPHAS,
    # ABLATION_BETAS,
    # ABLATION_FEATURE_SETS,
    # ABLATION_FINETUNE_STEPS,
    # ABLATION_TEMPERATURES,
    # ALL_SUBJECT_IDS,
    BONFERRONI_ALPHA,
    # KD_DEFAULTS,
    KD_RESULTS_DIR,
    N_COMPARISONS,
    # PRIMARY_METRIC,
    STAT_ALPHA,
    ensure_dirs,
)

#  1. FOUR-WAY COMPARISON TABLE

CONDITIONS = ["teacher", "no_kd", "kd", "personal"]
METRICS = ["accuracy", "macro_f1", "auroc", "stress_recall", "stress_precision"]


def build_comparison_table(
    loso_results: list[dict],
) -> pd.DataFrame:
    """Build the four-way comparison table from LOSO results.

    Produces a DataFrame with one row per subject and columns for each
    condition × metric combination, plus mean ± std summary rows.

    Parameters
    ----------
    loso_results : list[dict]
        Per-fold results from ``kd_trainer.run_full_loso()``.

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, then {condition}_{metric} for each
        combination.
    """
    rows = []

    for fold in loso_results:
        sid = fold["meta"]["test_subject"]
        row = {"subject_id": sid}

        for cond in CONDITIONS:
            if cond not in fold:
                for metric in METRICS:
                    row[f"{cond}_{metric}"] = np.nan
                continue

            r = fold[cond]
            row[f"{cond}_accuracy"] = r.get("accuracy", np.nan)
            row[f"{cond}_macro_f1"] = r.get("macro_f1", np.nan)
            row[f"{cond}_auroc"] = r.get("auroc", np.nan)
            row[f"{cond}_stress_recall"] = r.get("stress_recall", np.nan)
            row[f"{cond}_stress_precision"] = r.get("stress_precision", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add mean ± std summary rows
    summary_mean = {"subject_id": "Mean"}
    summary_std = {"subject_id": "Std"}
    for col in df.columns:
        if col == "subject_id":
            continue
        vals = df[col].dropna()
        summary_mean[col] = vals.mean() if len(vals) > 0 else np.nan
        summary_std[col] = vals.std() if len(vals) > 0 else np.nan

    df = pd.concat([df, pd.DataFrame([summary_mean, summary_std])], ignore_index=True)

    return df


def print_comparison_table(df: pd.DataFrame) -> str:
    """Format the comparison table for console/dissertation output.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Four-Way Comparison — Cross-Modal KD")
    lines.append("=" * 80)

    # Header
    header = f"{'Subject':>8s}"
    for cond in CONDITIONS:
        header += f"  {cond + ' F1':>12s}"
    lines.append(header)
    lines.append("-" * 80)

    # Per-subject rows
    for _, row in df.iterrows():
        sid = row["subject_id"]
        line = f"{str(sid):>8s}"
        for cond in CONDITIONS:
            col = f"{cond}_macro_f1"
            val = row.get(col, np.nan)
            if isinstance(val, float) and np.isfinite(val):
                line += f"  {val:>12.3f}"
            else:
                line += f"  {'—':>12s}"
        lines.append(line)

    lines.append("=" * 80)
    return "\n".join(lines)


#  2. STATISTICAL TESTS


def run_statistical_tests(
    loso_results: list[dict],
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Run Wilcoxon signed-rank tests on paired LOSO results.

    Three comparisons (per plan):
      * (c) KD vs (b) no-KD = KD contribution
      * (d) personal vs (c) KD = personalisation contribution
      * (d) personal vs (b) no-KD = full pipeline benefit

    Uses Bonferroni correction: p < 0.05/3 = 0.0167.

    Parameters
    ----------
    loso_results : list[dict]
        Per-fold results from run_full_loso().
    metric : str
        Which metric to test (default "macro_f1").

    Returns
    -------
    pd.DataFrame
        One row per comparison with columns: comparison, statistic,
        p_value, p_corrected, significant, effect_size_r.
    """
    # Extract per-subject scores
    scores: dict[str, list[float]] = {cond: [] for cond in CONDITIONS}
    for fold in loso_results:
        for cond in CONDITIONS:
            if cond in fold:
                scores[cond].append(fold[cond].get(metric, np.nan))
            else:
                scores[cond].append(np.nan)

    comparisons = [
        ("kd vs no_kd", "kd", "no_kd"),
        ("personal vs kd", "personal", "kd"),
        ("personal vs no_kd", "personal", "no_kd"),
        # NOTE: (a) teacher vs (d) personal ("residual modality gap") is intentionally
        # EXCLUDED from Wilcoxon tests.  The teacher operates on a
        # different input modality (raw multi-channel signals vs HRV
        # features), so a paired statistical comparison is methodologically
        # questionable as it conflates modality gap with model capacity.
        # The gap is reported descriptively in the comparison table.
    ]

    rows = []
    for name, cond_a, cond_b in comparisons:
        a = np.array(scores[cond_a])
        b = np.array(scores[cond_b])

        # Remove NaN pairs
        valid = np.isfinite(a) & np.isfinite(b)
        a_valid, b_valid = a[valid], b[valid]

        if len(a_valid) < 5:
            rows.append(
                {
                    "comparison": name,
                    "n_pairs": len(a_valid),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "p_corrected": np.nan,
                    "significant": False,
                    "effect_size_r": np.nan,
                    "mean_diff": np.nan,
                }
            )
            continue

        # Wilcoxon signed-rank (two-sided)
        try:
            stat, p_val = wilcoxon(a_valid, b_valid, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, p_val = 0.0, 1.0

        p_corrected = min(p_val * N_COMPARISONS, 1.0)  # Bonferroni

        # Rank-biserial effect size: r = Z / sqrt(N)
        # scipy.stats.wilcoxon returns T = min(W+, W−).  The normal
        # approximation Z = (W+ − μ) / σ requires W+ (sum of positive
        # ranks), not the minimum.  Reconstruct W+ as the larger of
        # (stat, total − stat).
        n = len(a_valid)
        w_total = n * (n + 1) / 2
        w_plus = max(stat, w_total - stat)
        z = (w_plus - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        effect_r = min(abs(z) / np.sqrt(n), 1.0)  # clamp to [0, 1]

        rows.append(
            {
                "comparison": name,
                "n_pairs": n,
                "statistic": float(stat),
                "p_value": float(p_val),
                "p_corrected": float(p_corrected),
                "significant": p_corrected < STAT_ALPHA,
                "effect_size_r": float(effect_r),
                "mean_diff": float(np.mean(a_valid - b_valid)),
            }
        )

    return pd.DataFrame(rows)


def print_statistical_tests(df: pd.DataFrame) -> str:
    """Format statistical test results for console output."""
    lines = []
    lines.append("=" * 67)
    lines.append("Statistical Tests — Wilcoxon Signed-Rank (Bonferroni corrected)")
    lines.append(
        f"  α = {STAT_ALPHA}, corrections = {N_COMPARISONS}, "
        f"α_corrected = {BONFERRONI_ALPHA:.4f}"
    )
    lines.append("=" * 67)

    for _, row in df.iterrows():
        sig = "✓ SIG" if row["significant"] else "  n.s."
        lines.append(
            f"  {row['comparison']:>20s}: "
            f"W={row['statistic']:>8.1f}, "
            f"p={row['p_value']:.4f} (corrected={row['p_corrected']:.4f}), "
            f"r={row['effect_size_r']:.3f}, "
            f"Δ={row['mean_diff']:+.3f}  {sig}"
        )

    lines.append("=" * 67)
    return "\n".join(lines)


# =====================================================================
#  3. CONFUSION MATRICES
# =====================================================================


def extract_confusion_matrices(
    loso_results: list[dict],
) -> dict[str, np.ndarray]:
    """Aggregate confusion matrices across all LOSO folds.

    For each condition, sums per-fold confusion matrices into a single
    global matrix.  Requires that per-fold results include predictions
    and labels (added to results by run_loso_fold if detailed=True).

    Parameters
    ----------
    loso_results : list[dict]
        Must contain per-fold 'predictions' and 'eval_labels' keys.

    Returns
    -------
    dict[str, np.ndarray]
        condition_name → (n_classes, n_classes) confusion matrix.
    """
    # This function works with the prediction arrays stored during
    # run_full_loso.  If those aren't available, it returns empty dicts.
    matrices = {}

    for cond in CONDITIONS:
        all_preds = []
        all_labels = []

        for fold in loso_results:
            preds_key = f"{cond}_preds"
            labels_key = "eval_labels"
            if preds_key in fold and labels_key in fold:
                all_preds.append(fold[preds_key])
                all_labels.append(fold[labels_key])

        if all_preds:
            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            matrices[cond] = confusion_matrix(labels, preds)

    return matrices


#  4. SAVE / LOAD RESULTS


def save_results(
    loso_results: list[dict],
    output_dir: str = KD_RESULTS_DIR,
    name: str = "main",
) -> str:
    """Save LOSO results to JSON.

    Parameters
    ----------
    loso_results : list[dict]
    output_dir : str
    name : str
        Experiment name for the filename.

    Returns
    -------
    str
        Path to saved file.
    """
    ensure_dirs()
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    path = os.path.join(output_dir, f"kd_results_{name}.json")
    with open(path, "w") as f:
        json.dump(_convert(loso_results), f, indent=2)

    print(f"  Results saved: {path}")
    return path


def load_results(path: str) -> list[dict[Any, Any]]:
    """Load LOSO results from JSON."""
    with open(path) as f:
        return cast(list[dict[Any, Any]], json.load(f))


#  5. FULL EVALUATION REPORT


def generate_evaluation_report(
    loso_results: list[dict],
    output_dir: str = KD_RESULTS_DIR,
    name: str = "main",
) -> dict:
    """Generate the complete evaluation report.

    Produces:
      * Four-way comparison table (DataFrame + formatted string)
      * Statistical tests (DataFrame + formatted string)
      * Saves results to JSON

    Parameters
    ----------
    loso_results : list[dict]
    output_dir : str
    name : str

    Returns
    -------
    dict with keys: comparison_table, stats_tests, loso_results
    """
    ensure_dirs()

    # Comparison table
    comp_df = build_comparison_table(loso_results)
    comp_str = print_comparison_table(comp_df)
    print(comp_str)

    # Statistical tests
    stats_df = run_statistical_tests(loso_results)
    stats_str = print_statistical_tests(stats_df)
    print(stats_str)

    # Save
    save_path = save_results(loso_results, output_dir, name)

    # Save tables as CSV
    comp_df.to_csv(os.path.join(output_dir, f"comparison_{name}.csv"), index=False)
    stats_df.to_csv(os.path.join(output_dir, f"stats_{name}.csv"), index=False)

    return {
        "comparison_table": comp_df,
        "stats_tests": stats_df,
        "results_path": save_path,
    }

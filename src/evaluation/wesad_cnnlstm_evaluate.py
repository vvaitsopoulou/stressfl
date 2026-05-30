# CNN-LSTM Evaluation

# Post-training evaluation module for the WESAD CNN-LSTM pipeline.
# Loads saved LOSO results or re-runs inference from model checkpoints
# to produce comprehensive evaluation outputs.

# Two modes of operation:
#   1. FROM RESULTS JSON: Load the loso_results.json saved by the training
#      script and compute additional aggregate statistics, tables, and plots.
#      This is faster as no GPU isneeded
#   2. FROM CHECKPOINTS: Re-run inference on test subjects using saved model
#      checkpoints.  Produces per-window predictions for detailed analysis

# Metrics
#   Primary:     Macro-F1 score (handles class imbalance)
#   Secondary:   Per-class F1, per-class recall (stress recall is clinically
#                important to minimise false negatives)
#   Tertiary:    Accuracy (for literature comparability but not a primary metric due to class imbalance
#   Per-fold:    All of the above + confusion matrix
#   Aggregate:   Mean ± std (ddof=1) across folds.
#   Global:      Concatenated predictions across all folds → single confusion
#                matrix and classification report.  Shows overall error patterns.

# References:
#   Prajod & André (2022): LOSO evaluation, macro F1
#   Moser et al. (2024): stress detection evaluation methodology
#   Schmidt et al. (2018): WESAD benchmark evaluation protocol

import json
import os
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
import torch

from src.data.wesad_dl_dataset import (
    ALL_SUBJECT_IDS,
    WESADDLDataset,
)
from src.models.wesad_cnnlstm_model import (
    build_chest_model,
    build_wrist_model,
)
from src.training.wesad_cnnlstm_train import TrainConfig, _collate_fn

#  Mode 1: Evaluate from saved results JSON


def evaluate_from_results(
    results_path: str,
    output_dir: str | None = None,
) -> dict:
    """Load loso_results.json and compute comprehensive evaluation.
    Uses the per-fold metrics and confusion matrices already computed
    by the training script.

    Parameters:
    results_path : str
        Path to loso_results.json from wesad_cnnlstm_train.py.
    output_dir : str | None
        Where to save evaluation outputs.  Defaults to same directory
        as results_path.

    Returns:
    dict with evaluation report.
    """
    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)

    fold_results = results["fold_results"]
    config = results.get("config", {})
    n_classes = config.get("n_classes", 2)

    # Class names for display
    if n_classes == 2:
        class_names = ["Non-Stress", "Stress"]
    else:
        class_names = ["Baseline", "Stress", "Amusement"]

    print("  CNN-LSTM EVALUATION REPORT")
    print("\n")
    print(f"  Source: {results_path}")
    print(f"  Folds: {len(fold_results)}")
    print(f"  Classes: {class_names}")
    print(f"  Device: {config.get('device_type', 'unknown')}")
    print(f"  Window: {config.get('window_sec', '?')}s")
    print()

    # Per-fold table
    _print_per_fold_table(fold_results)

    # aggregate metrics (mean ± std with ddof=1)
    aggregate = _compute_aggregate(fold_results)
    _print_aggregate(aggregate)

    # Global confusion matrix
    # Reconstruct from per-fold confusion matrices
    global_cm = _compute_global_confusion_matrix(fold_results, n_classes)
    _print_confusion_matrix(global_cm, class_names, title="GLOBAL CONFUSION MATRIX")

    # Per-class metrics from global CM
    per_class = _compute_per_class_from_cm(global_cm, class_names)
    _print_per_class(per_class, class_names)

    # Global Kappa and MCC
    # reconstruct flat arrays from the global CM
    # so we can use sklearn directly with checkpoint mode)
    global_true, global_pred = [], []
    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            count = int(global_cm[true_idx, pred_idx])
            global_true.extend([true_idx] * count)
            global_pred.extend([pred_idx] * count)

    global_kappa = cohen_kappa_score(global_true, global_pred)
    global_mcc = matthews_corrcoef(global_true, global_pred)
    print(f"  Global κ={global_kappa:.4f}  MCC={global_mcc:.4f}\n")

    # Loss curve analysis
    loss_analysis = _analyse_loss_curves(fold_results, output_dir)

    # Inter-subject variability analysis (FL motivation)
    variability = _analyse_variability(fold_results)

    # Compile full report
    report = {
        "config": config,
        "n_folds": len(fold_results),
        "aggregate": aggregate,
        "global_metrics": {
            "cohen_kappa": global_kappa,
            "matthews_corrcoef": global_mcc,
        },
        "per_fold": fold_results,
        "global_confusion_matrix": global_cm.tolist(),
        "per_class_metrics": per_class,
        "variability_analysis": variability,
        "loss_analysis": loss_analysis,
    }

    # Save report
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved → {report_path}")

    return report


#  Mode 2: Re-run inference from checkpoints


def evaluate_from_checkpoints(
    results_dir: str,
    windowed_dir: str = "data/dl_windowed/",
    config_override: TrainConfig | None = None,
) -> dict:
    """Re-run inference from saved model checkpoints.

    Produces per-window predictions for detailed error analysis.

    Parameters:
    results_dir : str
        Directory containing fold_S{id}/ subdirectories with best_model.pt
        and a config.json.
    windowed_dir : str
        Root of windowed data.
    config_override : TrainConfig | None
        Override the saved config (e.g. for evaluating on a different
        window length).
    """
    # Load config
    config_path = os.path.join(results_dir, "config.json")
    if config_override is not None:
        config = config_override
    elif os.path.exists(config_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)

        # Reconstruct TrainConfig from saved JSON.
        # Two problems to handle:
        #   1. JSON may contain keys that aren't TrainConfig parameters
        #      (e.g. if config was extended after this run).
        #   2. Types may be wrong: JSON serialises tuples as strings
        #      (e.g. scale_range becomes "(0.9, 1.1)") etc
        import inspect

        valid_params = set(inspect.signature(TrainConfig.__init__).parameters.keys())
        valid_params.discard("self")

        filtered = {k: v for k, v in cfg_dict.items() if k in valid_params}

        # Fix type conversions from JSON serialisation (default=str)
        if "scale_range" in filtered and isinstance(filtered["scale_range"], str):
            # "(0.9, 1.1)" → (0.9, 1.1)
            import ast

            filtered["scale_range"] = tuple(ast.literal_eval(filtered["scale_range"]))

        if "label_map" in filtered and isinstance(filtered["label_map"], str):
            import ast

            filtered["label_map"] = ast.literal_eval(filtered["label_map"])

        # JSON stores dict keys as strings and label_map needs float keys
        if "label_map" in filtered and isinstance(filtered["label_map"], dict):
            filtered["label_map"] = {
                float(k): int(v) for k, v in filtered["label_map"].items()
            }

        config = TrainConfig(**filtered)
    else:
        raise FileNotFoundError(
            f"No config.json found in {results_dir} and no config_override provided."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_fn = build_chest_model if config.device_type == "chest" else build_wrist_model

    n_classes = config.n_classes
    if n_classes == 2:
        class_names = ["Non-Stress", "Stress"]
    else:
        class_names = ["Baseline", "Stress", "Amusement"]

    all_classes = list(range(n_classes))

    print("\n")
    print("  CNN-LSTM CHECKPOINT EVALUATION")
    print("\n")
    print(f"  Results dir: {results_dir}")
    print(f"  Device: {device}")
    print()

    all_preds = []
    all_labels = []
    fold_results = []

    for test_sid in ALL_SUBJECT_IDS:
        fold_dir = os.path.join(results_dir, f"fold_S{test_sid}")
        model_path = os.path.join(fold_dir, "best_model.pt")

        if not os.path.exists(model_path):
            print(f"    No checkpoint for S{test_sid} — skipping")
            continue

        # Build model and load checkpoint

        build_kwargs: dict[str, Any] = {
            "n_classes": config.n_classes,
            "lstm_hidden": config.lstm_hidden,
            "lstm_layers": config.lstm_layers,
            "cnn_dropout": config.cnn_dropout,
            "fc_dropout": config.fc_dropout,
            "n_conv_layers": config.n_conv_layers,
            "n_filters_override": config.n_filters_override,
        }
        if config.target_time is not None:
            build_kwargs["target_time"] = config.target_time

        model = build_fn(**build_kwargs).to(device)
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )
        model.eval()

        # Build test dataset (non-overlapping windows only)
        test_ds = WESADDLDataset(
            subject_ids=[test_sid],
            windowed_dir=windowed_dir,
            device_type=config.device_type,
            window_sec=config.window_sec,
            mode="test",
            label_map=config.label_map,
            noise_std=0.0,
            scale_range=(1.0, 1.0),
            signals_to_load=config.signals_to_load,
        )

        if len(test_ds) == 0:
            print(f"    No test windows for S{test_sid} — skipping")
            continue

        loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=0,
        )

        # Inference
        fold_preds = []
        fold_labels = []

        with torch.no_grad():
            for signals, labels in loader:
                signals = {k: v.to(device) for k, v in signals.items()}
                logits = model(signals)
                preds = logits.argmax(dim=1).cpu().numpy()
                fold_preds.extend(preds.tolist())
                fold_labels.extend(labels.numpy().tolist())

        fold_preds = np.array(fold_preds)
        fold_labels = np.array(fold_labels)

        # Per-fold metrics
        acc = accuracy_score(fold_labels, fold_preds)
        f1_macro = f1_score(
            fold_labels,
            fold_preds,
            average="macro",
            zero_division=0,
            labels=all_classes,
        )
        f1_weighted = f1_score(
            fold_labels,
            fold_preds,
            average="weighted",
            zero_division=0,
            labels=all_classes,
        )
        prec = precision_score(
            fold_labels,
            fold_preds,
            average="macro",
            zero_division=0,
            labels=all_classes,
        )
        rec = recall_score(
            fold_labels,
            fold_preds,
            average="macro",
            zero_division=0,
            labels=all_classes,
        )
        cm = confusion_matrix(fold_labels, fold_preds, labels=all_classes)

        print(
            f"  S{test_sid}: acc={acc:.4f}  F1m={f1_macro:.4f}  "
            f"F1w={f1_weighted:.4f}  n={len(test_ds)}"
        )

        fold_results.append(
            {
                "subject": test_sid,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision": prec,
                "recall": rec,
                "confusion_matrix": cm.tolist(),
                "n_test": len(test_ds),
            }
        )

        all_preds.extend(fold_preds.tolist())
        all_labels.extend(fold_labels.tolist())

    # Aggregate
    if not fold_results:
        print("     No valid folds")
        return {}

    aggregate = _compute_aggregate(fold_results)

    # Global metrics from concatenated predictions
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    global_cm = confusion_matrix(all_labels_np, all_preds_np, labels=all_classes)
    global_acc = accuracy_score(all_labels_np, all_preds_np)
    global_f1_macro = f1_score(
        all_labels_np,
        all_preds_np,
        average="macro",
        zero_division=0,
        labels=all_classes,
    )
    global_kappa = cohen_kappa_score(all_labels_np, all_preds_np)
    global_mcc = matthews_corrcoef(all_labels_np, all_preds_np)
    global_report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names=class_names,
        labels=all_classes,
        zero_division=0,
    )

    print(
        f"\n  Global: acc={global_acc:.4f}  F1m={global_f1_macro:.4f}  "
        f"κ={global_kappa:.4f}  MCC={global_mcc:.4f}"
    )
    print()

    _print_per_fold_table(fold_results)
    _print_aggregate(aggregate)
    _print_confusion_matrix(global_cm, class_names, title="GLOBAL CONFUSION MATRIX")

    print("\n  Global Classification Report:")
    print(global_report)

    # Save
    per_class = _compute_per_class_from_cm(global_cm, class_names)
    variability = _analyse_variability(fold_results)

    report = {
        "config": config.to_dict() if hasattr(config, "to_dict") else {},
        "n_folds": len(fold_results),
        "aggregate": aggregate,
        "global_metrics": {
            "accuracy": global_acc,
            "f1_macro": global_f1_macro,
            "cohen_kappa": global_kappa,
            "matthews_corrcoef": global_mcc,
        },
        "global_confusion_matrix": global_cm.tolist(),
        "global_classification_report": global_report,
        "per_class_metrics": per_class,
        "per_fold": fold_results,
        "variability_analysis": variability,
    }

    report_path = os.path.join(results_dir, "checkpoint_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved → {report_path}")

    return report


#  Helper: per-fold table
def _print_per_fold_table(fold_results: list[dict]):
    """Print a formatted per-fold metrics table."""
    print("  PER-FOLD METRICS")
    print(
        f"  {'Subject':<10} {'Acc':<8} {'F1(m)':<8} {'F1(w)':<8} "
        f"{'Prec':<8} {'Rec':<8} {'N_test':<8}"
    )
    print("  " + "─" * 56)

    for r in fold_results:
        print(
            f"  S{r['subject']:<9} {r['accuracy']:<8.4f} {r['f1_macro']:<8.4f} "
            f"{r['f1_weighted']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} "
            f"{r.get('n_test', '?'):<8}"
        )
    print()


#  Helper: aggregate statistics


def _compute_aggregate(fold_results: list[dict]) -> dict:
    """Compute mean ± std (ddof=1) across folds."""
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]
    aggregate = {}

    for m in metrics:
        values = [r[m] for r in fold_results if m in r]
        if values:
            aggregate[f"{m}_mean"] = float(np.mean(values))
            aggregate[f"{m}_std"] = float(np.std(values, ddof=1))
            aggregate[f"{m}_min"] = float(np.min(values))
            aggregate[f"{m}_max"] = float(np.max(values))
            aggregate[f"{m}_median"] = float(np.median(values))

    return aggregate


def _print_aggregate(aggregate: dict):
    print("  AGGREGATE METRICS (mean ± std, ddof=1)")
    print("\n")
    for base_metric in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]:
        mean_key = f"{base_metric}_mean"
        std_key = f"{base_metric}_std"
        min_key = f"{base_metric}_min"
        max_key = f"{base_metric}_max"
        if mean_key in aggregate:
            print(
                f"  {base_metric:<16s}: {aggregate[mean_key]:.4f} ± {aggregate[std_key]:.4f}  "
                f"[{aggregate[min_key]:.4f}, {aggregate[max_key]:.4f}]"
            )
    print()


#  Helper: confusion matrix operations


def _compute_global_confusion_matrix(
    fold_results: list[dict], n_classes: int
) -> np.ndarray:
    # Sum per-fold confusion matrices into a global one.
    global_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for r in fold_results:
        cm = np.array(r["confusion_matrix"])
        # Every fold uses the same label_map and all_classes list,
        # so every confusion matrix must be exactly n_classes × n_classes.
        # A mismatch means something is wrong upstream — fail loudly.
        assert cm.shape == (n_classes, n_classes), (
            f"Fold S{r.get('subject', '?')} confusion matrix shape {cm.shape} "
            f"doesn't match n_classes={n_classes}. Check label_map consistency."
        )
        global_cm += cm
    return global_cm


def _print_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = ""):
    # Print a labelled confusion matrix.
    if title:
        print(f"  {title}")
    n = len(class_names)
    # Header
    header = "  " + " " * 14 + "".join(f"{'Pred ' + cn:<12}" for cn in class_names)
    print(header)
    print("  " + "─" * (14 + 12 * n))
    for i, name in enumerate(class_names):
        row_str = "".join(f"{cm[i, j]:<12}" for j in range(n))
        print(f"  {'True ' + name:<14}{row_str}")
    print()


def _compute_per_class_from_cm(cm: np.ndarray, class_names: list[str]) -> dict:
    # Compute per-class precision, recall, F1 from the confusion matrix.
    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(cm[i, :].sum())

        # Sensitivity = recall (same thing, clinical terminology)
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "sensitivity": round(recall, 4),  # clinical alias
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    return per_class


def _print_per_class(per_class: dict, class_names: list[str]):
    # Print per-class metrics table.
    print("  PER-CLASS METRICS (from global confusion matrix)")
    print(
        f"  {'Class':<14} {'Prec':<8} {'Recall':<8} {'Spec':<8} "
        f"{'F1':<8} {'Support':<10}"
    )
    print("  " + "─" * 54)
    for name in class_names:
        m = per_class[name]
        print(
            f"  {name:<14} {m['precision']:<8.4f} {m['recall']:<8.4f} "
            f"{m['specificity']:<8.4f} {m['f1']:<8.4f} {m['support']:<10}"
        )
    print()


#  Helper: loss curve analysis


def _analyse_loss_curves(fold_results: list[dict], output_dir: str) -> dict:
    """Analyse training/validation loss curves for overfitting diagnostics.

    Loads loss_curves.json from each fold directory and computes:
      - Mean final train/val loss across folds
      - Mean convergence epoch (early stopping)
      - Train-val loss gap at convergence (overfitting indicator)
    """
    train_losses_final = []
    val_losses_final = []
    convergence_epochs = []
    loss_gaps = []

    for r in fold_results:
        sid = r["subject"]
        curves_path = os.path.join(output_dir, f"fold_S{sid}", "loss_curves.json")

        if not os.path.exists(curves_path):
            # Try parent directory
            curves_path = os.path.join(
                os.path.dirname(output_dir), f"fold_S{sid}", "loss_curves.json"
            )

        if os.path.exists(curves_path):
            with open(curves_path) as f:
                curves = json.load(f)

            tl = curves.get("train_loss", [])
            vl = curves.get("val_loss", [])

            if tl and vl:
                # Find best val loss epoch
                best_epoch = int(np.argmin(vl))
                train_losses_final.append(tl[best_epoch])
                val_losses_final.append(vl[best_epoch])
                convergence_epochs.append(best_epoch)
                loss_gaps.append(vl[best_epoch] - tl[best_epoch])

    analysis = {}
    if train_losses_final:
        analysis = {
            "mean_train_loss_at_best": float(np.mean(train_losses_final)),
            "mean_val_loss_at_best": float(np.mean(val_losses_final)),
            "mean_convergence_epoch": float(np.mean(convergence_epochs)),
            "std_convergence_epoch": (
                float(np.std(convergence_epochs, ddof=1))
                if len(convergence_epochs) > 1
                else 0.0
            ),
            "mean_loss_gap": float(np.mean(loss_gaps)),
            "std_loss_gap": (
                float(np.std(loss_gaps, ddof=1)) if len(loss_gaps) > 1 else 0.0
            ),
        }

        print("  LOSS CURVE ANALYSIS")
        print("\n")
        print(
            f"  Mean train loss at best epoch: {analysis['mean_train_loss_at_best']:.4f}"
        )
        print(
            f"  Mean val loss at best epoch:   {analysis['mean_val_loss_at_best']:.4f}"
        )
        print(
            f"  Mean loss gap (val - train):   {analysis['mean_loss_gap']:.4f} ± {analysis['std_loss_gap']:.4f}"
        )
        print(
            f"  Mean convergence epoch:        {analysis['mean_convergence_epoch']:.1f} ± {analysis['std_convergence_epoch']:.1f}"
        )

        if analysis["mean_loss_gap"] > 0.5:
            print(
                "     Large loss gap suggests overfitting - consider more regularisation"
            )
        elif analysis["mean_loss_gap"] < 0.05:
            print("     Small loss gap - model is well-regularised")
        print()

    return analysis


#  Helper: inter-subject variability (FL motivation)


def _analyse_variability(fold_results: list[dict]) -> dict:
    """Analyse inter-subject variability in performance.

    High std across LOSO folds means the model performs very differently
    on different subjects — exactly the problem FL is designed to solve.
    """
    f1s = [r["f1_macro"] for r in fold_results if "f1_macro" in r]

    if not f1s:
        return {}

    # Identify best/worst subjects
    sorted_by_f1 = sorted(fold_results, key=lambda r: r.get("f1_macro", 0))
    worst = sorted_by_f1[0]
    best = sorted_by_f1[-1]

    # Coefficient of variation (std/mean) - normalised measure of spread
    f1_cv = float(np.std(f1s, ddof=1) / np.mean(f1s)) if np.mean(f1s) > 0 else 0.0

    variability = {
        "f1_macro_range": float(max(f1s) - min(f1s)),
        "f1_macro_cv": f1_cv,
        "best_subject": best["subject"],
        "best_f1_macro": best.get("f1_macro", 0),
        "worst_subject": worst["subject"],
        "worst_f1_macro": worst.get("f1_macro", 0),
        "subjects_below_0_5": [
            r["subject"] for r in fold_results if r.get("f1_macro", 0) < 0.5
        ],
    }

    print("  INTER-SUBJECT VARIABILITY (FL MOTIVATION)")
    print("\n")
    print(
        f"  F1 macro range: {variability['f1_macro_range']:.4f} "
        f"({variability['worst_f1_macro']:.4f} – {variability['best_f1_macro']:.4f})"
    )
    print(f"  Coefficient of variation: {variability['f1_macro_cv']:.4f}")
    print(
        f"  Best subject:  S{variability['best_subject']} (F1={variability['best_f1_macro']:.4f})"
    )
    print(
        f"  Worst subject: S{variability['worst_subject']} (F1={variability['worst_f1_macro']:.4f})"
    )

    if variability["subjects_below_0_5"]:
        print(f"  Subjects below F1=0.5: {variability['subjects_below_0_5']}")
        print("  → These subjects have highly atypical physiology — FL personalisation")
        print("    should improve their performance specifically.")

    if variability["f1_macro_range"] > 0.3:
        print("  → High inter-subject variability confirms the motivation for")
        print(
            "    Federated Learning: a centralised model cannot serve all subjects equally."
        )
    print()

    return variability


#  CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WESAD CNN-LSTM Evaluation")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing loso_results.json and fold checkpoints",
    )
    parser.add_argument(
        "--mode",
        default="json",
        choices=["json", "checkpoint"],
        help="Evaluation mode: 'json' (fast, from results file) or "
        "'checkpoint' (re-run inference from saved models)",
    )
    parser.add_argument(
        "--windowed-dir",
        default="data/dl_windowed/",
        help="Path to windowed data (only needed for checkpoint mode)",
    )

    args = parser.parse_args()

    if args.mode == "json":
        results_path = os.path.join(args.results_dir, "loso_results.json")
        if not os.path.exists(results_path):
            print(f"    {results_path} not found")
        else:
            evaluate_from_results(results_path, output_dir=args.results_dir)
    else:
        evaluate_from_checkpoints(
            results_dir=args.results_dir,
            windowed_dir=args.windowed_dir,
        )

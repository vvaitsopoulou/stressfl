#  Pilot Evaluator: Single-Participant Deployment Validation

# Purpose:
# Evaluate the cross-modal KD student on real Fitbit Charge 6 HR data
# from a single-participant pilot study.

# Pipeline
# 1–4. Load JSON, preprocess, extract 6 HRV features per 16 s window
#      (8 s step), normalise with WESAD training-set scaler.
# 5–7. Load best global KD checkpoint, freeze base, fine-tune head on
#      calibration windows (first 15 minutes of structured session).
# 8.   Evaluate three conditions on evaluation windows:
#        (b') Student without KD
#        (c') KD Student (global, no personalisation)
#        (d') KD Student + personalised head

# Reporting:
# * Per-phase accuracy, confusion matrices
# * Prediction timeline (HR + stress probability + labels vs time)
# * Data quality report (gaps, excluded windows, sampling statistics)

# N=1 Limitation:
# This is "a single-participant feasibility study demonstrating the
# end-to-end deployment pipeline". Results are illustrative, not generalisable.

# References:
# [Fuller22]    Fuller et al. 2022 — Fitbit HR accuracy meta-analysis
# [Bottcher22]  Bottcher et al. 2022 — Fitbit data quality
# [Collins21]   Collins et al. 2021 — FedRep base/head split

from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from src.kd.fitbit_preprocessor import (
    generate_quality_report,
    preprocess_fitbit_session,
)
from src.kd.kd_config import (
    DEVICE,
    FINETUNE_LR,
    FINETUNE_STEPS_DEFAULT,
    # KD_CHECKPOINT_DIR,
    KD_PAIRED_DIR,
    KD_RESULTS_DIR,
    # N_PRIMARY_FEATURES,
    # PRIMARY_HRV_FEATURES,
    ensure_dirs,
)
from src.kd.kd_trainer import (
    fit_scaler,
    load_paired_data,
    personalise_head,
)
from src.kd.preprocess_pilot import preprocess_two_session_pilot, summarise
from src.kd.student_model import StudentModel, build_student

#  1. LOAD WESAD SCALER


def load_wesad_scaler(
    train_subject_ids: list[int],
    paired_dir: str = KD_PAIRED_DIR,
    use_noisy: bool = True,
) -> StandardScaler:
    """Fit a Z-score scaler on WESAD training data.

    The Fitbit features must be normalised using WESAD training-set
    statistics (not Fitbit data statistics) to match the student's
    training distribution."

    Parameters
    ----------
    train_subject_ids : list[int]
        WESAD subject IDs used for training the deployed student.
        Typically all 15 subjects (since the pilot participant is not
        a WESAD subject).
    paired_dir : str
        Directory containing paired .npz files.
    use_noisy : bool
        If True, fit on noise-degraded features (the deployed student was trained on noisy features).

    Returns
    -------
    StandardScaler
        Fitted on WESAD training features.
    """
    train_data = load_paired_data(train_subject_ids, paired_dir, use_noisy)
    scaler = fit_scaler(train_data["features"])
    print(
        f"  WESAD scaler fitted on {len(train_data['features'])} windows "
        f"from {len(train_subject_ids)} subjects"
    )
    return scaler


#  2. EVALUATE CONDITIONS


def _evaluate_condition(
    model: StudentModel,
    features: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    device: str = DEVICE,
) -> dict:
    """Evaluate a single model on the evaluation windows.

    Only uses windows that are both quality-valid AND have labelled
    ground truth (label ≥ 0).

    Parameters
    ----------
    model : StudentModel
    features : np.ndarray (N, 6)
        Z-score normalised features.
    labels : np.ndarray (N,)
        Ground-truth labels (-1 for unlabelled).
    quality : np.ndarray of bool (N,)
    device : str

    Returns
    -------
    dict with accuracy, macro_f1, auroc, stress_recall, stress_precision,
    preds, probs, confusion_matrix.
    """
    # Only evaluate on valid, labelled windows
    valid_mask = quality & (labels >= 0)
    if valid_mask.sum() == 0:
        return {"error": "No valid labelled evaluation windows"}

    eval_feats = features[valid_mask]
    eval_labels = labels[valid_mask]

    model.eval()
    with torch.no_grad():
        feats_t = torch.tensor(eval_feats, dtype=torch.float32, device=device)
        logits = model(feats_t).cpu()
        preds = logits.argmax(dim=1).numpy()
        probs = F.softmax(logits, dim=1)[:, 1].numpy()

    acc = accuracy_score(eval_labels, preds)
    f1 = f1_score(eval_labels, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(eval_labels, probs)
    except ValueError:
        auroc = 0.0

    s_recall = recall_score(eval_labels, preds, pos_label=1, zero_division=0)
    s_precision = precision_score(eval_labels, preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(eval_labels, preds, labels=[0, 1])

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "auroc": float(auroc),
        "stress_recall": float(s_recall),
        "stress_precision": float(s_precision),
        "confusion_matrix": cm.tolist(),
        "preds": preds.tolist(),
        "probs": probs.tolist(),
        "eval_labels": eval_labels.tolist(),
        "n_eval": int(valid_mask.sum()),
    }


#  3. PER-PHASE BREAKDOWN


def _per_phase_metrics(
    model: StudentModel,
    features: np.ndarray,
    labels: np.ndarray,
    phase_names: np.ndarray,
    quality: np.ndarray,
    device: str = DEVICE,
) -> dict[str, dict]:
    """Compute metrics for each protocol phase separately.

    Parameters
    ----------
    model : StudentModel
    features, labels, phase_names, quality : arrays from preprocessing
    device : str

    Returns
    -------
    dict[phase_name → metrics_dict]
    """
    unique_phases = sorted(set(phase_names))
    phase_metrics = {}

    model.eval()
    for phase in unique_phases:
        mask = (phase_names == phase) & quality & (labels >= 0)
        if mask.sum() == 0:
            continue

        p_feats = features[mask]
        p_labels = labels[mask]

        with torch.no_grad():
            feats_t = torch.tensor(p_feats, dtype=torch.float32, device=device)
            logits = model(feats_t).cpu()
            preds = logits.argmax(dim=1).numpy()

        phase_metrics[phase] = {
            "n_windows": int(mask.sum()),
            "accuracy": float(accuracy_score(p_labels, preds)),
            "macro_f1": float(
                f1_score(p_labels, preds, average="macro", zero_division=0)
            ),
            "true_label": int(
                np.bincount(p_labels).argmax()
            ),  # majority label (phases may straddle boundaries)
        }

    return phase_metrics


#  4. PREDICTION TIMELINE


def _build_prediction_timeline(
    model: StudentModel,
    features: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    phase_names: np.ndarray,
    quality: np.ndarray,
    hr_uniform: np.ndarray,
    t_uniform: np.ndarray,
    device: str = DEVICE,
) -> dict:
    """Build a timeline of HR, predictions, and labels for plotting.

    Parameters
    ----------
    model : StudentModel
    features, timestamps, labels, phase_names, quality : from preprocessing
    hr_uniform, t_uniform : uniform 1 Hz HR data
    device : str

    Returns
    -------
    dict with:
        window_times : (N,) window start times
        stress_probs : (N,) P(stress) per window (NaN for invalid)
        preds : (N,) predicted labels (NaN for invalid)
        true_labels : (N,) ground-truth labels
        phase_names : (N,) phase names
        hr_times : (M,) 1 Hz timestamps
        hr_values : (M,) 1 Hz HR values
    """
    n_windows = len(features)
    stress_probs = np.full(n_windows, np.nan)
    pred_labels = np.full(n_windows, np.nan)

    # Only predict on valid windows
    valid_mask = quality
    if valid_mask.any():
        model.eval()
        with torch.no_grad():
            feats_t = torch.tensor(
                features[valid_mask],
                dtype=torch.float32,
                device=device,
            )
            logits = model(feats_t).cpu()
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
            preds = logits.argmax(dim=1).numpy()

        stress_probs[valid_mask] = probs
        pred_labels[valid_mask] = preds

    return {
        "window_times": timestamps.tolist(),
        "stress_probs": stress_probs.tolist(),
        "preds": pred_labels.tolist(),
        "true_labels": labels.tolist(),
        "phase_names": phase_names.tolist(),
        "hr_times": t_uniform.tolist(),
        "hr_values": hr_uniform.tolist(),
    }


#  5. MAIN: FULL PILOT EVALUATION


def evaluate_fitbit_pilot(
    json_path: str,
    kd_checkpoint_path: str,
    no_kd_checkpoint_path: str | None = None,
    session_start: str | None = None,
    train_subject_ids: list[int] | None = None,
    paired_dir: str = KD_PAIRED_DIR,
    finetune_lr: float = FINETUNE_LR,
    finetune_steps: int = FINETUNE_STEPS_DEFAULT,
    device: str = DEVICE,
    output_dir: str = KD_RESULTS_DIR,
    two_session_pilot=False,
) -> dict:
    """Run the full Fitbit pilot evaluation pipeline.

    This is the top-level function that executes the complete Phase 5
    pipeline as specified in the plan.

    Parameters
    ----------
    json_path : str
        Path to Fitbit intraday HR JSON file.
    kd_checkpoint_path : str
        Path to the best global KD student checkpoint (.pt file
        containing model state_dict).
    no_kd_checkpoint_path : str, optional
        Path to a no-KD student checkpoint for condition (b').
        If None, a fresh (untrained) student is used as baseline.
    session_start : str, optional
        Session start time (HH:MM:SS) for timestamp rebasing.
    train_subject_ids : list[int], optional
        WESAD subjects used for training.  Default: all 15.
    paired_dir : str
        Directory with paired .npz files (for WESAD scaler).
    finetune_lr : float
        Head fine-tuning learning rate.
    finetune_steps : int
        Number of head fine-tuning gradient steps.
    device : str
    output_dir : str

    Returns
    -------
    dict with all results, timelines, and quality reports.
    """
    from src.kd.kd_config import ALL_SUBJECT_IDS

    if train_subject_ids is None:
        train_subject_ids = ALL_SUBJECT_IDS

    ensure_dirs()

    print("=" * 60)
    print("Fitbit Pilot Evaluation — Phase 5")
    print("=" * 60)

    #   Steps 1-3: Preprocess Fitbit data
    print("\n1. Preprocessing Fitbit HR data...")
    if two_session_pilot:
        preproc = preprocess_two_session_pilot(json_path)
        summarise(preproc)
        quality_report = "(see two-session summary above)"
    else:
        preproc = preprocess_fitbit_session(json_path, session_start)
        quality_report = generate_quality_report(preproc)
        print(quality_report)

    #  ] Step 4: Normalise with WESAD scaler
    print("\n2. Loading WESAD scaler...")
    scaler = load_wesad_scaler(train_subject_ids, paired_dir, use_noisy=True)
    features_raw = preproc["features"]
    features_norm = scaler.transform(features_raw)

    labels = preproc["labels"]
    quality = preproc["quality"]
    cal_mask = preproc["cal_mask"]
    eval_mask = preproc["eval_mask"]
    phase_names = preproc["phase_names"]
    timestamps = preproc["timestamps"]

    # Calibration / evaluation feature sets
    cal_feats = features_norm[cal_mask & quality & (labels >= 0)]
    cal_labels = labels[cal_mask & quality & (labels >= 0)]
    print(
        f"  Calibration: {len(cal_feats)} valid labelled windows "
        f"(stress={int((cal_labels == 1).sum())}, "
        f"non-stress={int((cal_labels == 0).sum())})"
    )

    if len(cal_feats) == 0:
        raise RuntimeError(
            "No valid labelled calibration windows available for head fine-tuning."
        )

    #   Steps 5-6: Load checkpoint and personalise
    print("\n3. Loading KD student checkpoint...")
    model_kd = build_student().to(device)
    checkpoint = torch.load(kd_checkpoint_path, map_location=device, weights_only=True)
    model_kd.load_state_dict(checkpoint)
    print(f"  Loaded: {kd_checkpoint_path}")

    # Condition (b'): no-KD student
    if no_kd_checkpoint_path is not None and os.path.exists(no_kd_checkpoint_path):
        model_no_kd = build_student().to(device)
        ckpt_no_kd = torch.load(
            no_kd_checkpoint_path, map_location=device, weights_only=True
        )
        model_no_kd.load_state_dict(ckpt_no_kd)
        print(f"  No-KD loaded: {no_kd_checkpoint_path}")
    else:
        raise FileNotFoundError(
            "no_kd_checkpoint_path is required for condition (b'). "
            "A randomly initialised model is NOT a valid CE-only baseline. "
            f"Got: {no_kd_checkpoint_path!r}"
        )

    #   Step 7: Fine-tune head on calibration windows
    print(f"\n4. Personalising head ({finetune_steps} steps, lr={finetune_lr})...")
    model_personal = copy.deepcopy(model_kd)
    personalise_head(
        model_personal,
        cal_feats,
        cal_labels,
        lr=finetune_lr,
        n_steps=finetune_steps,
        device=device,
    )
    print("  Head fine-tuned on calibration data")

    #   Step 8: Evaluate three conditions on evaluation windows
    print("\n5. Evaluating on evaluation windows...")

    # Use eval_mask for evaluation — only windows AFTER calibration period
    eval_features = features_norm[eval_mask]
    eval_labels = labels[eval_mask]
    eval_quality = quality[eval_mask]
    eval_phases = phase_names[eval_mask]

    results: dict[str, Any] = {}
    conditions = [
        ("no_kd", model_no_kd),
        ("kd", model_kd),
        ("personal", model_personal),
    ]

    for cond_name, model in conditions:
        r = _evaluate_condition(
            model,
            eval_features,
            eval_labels,
            eval_quality,
            device,
        )
        results[cond_name] = r
        if "error" not in r:
            print(
                f"  {cond_name:10s}: F1={r['macro_f1']:.3f}, "
                f"Acc={r['accuracy']:.3f}, "
                f"StressRecall={r['stress_recall']:.3f}"
            )

    #   Per-phase breakdown (for the personalised model)
    print("\n6. Per-phase breakdown (KD + personal):")
    phase_results = _per_phase_metrics(
        model_personal,
        eval_features,
        eval_labels,
        eval_phases,
        eval_quality,
        device,
    )
    for phase, pm in phase_results.items():
        print(
            f"  {phase:30s}: n={pm['n_windows']:3d}, "
            f"F1={pm['macro_f1']:.3f}, "
            f"Acc={pm['accuracy']:.3f}, "
            f"true_label={pm['true_label']}"
        )
    results["per_phase"] = phase_results

    #   Prediction timeline (for visualisation)
    print("\n7. Building prediction timeline...")
    # In two-session mode, hr_uniform/t_uniform are dicts keyed by session.
    # Concatenate them so the timeline can plot both sessions on a continuous axis.
    if isinstance(preproc["hr_uniform"], dict):
        import numpy as np

        cal_t = preproc["t_uniform"]["cal"]
        eval_t = preproc["t_uniform"]["eval"]
        cal_hr = preproc["hr_uniform"]["cal"]
        eval_hr = preproc["hr_uniform"]["eval"]
        # Offset eval session times so they sit after cal on the timeline
        eval_t_shifted = eval_t + (cal_t[-1] + 60.0)  # 60 s gap separator
        t_uniform = np.concatenate([cal_t, eval_t_shifted])
        hr_uniform = np.concatenate([cal_hr, eval_hr])
    else:
        t_uniform = preproc["t_uniform"]
        hr_uniform = preproc["hr_uniform"]

    timeline = _build_prediction_timeline(
        model_personal,
        features_norm,
        timestamps,
        labels,
        phase_names,
        quality,
        hr_uniform,
        t_uniform,
        device,
    )
    results["timeline"] = timeline

    #   Package results
    results["meta"] = {
        "json_path": json_path,
        "kd_checkpoint": kd_checkpoint_path,
        "no_kd_checkpoint": no_kd_checkpoint_path,
        "finetune_lr": finetune_lr,
        "finetune_steps": finetune_steps,
        "n_cal_windows": len(cal_feats),
        "n_eval_windows": int(
            eval_mask.sum()
        ),  # all eval windows (incl. invalid/unlabelled)
        "n_valid_eval": int((eval_mask & quality & (labels >= 0)).sum()),
        "session_start": session_start,
    }
    results["quality_report"] = quality_report
    results["sampling_report"] = preproc["sampling_report"]
    results["gap_report"] = preproc["gap_report"]

    #   Save results
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "fitbit_pilot_results.json")

    def _convert(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
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

    with open(save_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"\n  Results saved: {save_path}")

    #   Summary
    print(f"\n{'=' * 60}")
    print("Fitbit Pilot Summary")
    print(f"{'=' * 60}")
    print("  N=1 limitation: single-participant feasibility study")
    print(
        f"  Calibration: {len(cal_feats)} windows "
        f"({int((cal_labels == 1).sum())} stress, "
        f"{int((cal_labels == 0).sum())} non-stress)"
    )
    for cond_name in ["no_kd", "kd", "personal"]:
        if cond_name in results and "error" not in results[cond_name]:
            r = results[cond_name]
            print(
                f"  {cond_name:10s}: F1={r['macro_f1']:.3f}, "
                f"Acc={r['accuracy']:.3f}, "
                f"AUROC={r['auroc']:.3f}"
            )
    print(f"{'=' * 60}")

    return results


#  6. VISUALISATION HELPERS
def plot_prediction_timeline(
    results: dict,
    output_path: str | None = None,
) -> None:
    """Plot the prediction timeline: HR + stress probability + labels.

    Parameters
    ----------
    results : dict
        Output of evaluate_fitbit_pilot().
    output_path : str, optional
        If provided, saves the figure to this path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib required for timeline plot")
        return

    timeline = results["timeline"]
    hr_t = np.array(timeline["hr_times"]) / 60  # to minutes
    hr_v = np.array(timeline["hr_values"])
    win_t = np.array(timeline["window_times"]) / 60
    probs = np.array(timeline["stress_probs"])
    true_labels = np.array(timeline["true_labels"])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Heart rate
    ax1.plot(hr_t, hr_v, color="#4477AA", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("HR (BPM)")
    ax1.set_title("Fitbit Pilot — Prediction Timeline")

    # Panel 2: Stress probability
    valid = np.isfinite(probs)
    ax2.plot(win_t[valid], probs[valid], color="#228833", linewidth=1)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("P(stress)")
    ax2.set_ylim(-0.05, 1.05)

    # Panel 3: True labels
    for i, t in enumerate(win_t):
        if true_labels[i] == 1:
            ax3.axvspan(t, t + 16 / 60, color="#EE6677", alpha=0.3)
        elif true_labels[i] == 0:
            ax3.axvspan(t, t + 16 / 60, color="#4477AA", alpha=0.15)
    ax3.set_ylabel("Protocol")
    ax3.set_xlabel("Time (minutes)")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Non-stress", "Stress"])

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Timeline saved: {output_path}")
    plt.close(fig)

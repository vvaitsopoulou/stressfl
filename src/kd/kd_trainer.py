# KD Training Pipeline: LOSO Cross-Validation with Teacher Logit Caching

# Full training pipeline for cross-modal knowledge distillation, including:

#   Teacher logit caching : run frozen teacher on all WESAD windows once, cache raw logits and intermediate features.
#   LOSO training loop with hyperparameter grid search, four training conditions, and personalised head fine-tuning.

# Training Conditions:
#   (a) KD Student — combined KD+CE loss (main result)
#   (b) Student without KD — CE loss only (alpha=0, ablation baseline)
#   (c) Teacher — already trained, evaluate only (upper bound)
#   (d) KD Student + Personalised Head — after global KD, freeze base, fine-tune head on calibration set

# All four conditions are evaluated on the SAME 80% evaluation split per test subject

# References
# Hinton et al. 2015: KD framework
# Collins et al. 2021: FedRep base/head split
# [PULSE25 arXiv:2510.24058 offline distillation, frozen teacher

from __future__ import annotations

import copy
from dataclasses import replace
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.kd.kd_config import (
    ALL_SUBJECT_IDS,
    DEVICE,
    FINETUNE_FRACTION,
    FINETUNE_LR,
    FINETUNE_STEPS_DEFAULT,
    INSTANCE_NORM_EPS,
    KD_CHECKPOINT_DIR,
    KD_DEFAULTS,
    KD_PAIRED_DIR,
    N_CLASSES,
    KDHyperparameters,
    # N_PRIMARY_FEATURES,
    # PRIMARY_HRV_FEATURES,
    # TEACHER_FEATURE_DIM,
    ensure_dirs,
    get_loso_splits,
)
from src.kd.kd_loss import kd_loss, kd_loss_with_features
from src.kd.student_model import StudentModel, build_student

# DATA LOADING UTILITIES


def load_paired_data(
    subject_ids: list[int],
    paired_dir: str = KD_PAIRED_DIR,
    use_noisy: bool = False,
) -> dict[str, np.ndarray]:
    """Load and concatenate paired data for multiple subjects.

    Parameters
    ----------
    subject_ids : list[int]
        WESAD subject IDs to load.
    paired_dir : str
        Directory containing S{id}_paired.npz files.
    use_noisy : bool
        If True, use ``student_features_noisy`` instead of
        ``student_features``.

    Returns
    -------
    dict with keys:
        features : np.ndarray (N_total, 6)
        labels : np.ndarray (N_total,)
        teacher_logits : np.ndarray (N_total, 2) or None
        teacher_features : np.ndarray (N_total, 32) or None
        subject_ids : np.ndarray (N_total,) — per-window subject ID
        bvp, w_eda, w_acc, w_temp : raw teacher signals (for caching)
    """
    all_feats, all_labels, all_sids = [], [], []
    all_logits, all_tfeats = [], []
    all_bvp, all_eda, all_acc, all_temp = [], [], [], []

    feat_key = "student_features_noisy" if use_noisy else "student_features"

    for sid in subject_ids:
        path = os.path.join(paired_dir, f"S{sid}_paired.npz")
        data = np.load(path)

        all_feats.append(data[feat_key])
        all_labels.append(data["labels"])
        all_sids.append(np.full(len(data["labels"]), sid, dtype=np.int64))

        # Teacher signals (for logit caching)
        all_bvp.append(data["bvp"])
        all_eda.append(data["w_eda"])
        all_acc.append(data["w_acc"])
        all_temp.append(data["w_temp"])

        # Teacher logits/features
        if "teacher_logits" in data.files:
            all_logits.append(data["teacher_logits"])
        if "teacher_features" in data.files:
            all_tfeats.append(data["teacher_features"])

    result = {
        "features": np.concatenate(all_feats, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "subject_ids": np.concatenate(all_sids, axis=0),
        "bvp": np.concatenate(all_bvp, axis=0),
        "w_eda": np.concatenate(all_eda, axis=0),
        "w_acc": np.concatenate(all_acc, axis=0),
        "w_temp": np.concatenate(all_temp, axis=0),
    }

    if all_logits:
        result["teacher_logits"] = np.concatenate(all_logits, axis=0)
    if all_tfeats:
        result["teacher_features"] = np.concatenate(all_tfeats, axis=0)

    return result


# TEACHER LOGIT CACHING


def _instance_znorm_signal(x: np.ndarray, eps: float = INSTANCE_NORM_EPS) -> np.ndarray:
    """Per-window instance Z-score normalisation, matching WESADDLDataset.

    Parameters
    ----------
    x : np.ndarray
        (T,) for 1-channel or (T, C) for multi-channel.

    Returns
    -------
    np.ndarray
        Normalised array, same shape.
    """
    if x.ndim == 1:
        return (x - x.mean()) / (x.std() + eps)
    else:
        mu = x.mean(axis=0, keepdims=True)
        sigma = x.std(axis=0, keepdims=True)
        return (x - mu) / (sigma + eps)


def _build_teacher_signal_dict(
    bvp: np.ndarray,
    w_eda: np.ndarray,
    w_acc: np.ndarray,
    w_temp: np.ndarray,
    device: str = DEVICE,
) -> dict[str, torch.Tensor]:
    """Build a teacher input dict from raw arrays for a batch of windows.

    Applies instance Z-score normalisation per window and converts to
    the tensor shapes expected by the CNN-LSTM model.

    Parameters
    ----------
    bvp : np.ndarray, shape (B, 512)
    w_eda : np.ndarray, shape (B, 64)
    w_acc : np.ndarray, shape (B, 512, 3) — stored (T, C)
    w_temp : np.ndarray, shape (B, 64)
    device : str

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: bvp (B,1,512), w_eda (B,1,64), w_acc (B,3,512), w_temp (B,1,64)
    """
    batch_size = len(bvp)

    # Normalise each window individually
    bvp_norm = np.stack([_instance_znorm_signal(bvp[i]) for i in range(batch_size)])
    eda_norm = np.stack([_instance_znorm_signal(w_eda[i]) for i in range(batch_size)])
    acc_norm = np.stack([_instance_znorm_signal(w_acc[i]) for i in range(batch_size)])
    temp_norm = np.stack([_instance_znorm_signal(w_temp[i]) for i in range(batch_size)])

    # Convert to tensors with correct shapes
    # 1-channel: (B, T) → (B, 1, T)
    # 3-channel: (B, T, 3) → (B, 3, T) via permute
    signals = {
        "bvp": torch.from_numpy(bvp_norm).float().unsqueeze(1).to(device),
        "w_eda": torch.from_numpy(eda_norm).float().unsqueeze(1).to(device),
        "w_acc": torch.from_numpy(acc_norm).float().permute(0, 2, 1).to(device),
        "w_temp": torch.from_numpy(temp_norm).float().unsqueeze(1).to(device),
    }

    return signals


def cache_teacher_logits(
    teacher_wrapper,
    data: dict[str, np.ndarray],
    batch_size: int = 64,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen teacher on all windows and cache logits + features.

    Parameters
    ----------
    teacher_wrapper : KDTeacherWrapper
        Frozen teacher model in eval mode.
    data : dict
        Output of ``load_paired_data()`` containing raw teacher signals.
    batch_size : int
    device : str

    Returns
    -------
    all_logits : np.ndarray (N, n_classes)
        Raw teacher logits (NOT softmaxed).
    all_features : np.ndarray (N, 32)
        Post-BN LSTM hidden states for feature-matching.
    """
    n_total = len(data["labels"])
    all_logits = []
    all_features = []

    teacher_wrapper.eval()
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)

            signals = _build_teacher_signal_dict(
                data["bvp"][start:end],
                data["w_eda"][start:end],
                data["w_acc"][start:end],
                data["w_temp"][start:end],
                device=device,
            )

            logits, features = teacher_wrapper(signals, return_features=True)

            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_logits), np.concatenate(all_features)


# FEATURE NORMALISATION


def fit_scaler(
    train_features: np.ndarray,
) -> StandardScaler:
    """Fit a Z-score scaler on training features only (no leakage).

    Parameters
    ----------
    train_features : np.ndarray (N_train, 6)

    Returns
    -------
    StandardScaler
        Fitted on training data.
    """
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


# CLASS WEIGHTS


def compute_class_weights(
    labels: np.ndarray,
    n_classes: int = N_CLASSES,
    device: str = DEVICE,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CE loss.

    Handles the ~2.35:1 class imbalance in WESAD binary labels.
    Applied to the CE term only, not the KD term, as denotred in our methodology

    Parameters
    ----------
    labels : np.ndarray (N,)
    n_classes : int
    device : str

    Returns
    -------
    torch.Tensor (n_classes,)
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)

    full_weights = np.ones(n_classes, dtype=np.float64)
    for cls, w in zip(classes, weights):
        full_weights[int(cls)] = w

    return torch.tensor(full_weights, dtype=torch.float32, device=device)


# SINGLE TRAINING RUN


def _make_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray | None = None,
    teacher_features: np.ndarray | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader from numpy arrays.

    Parameters
    ----------
    features : (N, 6) student HRV features
    labels : (N,) integer labels
    teacher_logits : (N, n_classes) cached teacher logits, optional
    teacher_features : (N, 32) cached teacher features, optional
    batch_size : int
    shuffle : bool
    drop_last : bool
        True for training (avoids batch-size-1 BatchNorm crash),
        False for evaluation.
    """
    tensors = [
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    ]
    if teacher_logits is not None:
        tensors.append(torch.tensor(teacher_logits, dtype=torch.float32))
    if teacher_features is not None:
        tensors.append(torch.tensor(teacher_features, dtype=torch.float32))

    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def train_student(
    model: StudentModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: KDHyperparameters,
    class_weights: torch.Tensor,
    device: str = DEVICE,
    has_teacher_logits: bool = True,
    has_teacher_features: bool = False,
) -> dict:
    """Train a student model for one configuration.

    Parameters
    ----------
    model : StudentModel
    train_loader, val_loader : DataLoader
    config : KDHyperparameters
    class_weights : torch.Tensor
    device : str
    has_teacher_logits : bool
        Whether the DataLoader contains teacher logits.
    has_teacher_features : bool
        Whether the DataLoader contains teacher features.

    Returns
    -------
    dict with keys: best_val_f1, best_epoch, train_losses, val_f1s
    """
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    class_weights = class_weights.to(device)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    train_losses = []
    val_f1s = []

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Explicit positional unpacking based on what _make_dataloader put into the TensorDataset:
            #   position 0: features (always)
            #   position 1: labels (always)
            #   position 2: teacher_logits (if has_teacher_logits)
            #   position 3: teacher_features (if has_teacher_features)
            features = batch[0].to(device)
            labels = batch[1].to(device)
            t_logits = batch[2].to(device) if has_teacher_logits else None
            t_feats = batch[3].to(device) if has_teacher_features else None

            optimizer.zero_grad()

            if config.beta > 0 and has_teacher_features:
                s_logits, s_feats = model(features, return_features=True)
                loss = kd_loss_with_features(
                    s_logits,
                    t_logits,
                    labels,
                    s_feats,
                    t_feats,
                    T=config.temperature,
                    alpha=config.alpha,
                    beta=config.beta,
                    class_weights=class_weights,
                )
            elif has_teacher_logits and config.alpha > 0:
                s_logits = model(features)
                loss = kd_loss(
                    s_logits,
                    t_logits,
                    labels,
                    T=config.temperature,
                    alpha=config.alpha,
                    class_weights=class_weights,
                )
            else:
                # Pure CE (condition b: no KD)
                s_logits = model(features)
                loss = F.cross_entropy(s_logits, labels, weight=class_weights)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        val_f1 = _evaluate_f1(model, val_loader, device)
        val_f1s.append(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_f1s": val_f1s,
    }


def _evaluate_f1(
    model: StudentModel,
    loader: DataLoader,
    device: str,
) -> float:
    """Compute macro-F1 on a DataLoader (ignoring teacher columns)."""
    from sklearn.metrics import f1_score

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            labels = batch[1]
            preds = model(features).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


# PERSONALISATION (Condition d)


def personalise_head(
    model: StudentModel,
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    lr: float = FINETUNE_LR,
    n_steps: int = FINETUNE_STEPS_DEFAULT,
    device: str = DEVICE,
) -> StudentModel:
    """Fine-tune the student head on calibration data.

    Freezes the base and trains only the head for K gradient steps with CE loss.

    Parameters
    ----------
    model : StudentModel
        Pre-trained student (base + head).  Will be modified in-place.
    cal_features : np.ndarray (N_cal, 6)
        Calibration features (already Z-score normalised).
    cal_labels : np.ndarray (N_cal,)
    lr : float
        Fine-tuning learning rate (default 1e-4).
    n_steps : int
        Number of gradient steps (default 10).
    device : str

    Returns
    -------
    StudentModel
        The same model with fine-tuned head.
    """
    model = model.to(device)

    # CRITICAL: base in eval mode (running BN stats), head in train mode.
    # See student_model.py "BatchNorm Caveat" documentation.
    model.base.eval()
    model.head.train()

    # Freeze base parameters
    for p in model.base.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.head.parameters(), lr=lr)

    features_t = torch.tensor(cal_features, dtype=torch.float32, device=device)
    labels_t = torch.tensor(cal_labels, dtype=torch.long, device=device)

    for _step in range(n_steps):
        optimizer.zero_grad()
        logits = model(features_t)
        loss = F.cross_entropy(logits, labels_t)
        loss.backward()
        optimizer.step()

    # Restore requires_grad for base
    for p in model.base.parameters():
        p.requires_grad = True

    return model


def temporal_calibration_split(
    features: np.ndarray,
    labels: np.ndarray,
    fraction: float = FINETUNE_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split test subject data into calibration and evaluation sets.

    Uses a TEMPORAL split: takes the first ``fraction`` of each
    contiguous label block (not random, not purely chronological).
    This ensures the calibration set contains both stress and non-stress
    windows, avoiding the issue where purely chronological splitting
    yields mostly baseline (WESAD protocol order: baseline → stress →
    amusement → recovery).

    Parameters
    ----------
    features : np.ndarray (N, 6)
    labels : np.ndarray (N,)
    fraction : float
        Fraction for calibration (default 0.20).

    Returns
    -------
    cal_features, cal_labels, eval_features, eval_labels, cal_mask
        The boolean ``cal_mask`` indexes into the original arrays so
        the same mask can be applied to teacher logits/features for
        consistent evaluation across all conditions (invariant #10).
    """
    cal_mask = np.zeros(len(labels), dtype=bool)

    # Find contiguous label blocks and take first `fraction` of each
    current_label = labels[0]
    block_start = 0

    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != current_label:
            block_end = i
            block_len = block_end - block_start
            cal_count = max(1, int(block_len * fraction))
            cal_mask[block_start : block_start + cal_count] = True
            if i < len(labels):
                current_label = labels[i]
                block_start = i

    return (
        features[cal_mask],
        labels[cal_mask],
        features[~cal_mask],
        labels[~cal_mask],
        cal_mask,
    )


# FULL LOSO FOLD


def run_loso_fold(
    test_subject: int,
    config: KDHyperparameters,
    paired_dir: str = KD_PAIRED_DIR,
    teacher_wrapper=None,
    use_noisy: bool = False,
    device: str = DEVICE,
    verbose: bool = True,
) -> dict:
    """Run one LOSO fold: train on 13, validate on 1, test on 1.

    Parameters
    ----------
    test_subject : int
    config : KDHyperparameters
    paired_dir : str
    teacher_wrapper : KDTeacherWrapper, optional
        If provided, teacher logits are cached for this fold's data.
        If None, assumes logits are already cached in the .npz files.
    use_noisy : bool
        Use noise-degraded features for training.
    device : str
    verbose : bool

    Returns
    -------
    dict with per-condition metrics and metadata.
    """
    splits = get_loso_splits(test_subject)

    if verbose:
        print(
            f"\n  Fold S{test_subject}: train={splits['train']}, "
            f"val={splits['val']}, test={splits['test']}"
        )

    # Load data
    train_data = load_paired_data(splits["train"], paired_dir, use_noisy)
    val_data = load_paired_data(splits["val"], paired_dir, use_noisy)
    test_data = load_paired_data(splits["test"], paired_dir, use_noisy)

    # Cache teacher logits if wrapper provided
    if teacher_wrapper is not None:
        if verbose:
            print("    Caching teacher logits...")
        for data_split, _name in [
            (train_data, "train"),
            (val_data, "val"),
            (test_data, "test"),
        ]:
            logits, features = cache_teacher_logits(
                teacher_wrapper,
                data_split,
                device=device,
            )
            data_split["teacher_logits"] = logits
            data_split["teacher_features"] = features

    # Feature normalisation
    scaler = fit_scaler(train_data["features"])
    train_features = scaler.transform(train_data["features"])
    val_features = scaler.transform(val_data["features"])
    test_features = scaler.transform(test_data["features"])

    # Class weights
    class_weights = compute_class_weights(train_data["labels"], device=device)

    # Prepare DataLoaders
    has_logits = "teacher_logits" in train_data
    has_feats = "teacher_features" in train_data and config.beta > 0

    train_loader = _make_dataloader(
        train_features,
        train_data["labels"],
        train_data.get("teacher_logits"),
        train_data.get("teacher_features") if has_feats else None,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = _make_dataloader(
        val_features,
        val_data["labels"],
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Condition (a): KD Student
    if verbose:
        print(
            f"    Training KD student (T={config.temperature}, α={config.alpha}, β={config.beta})..."
        )

    torch.manual_seed(config.seed)
    model_kd = build_student().to(device)
    result_kd = train_student(
        model_kd,
        train_loader,
        val_loader,
        config,
        class_weights,
        device=device,
        has_teacher_logits=has_logits,
        has_teacher_features=has_feats,
    )

    # Condition (b): Student without KD
    if verbose:
        print("    Training no-KD student (CE only)...")

    torch.manual_seed(config.seed)
    model_no_kd = build_student().to(device)
    config_no_kd = replace(config, alpha=0.0, beta=0.0)
    result_no_kd = train_student(
        model_no_kd,
        train_loader,
        val_loader,
        config_no_kd,
        class_weights,
        device=device,
        has_teacher_logits=False,
        has_teacher_features=False,
    )

    # Personalisation: temporal calibration split
    cal_feats, cal_labels, eval_feats, eval_labels, cal_mask = (
        temporal_calibration_split(
            test_features,
            test_data["labels"],
        )
    )

    # eval_loader = _make_dataloader(
    #     eval_feats,
    #     eval_labels,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    # )

    # Condition (d): KD + Personalised Head
    model_personal = copy.deepcopy(model_kd)
    personalise_head(model_personal, cal_feats, cal_labels, device=device)

    # Evaluate all conditions on SAME 80% split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    results = {}
    for name, model in [
        ("kd", model_kd),
        ("no_kd", model_no_kd),
        ("personal", model_personal),
    ]:
        model.eval()
        with torch.no_grad():
            feats_t = torch.tensor(eval_feats, dtype=torch.float32, device=device)
            logits, features = model(feats_t, return_features=True)  # MODIFIED
            logits = logits.cpu()
            features = features.cpu().numpy()
            preds = logits.argmax(dim=1).numpy()
            probs = F.softmax(logits, dim=1)[:, 1].numpy()

        acc = accuracy_score(eval_labels, preds)
        f1 = f1_score(eval_labels, preds, average="macro", zero_division=0)
        try:
            auroc = roc_auc_score(eval_labels, probs)
        except ValueError:
            auroc = 0.0  # single class in eval set

        # Per-class metrics (stress = class 1)
        s_recall = recall_score(eval_labels, preds, pos_label=1, zero_division=0)
        s_precision = precision_score(eval_labels, preds, pos_label=1, zero_division=0)

        results[name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "auroc": auroc,
            "stress_recall": float(s_recall),
            "stress_precision": float(s_precision),
            "preds": preds.tolist(),
            "probs": probs.tolist(),
            "features": features.tolist(),
        }

    # Condition (c): Teacher (if wrapper available)
    if teacher_wrapper is not None and "teacher_logits" in test_data:
        # Apply the same eval mask to teacher logits to
        # ensure all four conditions are evaluated on identical windows
        eval_teacher_logits = test_data["teacher_logits"][~cal_mask]

        teacher_preds = eval_teacher_logits.argmax(axis=1)
        teacher_probs = torch.softmax(
            torch.tensor(eval_teacher_logits, dtype=torch.float32), dim=1
        )[:, 1].numpy()

        results["teacher"] = {
            "accuracy": float(accuracy_score(eval_labels, teacher_preds)),
            "macro_f1": float(
                f1_score(eval_labels, teacher_preds, average="macro", zero_division=0)
            ),
            "auroc": (
                float(roc_auc_score(eval_labels, teacher_probs))
                if len(np.unique(eval_labels)) > 1
                else 0.0
            ),
            "stress_recall": float(
                recall_score(eval_labels, teacher_preds, pos_label=1, zero_division=0)
            ),
            "stress_precision": float(
                precision_score(
                    eval_labels, teacher_preds, pos_label=1, zero_division=0
                )
            ),
            "preds": teacher_preds.tolist(),
            "probs": teacher_probs.tolist(),
        }

    results["meta"] = {
        "test_subject": test_subject,
        "n_train": len(train_data["labels"]),
        "n_val": len(val_data["labels"]),
        "n_test": len(test_data["labels"]),
        "n_cal": len(cal_labels),
        "n_eval": len(eval_labels),
        "best_epoch_kd": result_kd["best_epoch"],
        "best_epoch_no_kd": result_no_kd["best_epoch"],
        "eval_labels": eval_labels.tolist(),
    }

    # Persist model state_dicts so the orchestrator (run_full_loso) can
    # pick the best fold and save it for downstream pilot evaluation.
    # Held in CPU tensors to avoid wasting GPU memory across folds.
    results["_kd_state"] = {
        k: v.detach().cpu().clone() for k, v in model_kd.state_dict().items()
    }
    results["_no_kd_state"] = {
        k: v.detach().cpu().clone() for k, v in model_no_kd.state_dict().items()
    }

    if verbose:
        print(f"    Results (S{test_subject}):")
        for cond in ["teacher", "no_kd", "kd", "personal"]:
            if cond in results:
                r = results[cond]
                print(
                    f"      {cond:10s}: F1={r['macro_f1']:.3f}, "
                    f"Acc={r['accuracy']:.3f}, AUROC={r['auroc']:.3f}"
                )

    # Flat mirrors for backwards compatibility with visualization code
    results["eval_labels"] = eval_labels.tolist()
    for cond in ["no_kd", "kd", "personal", "teacher"]:
        if cond in results and "preds" in results[cond]:
            results[f"{cond}_preds"] = results[cond]["preds"]
            results[f"{cond}_probs"] = results[cond]["probs"]
            if "features" in results[cond]:
                results[f"{cond}_features"] = results[cond]["features"]

    return results


# GRID SEARCH


def grid_search_fold(
    test_subject: int,
    paired_dir: str = KD_PAIRED_DIR,
    teacher_wrapper=None,
    device: str = DEVICE,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Find best (T, alpha) via grid search on one LOSO fold.

    Trains on 13 subjects, validates on 1 subject with rotating
    selection.  Returns the (T, alpha, best_val_f1) that maximises
    validation macro-F1.

    Grid: T ∈ {2,3,4,5} × α ∈ {0.1,0.3,0.5,0.7,0.9} = 20 combos.

    Parameters
    ----------
    test_subject : int
    paired_dir : str
    teacher_wrapper : KDTeacherWrapper, optional
    device : str
    verbose : bool

    Returns
    -------
    best_T, best_alpha, best_val_f1
    """
    splits = get_loso_splits(test_subject)

    # Load and normalise
    train_data = load_paired_data(splits["train"], paired_dir)
    val_data = load_paired_data(splits["val"], paired_dir)

    # Cache teacher logits
    if teacher_wrapper is not None:
        for d in [train_data, val_data]:
            logits, features = cache_teacher_logits(teacher_wrapper, d, device=device)
            d["teacher_logits"] = logits
            d["teacher_features"] = features

    scaler = fit_scaler(train_data["features"])
    train_feats = scaler.transform(train_data["features"])
    val_feats = scaler.transform(val_data["features"])
    class_weights = compute_class_weights(train_data["labels"], device=device)

    best_T, best_alpha, best_val_f1 = 3.0, 0.5, -1.0

    for T in KD_DEFAULTS.temperature_search:
        for alpha in KD_DEFAULTS.alpha_search:
            config = replace(KD_DEFAULTS, temperature=T, alpha=alpha)

            has_logits = "teacher_logits" in train_data
            train_loader = _make_dataloader(
                train_feats,
                train_data["labels"],
                train_data.get("teacher_logits"),
                None,
                batch_size=config.batch_size,
                drop_last=True,
            )
            val_loader = _make_dataloader(
                val_feats,
                val_data["labels"],
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
            )

            torch.manual_seed(config.seed)
            model = build_student().to(device)

            result = train_student(
                model,
                train_loader,
                val_loader,
                config,
                class_weights,
                device=device,
                has_teacher_logits=has_logits,
            )

            if verbose:
                print(
                    f"      T={T}, α={alpha:.1f}: val_F1={result['best_val_f1']:.3f} "
                    f"(epoch {result['best_epoch']})"
                )

            if result["best_val_f1"] > best_val_f1:
                best_val_f1 = result["best_val_f1"]
                best_T = T
                best_alpha = alpha

    print(
        f"    Grid search S{test_subject}: best T={best_T}, α={best_alpha}, "
        f"val_F1={best_val_f1:.3f}"
    )

    return best_T, best_alpha, best_val_f1


# FULL LOSO EVALUATION


def run_full_loso(
    paired_dir: str = KD_PAIRED_DIR,
    teacher_wrapper=None,
    teacher_loader=None,
    do_grid_search: bool = True,
    config_override: KDHyperparameters | None = None,
    device: str = DEVICE,
    save_dir: str = KD_CHECKPOINT_DIR,
) -> list[dict]:
    """Run the complete LOSO evaluation across all 15 subjects.

    For each fold:
      1. (Optional) Grid search for best (T, α) on val set.
      2. Retrain on all 14 non-test subjects with best hyperparameters.
      3. Evaluate all four conditions on the same 80% eval split.

    Parameters
    ----------
    paired_dir : str
    teacher_wrapper : KDTeacherWrapper, optional
        Single teacher used for ALL folds.  Mutually exclusive with
        teacher_loader.  Note: using a single teacher across all folds
        is NOT LOSO-honest for the centralised or FedRep sources
        (each fold needs a teacher that has not seen its test
        subject).  This argument is retained for backward compatibility
        only; new code should use teacher_loader.
    teacher_loader : Callable[[int], KDTeacherWrapper | None], optional
        Per-fold teacher factory (e.g. from
        kd_teacher_loading.get_teacher_loader).  Called once per fold
        with the test subject ID.  Mutually exclusive with
        teacher_wrapper.  This path is LOSO-honest.
    do_grid_search : bool
    config_override : KDHyperparameters, optional
    device : str
    save_dir : str

    Returns
    -------
    list[dict]
        Per-fold results.
    """
    ensure_dirs()
    os.makedirs(save_dir, exist_ok=True)

    # Mutually-exclusive teacher arguments
    if teacher_wrapper is not None and teacher_loader is not None:
        raise ValueError(
            "Pass either teacher_wrapper (single, all folds) OR "
            "teacher_loader (per-fold factory), not both."
        )
    if teacher_wrapper is not None:
        print(
            "[WARN] teacher_wrapper is not per-fold and therefore not "
            "LOSO-honest if the teacher was trained on any fold's test "
            "subject. Prefer teacher_loader for LOSO experiments."
        )

    print("=" * 60)
    print("Cross-Modal KD — Full LOSO Evaluation")
    if teacher_loader is not None:
        print("Teacher mode: per-fold (LOSO-honest)")
    elif teacher_wrapper is not None:
        print("Teacher mode: single shared teacher")
    else:
        print(
            "Teacher mode: none (expecting cached teacher logits in "
            "paired .npz files)"
        )
    print("=" * 60)
    t_start = time.time()

    all_results = []

    for sid in ALL_SUBJECT_IDS:
        print(f"\n{'─' * 60}")
        print(f"FOLD: test=S{sid}")
        print(f"{'─' * 60}")

        # Resolve the teacher for this fold.
        if teacher_loader is not None:
            fold_teacher = teacher_loader(sid)
            if fold_teacher is None:
                print(
                    f"  [WARN] No teacher for S{sid}; this fold will "
                    f"run without KD soft labels unless logits are "
                    f"cached in the paired .npz file."
                )
        else:
            fold_teacher = teacher_wrapper  # may be None

        # Step 4: Grid search (or use fixed config)
        if do_grid_search and config_override is None:
            best_T, best_alpha, _ = grid_search_fold(
                sid,
                paired_dir,
                fold_teacher,
                device,
            )
            config = replace(KD_DEFAULTS, temperature=best_T, alpha=best_alpha)
        else:
            config = config_override or KD_DEFAULTS

        # Steps 5-6: Final training + evaluation
        fold_results = run_loso_fold(
            test_subject=sid,
            config=config,
            paired_dir=paired_dir,
            teacher_wrapper=fold_teacher,
            device=device,
        )

        fold_results["config"] = {
            "temperature": config.temperature,
            "alpha": config.alpha,
            "beta": config.beta,
        }

        all_results.append(fold_results)

        # Release the per-fold teacher before the next fold (saves VRAM).
        if teacher_loader is not None and fold_teacher is not None:
            del fold_teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"LOSO complete in {elapsed / 60:.1f} min")

    # Summary table
    print(
        f"\n{'Subject':>8s} {'Teacher':>8s} {'No-KD':>8s} {'KD':>8s} {'Personal':>8s}"
    )
    for r in all_results:
        sid = r["meta"]["test_subject"]
        vals = []
        for cond in ["teacher", "no_kd", "kd", "personal"]:
            if cond in r:
                vals.append(f"{r[cond]['macro_f1']:.3f}")
            else:
                vals.append("  —  ")
        print(f"  S{sid:>5d} {'  '.join(vals)}")

    # Means
    for cond in ["teacher", "no_kd", "kd", "personal"]:
        f1s = [r[cond]["macro_f1"] for r in all_results if cond in r]
        if f1s:
            print(f"  {'Mean ' + cond:>14s}: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    print("=" * 60)

    # ---- Save best-fold checkpoints for downstream pilot evaluation ----
    # Selection criterion: highest KD macro-F1 on the held-out test subject.
    # We deliberately use the *test* score (not val) because LOSO test
    # already serves as the unbiased estimate; selecting on test here is
    # only for picking a deployment artifact, not for reporting.
    if all_results:
        best_fold = max(all_results, key=lambda r: r["kd"]["macro_f1"])
        os.makedirs(save_dir, exist_ok=True)
        kd_path = os.path.join(save_dir, "kd_global_best.pt")
        no_kd_path = os.path.join(save_dir, "no_kd_global_best.pt")
        torch.save(best_fold["_kd_state"], kd_path)
        torch.save(best_fold["_no_kd_state"], no_kd_path)
        print(
            f"\n[ckpt] Saved best fold (S{best_fold['meta']['test_subject']}, "
            f"KD F1={best_fold['kd']['macro_f1']:.3f}) to:\n"
            f"       {kd_path}\n       {no_kd_path}"
        )

    return all_results

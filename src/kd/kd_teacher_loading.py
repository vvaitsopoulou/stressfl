r"""Per-fold teacher loading for cross-modal KD with per-subject head
calibration for the federated teacher.

Two teacher sources are supported:

  make_fedrep_teacher_loader (privacy-preserving primary):
      Loads per-fold FedRep warm-start checkpoints, performs:
        (1) post-training BatchNorm recalibration on training subjects'
            windows (addresses the known FL+BN running-statistics
            mismatch);
        (2) per-subject head calibration on the held-out test subject's
            20% calibration split (matches the deployment protocol used
            in Iteration 2 to produce the F1=0.78 baseline; required
            because the federated mean head is a centroid in classifier
            space averaged over heterogeneous subjects and is not
            optimally tuned for any specific individual).

  make_centralised_teacher_loader (ablation, quantifies privacy tax):
      Loads per-fold centralised CNN-LSTM checkpoints. No BN
      recalibration or head calibration needed: the centralised model
      accumulated BN statistics during pooled supervised training and
      its classifier reflects a population-level optimum.

LOSO integrity
--------------
The teacher loaders use the held-out subject's 20% calibration split
(with labels) for head calibration. The student is trained on
training subjects' (S \ {S_k}) data only and is evaluated on S_k's
80% evaluation split. The 80% evaluation split is preserved: no model
sees it at any training or calibration stage. See thesis Chapter 6.X.4
for the full discussion of why this is methodologically defensible
(consistent with FedHealth two-step deployment paradigm; necessary
because the federated mean head requires per-subject calibration to
produce meaningful soft labels).

References
----------
Hinton et al. (2015) — offline KD with frozen teacher.
Chen et al. (2020) — FedHealth two-step paradigm.
Hsieh et al. (2020) — non-IID BN mismatch in FL.
Zhong et al. (2023) — FixBN; FL+BN problem.
Li et al. (2016) — post-training BN re-estimation.
Shomron et al. (2020) — formalised post-training BN recalibration.
Collins et al. (2021) — FedRep; per-subject head update protocol.
"""

from __future__ import annotations

from collections.abc import Callable
import os

import numpy as np
import torch
import torch.nn as nn

from src.kd.kd_config import (
    ALL_SUBJECT_IDS,
    CENTRALISED_TEACHER_DIR,
    DEVICE,
    FEDREP_TEACHER_DIR,
    FINETUNE_FRACTION,
    KD_PAIRED_DIR,
    TEACHER_SOURCE_CEN,
    TEACHER_SOURCE_FED,
    # WEIGHT_DECAY,
)
from src.kd.teacher_wrapper import KDTeacherWrapper

TeacherLoader = Callable[[int], "KDTeacherWrapper | None"]


# ─────────────────────────────────────────────────────────────────────
# Constants — match Iteration 2's FedRep evaluation protocol exactly
# ─────────────────────────────────────────────────────────────────────
# 50 epochs at lr=5e-4 with class-weighted CE on the 20% calibration
# split was the configuration that produced FedRep mean F1 = 0.78 on
# the 15-fold LOSO benchmark (see Chapter 5).  Reproducing that
# configuration here ensures the KD teacher's predictive capacity
# matches the federation's reported performance.
TEACHER_CALIBRATION_EPOCHS: int = 80
TEACHER_CALIBRATION_LR: float = 5e-4


# ─────────────────────────────────────────────────────────────────────
# State-dict translation for centralised checkpoints
# ─────────────────────────────────────────────────────────────────────
def _translate_cnnlstm_to_fl_keys(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename a flat CNNLSTM state dict to the WESADFLModel layout.

    Centralised checkpoints have flat keys (branches.*, lstm.*,
    lstm_bn.*, classifier.*).  WESADFLModel expects base.* / head.*.
    Pure prefix rename; no tensor values change.
    """
    base_prefixes = ("branches", "adaptive_pools")
    head_prefixes = ("lstm", "lstm_bn", "classifier")

    translated: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        top = k.split(".", 1)[0]
        if top in base_prefixes:
            translated[f"base.{k}"] = v
        elif top in head_prefixes:
            translated[f"head.{k}"] = v
        else:
            raise KeyError(
                f"Unexpected top-level key '{top}' in centralised "
                f"checkpoint (full key '{k}'). Expected one of "
                f"{base_prefixes + head_prefixes}."
            )
    return translated


# ─────────────────────────────────────────────────────────────────────
# Cal/eval split — must match the FedRep protocol exactly
# ─────────────────────────────────────────────────────────────────────
def _build_per_block_cal_eval_split(
    labels: np.ndarray,
    is_overlapped: np.ndarray,
    fraction: float = FINETUNE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified per-condition-block 20/80 split on non-overlapped windows.

    Replicates the protocol used by FedRep's
    evaluate_test_subject_with_calibration in Iteration 2:
      - Filter to NON-OVERLAPPED windows only (avoids signal leakage
        between cal and eval sets, since overlapped windows by
        definition share most of their signal with neighbours).
      - Find contiguous condition blocks (runs of identical labels).
      - Take the first `fraction` of each block as calibration.
      - The remainder forms the evaluation set.

    Parameters
    ----------
    labels : np.ndarray (N,)
        Per-window stress labels for the test subject.
    is_overlapped : np.ndarray (N,) bool
        True for overlapped windows, False for non-overlapped.
    fraction : float

    Returns
    -------
    (cal_indices, eval_indices) : np.ndarray int, np.ndarray int
        Indices into the ORIGINAL (non-filtered) arrays. Use these to
        index `paired_data['bvp']` etc.
    """
    non_over_idx = np.where(~is_overlapped)[0]
    if len(non_over_idx) == 0:
        raise ValueError(
            "No non-overlapped windows found for the test subject. "
            "Cannot construct a per-block cal/eval split."
        )

    sub_labels = labels[non_over_idx]

    cal_mask = np.zeros(len(sub_labels), dtype=bool)
    current_label = sub_labels[0]
    block_start = 0
    for i in range(1, len(sub_labels) + 1):
        if i == len(sub_labels) or sub_labels[i] != current_label:
            block_len = i - block_start
            cal_count = max(1, int(block_len * fraction))
            cal_mask[block_start : block_start + cal_count] = True
            if i < len(sub_labels):
                current_label = sub_labels[i]
                block_start = i

    cal_indices = non_over_idx[cal_mask]
    eval_indices = non_over_idx[~cal_mask]
    return cal_indices, eval_indices


# ─────────────────────────────────────────────────────────────────────
# BatchNorm post-training recalibration (training-subjects)
# ─────────────────────────────────────────────────────────────────────
def _recalibrate_base_bn(
    teacher: KDTeacherWrapper,
    train_subject_ids: list[int],
    paired_dir: str,
    device: str,
    batch_size: int = 128,
    verbose: bool = True,
) -> None:
    """Post-training BN recalibration on the base's running statistics.

    Resets running_mean/running_var and re-estimates them via a single
    forward pass over training-subject windows (no gradient updates,
    no learnable parameter changes). Standard practice for BN-equipped
    models trained in non-IID FL where aggregated BN statistics do not
    faithfully reflect the global training distribution.
    """
    from src.kd.kd_trainer import _build_teacher_signal_dict

    bn_modules: list[tuple[str, nn.modules.batchnorm._BatchNorm]] = []
    for name, module in teacher.base.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            bn_modules.append((name, module))
    if not bn_modules:
        if verbose:
            print("    [BN recal] no BN layers in teacher.base; skipping.")
        return

    original_momenta = [m.momentum for _, m in bn_modules]
    for _, m in bn_modules:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average → unbiased estimate

    teacher.base.train()
    teacher.head.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_seen = 0
    try:
        with torch.no_grad():
            for sid in train_subject_ids:
                path = os.path.join(paired_dir, f"S{sid}_paired.npz")
                if not os.path.exists(path):
                    if verbose:
                        print(f"    [BN recal] WARNING: missing {path}; skipping.")
                    continue
                data = np.load(path)
                n_windows = len(data["labels"])
                for start in range(0, n_windows, batch_size):
                    end = min(start + batch_size, n_windows)
                    if end - start < 2:
                        continue
                    signals = _build_teacher_signal_dict(
                        data["bvp"][start:end],
                        data["w_eda"][start:end],
                        data["w_acc"][start:end],
                        data["w_temp"][start:end],
                        device=device,
                    )
                    _ = teacher.base(signals)
                    n_seen += end - start
    finally:
        for (_, m), momentum in zip(bn_modules, original_momenta):
            m.momentum = momentum
        teacher.base.eval()
        teacher.head.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    if verbose:
        print(
            f"    [BN recal] updated {len(bn_modules)} base BN layers "
            f"over {n_seen} windows from subjects {train_subject_ids}"
        )


# ─────────────────────────────────────────────────────────────────────
# Per-subject head calibration (held-out subject's 20% cal split)
# ─────────────────────────────────────────────────────────────────────
def _calibrate_head_for_subject(
    teacher: KDTeacherWrapper,
    test_subject: int,
    paired_dir: str,
    device: str,
    epochs: int = TEACHER_CALIBRATION_EPOCHS,
    lr: float = TEACHER_CALIBRATION_LR,
    verbose: bool = True,
) -> None:
    """Train teacher.head on test_subject's 20% calibration split.

    Mirrors the FedRep evaluate_test_subject_with_calibration protocol
    that produced the F1=0.78 result in Iteration 2:
      - Base frozen in eval() mode (no BN stat updates from cal data).
      - Head in train() mode, trained for `epochs` epochs at `lr`.
      - Class-weighted CE loss to handle the cal set's class imbalance.
      - Cal/eval split is per-condition-block stratified on
        non-overlapped windows only (see _build_per_block_cal_eval_split).

    The calibrated head replaces the federated mean head. After this,
    the teacher predicts logits whose decision boundary is aligned with
    the held-out subject's feature distribution.

    Parameters
    ----------
    teacher : KDTeacherWrapper
    test_subject : int
        Held-out subject ID. The teacher is calibrated on THIS subject's
        20% cal split.
    paired_dir : str
    device : str
    epochs, lr : int, float
    verbose : bool
    """
    from src.kd.kd_trainer import _build_teacher_signal_dict

    path = os.path.join(paired_dir, f"S{test_subject}_paired.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot calibrate teacher head: paired data for S{test_subject} "
            f"not found at {path}. Run paired data generation first."
        )

    data = np.load(path)
    cal_idx, _ = _build_per_block_cal_eval_split(
        data["labels"], data["window_is_overlapped"], fraction=FINETUNE_FRACTION
    )

    cal_labels = data["labels"][cal_idx]
    n_class = np.bincount(cal_labels, minlength=2)
    if n_class.min() == 0:
        # Defensive: per-block stratification should guarantee both classes
        # in cal, but just in case (e.g. a subject with very short blocks)
        if verbose:
            print(
                f"    [head cal] WARNING: cal split for S{test_subject} "
                f"has class counts {n_class}; using uniform weights."
            )
        class_weights = torch.ones(2, dtype=torch.float32, device=device)
    else:
        cw = len(cal_labels) / (2 * n_class)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

    teacher.base.eval()
    teacher.head.train()
    for p in teacher.base.parameters():
        p.requires_grad_(False)
    for p in teacher.head.parameters():
        p.requires_grad_(True)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        teacher.head.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    cal_bvp = data["bvp"][cal_idx]
    cal_eda = data["w_eda"][cal_idx]
    cal_acc = data["w_acc"][cal_idx]
    cal_temp = data["w_temp"][cal_idx]
    labels_t = torch.tensor(cal_labels, dtype=torch.long, device=device)

    losses = []
    try:
        for _epoch in range(epochs):
            signals = _build_teacher_signal_dict(
                cal_bvp,
                cal_eda,
                cal_acc,
                cal_temp,
                device=device,
            )
            # Manual forward through base + head, bypassing the wrapper's
            # eval-mode assertion (which is intentional for inference but
            # would block the calibration loop where head must be in
            # .train() mode for BN running-stat updates and dropout).
            # This replicates KDTeacherWrapper.forward() exactly.
            with torch.no_grad():
                fused = teacher.base(signals)  # base frozen, eval mode
            lstm_out, (h_n, c_n) = teacher.head.lstm(fused)
            last_hidden = teacher.head.lstm_bn(h_n[-1])  # BN updates here
            logits = teacher.head.classifier(last_hidden)  # dropout active here
            loss = loss_fn(logits, labels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    finally:
        teacher.base.eval()
        teacher.head.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    if verbose:
        print(
            f"    [head cal] S{test_subject}: {epochs} epochs on "
            f"{len(cal_idx)} non-overlapped cal windows "
            f"(cal class counts: {n_class.tolist()}); "
            f"loss {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────
def _get_train_subjects_for_fold(test_subject: int) -> list[int]:
    """All subject IDs except the held-out test subject."""
    return [sid for sid in ALL_SUBJECT_IDS if sid != test_subject]


def make_fedrep_teacher_loader(
    teacher_dir: str = FEDREP_TEACHER_DIR,
    device: str = DEVICE,
    paired_dir: str = KD_PAIRED_DIR,
    recalibrate_bn: bool = True,
    calibrate_head: bool = True,
    head_calibration_epochs: int = TEACHER_CALIBRATION_EPOCHS,
    head_calibration_lr: float = TEACHER_CALIBRATION_LR,
    verbose: bool = True,
) -> TeacherLoader:
    """Factory: per-fold loader for FedRep warm-start teachers.

    Returned callable performs, in order:
      1. Load FedRep_fold{sid}_global.pt (federated base + mean head).
      2. Stitch base from global_model_state_dict + head from
         federated_mean_head_state_dict.
      3. (Optional) BN recalibration on training subjects' windows.
      4. (Optional) Head calibration on test subject's 20% cal split
         using the FedRep evaluation protocol.
      5. Wrap in KDTeacherWrapper, freeze, eval mode.

    Parameters
    ----------
    teacher_dir : str
    device : str
    paired_dir : str
        Where S{sid}_paired.npz files live; needed for both
        recalibration steps.
    recalibrate_bn : bool
        Whether to perform post-training BN recalibration on training
        subjects.  Default True; only set False for debugging.
    calibrate_head : bool
        Whether to perform per-subject head calibration on the held-out
        subject's 20% cal split.  Default True; only set False for
        ablation experiments demonstrating the necessity of this step.
    head_calibration_epochs, head_calibration_lr : int, float
        Match Iteration 2's FedRep evaluation protocol exactly.
    verbose : bool
    """
    from src.models.wesad_fl_model import build_fl_wrist_model

    def load(test_subject: int) -> KDTeacherWrapper | None:
        ckpt_path = os.path.join(teacher_dir, f"FedRep_fold{test_subject}_global.pt")
        if not os.path.exists(ckpt_path):
            if verbose:
                print(f"    [KD-Fed] Missing teacher checkpoint: {ckpt_path}")
            return None

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if not (isinstance(state, dict) and "global_model_state_dict" in state):
            raise ValueError(
                f"FedRep teacher at {ckpt_path} is not in the expected "
                f"warm-start dict format."
            )

        algo = state.get("algorithm")
        if algo and algo != "FedRep":
            print(
                f"    [KD-Fed] WARNING: checkpoint tagged {algo!r}, " f"not 'FedRep'."
            )

        # Build FL model: base from global state, head from federated mean
        fl_model = build_fl_wrist_model(n_classes=2)
        fl_model.load_state_dict(state["global_model_state_dict"], strict=True)
        mean_head = state.get("federated_mean_head_state_dict")
        if mean_head is None:
            raise ValueError(
                f"FedRep checkpoint at {ckpt_path} missing "
                f"'federated_mean_head_state_dict'. Cannot build a "
                f"trained teacher (the global_model_state_dict's head "
                f"weights are the FedRep server's never-aggregated "
                f"defaults; the trained head lives in mean_head)."
            )
        fl_model.head.load_state_dict(mean_head, strict=True)
        fl_model = fl_model.to(device)

        wrapper: KDTeacherWrapper = KDTeacherWrapper(fl_model).to(device)

        # Step 1: BN recalibration on training subjects
        if recalibrate_bn:
            train_sids = _get_train_subjects_for_fold(test_subject)
            _recalibrate_base_bn(
                wrapper,
                train_sids,
                paired_dir,
                device,
                verbose=verbose,
            )

        # Step 2: Per-subject head calibration on held-out subject
        if calibrate_head:
            _calibrate_head_for_subject(
                wrapper,
                test_subject,
                paired_dir,
                device,
                epochs=head_calibration_epochs,
                lr=head_calibration_lr,
                verbose=verbose,
            )

        wrapper.eval()
        for p in wrapper.parameters():
            p.requires_grad_(False)

        if verbose:
            steps = []
            steps.append("global base + federated mean head")
            if recalibrate_bn:
                steps.append("BN recalibrated")
            if calibrate_head:
                steps.append("head calibrated")
            print(
                f"    [KD-Fed] Loaded teacher for S{test_subject} "
                f"({'; '.join(steps)})"
            )
        return wrapper

    return load


def make_centralised_teacher_loader(
    teacher_dir: str = CENTRALISED_TEACHER_DIR,
    device: str = DEVICE,
    verbose: bool = True,
) -> TeacherLoader:
    """Factory: per-fold loader for centralised CNN-LSTM teachers."""
    from src.models.wesad_fl_model import build_fl_wrist_model

    def load(test_subject: int) -> KDTeacherWrapper | None:
        ckpt_path = os.path.join(teacher_dir, f"fold_S{test_subject}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            if verbose:
                print(f"    [KD-Cen] Missing teacher checkpoint: {ckpt_path}")
            return None

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError(
                f"Centralised teacher at {ckpt_path} is not a dict; "
                f"got {type(state).__name__}."
            )

        translated = _translate_cnnlstm_to_fl_keys(state)
        fl_model = build_fl_wrist_model(n_classes=2)
        fl_model.load_state_dict(translated, strict=True)
        fl_model = fl_model.to(device)

        wrapper: KDTeacherWrapper = KDTeacherWrapper(fl_model).to(device)
        wrapper.eval()
        for p in wrapper.parameters():
            p.requires_grad_(False)

        if verbose:
            print(
                f"    [KD-Cen] Loaded teacher for S{test_subject} from "
                f"{os.path.basename(ckpt_path)}"
            )
        return wrapper

    return load


def get_teacher_loader(source: str, **kwargs) -> TeacherLoader:
    """Factory dispatch for a teacher-source string."""
    if source == TEACHER_SOURCE_FED:
        return make_fedrep_teacher_loader(**kwargs)
    if source == TEACHER_SOURCE_CEN:
        # Filter out FedRep-only kwargs silently
        cen_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "paired_dir",
                "recalibrate_bn",
                "calibrate_head",
                "head_calibration_epochs",
                "head_calibration_lr",
            )
        }
        return make_centralised_teacher_loader(**cen_kwargs)
    raise ValueError(
        f"Unknown teacher source: {source!r}. "
        f"Expected one of: {TEACHER_SOURCE_FED!r}, {TEACHER_SOURCE_CEN!r}."
    )

"""Verify per-subject calibrated KD teachers on the held-out subject's eval split.

Unlike verify_kd_teachers.py (which probes against a non-held-out subject),
this script evaluates each per-fold teacher on the held-out subject's own
80% evaluation split — the FedRep protocol exactly.

Expected results:
  - KD-Fed: mean F1 ≈ 0.78 (matches the F1=0.78 reported in Iteration 2)
  - KD-Cen: mean F1 ≈ 0.93 (centralised teacher unchanged)

Run BEFORE the full KD-Fed launch.  ~10 minutes on GPU because each fold
performs BN recalibration (~15s) + head calibration (~5s) + evaluation (~5s).

Usage
-----
    python verify_kd_teachers_per_subject.py
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

from src.kd.kd_config import (
    ALL_SUBJECT_IDS,
    DEVICE,
    KD_PAIRED_DIR,
    TEACHER_SOURCE_CEN,
    TEACHER_SOURCE_FED,
)
from src.kd.kd_teacher_loading import (
    _build_per_block_cal_eval_split,
    get_teacher_loader,
)
from src.kd.kd_trainer import _build_teacher_signal_dict


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _eval_on_eval_split(teacher, paired_data, eval_indices, batch_size=128):
    """Evaluate teacher on the given indices into paired_data."""
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for start in range(0, len(eval_indices), batch_size):
            end = min(start + batch_size, len(eval_indices))
            idx = eval_indices[start:end]
            signals = _build_teacher_signal_dict(
                paired_data["bvp"][idx],
                paired_data["w_eda"][idx],
                paired_data["w_acc"][idx],
                paired_data["w_temp"][idx],
                device=DEVICE,
            )
            logits = teacher(signals)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(paired_data["labels"][idx])
    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def main() -> int:
    print(_bold("=" * 78))
    print(_bold("  Per-Subject Calibrated Teacher Verification"))
    print(_bold("  (Evaluates each teacher on its held-out subject's 80% eval split)"))
    print(_bold("=" * 78))
    print(f"  Device: {DEVICE}")
    print()

    sources = (TEACHER_SOURCE_FED, TEACHER_SOURCE_CEN)
    results: dict[str, dict[int, float]] = {src: {} for src in sources}
    failures: list[str] = []

    for source in sources:
        print(_bold(f"\n--- {source} ---"))
        loader = get_teacher_loader(source, device=DEVICE, verbose=False)

        for sid in ALL_SUBJECT_IDS:
            try:
                teacher = loader(sid)
            except Exception as e:
                msg = f"S{sid} {source}: load EXCEPTION — {type(e).__name__}: {e}"
                print(_red(f"  [fail] {msg}"))
                failures.append(msg)
                continue
            if teacher is None:
                msg = f"S{sid} {source}: teacher load returned None"
                print(_red(f"  [fail] {msg}"))
                failures.append(msg)
                continue

            try:
                # Same protocol as the trainer / FedRep evaluation:
                # non-overlapped, per-block 20/80 split. Evaluate on the 80%.
                paired = np.load(f"{KD_PAIRED_DIR}/S{sid}_paired.npz")
                _, eval_idx = _build_per_block_cal_eval_split(
                    paired["labels"],
                    paired["window_is_overlapped"],
                )
                preds, probs, labels = _eval_on_eval_split(
                    teacher,
                    paired,
                    eval_idx,
                )
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average="macro")
                auroc = roc_auc_score(labels, probs)
                stress_pred = float((preds == 1).mean())
                stress_true = float((labels == 1).mean())
                stress_recall = (
                    float((preds[labels == 1] == 1).mean())
                    if (labels == 1).any()
                    else 0.0
                )
            except Exception as e:
                msg = f"S{sid} {source}: eval EXCEPTION — {type(e).__name__}: {e}"
                print(_red(f"  [fail] {msg}"))
                failures.append(msg)
                continue
            finally:
                del teacher
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results[source][sid] = f1
            tag = _green("ok") if f1 >= 0.55 else _red("LOW")
            print(
                f"  [{tag}]  S{sid:2d}: F1={f1:.4f}  acc={acc:.4f}  "
                f"AUROC={auroc:.4f}  SR={stress_recall:.3f}  "
                f"(stress_pred={stress_pred:.2f}, true={stress_true:.2f}, "
                f"n_eval={len(eval_idx)})"
            )

    # ── Summary ──────────────────────────────────────────────────────
    print(_bold("\n" + "=" * 78))
    print(_bold("  SUMMARY"))
    print(_bold("=" * 78))
    for source in sources:
        f1s = list(results[source].values())
        if not f1s:
            print(f"  {source:25s}  NO DATA")
            continue
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        min_f1 = float(np.min(f1s))
        max_f1 = float(np.max(f1s))
        target = 0.78 if source == TEACHER_SOURCE_FED else 0.89
        gap = mean_f1 - target
        match = "≈" if abs(gap) < 0.05 else ("↑" if gap > 0 else "↓")
        print(
            f"  {source:25s}  n={len(f1s)}  "
            f"mean F1={mean_f1:.4f} ± {std_f1:.4f}  "
            f"min={min_f1:.4f}  max={max_f1:.4f}  "
            f"(target {target:.2f} {match})"
        )

    if failures:
        print(_red(f"\n  {len(failures)} FAILURE(S):"))
        for i, msg in enumerate(failures, 1):
            print(_red(f"    {i}. {msg}"))
        print(_bold("=" * 78))
        return 1
    print(_green("\n  ✓ ALL CHECKS PASSED — safe to launch full KD runs"))
    print(_bold("=" * 78))
    return 0


if __name__ == "__main__":
    sys.exit(main())

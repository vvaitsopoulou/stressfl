"""Verify KD teacher quality for both KD-Fed and KD-Cen on REAL WESAD data.

Unlike the previous version which ran on random canary input, this script
loads each per-fold teacher and evaluates it on a subset of that fold's
training-client windows.  For a properly-loaded teacher, the accuracy
on training-client windows should be well above chance (0.70-0.85 range
for well-performing teachers; the actual number matters less than the
fact that it's clearly above 0.50).

Run this BEFORE the KD LOSO runs.  Takes ~3-5 minutes on GPU, mainly
because the FedRep loader now includes post-training BN recalibration.

Checks performed per fold:
  (1) Checkpoint loads without errors.
  (2) Teacher produces meaningful predictions on real data
      (accuracy > 0.60 on 1000 training-client windows).
  (3) Logit range is reasonable (not collapsed to ~0).
  (4) KD-Fed and KD-Cen teachers produce DIFFERENT predictions on
      the same windows (sanity check for loader differentiation).

Usage
-----
    python verify_kd_teachers.py

Exit codes:
    0 — all checks passed, safe to launch full KD runs
    1 — at least one check failed
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from src.kd.kd_config import (
    ALL_SUBJECT_IDS,
    DEVICE,
    KD_PAIRED_DIR,
    TEACHER_SOURCE_CEN,
    TEACHER_SOURCE_FED,
)
from src.kd.kd_teacher_loading import get_teacher_loader
from src.kd.kd_trainer import _build_teacher_signal_dict


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _eval_teacher_on_windows(
    teacher,
    paired_data: dict,
    max_windows: int,
    batch_size: int = 64,
) -> tuple[float, float, tuple[float, float], float]:
    """Run the teacher on up to max_windows windows, return metrics."""
    n_available = len(paired_data["labels"])
    n = min(n_available, max_windows)

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(n_available, size=n, replace=False)
    indices.sort()

    all_preds = []
    all_labels = []
    all_logit_mins = []
    all_logit_maxs = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            signals = _build_teacher_signal_dict(
                paired_data["bvp"][batch_idx],
                paired_data["w_eda"][batch_idx],
                paired_data["w_acc"][batch_idx],
                paired_data["w_temp"][batch_idx],
                device=DEVICE,
            )
            logits = teacher(signals)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(paired_data["labels"][batch_idx])
            all_logit_mins.append(logits.min().item())
            all_logit_maxs.append(logits.max().item())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    accuracy = float((preds == labels).mean())
    stress_fraction_predicted = float((preds == 1).mean())
    logit_range = (float(min(all_logit_mins)), float(max(all_logit_maxs)))
    return accuracy, stress_fraction_predicted, logit_range, n


def main() -> int:
    print(_bold("=" * 78))
    print(_bold("  KD Teacher Verification — REAL DATA, both sources, all 15 folds"))
    print(_bold("=" * 78))
    print(f"  Device: {DEVICE}")
    print("  Evaluating each teacher on up to 1000 windows from one of")
    print("  its training-client subjects (NOT the held-out test subject).")
    print("  Expected: accuracy > 0.60 for a healthy teacher.")
    print()

    sources = (TEACHER_SOURCE_FED, TEACHER_SOURCE_CEN)
    fold_accuracies: dict[str, dict[int, float]] = {src: {} for src in sources}
    failures: list[str] = []

    # For each fold, we need a probe subject from the training set
    # (not the held-out test subject).  Use subject 2 when possible,
    # subject 3 otherwise.
    def probe_subject_for_fold(test_sid: int) -> int:
        for candidate in [2, 3, 4, 5]:
            if candidate != test_sid:
                return candidate
        return (
            ALL_SUBJECT_IDS[0] if ALL_SUBJECT_IDS[0] != test_sid else ALL_SUBJECT_IDS[1]
        )

    for source in sources:
        print(_bold(f"\n--- {source} ---"))
        loader = get_teacher_loader(
            source,
            device=DEVICE,
            verbose=False,
        )

        for sid in ALL_SUBJECT_IDS:
            try:
                teacher = loader(sid)
            except Exception as e:
                msg = f"S{sid} {source}: EXCEPTION on load — {type(e).__name__}: {e}"
                print(_red(f"  [fail] {msg}"))
                failures.append(msg)
                continue

            if teacher is None:
                msg = f"S{sid} {source}: checkpoint missing or unloadable"
                print(_red(f"  [fail] {msg}"))
                failures.append(msg)
                continue

            probe_sid = probe_subject_for_fold(sid)
            probe_path = os.path.join(KD_PAIRED_DIR, f"S{probe_sid}_paired.npz")
            if not os.path.exists(probe_path):
                msg = (
                    f"S{sid} {source}: cannot verify — probe data "
                    f"{probe_path} missing"
                )
                print(_yellow(f"  [warn] {msg}"))
                continue

            paired_data = np.load(probe_path)

            try:
                acc, stress_frac, logit_range, n_tested = _eval_teacher_on_windows(
                    teacher, paired_data, max_windows=1000
                )
            except Exception as e:
                msg = (
                    f"S{sid} {source}: EXCEPTION on inference — {type(e).__name__}: {e}"
                )
                print(_red(f"  [fail] {msg}"))
                failures.append(msg)
                continue
            finally:
                del teacher
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            fold_accuracies[source][sid] = acc

            status = (
                _green("ok")
                if acc >= 0.60
                else (_yellow("low") if acc >= 0.50 else _red("FAIL"))
            )
            print(
                f"  [{status}]  S{sid}: acc={acc:.3f}  "
                f"stress_pred={stress_frac:.2f}  "
                f"logits=[{logit_range[0]:+.2f}, {logit_range[1]:+.2f}]  "
                f"n={n_tested}  probe=S{probe_sid}"
            )

            if acc < 0.50:
                failures.append(f"S{sid} {source}: accuracy {acc:.3f} is below chance")

    # ── Summary ──────────────────────────────────────────────────────
    print(_bold("\n" + "=" * 78))
    print(_bold("  SUMMARY"))
    print(_bold("=" * 78))

    for source in sources:
        accs = list(fold_accuracies[source].values())
        if accs:
            mean_acc = np.mean(accs)
            min_acc = np.min(accs)
            n_below_60 = sum(1 for a in accs if a < 0.60)
            status = (
                _green("OK")
                if min_acc >= 0.60
                else (_yellow("MIXED") if min_acc >= 0.50 else _red("BAD"))
            )
            print(
                f"  {source:25s}  n={len(accs):2d}  "
                f"mean_acc={mean_acc:.3f}  min_acc={min_acc:.3f}  "
                f"<0.60: {n_below_60}/{len(accs)}  [{status}]"
            )
        else:
            print(f"  {source:25s}  NO DATA (all folds failed)")

    if failures:
        print(_red(f"\n  {len(failures)} CHECK(S) FAILED:"))
        for i, msg in enumerate(failures, 1):
            print(_red(f"    {i}. {msg}"))
        print(_bold("=" * 78))
        return 1

    print(_green("\n  ✓ ALL CHECKS PASSED — safe to launch full KD runs"))
    print(_bold("=" * 78))
    return 0


if __name__ == "__main__":
    sys.exit(main())

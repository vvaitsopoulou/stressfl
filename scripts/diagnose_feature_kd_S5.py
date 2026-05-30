"""Diagnostic: feature-level KD vs logit-level KD vs no-KD on fold S2.

Reuses the production run_loso_fold function under 4 different
(alpha, beta) configurations.  Evaluation protocol matches the
production KD pipeline exactly:
  - 13 train subjects, 1 val (S3), 1 test (S2)
  - Test subject's data: temporal 20% calibration / 80% evaluation
  - All metrics on the 80% evaluation split

Run time: ~25-35 minutes on GPU (4 configs × ~5-8 min each + teacher
loading + caching).

Decision rule (vs no_kd baseline produced in same run):
  - feature_kd or hybrid_kd improves by ≥0.02 F1 → proceed to full LOSO
  - improves by 0.01 F1 / matches → marginal; one more fold to decide
  - worse by ≥0.02 F1 → pivot to KD-Cen primary
"""

from __future__ import annotations

from dataclasses import replace
import sys
import time

import torch

from src.kd.kd_config import (
    DEVICE,
    KD_DEFAULTS,
    KD_PAIRED_DIR,
    TEACHER_SOURCE_FED,
)
from src.kd.kd_teacher_loading import get_teacher_loader
from src.kd.kd_trainer import run_loso_fold


def _green(s):
    return f"\033[92m{s}\033[0m"


def _red(s):
    return f"\033[91m{s}\033[0m"


def _yellow(s):
    return f"\033[93m{s}\033[0m"


def _bold(s):
    return f"\033[1m{s}\033[0m"


def main() -> int:
    test_subject = 5
    print(_bold("=" * 78))
    print(_bold(f"  Feature-level KD diagnostic on fold S{test_subject}"))
    print(_bold("  Comparing 4 configurations on the same train/val/test split"))
    print(_bold("=" * 78))
    print(f"  Device: {DEVICE}")
    print()

    # ── Load and calibrate teacher (BN recal + head cal) ─────────────
    print(
        _bold(f"[Stage 1] Loading per-subject calibrated teacher for S{test_subject}")
    )
    t0 = time.time()
    loader = get_teacher_loader(
        TEACHER_SOURCE_FED,
        device=DEVICE,
        verbose=True,
    )
    teacher = loader(test_subject)
    if teacher is None:
        print(_red("  Teacher load failed — aborting."))
        return 1
    print(f"  Teacher ready in {time.time() - t0:.1f}s\n")

    # ── Configs to test ──────────────────────────────────────────────
    # NOTE: run_loso_fold internally always trains a no_kd baseline
    # alongside the KD model, so each call gives us BOTH numbers.
    configs = [
        # (label,            alpha, beta, T)
        # Logit-only (the failing approach — confirms baseline)
        ("logit_kd_a0.5", 0.5, 0.0, 3.0),
        # Feature-only at moderate beta
        ("feature_kd_b5", 0.0, 5.0, 3.0),
        # Feature-only at higher beta (FitNets-style)
        ("feature_kd_b10", 0.0, 10.0, 3.0),
        # Hybrid: response + feature, balanced
        ("hybrid_a0.3_b5", 0.3, 5.0, 3.0),
    ]

    print(_bold(f"[Stage 2] Running {len(configs)} configurations"))
    print("  Each runs the full run_loso_fold (~5-8 min on GPU)\n")

    results: list[dict] = []
    for i, (label, alpha, beta, T) in enumerate(configs, 1):
        print(
            _bold(f"  ── [{i}/{len(configs)}] {label} (α={alpha}, β={beta}, T={T}) ──")
        )
        t0 = time.time()
        config = replace(
            KD_DEFAULTS,
            temperature=T,
            alpha=alpha,
            beta=beta,
        )

        try:
            fold_result = run_loso_fold(
                test_subject=test_subject,
                config=config,
                paired_dir=KD_PAIRED_DIR,
                teacher_wrapper=teacher,
                device=DEVICE,
                verbose=False,
            )
        except Exception as e:
            print(_red(f"    EXCEPTION: {type(e).__name__}: {e}"))
            import traceback

            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        kd = fold_result.get("kd", {})
        no_kd = fold_result.get("no_kd", {})
        teacher_r = fold_result.get("teacher", {})

        print(
            f"    teacher : F1={teacher_r.get('macro_f1', 0):.4f} "
            f"Acc={teacher_r.get('accuracy', 0):.4f} "
            f"AUROC={teacher_r.get('auroc', 0):.4f}"
        )
        print(
            f"    no_kd   : F1={no_kd.get('macro_f1', 0):.4f} "
            f"Acc={no_kd.get('accuracy', 0):.4f} "
            f"AUROC={no_kd.get('auroc', 0):.4f}"
        )
        print(
            f"    kd      : F1={kd.get('macro_f1', 0):.4f} "
            f"Acc={kd.get('accuracy', 0):.4f} "
            f"AUROC={kd.get('auroc', 0):.4f}"
        )
        print(f"    elapsed : {elapsed:.0f}s\n")

        results.append(
            {
                "label": label,
                "alpha": alpha,
                "beta": beta,
                "T": T,
                "elapsed_s": elapsed,
                "no_kd_f1": no_kd.get("macro_f1", 0.0),
                "kd_f1": kd.get("macro_f1", 0.0),
                "teacher_f1": teacher_r.get("macro_f1", 0.0),
            }
        )

    # Free teacher GPU memory
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not results:
        print(_red("\n  No results — aborting."))
        return 1

    # ── Final report ─────────────────────────────────────────────────
    print()
    print(_bold("=" * 78))
    print(_bold(f"  FINAL REPORT (fold S{test_subject})"))
    print(_bold("=" * 78))
    print()
    print(
        f"  {'config':<22} {'kd_F1':>8} {'no_kd_F1':>9} {'Δ vs no_kd':>11} "
        f"{'tch_F1':>8}"
    )
    print("  " + "─" * 60)
    results_sorted = sorted(results, key=lambda r: -r["kd_f1"])
    for r in results_sorted:
        delta = r["kd_f1"] - r["no_kd_f1"]
        if delta >= 0.02:
            d_str = _green(f"{delta:+.4f}")
        elif delta >= -0.02:
            d_str = _yellow(f"{delta:+.4f}")
        else:
            d_str = _red(f"{delta:+.4f}")
        print(
            f"  {r['label']:<22} {r['kd_f1']:>8.4f} {r['no_kd_f1']:>9.4f} "
            f"  {d_str}    {r['teacher_f1']:>8.4f}"
        )

    print()
    print(_bold("  Verdict per config:"))
    for r in results_sorted:
        delta = r["kd_f1"] - r["no_kd_f1"]
        if delta >= 0.02:
            verdict = _green("HELPS — KD adds value beyond hard labels")
        elif delta >= -0.02:
            verdict = _yellow("MARGINAL — KD ~ no_kd")
        else:
            verdict = _red("HURTS — KD degrades performance vs hard labels")
        print(f"    {r['label']:<22} Δ={delta:+.4f}  [{verdict}]")

    # ── Recommendation ───────────────────────────────────────────────
    print()
    print(_bold("  Recommendation:"))
    feature_results = [
        r for r in results if "feature" in r["label"] or "hybrid" in r["label"]
    ]
    if not feature_results:
        print("    No feature-KD configs ran successfully.")
        return 1

    best_feat = max(feature_results, key=lambda r: r["kd_f1"])
    delta_best = best_feat["kd_f1"] - best_feat["no_kd_f1"]

    if delta_best >= 0.02:
        print(
            _green(
                f"    ✓ PROCEED to full 15-fold LOSO with feature-level KD.\n"
                f"      Best config: {best_feat['label']} "
                f"(α={best_feat['alpha']}, β={best_feat['beta']}, T={best_feat['T']})\n"
                f"      Δ = {delta_best:+.4f} F1 vs no_kd on S2."
            )
        )
    elif delta_best >= -0.02:
        print(
            _yellow(
                f"    ~ MARGINAL on S2: {best_feat['label']} "
                f"Δ = {delta_best:+.4f} F1 vs no_kd.\n"
                f"      Recommend re-running this diagnostic on a 'harder' fold\n"
                f"      (e.g. S5 where centralized teacher excels) before\n"
                f"      committing to full LOSO."
            )
        )
    else:
        print(
            _red(
                f"    ✗ PIVOT recommended.\n"
                f"      Best feature-KD config ({best_feat['label']}) hurts "
                f"by {delta_best:+.4f} F1 on S2.\n"
                f"      Even feature-level distillation from FedRep does not\n"
                f"      transfer to the cross-modal student. Recommend:\n"
                f"        • Use centralized teacher for KD primary result\n"
                f"        • Frame KD-Fed as honest negative finding in thesis\n"
                f"        • Iteration 2's federated model remains valuable\n"
                f"          for users with research-grade hardware\n"
                f"        • The modality gap requires teachers that learned\n"
                f"          on pooled data."
            )
        )

    print(_bold("=" * 78))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Saves and loads per-fold, per-algorithm results in HDF5 format.
#
# Adapts from:
#   - PFLIB result_utils.py          : h5py structure, read_data_then_delete(),
#                                      average_data()
#   - PFLIB serverbase.py save_results() : the rs_test_acc pattern
#
# Modifications on top of PFLlib:
#   - Per-subject F1 array (15-element array per round) for variance analysis
#   - Macro-F1 and stress recall datasets alongside accuracy
#   - Fold metadata (test subject, val subject, algorithm, hyperparams)
#   - Load/query utilities for cross-fold analysis
#
# HDF5 Structure (per file):
#   /{algorithm}_{fold_id}/
#       rs_test_f1          : (n_eval_rounds,) macro-F1 per eval round
#       rs_test_acc         : (n_eval_rounds,) accuracy per eval round
#       rs_test_stress_recall : (n_eval_rounds,) stress recall per eval round
#       rs_train_loss       : (n_eval_rounds,) mean train loss per eval round
#       rs_per_subject_f1   : (n_eval_rounds, n_clients) per-subject F1
#       final_per_subject_f1: (n_clients,) final round per-subject F1
#       final_per_subject_acc: (n_clients,) final round per-subject acc
#       final_per_subject_stress_recall: (n_clients,)
#       subject_ids         : (n_clients,) subject IDs (for column mapping)
#       attrs:
#           algorithm, fold_id, test_subject, val_subject,
#           global_rounds, local_epochs, learning_rate, ...

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any
import warnings

import h5py
import numpy as np

# In-memory result accumulator (used during training)


@dataclass
class FoldResultAccumulator:
    """Accumulates results during training for one fold of one algorithm.

    Mirrors PFLlib's rs_test_acc / rs_train_loss pattern but extended
    with macro-F1, stress recall, and per-subject tracking.

    Usage:
        acc = FoldResultAccumulator(algorithm="FedRep", fold_id=2, ...)
        # After each evaluation round:
        acc.append_round(macro_f1=0.85, accuracy=0.90, stress_recall=0.82,
                         train_loss=0.45, per_subject_f1={2: 0.88, 3: 0.82, ...})
        # After training:
        acc.save("results/")
    """

    algorithm: str
    fold_id: int  # test subject ID
    test_subject: int = 0
    val_subject: int = 0
    subject_ids: list[int] = field(default_factory=list)

    # Hyperparameters (stored as metadata)
    hyperparams: dict[str, Any] = field(default_factory=dict)

    # Per-round accumulation lists
    rs_test_f1: list[float] = field(default_factory=list)
    rs_test_acc: list[float] = field(default_factory=list)
    rs_test_stress_recall: list[float] = field(default_factory=list)
    rs_train_loss: list[float] = field(default_factory=list)
    rs_per_subject_f1: list[dict[int, float]] = field(default_factory=list)

    # Final-round per-subject metrics (set at end of training)
    final_per_subject_f1: dict[int, float] = field(default_factory=dict)
    final_per_subject_acc: dict[int, float] = field(default_factory=dict)
    final_per_subject_stress_recall: dict[int, float] = field(default_factory=dict)

    # Held-out test subject LOSO metrics (set at end of training by the
    # server's evaluate_test_subject_with_calibration() step).  Under the
    # held-out-with-calibration protocol the test subject is NOT an FL
    # client, so these cannot be recovered from final_per_subject_* above
    # (which only covers training clients).  These fields ARE the LOSO
    # fold metric for FedAvg / FedRep / FedBABU / PeFLL.  -1.0 is the
    # sentinel "not yet set" value — save() will skip persistence if
    # test_subject_n_samples <= 0, and load_fold_result will use the same
    # sentinel so a missing HDF5 dataset still round-trips cleanly.
    test_subject_f1: float = -1.0
    test_subject_acc: float = -1.0
    test_subject_stress_recall: float = -1.0
    test_subject_stress_f1: float = -1.0
    test_subject_per_class_f1: list[float] = field(default_factory=list)
    test_subject_n_samples: int = 0

    def __post_init__(self):
        if self.test_subject == 0:
            self.test_subject = self.fold_id

    def append_round(
        self,
        macro_f1: float,
        accuracy: float,
        stress_recall: float,
        train_loss: float,
        per_subject_f1: dict[int, float] | None = None,
    ):
        """Record metrics from one evaluation round.

        Parameters:
        macro_f1 : float
            Mean macro-F1 across all clients.
        accuracy : float
            Mean accuracy across all clients.
        stress_recall : float
            Mean stress recall across all clients.
        train_loss : float
            Mean training loss across all clients.
        per_subject_f1 : dict[int, float] | None
            {subject_id: macro_f1} for each client.
        """
        self.rs_test_f1.append(macro_f1)
        self.rs_test_acc.append(accuracy)
        self.rs_test_stress_recall.append(stress_recall)
        self.rs_train_loss.append(train_loss)
        # Empty dict → row of zeros in the saved matrix.
        self.rs_per_subject_f1.append(per_subject_f1 or {})

    def set_final_metrics(
        self,
        per_subject_f1: dict[int, float],
        per_subject_acc: dict[int, float],
        per_subject_stress_recall: dict[int, float],
    ):
        """Store final-round per-subject metrics for the results table."""
        self.final_per_subject_f1 = per_subject_f1
        self.final_per_subject_acc = per_subject_acc
        self.final_per_subject_stress_recall = per_subject_stress_recall

    def set_test_subject_metrics(self, metrics):
        """Store the held-out test subject's post-calibration LOSO metrics.

        Called once per run, at the end of training, by the server's
        evaluate_test_subject_with_calibration() step.  Under the
        held-out-with-calibration protocol (Collins et al. 2021 §5.3),
        this is THE LOSO fold metric — not the mean over training
        clients tracked in rs_test_f1.

        Parameters
        ----------
        metrics : ClientMetrics
            Metrics computed on the test subject's 80% evaluation split
            after the algorithm-appropriate calibration step
            (no-op for FedAvg; K-epoch head-only training for FedRep /
            FedBABU / FedRepDP; hypernet-generated head for PeFLL).
        """
        # Guard against accidentally being passed the wrong subject's
        # metrics.  Not fatal because the caller might legitimately
        # re-key during testing, but worth a clear warning.
        if metrics.subject_id != self.test_subject:
            warnings.warn(
                f"set_test_subject_metrics received metrics for "
                f"S{metrics.subject_id} but this accumulator's "
                f"test_subject is S{self.test_subject}.  Storing anyway.",
                stacklevel=2,
            )

        self.test_subject_f1 = float(metrics.macro_f1)
        self.test_subject_acc = float(metrics.accuracy)
        self.test_subject_stress_recall = float(metrics.stress_recall)
        self.test_subject_stress_f1 = float(metrics.stress_f1)
        self.test_subject_per_class_f1 = list(metrics.per_class_f1)
        self.test_subject_n_samples = int(metrics.n_samples)

    # Save / Load

    def save(self, result_dir: str = "results/", overwrite: bool = True):
        """Save to HDF5 file.

        File path: {result_dir}/{algorithm}_fold{fold_id}.h5

        Keep PFLlib's h5py structure

        Parameters:
        result_dir : str
            Directory for result files.
        overwrite : bool
            If True (default), overwrite existing files with a warning.
            If False, raise FileExistsError on collision.
            Default is True because during development we re-run
            folds constantly.  For final runs, we can setit to False.
        """
        os.makedirs(result_dir, exist_ok=True)
        file_path = os.path.join(result_dir, f"{self.algorithm}_fold{self.fold_id}.h5")
        group_name = f"{self.algorithm}_{self.fold_id}"

        # Overwrite protection
        if os.path.exists(file_path):
            if not overwrite:
                raise FileExistsError(
                    f"Result file already exists: {file_path}. "
                    f"Delete manually or pass overwrite=True."
                )
            print(f" Overwriting existing result file: {file_path}")

        # Warn if subject_ids not set (per-subject data will be empty)
        if not self.subject_ids:
            warnings.warn(
                f"subject_ids is empty for {self.algorithm} fold {self.fold_id}. "
                f"Per-subject datasets (rs_per_subject_f1, final_per_subject_*) "
                f"will not be saved.  Did you forget to set subject_ids after "
                f"set_clients()?",
                stacklevel=2,
            )

        with h5py.File(file_path, "w") as hf:
            grp = hf.create_group(group_name)

            # Per-round metrics (PFLlib-compatible flat arrays)
            if self.rs_test_f1:
                grp.create_dataset("rs_test_f1", data=np.array(self.rs_test_f1))
            if self.rs_test_acc:
                grp.create_dataset("rs_test_acc", data=np.array(self.rs_test_acc))
            if self.rs_test_stress_recall:
                grp.create_dataset(
                    "rs_test_stress_recall",
                    data=np.array(self.rs_test_stress_recall),
                )
            if self.rs_train_loss:
                grp.create_dataset("rs_train_loss", data=np.array(self.rs_train_loss))

            # Per-subject F1 per round: (n_rounds, n_subjects) matrix
            if self.rs_per_subject_f1 and self.subject_ids:
                n_rounds = len(self.rs_per_subject_f1)
                n_subjects = len(self.subject_ids)
                matrix = np.zeros((n_rounds, n_subjects), dtype=np.float64)
                for r, round_dict in enumerate(self.rs_per_subject_f1):
                    for j, sid in enumerate(self.subject_ids):
                        matrix[r, j] = round_dict.get(sid, 0.0)
                grp.create_dataset("rs_per_subject_f1", data=matrix)

            # Subject ID mapping (for interpreting column indices)
            if self.subject_ids:
                grp.create_dataset("subject_ids", data=np.array(self.subject_ids))

            # Final per-subject metrics
            if self.final_per_subject_f1:
                arr = np.array(
                    [self.final_per_subject_f1.get(s, 0.0) for s in self.subject_ids]
                )
                grp.create_dataset("final_per_subject_f1", data=arr)

            if self.final_per_subject_acc:
                arr = np.array(
                    [self.final_per_subject_acc.get(s, 0.0) for s in self.subject_ids]
                )
                grp.create_dataset("final_per_subject_acc", data=arr)

            if self.final_per_subject_stress_recall:
                arr = np.array(
                    [
                        self.final_per_subject_stress_recall.get(s, 0.0)
                        for s in self.subject_ids
                    ]
                )
                grp.create_dataset("final_per_subject_stress_recall", data=arr)

            # Held-out test subject LOSO metrics.  Under the
            # held-out-with-calibration protocol these ARE the fold
            # result; the rs_* / final_per_subject_* above are only
            # training-client monitoring.  Written only if actually set
            # (test_subject_n_samples > 0); a sentinel of 0 indicates
            # the server never called set_test_subject_metrics().
            if self.test_subject_n_samples > 0:
                grp.create_dataset(
                    "test_subject_f1", data=np.float64(self.test_subject_f1)
                )
                grp.create_dataset(
                    "test_subject_acc", data=np.float64(self.test_subject_acc)
                )
                grp.create_dataset(
                    "test_subject_stress_recall",
                    data=np.float64(self.test_subject_stress_recall),
                )
                grp.create_dataset(
                    "test_subject_stress_f1",
                    data=np.float64(self.test_subject_stress_f1),
                )
                grp.create_dataset(
                    "test_subject_per_class_f1",
                    data=np.array(self.test_subject_per_class_f1, dtype=np.float64),
                )
                grp.attrs["test_subject_n_samples"] = int(self.test_subject_n_samples)

            # Metadata as HDF5 attributes
            grp.attrs["algorithm"] = self.algorithm
            grp.attrs["fold_id"] = self.fold_id
            grp.attrs["test_subject"] = self.test_subject
            grp.attrs["val_subject"] = self.val_subject
            grp.attrs["n_eval_rounds"] = len(self.rs_test_f1)

            for k, v in self.hyperparams.items():
                try:
                    grp.attrs[k] = v
                except TypeError:
                    grp.attrs[k] = str(v)

        print(f"  Results saved: {file_path}")
        return file_path


# Loading utilities (for analysis)


@dataclass
class FoldResult:
    algorithm: str
    fold_id: int
    test_subject: int
    val_subject: int

    rs_test_f1: np.ndarray  # (n_rounds,)
    rs_test_acc: np.ndarray
    rs_test_stress_recall: np.ndarray
    rs_train_loss: np.ndarray
    rs_per_subject_f1: np.ndarray  # (n_rounds, n_subjects)
    subject_ids: np.ndarray  # (n_subjects,)

    final_per_subject_f1: np.ndarray
    final_per_subject_acc: np.ndarray
    final_per_subject_stress_recall: np.ndarray

    # Held-out test subject LOSO metrics (written under the
    # held-out-with-calibration protocol).  Sentinel -1.0 means "not
    # present in this HDF5 file", which will be the case for pre-
    # protocol-change result files.  Consumers should check
    # `has_test_subject_metrics` before using them, or use the
    # backward-compatible `test_subject_final_f1` / `test_subject_loso_f1`
    # properties which handle the fallback.
    test_subject_f1: float = -1.0
    test_subject_acc: float = -1.0
    test_subject_stress_recall: float = -1.0
    test_subject_stress_f1: float = -1.0
    test_subject_per_class_f1: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    test_subject_n_samples: int = 0

    @property
    def best_f1(self) -> float:
        # Best macro-F1 achieved across all evaluation rounds
        return float(self.rs_test_f1.max()) if len(self.rs_test_f1) > 0 else 0.0

    @property
    def final_f1(self) -> float:
        # Macro-F1 at the final evaluation round
        return float(self.rs_test_f1[-1]) if len(self.rs_test_f1) > 0 else 0.0

    @property
    def has_test_subject_metrics(self) -> bool:
        """True if this file contains post-calibration LOSO metrics.

        False for pre-protocol-change files (which relied on the
        test-subject-as-client lookup in final_per_subject_f1).
        """
        return self.test_subject_n_samples > 0 and self.test_subject_f1 >= 0.0

    @property
    def test_subject_loso_f1(self) -> float:
        """Post-calibration LOSO F1 for the held-out test subject.

        This is THE LOSO fold metric under the held-out-with-calibration
        protocol.  Raises if the file was written before the protocol
        change (use `has_test_subject_metrics` to gate, or use
        `test_subject_final_f1` for a backward-compatible view).
        """
        if not self.has_test_subject_metrics:
            raise ValueError(
                f"FoldResult for {self.algorithm} fold {self.fold_id} "
                f"has no post-calibration LOSO metrics.  This file was "
                f"probably produced before the held-out-with-calibration "
                f"protocol was introduced.  Re-run this fold to get the "
                f"correct LOSO number."
            )
        return float(self.test_subject_f1)

    @property
    def test_subject_final_f1(self) -> float:
        """Final F1 for the test subject, preferring the new protocol.

        New protocol (held-out-with-calibration): returns the post-
        calibration LOSO F1 from the dedicated HDF5 field.  This is
        the correct LOSO fold metric.

        Legacy protocol (pre-fix): falls back to looking up the test
        subject in the per-client final_per_subject_f1 array.  Under
        the new protocol the test subject is not an FL client, so this
        fallback returns 0.0; check `has_test_subject_metrics` first
        if you need to distinguish "no LOSO metric written" from
        "LOSO metric was legitimately 0".
        """
        if self.has_test_subject_metrics:
            return float(self.test_subject_f1)
        # Legacy fallback: look up test_subject among per-client results
        if len(self.final_per_subject_f1) > 0 and len(self.subject_ids) > 0:
            idx = np.where(self.subject_ids == self.test_subject)[0]
            if len(idx) > 0:
                return float(self.final_per_subject_f1[idx[0]])
        return 0.0


def load_fold_result(file_path: str) -> FoldResult:
    """Load results from an HDF5 file produced by FoldResultAccumulator.save().

    Adapting from PFLIB result_utils.py read_data_then_delete().

    Parameters:
    file_path : str
        Path to the HDF5 file.

    Returns:
    FoldResult
    """
    with h5py.File(file_path, "r") as hf:  # expected to mqatch exactly one group
        group_names = list(hf.keys())
        if len(group_names) == 0:
            raise ValueError(f"No groups found in {file_path}")
        grp = hf[group_names[0]]

        def _get(name, default_shape=(0,)):
            if name in grp:
                return np.array(grp[name])
            return np.zeros(default_shape)

        def _get_scalar(name, default=-1.0):
            """Read a 0-d dataset as a Python float."""
            if name in grp:
                return float(np.asarray(grp[name]).item())
            return default

        result = FoldResult(
            algorithm=str(grp.attrs.get("algorithm", "unknown")),
            fold_id=int(grp.attrs.get("fold_id", 0)),
            test_subject=int(grp.attrs.get("test_subject", 0)),
            val_subject=int(grp.attrs.get("val_subject", 0)),
            rs_test_f1=_get("rs_test_f1"),
            rs_test_acc=_get("rs_test_acc"),
            rs_test_stress_recall=_get("rs_test_stress_recall"),
            rs_train_loss=_get("rs_train_loss"),
            rs_per_subject_f1=_get("rs_per_subject_f1", default_shape=(0, 0)),
            subject_ids=_get("subject_ids"),
            final_per_subject_f1=_get("final_per_subject_f1"),
            final_per_subject_acc=_get("final_per_subject_acc"),
            final_per_subject_stress_recall=_get("final_per_subject_stress_recall"),
            # Held-out test subject LOSO metrics (may be absent in
            # pre-protocol-change files; sentinel -1.0 means "not set").
            test_subject_f1=_get_scalar("test_subject_f1", default=-1.0),
            test_subject_acc=_get_scalar("test_subject_acc", default=-1.0),
            test_subject_stress_recall=_get_scalar(
                "test_subject_stress_recall", default=-1.0
            ),
            test_subject_stress_f1=_get_scalar("test_subject_stress_f1", default=-1.0),
            test_subject_per_class_f1=_get(
                "test_subject_per_class_f1", default_shape=(0,)
            ),
            test_subject_n_samples=int(grp.attrs.get("test_subject_n_samples", 0)),
        )

    return result


def load_all_fold_results(
    result_dir: str = "results/",
    algorithm: str | None = None,
) -> list[FoldResult]:
    """Load all fold results from a directory.

    Parameters
    ----------
    result_dir : str
        Directory containing HDF5 result files.
    algorithm : str | None
        If provided, only load files matching this algorithm name.

    Returns:
    list[FoldResult]
        Sorted by fold_id.
    """
    results: list[FoldResult] = []
    if not os.path.exists(result_dir):
        return results

    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".h5"):
            continue
        if algorithm is not None and not fname.startswith(algorithm):
            continue

        try:
            result = load_fold_result(os.path.join(result_dir, fname))
            results.append(result)
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")

    results.sort(key=lambda r: r.fold_id)
    return results


# Cross-fold analysis utilities


def summarise_across_folds(results: list[FoldResult]) -> dict[str, Any]:
    """Compute mean ± std across LOSO folds.

    Adapting again PFLIB_result_utils.py average_data()
    Parameters:
    results : list[FoldResult]
        One per LOSO fold (should be 15 for full WESAD evaluation).

    Returns:
    dict with keys:
        mean_test_f1, std_test_f1            - held-out LOSO macro-F1
        mean_test_acc, std_test_acc          - held-out LOSO accuracy
        mean_test_stress_recall, std_*       - held-out LOSO stress recall
        per_fold_test_f1                     - list of per-fold LOSO F1
        mean_f1, std_f1                      - training-client mean (monitor only)
        mean_best_f1, std_best_f1            - best training-client mean per fold
        mean_acc, std_acc                    - training-client mean accuracy
        mean_stress_recall, std_stress_recall- training-client mean stress recall
        n_folds                              - number of folds summarised
    """
    # The test subject's post-calibration LOSO metric IS the fold
    # result under the held-out-with-calibration protocol.
    # test_subject_final_f1 prefers the new dedicated field and falls
    # back to the old per-client lookup so pre-protocol result files
    # still produce a value.
    fold_test_f1s = [r.test_subject_final_f1 for r in results]

    # LOSO accuracy and stress recall come from the dedicated
    # test_subject_* fields.  If a given result was written before the
    # protocol change, these will be sentinel -1.0; filter those out
    # before averaging so they don't drag the mean below zero.
    fold_test_accs = [r.test_subject_acc for r in results if r.has_test_subject_metrics]
    fold_test_stress_recalls = [
        r.test_subject_stress_recall for r in results if r.has_test_subject_metrics
    ]

    # Training-client monitoring statistics (mean across the 13 FL
    # clients, per round).  These are NOT the LOSO result but are
    # useful for convergence diagnostics and plots.
    fold_mean_f1s = [r.final_f1 for r in results]
    fold_best_f1s = [r.best_f1 for r in results]

    fold_accs = [
        float(r.rs_test_acc[-1]) if len(r.rs_test_acc) > 0 else 0.0 for r in results
    ]

    fold_stress_recalls = [
        float(r.rs_test_stress_recall[-1]) if len(r.rs_test_stress_recall) > 0 else 0.0
        for r in results
    ]

    summary: dict[str, Any] = {
        # Test subject's LOSO metric per fold (primary result)
        "mean_test_f1": float(np.mean(fold_test_f1s)) if fold_test_f1s else 0.0,
        "std_test_f1": float(np.std(fold_test_f1s)) if fold_test_f1s else 0.0,
        "per_fold_test_f1": fold_test_f1s,
        # Mean F1 across training clients per fold (monitoring only)
        "mean_f1": float(np.mean(fold_mean_f1s)) if fold_mean_f1s else 0.0,
        "std_f1": float(np.std(fold_mean_f1s)) if fold_mean_f1s else 0.0,
        # Best F1 achieved (for convergence comparison)
        "mean_best_f1": float(np.mean(fold_best_f1s)) if fold_best_f1s else 0.0,
        "std_best_f1": float(np.std(fold_best_f1s)) if fold_best_f1s else 0.0,
        # Training-client mean accuracy / stress recall (monitoring only)
        "mean_acc": float(np.mean(fold_accs)) if fold_accs else 0.0,
        "std_acc": float(np.std(fold_accs)) if fold_accs else 0.0,
        "mean_stress_recall": (
            float(np.mean(fold_stress_recalls)) if fold_stress_recalls else 0.0
        ),
        "std_stress_recall": (
            float(np.std(fold_stress_recalls)) if fold_stress_recalls else 0.0
        ),
        # Number of folds
        "n_folds": len(results),
    }

    # LOSO accuracy and stress recall — only populated if at least one
    # result file contains test_subject_* fields.
    if fold_test_accs:
        summary["mean_test_acc"] = float(np.mean(fold_test_accs))
        summary["std_test_acc"] = float(np.std(fold_test_accs))
        summary["n_folds_with_loso"] = len(fold_test_accs)
    if fold_test_stress_recalls:
        summary["mean_test_stress_recall"] = float(np.mean(fold_test_stress_recalls))
        summary["std_test_stress_recall"] = float(np.std(fold_test_stress_recalls))

    return summary


def print_cross_fold_summary(algorithm: str, summary: dict[str, Any]):
    print(f"\n  ~~~~~ {algorithm} - {summary['n_folds']}-fold LOSO Summary ~~~~~")
    print(
        f"  Test Subject F1:  {summary['mean_test_f1']:.4f} ± "
        f"{summary['std_test_f1']:.4f}"
    )
    print(f"  Mean Client F1:   {summary['mean_f1']:.4f} ± " f"{summary['std_f1']:.4f}")
    print(
        f"  Accuracy:         {summary['mean_acc']:.4f} ± " f"{summary['std_acc']:.4f}"
    )
    print(
        f"  Stress Recall:    {summary['mean_stress_recall']:.4f} ± "
        f"{summary['std_stress_recall']:.4f}"
    )
    print(
        f"  Best F1 (any round): {summary['mean_best_f1']:.4f} ± "
        f"{summary['std_best_f1']:.4f}"
    )

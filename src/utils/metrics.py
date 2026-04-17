# Centralised evaluation logic for all FL methods.
#
# Computes: macro-F1 (primary metric), per-class recall (stress recall
# is clinically critical), accuracy (for literature comparability),
# and per-subject scores stored separately for variance analysis.
#
# Called by every server's evaluate() method.
#
# Adapts from:
#   - PFLIB serverbase.py evaluate() / test_metrics()
#   - PFLIB clientbase.py test_metrics()
#
# Extensions beyond PFLlib:
#   - Macro-F1 as primary metric (PFLlib only tracks accuracy and AUC)
#   - Per-class recall with emphasis on stress recall
#   - Per-subject score tracking for variance/fairness analysis
#   - Confusion matrix generation for per-fold analysis

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

# Data classes for structured metric results


@dataclass
class ClientMetrics:
    """Evaluation metrics for a single FL client (subject)."""

    subject_id: int
    n_samples: int

    # Primary metrics
    macro_f1: float
    accuracy: float

    # Per-class recall (index = class id)
    # For binary: [baseline_recall, stress_recall]
    # For 3-class: [baseline_recall, stress_recall, amusement_recall]
    per_class_recall: list[float]
    per_class_precision: list[float]
    per_class_f1: list[float]

    # Stress-specific
    @property
    def stress_recall(self) -> float:
        """Recall for the stress class (class index 1)."""
        return self.per_class_recall[1] if len(self.per_class_recall) > 1 else 0.0

    @property
    def stress_precision(self) -> float:
        """Precision for the stress class (class index 1)."""
        return self.per_class_precision[1] if len(self.per_class_precision) > 1 else 0.0

    @property
    def stress_f1(self) -> float:
        """F1 for the stress class (class index 1)."""
        return self.per_class_f1[1] if len(self.per_class_f1) > 1 else 0.0


@dataclass
class FoldMetrics:
    """Aggregated metrics across all clients in a LOSO fold.

    Stores both aggregate statistics and per-subject breakdown
    for variance analysis and fairness evaluation.
    """

    fold_id: int  # test subject ID for this fold

    # Per-subject metrics (one ClientMetrics per participating client)
    per_subject: list[ClientMetrics] = field(default_factory=list)

    # Aggregate statistics (computed from per_subject)
    mean_macro_f1: float = 0.0
    std_macro_f1: float = 0.0
    mean_accuracy: float = 0.0
    mean_stress_recall: float = 0.0

    # Test subject's metrics
    test_subject_metrics: ClientMetrics | None = None

    def compute_aggregates(self):
        """Compute aggregate statistics from per-subject metrics."""
        if not self.per_subject:
            return

        f1s = [m.macro_f1 for m in self.per_subject]
        accs = [m.accuracy for m in self.per_subject]
        stress_recalls = [m.stress_recall for m in self.per_subject]

        self.mean_macro_f1 = float(np.mean(f1s))
        self.std_macro_f1 = float(np.std(f1s))
        self.mean_accuracy = float(np.mean(accs))
        self.mean_stress_recall = float(np.mean(stress_recalls))


# Core evaluation functions


def evaluate_client(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    subject_id: int,
    num_classes: int = 2,
) -> ClientMetrics:
    """Evaluate a single client's model on their data.

    This replaces PFLIB_clientbase.py test_metrics() with extended metrics.
    Returns a ClientMetrics dataclass

    IMPORTANT: The caller (typically server_base.py evaluate()) is
    responsible for passing the correct dataloader.  For the LOSO fold
    result, pass the test subject's load_test_data().  For monitoring
    training clients, pass their load_test_data() (local non-overlapped
    windows).  Passing the wrong loader (e.g. val instead of test) will
    silently produce wrong numbers.

    Parameters:
    model : torch.nn.Module
        The client's model (personalised or global).
    dataloader : DataLoader
        The client's test DataLoader (from FLClientDataManager.load_test_data()).
    device : torch.device
        Computation device.
    subject_id : int
        This client's subject ID (for logging and result storage).
    num_classes : int
        Number of classes

    Returns:
    ClientMetrics
        Structured metrics for this client.
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            # Handle dict-of-signals input (WESADDLDataset format)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                signals, labels = batch
            else:
                raise ValueError(
                    f"Unexpected batch format: {type(batch)}. "
                    f"Expected (signals_dict, labels) tuple."
                )

            # Move signals to device (non_blocking benefits from pin_memory)
            if isinstance(signals, dict):
                signals = {
                    k: v.to(device, non_blocking=True) for k, v in signals.items()
                }
            else:
                signals = signals.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            # Forward pass
            logits = model(signals)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        # No data —> return zero metrics
        return ClientMetrics(
            subject_id=subject_id,
            n_samples=0,
            macro_f1=0.0,
            accuracy=0.0,
            per_class_recall=[0.0] * num_classes,
            per_class_precision=[0.0] * num_classes,
            per_class_f1=[0.0] * num_classes,
        )

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    n_samples = len(all_labels)

    # Class labels present in this evaluation set
    labels_range = list(range(num_classes))

    # Macro-F1 (primary metric)
    macro_f1 = f1_score(
        all_labels, all_preds, average="macro", labels=labels_range, zero_division=0.0
    )

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    per_class_recall = recall_score(
        all_labels, all_preds, average=None, labels=labels_range, zero_division=0.0
    ).tolist()

    per_class_precision = precision_score(
        all_labels, all_preds, average=None, labels=labels_range, zero_division=0.0
    ).tolist()

    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=labels_range, zero_division=0.0
    ).tolist()

    return ClientMetrics(
        subject_id=subject_id,
        n_samples=n_samples,
        macro_f1=float(macro_f1),
        accuracy=float(accuracy),
        per_class_recall=per_class_recall,
        per_class_precision=per_class_precision,
        per_class_f1=per_class_f1,
    )


def compute_fold_metrics(
    client_metrics_list: list[ClientMetrics],
    test_subject_id: int,
) -> FoldMetrics:
    """Aggregate per-client metrics into fold-level results.

    Called by the server's evaluate() method after collecting
    ClientMetrics from all clients.

    Parameters:
    client_metrics_list : list[ClientMetrics]
        Metrics from each participating client.
    test_subject_id : int
        The test subject ID for this fold (for LOSO reporting).

    Returns:
    FoldMetrics
        Aggregated fold metrics with per-subject breakdown.
    """
    fold = FoldMetrics(fold_id=test_subject_id)
    fold.per_subject = client_metrics_list

    # Extract the test subject's metrics
    for cm in client_metrics_list:
        if cm.subject_id == test_subject_id:
            fold.test_subject_metrics = cm
            break

    fold.compute_aggregates()
    return fold


def compute_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> np.ndarray:
    """Compute the confusion matrix for a model on a dataset.
    Returns:
    np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
        Rows = true labels, columns = predicted labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            signals, labels = batch

            if isinstance(signals, dict):
                signals = {k: v.to(device) for k, v in signals.items()}
            else:
                signals = signals.to(device)

            labels = labels.to(device)
            logits = model(signals)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        return np.zeros((num_classes, num_classes), dtype=int)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


# Logging / printing utilities


def print_client_metrics(cm: ClientMetrics, prefix: str = ""):
    """Print a single client's metrics in a readable format."""
    tag = f"{prefix}S{cm.subject_id}" if prefix else f"S{cm.subject_id}"
    print(
        f"  {tag}: F1={cm.macro_f1:.4f}  Acc={cm.accuracy:.4f}  "
        f"StressRecall={cm.stress_recall:.4f}  "
        f"StressPrecision={cm.stress_precision:.4f}  "
        f"n={cm.n_samples}"
    )


def print_fold_metrics(fm: FoldMetrics, round_num: int | None = None):
    """Print fold-level metrics summary."""
    header = f"Fold S{fm.fold_id}"
    if round_num is not None:
        header += f" (round {round_num})"

    print(f"\n  ── {header} ──")
    print(f"  Mean Macro-F1:      {fm.mean_macro_f1:.4f} ± {fm.std_macro_f1:.4f}")
    print(f"  Mean Accuracy:      {fm.mean_accuracy:.4f}")
    print(f"  Mean Stress Recall: {fm.mean_stress_recall:.4f}")

    if fm.test_subject_metrics is not None:
        print(f"  Test subject S{fm.fold_id}:")
        print_client_metrics(fm.test_subject_metrics, prefix="    ")

    # Per-subject breakdown
    f1s = [m.macro_f1 for m in fm.per_subject]
    if f1s:
        print(
            f"  Per-subject F1: min={min(f1s):.4f}  max={max(f1s):.4f}  "
            f"range={max(f1s)-min(f1s):.4f}"
        )
    else:
        print("  Per-subject F1: no clients evaluated")

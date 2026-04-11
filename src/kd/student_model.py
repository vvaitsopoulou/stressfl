# Student Model for Cross-Modal Knowledge Distillation

# lightweight FedRep-compatible MLP that takes HRV features
# and predicts stress.  This is the student that learns from the CNN-LSTM
# teacher's soft labels via knowledge distillation.

# Architecture
#     StudentBase (shared, federatable):
#       Input(n_features) → Linear(64) → ReLU → BatchNorm(64) → Dropout(0.2)
#                         → Linear(32) → ReLU → BatchNorm(32) → Dropout(0.2)
#       Output: (batch, 32) intermediate representation

#     StudentHead (personal, local):
#       Linear(32, 16) → ReLU → Linear(16, n_classes)
#       Output: (batch, n_classes) logits

# Total: ~3,500–5,000 parameters (vs teacher's ~50K–100K = 10–20×
# compression).

# Design Decisions:
#   MLP over LSTM: The student input is a flat feature vector per
#   window (6 HRV features), not a time series.  There is no temporal
#   dimension to model because each window is independent.  Ay et al. (2022,
#   IJCNN) demonstrated comparable performance with MLPs on pre-extracted
#   HRV features.

#   BatchNorm: Stabilises training with the small effective batch
#   sizes typical in LOSO cross-validation, where training data comes
#   from only 13 subjects (Ioffe & Szegedy 2015).

#   Dropout 0.2: Lighter than the teacher's 0.5 because the student
#   has far fewer parameters and is inherently less prone to overfitting.
#   Holda et al. (2024, Sensors) used similar light dropout for
#   compressed biosignal models.

#   Base/Head split at 32→16: The base (input→64→32) learns universal
#   HRV-to-stress mappings sharable across subjects in FedRep.  The head
#   (32→16→2) learns personal decision boundaries that adapt to
#   individual stress responses (Collins et al. 2021, ICML).

#   Intermediate dim = 32: Matches the teacher's post-BatchNorm LSTM
#   hidden state dimension (TEACHER_FEATURE_DIM = 32 in kd_config.py).
#   This enables feature-matching loss without requiring a projection layer
#   as a more efficient and simple design (MemKD 2025).

# FedRep Compatibility:
# The base/head split serves triple duty:

#   During FL training: base shared across clients, head personal
#     (inter-subject heterogeneity).
#   During KD: teacher representations distilled to student base,
#     head personal (inter-device heterogeneity).
#   During deployment: base frozen, head fine-tuned on user data
#     (inter-individual personalisation).

# This mirrors the existing WESADFLModel interface:
#   - model.base.parameters()  → shared parameters for server aggregation
#   - model.head.parameters() → local parameters never communicated

# BatchNorm Caveat During Head Fine-Tuning!!!!
# The base contains BatchNorm1d layers.  During personalisation
# (head-only fine-tuning with base frozen), the trainer must set
# model.base.eval() while keeping model.head.train().  This
# ensures BatchNorm uses its learned running statistics rather than
# batch statistics.  If left in training mode, a batch of size 1will raise
# ValueError: Expected more than 1 value per channel.  The
# trainer should also use drop_last=True as a second safeguard.

# Interface:
#   StudentBase(nn.Module): forward(x) → (batch, 32)
#   StudentHead(nn.Module): forward(h) → (batch, n_classes)
#   StudentModel(nn.Module): composes base + head, exposes ``.base`` and ``.head``
#   build_student(n_features, n_classes): factory function

# References
# ----------
# Collins et al. 2021, ICML -> FedRep base/head split
# Ay et al. 2022, IJCNN -> MLP on HRV features
# Ioffe & Szegedy 2015 -> Batch Normalization
# Holda et al. 2024, Sensors -> compressed biosignal models
# MemKD 2025, arXiv:2601.04264 -> feature matching in KD
# Hinton et al. 2015, arXiv:1503.02531 -> KD framework

from __future__ import annotations

import torch
import torch.nn as nn

from kd.kd_config import (
    N_CLASSES,
    N_PRIMARY_FEATURES,
    STUDENT_BASE_HIDDEN_1,
    STUDENT_BASE_HIDDEN_2,
    STUDENT_DROPOUT,
    STUDENT_HEAD_HIDDEN,
    TEACHER_FEATURE_DIM,
)

# BASE MODULE (shared, federatable)


class StudentBase(nn.Module):
    """Shared representation: HRV features → intermediate embedding.

    This module learns universal HRV-to-stress mappings that generalise
    across subjects.  In a FedRep setting, this is the part that would
    be aggregated on the server.

    Architecture::

        Linear(n_features, 64) → ReLU → BatchNorm(64) → Dropout(0.2)
        Linear(64, 32)         → ReLU → BatchNorm(32) → Dropout(0.2)

    The output dimensionality (32) matches the teacher's post-BatchNorm
    LSTM hidden state, enabling feature-matching loss without a
    projection layer.

    Parameters
    ----------
    n_features : int
        Number of input features.  Default 6 (primary HRV set).
        Configurable for feature-set ablation (6, 5, 1).
    hidden_1 : int
        First hidden layer width (default 64).
    hidden_2 : int
        Second hidden layer / output width (default 32).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        n_features: int = N_PRIMARY_FEATURES,
        hidden_1: int = STUDENT_BASE_HIDDEN_1,
        hidden_2: int = STUDENT_BASE_HIDDEN_2,
        dropout: float = STUDENT_DROPOUT,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_1),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_2),
            nn.Dropout(p=dropout),
        )

        self.output_dim = hidden_2

        # Weight initialisation: Kaiming/He for ReLU layers
        # (matches teacher's CNN branch initialisation convention)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for Linear layers, standard for BN."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared intermediate representation.

        Parameters
        ----------
        x : torch.Tensor
            (batch, n_features) — HRV feature vector.

        Returns
        -------
        torch.Tensor
            (batch, hidden_2) — intermediate representation (dim=32).
        """
        return self.net(x)


# HEAD MODULE (personal, local)
class StudentHead(nn.Module):
    """Personal classification head: embedding → stress prediction.

    This module learns personal decision boundaries that adapt to an
    individual's stress response patterns.  In FedRep, this part stays
    local and is never communicated to the server.

    Architecture:

        Linear(32, 16) → ReLU → Linear(16, n_classes)

    No BatchNorm or Dropout in the head as it has only ~34+18 = 546
    parameters for binary classification, making regularisation
    unnecessary and potentially harmful with the few calibration
    samples available during personalisation (K=10–50 gradient steps,
    ~20% of one subject's data).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the base output (default 32).
    hidden : int
        Hidden layer width (default 16).
    n_classes : int
        Number of output classes (default 2).
    """

    def __init__(
        self,
        input_dim: int = STUDENT_BASE_HIDDEN_2,
        hidden: int = STUDENT_HEAD_HIDDEN,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for the classifier (matches teacher)."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Classify from intermediate representation.

        Parameters
        ----------
        h : torch.Tensor
            (batch, input_dim) — output of StudentBase.

        Returns
        -------
        torch.Tensor
            (batch, n_classes) — raw logits for CrossEntropyLoss.
        """
        return self.classifier(h)


# COMPOSED MODEL


class StudentModel(nn.Module):
    """Complete student: base + head with FedRep-compatible interface.

    Exposes ``.base`` (StudentBase) and ``.head`` (StudentHead) as named
    sub-modules, matching the interface expected by:
      * ``kd_trainer.py``: separate optimisers for base vs head
      * ``client_rep.py``: ``model.base.parameters()`` for aggregation
      * ``fitbit_evaluator.py``: freeze base, fine-tune head

    The forward() method calls base → head in sequence, identical to
    WESADFLModel's forward pattern.

    Parameters
    ----------
    n_features : int
        Number of input features (default 6).
    n_classes : int
        Number of output classes (default 2).
    base_hidden_1 : int
        First hidden layer in base (default 64).
    base_hidden_2 : int
        Second hidden layer / intermediate dim in base (default 32).
    head_hidden : int
        Hidden layer in head (default 16).
    dropout : float
        Dropout in base (default 0.2).
    """

    def __init__(
        self,
        n_features: int = N_PRIMARY_FEATURES,
        n_classes: int = N_CLASSES,
        base_hidden_1: int = STUDENT_BASE_HIDDEN_1,
        base_hidden_2: int = STUDENT_BASE_HIDDEN_2,
        head_hidden: int = STUDENT_HEAD_HIDDEN,
        dropout: float = STUDENT_DROPOUT,
    ):
        super().__init__()

        self.base = StudentBase(
            n_features=n_features,
            hidden_1=base_hidden_1,
            hidden_2=base_hidden_2,
            dropout=dropout,
        )

        self.head = StudentHead(
            input_dim=base_hidden_2,
            hidden=head_hidden,
            n_classes=n_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: base → head.

        Parameters
        ----------
        x : torch.Tensor
            (batch, n_features) — HRV feature vector.
        return_features : bool
            If True, also return the intermediate representation from
            the base (dim=32) for feature-matching loss.

        Returns
        -------
        logits : torch.Tensor
            (batch, n_classes) — raw logits.
        features : torch.Tensor, optional
            (batch, 32) — intermediate representation, only if
            ``return_features=True``.
        """
        features = self.base(x)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits


# FACTORY FUNCTION


def build_student(
    n_features: int = N_PRIMARY_FEATURES,
    n_classes: int = N_CLASSES,
    **kwargs,
) -> StudentModel:
    """Build a student model with default or custom architecture.

    This is the primary entry point for creating student models.
    Mirrors the ``build_wrist_model()`` / ``build_fl_wrist_model()``
    factory pattern used in the teacher codebase.

    Parameters
    ----------
    n_features : int
        Number of input features.  Default 6 (primary HRV).
        For feature-set ablation: 5 (Fitbit), 1 (minimal).
    n_classes : int
        Number of output classes.  Default 2 (binary).
    **kwargs
        Forwarded to StudentModel (base_hidden_1, base_hidden_2,
        head_hidden, dropout).

    Returns
    -------
    StudentModel
    """
    return StudentModel(
        n_features=n_features,
        n_classes=n_classes,
        **kwargs,
    )


# DIAGNOSTIC UTILITIES


def print_student_summary(model: StudentModel) -> dict[str, int]:
    """Print parameter counts for base and head.

    Mirrors ``wesad_fl_model.print_split_summary()``.

    Parameters
    ----------
    model : StudentModel

    Returns
    -------
    dict with keys: base_params, head_params, total_params.
    """
    base_params = sum(p.numel() for p in model.base.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    total = base_params + head_params
    base_pct = 100 * base_params / total if total > 0 else 0
    head_pct = 100 * head_params / total if total > 0 else 0

    print("\n  StudentModel Summary")
    print(f"  {'─' * 69}")
    print(f"  Base (shared):    {base_params:,} params ({base_pct:.1f}%)")
    print(f"  Head (personal):  {head_params:,} params ({head_pct:.1f}%)")
    print(f"  Total:            {total:,} params")
    print(f"  Compression vs teacher (~80K): {80000 / max(total, 1):.0f}×")
    print(f"  {'─' * 69}")

    # Detail
    print("\n  Base layers:")
    for name, param in model.base.named_parameters():
        print(f"    {name}: {list(param.shape)} ({param.numel():,})")

    print("\n  Head layers:")
    for name, param in model.head.named_parameters():
        print(f"    {name}: {list(param.shape)} ({param.numel():,})")

    return {
        "base_params": base_params,
        "head_params": head_params,
        "total_params": total,
    }


def verify_feature_matching_compatibility(model: StudentModel) -> bool:
    """Verify that the student's intermediate dim matches the teacher's.

    The feature-matching loss requires:
      student base output dim == teacher post-BN LSTM hidden dim

    Both should be 32.  This function checks at runtime.

    Parameters
    ----------
    model : StudentModel

    Returns
    -------
    bool
        True if dimensions match.
    """
    student_dim = model.base.output_dim
    teacher_dim = TEACHER_FEATURE_DIM

    match = student_dim == teacher_dim
    if match:
        print(
            f"  Feature-matching compatible: student={student_dim}, "
            f"teacher={teacher_dim}"
        )
    else:
        print(
            f"  Feature-matching INCOMPATIBLE: student={student_dim}, "
            f"teacher={teacher_dim}"
        )
    return bool(match)

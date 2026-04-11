# Knowledge Distillation Loss Functions

# Implement the combined distillation + hard-label loss for training the
# student model, with optional feature-matching regularisation.

# Loss Formulations

# Response-Based Distillation Loss (Hinton et al. 2015):

#   L_total = α · T² · KL(log_softmax(z_s / T) || softmax(z_t / T))
#             + (1 - α) · CE(z_s, y)
#
# where:
#   z_s = student logits
#   z_t = cached teacher logits (raw, not softmaxed)
#   T   = temperature
#   α   = distillation weight
#   y   = hard ground-truth labels

#   - softmax(z_t / T) produces the teacher's softened target distribution
#   - log_softmax(z_s / T) is used for the student side of KL divergence
#   - T² compensates for smaller gradients when T is high

# Feature-Matching Loss ( MemKD 2025):

#   L_feat = β · MSE(f_s, f_t)
#
# where:
#   f_s = student intermediate representation
#         (StudentBase output, dim = 32)
#   f_t = teacher penultimate representation
#         (post-BN LSTM hidden state, dim = 32)
#   β   = feature-matching weight

# Combined Loss (when β > 0):

#   L = L_total + L_feat

# Binary Classification Note:
# With only 2 classes, the soft probability distribution is effectively a
# single confidence value. For example at T=3, a teacher confidence of 0.95 becomes
# ~0.77, providing a gentler gradient. According to literature, we must not exceed T=5 for
# binary tasks as higher T makes the distribution nearly uniform. Feature-matching may add more value here than in
# typical multi-class KD because the soft output carries limited information (MemKD 2025).

# IMPORTANT: Teacher Logits Must Be Raw!!
# teacher_logits must be raw logits (pre-softmax) because if softmaxed
# probabilities are passed instead, the loss will compute without error
# but will be silently wrong as softmax of already-soft values
# produces a near-uniform distribution, destroying the KD signal.

# References
# ----------
# Hinton et al. 2015, arXiv:1503.02531: foundational KD
# MemKD 2025, arXiv:2601.04264: feature matching for TS KD
# Gou et al. 2021, IJCV: KD survey, T² gradient compensation

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    T: float,
    alpha: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined response-based KD loss + hard-label CE.

    Parameters
    ----------
    student_logits : torch.Tensor
        (batch, n_classes) — raw student logits (before softmax).
    teacher_logits : torch.Tensor
        (batch, n_classes) — raw cached teacher logits (before softmax).
        Must NOT be softmaxed — see module docstring.
    hard_labels : torch.Tensor
        (batch,) — integer ground-truth class labels.
    T : float
        Temperature for softening distributions.  Should not exceed 5
        for binary classification (plan recommendation).
    alpha : float
        Distillation weight.  alpha=0 → pure CE, alpha=1 → pure KD.
    class_weights : torch.Tensor, optional
        (n_classes,) — per-class weights for the CE term to handle
        class imbalance.  Applied to the CE term only (not the KD term),
        following the plan's specification: "class weights in CE loss
        (not KD term)".

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if T > 5:
        warnings.warn(
            f"Temperature T={T} exceeds 5.  For binary classification, "
            f"T>5 makes the softened distribution nearly uniform, "
            f"destroying the KD signal.  The plan recommends T ≤ 5.",
            UserWarning,
            stacklevel=2,
        )

    # KL divergence on softened distributions
    # F.kl_div expects log-probabilities as input, probabilities as target.
    if alpha > 0:
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # ∂KL/∂z_s scales as 1/T², so multiplying by T² restores the
        # gradient magnitude to be independent of T (Hinton et al. 2015).
        loss_kd = T * T * F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    else:
        loss_kd = torch.tensor(0.0, device=student_logits.device)

    # hard-label cross-entropy
    if alpha < 1:
        # Defensive dtype/device cast for class_weights to prevent
        # silent precision issues (float64 from numpy) or device mismatches.
        if class_weights is not None:
            class_weights = class_weights.to(
                dtype=student_logits.dtype,
                device=student_logits.device,
            )
        loss_ce = F.cross_entropy(
            student_logits,
            hard_labels,
            weight=class_weights,
        )
    else:
        loss_ce = torch.tensor(0.0, device=student_logits.device)

    # Combined
    loss = alpha * loss_kd + (1.0 - alpha) * loss_ce

    return loss


def kd_loss_with_features(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined KD loss + feature-matching loss.

    Extends ``kd_loss`` with an additional MSE term on intermediate
    representations

    Parameters
    ----------
    student_logits, teacher_logits, hard_labels, T, alpha, class_weights
        Same as ``kd_loss``.
    student_features : torch.Tensor
        (batch, dim) — student's intermediate representation from
        StudentBase (dim=32).
    teacher_features : torch.Tensor
        (batch, dim) — teacher's post-BN LSTM hidden state from
        KDTeacherWrapper (dim=32).
    beta : float
        Feature-matching weight.  beta=0 → response-only KD.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    # Response-based loss (KD + CE)
    loss_response = kd_loss(
        student_logits,
        teacher_logits,
        hard_labels,
        T,
        alpha,
        class_weights,
    )

    # Feature-matching loss
    if beta > 0:
        loss_feat = beta * F.mse_loss(student_features, teacher_features)
    else:
        loss_feat = torch.tensor(0.0, device=student_features.device)

    return loss_response + loss_feat

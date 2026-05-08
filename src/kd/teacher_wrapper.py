# KD Teacher Wrapper for Feature Extraction Without Modifying Production Code

# Subclass that wraps a WESADFLModel and overrides forward() to optionally return the intermediate representation
# (post-BatchNorm LSTM hidden state, dim=32) alongside logits.

# This avoids modifying wesad_fl_model.py or  wesad_cnnlstm_model.py while enabling feature-matching
# loss and teacher logit caching.

# The wrapper shares parameters with the original model (no copying)

# Architecture:
# The wrapper reaches into the head's internal sub-modules::

#     self.head.lstm        → LSTM temporal modelling
#     self.head.lstm_bn     → BatchNorm on final hidden state
#     self.head.classifier  → Dropout + Linear(32, 2)

# to extract the post-BatchNorm hidden state (dim=32) before it enters
# the classifier.  This is the representation that matches the student's
# intermediate dim=32.

# References
# ----------
# Hinton et al. 2015: offline distillation with frozen teacher
# Gou et al. 2021, IJCV: KD survey
# arXiv:2510.24058: frozen EDA teacher

from __future__ import annotations

import torch
import torch.nn as nn

from src.kd.kd_config import (
    TEACHER_LSTM_HIDDEN,
    TEACHER_TARGET_TIME,
)


class KDTeacherWrapper(nn.Module):
    """Wrapper for frozen teacher with intermediate feature extraction.

    Parameters:
    fl_model : WESADFLModel
        A pre-trained FL model with ``.base`` (WESADBase) and ``.head``
        (WESADHead) sub-modules.  The wrapper shares parameters with
        this model so no weight copying occurs.

    Usage:
    >>> from wesad_fl_model import build_fl_wrist_model
    >>> fl_model = build_fl_wrist_model()
    >>> fl_model.load_state_dict(torch.load("checkpoint.pt"))
    >>> wrapper = KDTeacherWrapper(fl_model)
    >>> wrapper.eval()
    >>> for p in wrapper.parameters(): p.requires_grad = False
    >>> logits, features = wrapper(signals, return_features=True)
    """

    def __init__(self, fl_model):
        super().__init__()
        self.base = fl_model.base
        self.head = fl_model.head

        # Runtime assertions (per kd_config.py documentation)
        assert self.head.lstm.hidden_size == TEACHER_LSTM_HIDDEN, (
            f"Teacher LSTM hidden size mismatch: expected {TEACHER_LSTM_HIDDEN}, "
            f"got {self.head.lstm.hidden_size}"
        )
        assert self.base.target_time == TEACHER_TARGET_TIME, (
            f"Teacher target_time mismatch: expected {TEACHER_TARGET_TIME}, "
            f"got {self.base.target_time}"
        )

    def forward(
        self,
        signals: dict[str, torch.Tensor],
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional intermediate feature extraction.

        Parameters:
        signals : dict[str, torch.Tensor]
            signal_name → (batch, channels, time_steps).
            Must include all 4 wrist signals: bvp, w_eda, w_acc, w_temp.
        return_features : bool
            If True, also return the post-BatchNorm LSTM hidden state
            (dim=32) for feature-matching loss.

        Returns:
        logits : torch.Tensor
            (batch, n_classes) — raw logits.
        features : torch.Tensor, optional
            (batch, 32) — intermediate representation, only if
            ``return_features=True``.
        """
        # The teacher MUST be in eval mode during inference to prevent
        # BatchNorm from updating running statistics with student data.
        assert not self.training, (
            "KDTeacherWrapper must be in eval mode during inference. "
            "Call wrapper.eval() before use."
        )

        # CNN base: extract shared features from all modalities
        fused = self.base(signals)

        # LSTM: temporal modelling
        lstm_out, (h_n, c_n) = self.head.lstm(fused)

        # Post-BatchNorm hidden state (the feature-matching target)
        last_hidden = self.head.lstm_bn(h_n[-1])  # (batch, 32)

        # Classification
        logits = self.head.classifier(last_hidden)  # (batch, 2)

        if return_features:
            return logits, last_hidden
        return logits

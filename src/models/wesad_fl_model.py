# Base/head wrapper for federated learning model splitting.
#
# Wraps an existing CNNLSTM instance and exposes .base and .head as
# nn.Module sub-modules, as required by PFLlib's FedRep, FedBABU,
# and Ditto implementations.
#
# does NOT modify the original CNNLSTM class, the standalone
# wesad_cnnlstm_train.py and wesad_cnnlstm_evaluate.py continue
# to work unchanged.
#
# Adapts from:
#   - wesad_cnnlstm_model.py       : the wrapped model
#   - PFLIB clientrep.py            : model.base.parameters(), model.head.parameters()
#   - PFLIB clientbabu.py           : model.head requires_grad = False
#   - PFLIB serverrep.py            : uploads client.model.base only
#
# THE SPLIT DECISION
#
# model.base = CNN branches (self.branches ModuleDict)
#            + adaptive pools (self.adaptive_pools ModuleDict)
#            + temperature scalar processing
#
#   These are the shared physiological feature extractors that benefit
#   from federation.  CNN branches learn modality-specific physiological
#   features (BVP cardiac rhythm, EDA electrodermal response, ACC motion
#   patterns) that are universal across subjects.
#
# model.head = LSTM (self.lstm)
#            + BatchNorm (self.lstm_bn)
#            + classifier (self.classifier Sequential)
#
#   These are the temporal and decision layer that personalise
#   to each subject's stress response dynamics.  The LSTM learns the
#   temporal structure of how a specific individual transitions into and
#   out of stress.
#
# Literature justification:
#   Collins et al. 2021 (FedRep): "the representation should capture
#   shared structure while the head captures user-specific variation."
#
# TEMPERATURE DECISION
#   Temperature (self.use_temp) is kept in model.base.  The temperature
#   scalar is concatenated to the fused CNN output before it reaches the
#   LSTM, so it is a feature contribution to the shared representation,
#   not a classifier parameter.
#
#   Only base parameters are communicated per round → major communication
#   reduction compared to full-model FedAvg, in accordance with pfl literature

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.wesad_cnnlstm_model import (
    CNNLSTM,
    build_chest_model,
    build_wrist_model,
)

# Base module: CNN feature extraction (shared across clients)


class WESADBase(nn.Module):
    """Shared representation: CNN branches + adaptive pooling + temp fusion.

    Takes the raw signal dict and produces a fused feature tensor
    ready for the LSTM.  This is the module that gets aggregated on
    the server in FedRep/FedBABU.

    The forward pass replicates the first half of CNNLSTM.forward()
    using the nn.Module objects from the wrapped CNNLSTM instance.

    Output shape: (batch, target_time, total_features)
    """

    def __init__(self, cnnlstm: CNNLSTM):
        super().__init__()
        self.branches = cnnlstm.branches
        self.adaptive_pools = cnnlstm.adaptive_pools

        # Metadata needed for the forward pass (not learnable)
        self.branch_names = cnnlstm.branch_names
        self.use_temp = cnnlstm.use_temp
        self.temp_signal_name = cnnlstm.temp_signal_name
        self.target_time = cnnlstm.target_time

    def forward(self, signals: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract shared features from raw signals.

        Parameters:
        signals : dict[str, torch.Tensor]
            signal_name → (batch, channels, time_steps)

        Returns:
        torch.Tensor
            (batch, target_time, total_features) — ready for LSTM input.
        """
        branch_outputs = []

        for sig_name in self.branch_names:
            if sig_name not in signals:
                raise ValueError(
                    f"Signal '{sig_name}' expected but not found in input. "
                    f"Available: {list(signals.keys())}"
                )

            x = signals[sig_name]  # (batch, C, T)
            x = self.branches[sig_name](x)  # (batch, n_filters, compressed_T)

            # adaptive pooling must compress, not upsample
            compressed_T = x.shape[2]
            if compressed_T < self.target_time:
                raise RuntimeError(
                    f"Branch '{sig_name}': CNN output has {compressed_T} timesteps "
                    f"but target_time={self.target_time}. Adaptive pooling would "
                    f"upsample. Reduce target_time or use a longer window."
                )

            x = self.adaptive_pools[sig_name](x)  # (batch, n_filters, target_time)
            branch_outputs.append(x)

        # Fuse branches along feature axis
        fused = torch.cat(
            branch_outputs, dim=1
        )  # (batch, total_cnn_features, target_time)

        # Temperature scalar feature (no learnable parameters)
        if self.use_temp and self.temp_signal_name in signals:
            temp = signals[self.temp_signal_name]  # (batch, 1, T_temp)
            temp_mean = temp.mean(dim=2, keepdim=True)  # (batch, 1, 1)
            temp_expanded = temp_mean.expand(
                -1, 1, self.target_time
            )  # (batch, 1, target_time)
            fused = torch.cat(
                [fused, temp_expanded], dim=1
            )  # (batch, total_features, target_time)

        # Transpose for LSTM: (batch, time, features)
        fused = fused.permute(0, 2, 1)  # (batch, target_time, total_features)

        return fused


# Head module: LSTM temporal modelling + classification (local per client)


class WESADHead(nn.Module):
    """Personal head: LSTM + BatchNorm + classifier.

    Takes the fused feature tensor from WESADBase and produces logits.
    This module stays local in FedRep (never sent to server) and is
    frozen during FL rounds in FedBABU.

    Input shape:  (batch, target_time, total_features)
    Output shape: (batch, n_classes)
    """

    def __init__(self, cnnlstm: CNNLSTM):
        super().__init__()
        # temporal + classification sub-modules
        self.lstm = cnnlstm.lstm
        self.lstm_bn = cnnlstm.lstm_bn
        self.classifier = cnnlstm.classifier

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Temporal modelling and classification.

        Parameters:
        fused : torch.Tensor
            (batch, target_time, total_features) from WESADBase.

        Returns:
        torch.Tensor
            (batch, n_classes) — raw logits for CrossEntropyLoss.
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(fused)
        # Use hidden state from last timestep of last layer
        last_hidden = h_n[-1]  # (batch, lstm_hidden)

        # BatchNorm on final hidden state
        last_hidden = self.lstm_bn(last_hidden)

        # Classification
        logits = self.classifier(last_hidden)  # (batch, n_classes)

        return logits


# FL Model wrapper: composes base + head with the PFLlib interface


class WESADFLModel(nn.Module):
    """Federated learning wrapper for CNNLSTM with base/head split.

    Exposes .base (WESADBase) and .head (WESADHead) as named
    sub-modules, matching the interface expected by:
      - PFLIB clientrep.py:  model.base.parameters(), model.head.parameters()
      - PFLIB clientbabu.py: model.head requires_grad = False
      - PFLIB serverrep.py:  client.model.base (for aggregation)

    The forward() method calls base then head in sequence, producing
    the same output as the original CNNLSTM.forward().

    Parameters:
    cnnlstm : CNNLSTM
        An existing CNNLSTM instance to wrap.  The wrapper takes
        ownership of the sub-modules (branches, pools, lstm, etc.)
        by re-registering them under .base and .head.  The original
        CNNLSTM object will not be used after wrapping.

    Usage
    -----
    >>> from wesad_cnnlstm_model import build_wrist_model
    >>> cnnlstm = build_wrist_model(n_classes=2)
    >>> fl_model = WESADFLModel(cnnlstm)
    >>> # FedRep: separate optimisers
    >>> opt_base = AdamW(fl_model.base.parameters(), lr=1e-3)
    >>> opt_head = AdamW(fl_model.head.parameters(), lr=1e-3)
    >>> # FedBABU: freeze head
    >>> for p in fl_model.head.parameters(): p.requires_grad = False
    >>> # Server aggregation: send only base
    >>> server_model = copy.deepcopy(fl_model.base)
    """

    def __init__(self, cnnlstm: CNNLSTM):
        super().__init__()

        # The base and head constructors re-register the sub-modules
        # from cnnlstm under their own parameter trees.  After this,
        # fl_model.base.parameters() returns CNN params and
        # fl_model.head.parameters() returns LSTM+classifier params.
        self.base = WESADBase(cnnlstm)
        self.head = WESADHead(cnnlstm)

    def forward(self, signals: dict[str, torch.Tensor]) -> torch.Tensor:
        """Full forward pass: base → head.

        Produces identical output to the original CNNLSTM.forward().

        Parameters:
        signals : dict[str, torch.Tensor]
            signal_name → (batch, channels, time_steps)

        Returns:
        torch.Tensor
            (batch, n_classes) — raw logits.
        """
        fused = self.base(signals)
        logits = self.head(fused)
        return logits


def build_fl_wrist_model(
    n_classes: int = 2,
    lstm_hidden: int = 32,
    lstm_layers: int = 2,
    cnn_dropout: float = 0.5,
    fc_dropout: float = 0.5,
    n_conv_layers: int = 1,
    n_filters_override: int | None = None,
    target_time: int = 8,
) -> WESADFLModel:
    """Build a wrist FL model with base/head split.

    Convenience function that builds a CNNLSTM via build_wrist_model()
    and wraps it in WESADFLModel.

    All parameters are forwarded to build_wrist_model().
    """
    cnnlstm = build_wrist_model(
        n_classes=n_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        cnn_dropout=cnn_dropout,
        fc_dropout=fc_dropout,
        n_conv_layers=n_conv_layers,
        n_filters_override=n_filters_override,
        target_time=target_time,
    )
    return WESADFLModel(cnnlstm)


def build_fl_chest_model(
    n_classes: int = 2,
    lstm_hidden: int = 32,
    lstm_layers: int = 2,
    cnn_dropout: float = 0.5,
    fc_dropout: float = 0.5,
    n_conv_layers: int = 1,
    n_filters_override: int | None = None,
    target_time: int = 5,
) -> WESADFLModel:
    """Build a chest FL model with base/head split.

    Convenience function that builds a CNNLSTM via build_chest_model()
    and wraps it in WESADFLModel.
    """
    cnnlstm = build_chest_model(
        n_classes=n_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        cnn_dropout=cnn_dropout,
        fc_dropout=fc_dropout,
        n_conv_layers=n_conv_layers,
        n_filters_override=n_filters_override,
        target_time=target_time,
    )
    return WESADFLModel(cnnlstm)


def build_fl_model_from_config(config) -> WESADFLModel:
    """Build an FL model from an FLConfig dataclass.

    Parameters:
    config : FLConfig
        Configuration with device_type, n_classes, lstm_hidden, etc.
        If config has a target_time attribute, it is forwarded to the
        factory function.  Otherwise, the per-device default is used
        (wrist=8, chest=5).

    Returns:
    WESADFLModel
    """
    build_fn = (
        build_fl_chest_model if config.device_type == "chest" else build_fl_wrist_model
    )

    kwargs = {
        "n_classes": config.n_classes,
        "lstm_hidden": config.lstm_hidden,
        "lstm_layers": config.lstm_layers,
        "cnn_dropout": config.cnn_dropout,
        "fc_dropout": config.fc_dropout,
        "n_conv_layers": config.n_conv_layers,
    }

    # Forward target_time if present in config (needed for window ablations)
    if hasattr(config, "target_time") and config.target_time is not None:
        kwargs["target_time"] = config.target_time

    return build_fn(**kwargs)


def wrap_pretrained_model(cnnlstm: CNNLSTM) -> WESADFLModel:
    """Wrap a pre-trained CNNLSTM for federated learning.

    load a pre-trained CNNLSTM checkpoint,
    then wrap it to get the base/head split for FedRep initialisation.

    This is the FedHealth two-step paradigm (Chen et al. 2020):
    pre-trained model → FL with personalisation.

    Parameters:
    cnnlstm : CNNLSTM
        A pre-trained CNNLSTM instance ( from wesad_cnnlstm_train.py).

    Returns:
    WESADFLModel
        The same model with base/head interface.  All weights are
        preserved from the pre-trained checkpoint.
    """
    return WESADFLModel(cnnlstm)


# Diagnostic utilities


def print_split_summary(model: WESADFLModel):
    """Print parameter counts for base and head."""
    base_params = sum(p.numel() for p in model.base.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    total = base_params + head_params
    base_pct = 100 * base_params / total if total > 0 else 0
    head_pct = 100 * head_params / total if total > 0 else 0

    print("\n WESADFLModel Split Summary")
    print(f"  Base (shared, communicated):  {base_params:,} params ({base_pct:.1f}%)")
    print(f"  Head (local, personalised):   {head_params:,} params ({head_pct:.1f}%)")
    print(f"  Total:                        {total:,} params")
    print(f"  Communication reduction vs full model: {head_pct:.1f}%")

    # Detail by sub-module
    print("\n  Base sub-modules:")
    for name, param in model.base.named_parameters():
        print(f"    {name}: {param.shape} ({param.numel():,})")

    print("\n  Head sub-modules:")
    for name, param in model.head.named_parameters():
        print(f"    {name}: {param.shape} ({param.numel():,})")


def verify_equivalence(
    signals: dict[str, torch.Tensor],
    build_fn=None,
    build_kwargs: dict | None = None,
    atol: float = 1e-6,
) -> bool:
    """Verify that WESADFLModel produces identical output to CNNLSTM.

    This function builds TWO independent models from scratch
    with the same random seed, wraps one in WESADFLModel, and compares
    their outputs to test equivalence.

    Debugged fix: do not pass the same CNNLSTM instance that was used to construct
    the WESADFLModel because after wrapping both CNNLSTM's sub-modules and the wrappee share the same
    tensors in memory and comparison would be trivially True.

    Parameters:
    signals : dict[str, torch.Tensor]
        A sample input batch.
    build_fn : callable, optional
        Factory function (e.g. build_wrist_model).  Defaults to build_wrist_model.
    build_kwargs : dict, optional
        Keyword arguments for the factory function.
    atol : float
        Absolute tolerance for floating-point comparison.

    Returns:
    bool
        True if outputs match within tolerance.

    """

    if build_fn is None:
        build_fn = build_wrist_model
    if build_kwargs is None:
        build_kwargs = {"n_classes": 2}

    # Build reference model with fixed seed
    torch.manual_seed(0)
    reference = build_fn(**build_kwargs)

    # Build identical model and wrap it
    torch.manual_seed(0)
    to_wrap = build_fn(**build_kwargs)
    fl_model = WESADFLModel(to_wrap)

    # Compare forward pass outputs
    reference.eval()
    fl_model.eval()

    with torch.no_grad():
        ref_out = reference(signals)
        fl_out = fl_model(signals)

    match = torch.allclose(fl_out, ref_out, atol=atol)

    # Also verify parameter counts match
    ref_count = sum(p.numel() for p in reference.parameters())
    fl_count = sum(p.numel() for p in fl_model.parameters())
    count_match = ref_count == fl_count

    if match and count_match:
        print("    Equivalence verified: WESADFLModel output matches CNNLSTM")
        print(
            f"    Parameter count: {ref_count:,} (reference) == {fl_count:,} (FL wrapper)"
        )
    else:
        if not match:
            max_diff = (fl_out - ref_out).abs().max().item()
            print(f"  Output equivalence FAILED: max difference = {max_diff}")
        if not count_match:
            print(
                f"  Parameter count MISMATCH: {ref_count:,} (reference) vs {fl_count:,} (FL wrapper)"
            )

    return bool(match) and count_match

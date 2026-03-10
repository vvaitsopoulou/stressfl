# CNN-LSTM Model

# Multi-branch CNN-LSTM architecture for wearable stress detection.
# Each physiological modality has its own CNN branch with biologically-
# justified kernel and pooling sizes.  Branch outputs are temporally
# standardised via adaptive average pooling, then fused and passed to
# a shared LSTM for temporal modelling.

# Architecture Flow
# 1. Input: dict of signal tensors, one per modality (batch, C, T)
# 2. CNN branches: one per modality: Conv1d → BatchNorm → ReLU → MaxPool → Dropout
#    Each branch has kernel/pool sizes derived from the signal's physiological
#    periodicity (Hwang et al., 2018).
# 3. Adaptive pooling: nn.AdaptiveAvgPool1d(target_time) standardises all
#    branch outputs to a common temporal dimension before fusion.
# 4. TEMP scalar: mean temperature of the window is concatenated to the
#    fused features at each timestep (no CNN branch because temperature changes
#    over minutes, not seconds).
# 5. Feature fusion: concatenate all branch outputs along channel axis →
#    (batch, total_features, target_time)
# 6. LSTM: 2-layer LSTM reads the fused sequence.
# 7. Classifier: Dense → Dropout → output logits.

# CNN Branch Kernel/Pool Sizing (from literature):
# Signal    Hz    Kernel         Pool           Biological Basis
# ──────    ──    ──────         ────           ────────────────
# ECG       256   154 (0.6s)     205 (0.8s)     PQRST = 0.57–0.67s; heartbeat = 0.8s
#                                                (Hwang et al., 2018)
# BVP       32    19  (0.6s)     26  (0.8s)     Optical analogue of ECG — same cardiac
#                                                cycle (Hwang principle)
# EMG       64    32  (0.5s)     64  (1.0s)     Rectified envelope: stress-related muscle
#                                                tension manifests as tonic contractions
#                                                (0.5–2s) and phasic bursts (100–500ms).
#                                                Kernel covers one burst; pool covers one
#                                                activation-relaxation cycle.
# EDA       8     16  (2.0s)     8   (1.0s)     SCR onset 1–3s (Moser et al., 2024)
# EDA wrist 4     8   (2.0s)     4   (1.0s)     Same SCR physiology at lower rate
# RESP      32    48  (1.5s)     64  (2.0s)     Kernel covers one inhale-exhale phase
#                                                under stress (~1.5s).  Pool covers one
#                                                full stressed breath cycle (~2s).
#                                                Plan's original 3s/5s values were for
#                                                60s windows; at 16s they produce only
#                                                2 post-pool steps.  Adapted to stressed
#                                                breathing rate (20–30 bpm = 2–3s/cycle).
# ACC       32    19  (0.6s)     26  (0.8s)     Motion at cardiac/respiratory scale

# Weight Initialization (Moser et al., 2024)
#   CNN:  Kaiming/He (PyTorch default for ReLU layers)
#   LSTM recurrent: orthogonal initialization
#   LSTM input: Xavier/Glorot initialization
#   Biases: 0, EXCEPT forget gate bias → 1 (mitigates vanishing gradients)

# References
#   Hwang et al. (2018) — Deep ECGNet: kernel/pool from ECG periodicity
#   Ordóñez & Roggen (2016) — DeepConvLSTM: ReLU, Tanh, 2 LSTM layers,
#     dropout p=0.5, 64 filters
#   Prajod & André (2022) — Deep ECGNet adaptation on WESAD
#   Moser et al. (2024) — weight initialization, forget gate bias = 1

import torch
import torch.nn as nn

#  CNN Branch (one per modality)


class CNNBranch(nn.Module):
    """Single-modality 1D CNN feature extractor.

    Architecture per branch: Conv1d → BatchNorm1d → ReLU → MaxPool1d → Dropout

    Parameters
    in_channels : int
        Number of input channels (1 for scalar signals, 3 for ACC).
    n_filters : int
        Number of convolutional filters (feature maps).
    kernel_size : int
        Convolution kernel size in samples (biologically justified).
    pool_size : int
        Max pooling kernel size in samples (biologically justified).
    dropout : float
        Dropout probability after pooling.
    n_conv_layers : int
        Number of conv layers.  Default 1 (Deep ECGNet baseline).
        Ablation: 2 or 3 (Ordóñez & Roggen show diminishing returns).
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        pool_size: int,
        dropout: float = 0.5,
        n_conv_layers: int = 1,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(n_conv_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        current_channels, n_filters, kernel_size=kernel_size, padding=0
                    ),  # valid convolution, no padding
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                ]
            )
            current_channels = n_filters
            # Only pool after the last conv layer to avoid over-compression
            # on short windows (10s ablation).  Additional conv layers
            # reduce the sequence by (kernel_size - 1) each, which is
            # manageable, but stacking multiple pools would collapse it.
            if i == n_conv_layers - 1:
                layers.append(nn.MaxPool1d(kernel_size=pool_size))

        layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)
        self.out_channels = n_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        x : (batch, in_channels, time_steps)

        Returns
        (batch, n_filters, compressed_time)
        """
        return self.net(x)


#  Branch configuration definitions

# Default branch configs: biologically justified kernel adn pool sizes.
# These are the BASELINE values.  Ablation overrides n_filters, n_conv_layers, etc

CHEST_BRANCH_CONFIGS = {
    "ecg": {"in_channels": 1, "n_filters": 50, "kernel_size": 154, "pool_size": 205},
    "emg": {"in_channels": 1, "n_filters": 32, "kernel_size": 32, "pool_size": 64},
    "c_eda": {"in_channels": 1, "n_filters": 32, "kernel_size": 16, "pool_size": 8},
    "resp": {"in_channels": 1, "n_filters": 32, "kernel_size": 48, "pool_size": 64},
    "c_acc": {"in_channels": 3, "n_filters": 32, "kernel_size": 19, "pool_size": 26},
}

WRIST_BRANCH_CONFIGS = {
    "bvp": {"in_channels": 1, "n_filters": 50, "kernel_size": 19, "pool_size": 26},
    "w_eda": {"in_channels": 1, "n_filters": 32, "kernel_size": 8, "pool_size": 4},
    "w_acc": {"in_channels": 3, "n_filters": 32, "kernel_size": 19, "pool_size": 26},
}

#  Main model


class CNNLSTM(nn.Module):
    """Multi-branch CNN-LSTM for wearable stress detection.

    Parameters
    ----------
    branch_configs : dict[str, dict]
        Per-signal CNN branch configuration.
        Keys = signal names matching the Dataset output.
        Values = dict with in_channels, n_filters, kernel_size, pool_size.
    n_classes : int
        Number of output classes (2 for binary, 3 for 3-class).
    lstm_hidden : int
        Hidden units in each LSTM layer.  Default 32 (Deep ECGNet).
    lstm_layers : int
        Number of LSTM layers.  Default 2 (Ordóñez & Roggen).
    lstm_dropout : float
        Dropout between LSTM layers (PyTorch built-in).  Default 0.2
        (Prajod & André, 2022; Hwang et al., 2018).  Applied only
        when lstm_layers > 1.
    fc_dropout : float
        Dropout before the classification head.
    cnn_dropout : float
        Dropout after each CNN branch.
    n_conv_layers : int
        Number of conv layers per branch.
    target_time : int
        Temporal dimension after adaptive pooling — the sequence length
        the LSTM reads.  All branches are pooled to this length before
        fusion.

        CRITICAL: target_time must be ≤ the minimum post-CNN-pool
        sequence length across ALL branches.  If any branch produces
        fewer timesteps than target_time, the forward pass raises a
        RuntimeError (adaptive pooling would upsample, not compress).

        The chest and wrist models have DIFFERENT optimal defaults
        because RESP (a chest-only signal with large kernel/pool) is
        the chest bottleneck, while the wrist model is unconstrained:
          - Chest default: target_time=5  (safe across all ablations)
          - Wrist default: target_time=8  (safe across all ablations)

        Use the factory functions build_chest_model / build_wrist_model
        which set the correct default per device.  If constructing
        CNNLSTM directly, choose target_time based on the signal set.

        For ablation: keep target_time constant within each ablation
        so only the intended variable changes (window length, CNN
        layers, etc.).
    use_temp : bool
        Whether to include temperature as a scalar feature concatenated
        to the fused representation.  Default True.
    temp_signal_name : str
        Name of the temperature signal in the input dict.
    """

    def __init__(
        self,
        branch_configs: dict[str, dict],
        n_classes: int = 2,  # TODO I need to add class flexibility
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,  # Prajod & André (2022), Hwang et al. (2018)
        fc_dropout: float = 0.5,
        cnn_dropout: float = 0.5,
        n_conv_layers: int = 1,
        target_time: int = 5,  # conservative default, we use factory functions instead
        use_temp: bool = True,
        temp_signal_name: str = "w_temp",
    ):
        super().__init__()

        self.branch_names = []  # signal names that have CNN branches
        self.use_temp = use_temp
        self.temp_signal_name = temp_signal_name
        self.target_time = target_time

        # Build CNN branches
        self.branches = nn.ModuleDict()
        self.adaptive_pools = nn.ModuleDict()

        total_features = 0

        for sig_name, cfg in branch_configs.items():
            # Skip temperature its handled as scalar, not CNN branch
            if sig_name in ("w_temp", "c_temp"):
                continue

            branch = CNNBranch(
                in_channels=cfg["in_channels"],
                n_filters=cfg["n_filters"],
                kernel_size=cfg["kernel_size"],
                pool_size=cfg["pool_size"],
                dropout=cnn_dropout,
                n_conv_layers=n_conv_layers,
            )
            self.branches[sig_name] = branch
            self.adaptive_pools[sig_name] = nn.AdaptiveAvgPool1d(target_time)
            self.branch_names.append(sig_name)
            total_features += cfg["n_filters"]

        # TEMP scalar feature (+1 to fused features if used)
        if use_temp:
            total_features += 1  # mean temperature as scalar

        # LSTM
        # Input: (batch, target_time, total_features) after transpose
        self.lstm = nn.LSTM(
            input_size=total_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # BatchNorm on LSTM output
        # Applied to the hidden state at the final timestep.
        #  Hwang et al. (2018) include batch normalization
        # between LSTM layers to prevent exploding/vanishing gradients.
        self.lstm_bn = nn.BatchNorm1d(lstm_hidden)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(lstm_hidden, n_classes),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Apply literature-justified weight initialization.

        CNN: Kaiming/He (PyTorch default for Conv1d ).
        LSTM recurrent: orthogonal (Ordóñez & Roggen, 2016, Moser, 2024).
        LSTM input: Xavier/Glorot (Moser et al., 2024).
        Biases: 0, EXCEPT forget gate bias → 1 (Moser et al., 2024;
          mitigates vanishing gradients in the forget gate).

        PyTorch LSTM stores all 4 gate weight matrices concatenated:
          [input_gate | forget_gate | cell_gate | output_gate]
        Each has size hidden_size, so forget gate bias occupies
        indices [hidden_size : 2*hidden_size] in bias_ih and bias_hh.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                # Input-to-hidden weights: Xavier initialization
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                # Hidden-to-hidden (recurrent) weights: orthogonal
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                # All biases to 0 first
                nn.init.zeros_(param.data)
                # Then set forget gate bias to 1
                # Forget gate is the second of 4 gates in the concatenated bias
                hidden_size = self.lstm.hidden_size
                param.data[hidden_size : 2 * hidden_size] = 1.0

        # Linear layers: Xavier
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        signals: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters:
        signals : dict[str, torch.Tensor]
            signal_name → (batch, channels, time_steps) as produced
            by WESADDLDataset.__getitem__.

        Returns:
        logits : torch.Tensor
            (batch, n_classes) — raw logits (pass to CrossEntropyLoss).
        """
        branch_outputs = []

        # CNN branches
        for sig_name in self.branch_names:
            if sig_name not in signals:
                raise ValueError(
                    f"Signal '{sig_name}' expected but not found in input. "
                    f"Available: {list(signals.keys())}"
                )

            x = signals[sig_name]  # (batch, C, T)
            x = self.branches[sig_name](x)  # (batch, n_filters, compressed_T)

            # Guard: adaptive pooling must compress, never upsample.
            # If the CNN output sequence is shorter than target_time,
            # AdaptiveAvgPool would duplicate timesteps creating
            # meaningless repeated values that degrade LSTM learning.
            # This happens when window_sec is too short for the current
            # architecture (e.g. 10s window with 2-layer CNN on ECG).
            # TODO check Fix: reduce target_time for shorter window ablations.
            compressed_T = x.shape[2]
            if compressed_T < self.target_time:
                raise RuntimeError(
                    f"Branch '{sig_name}': CNN output has {compressed_T} timesteps "
                    f"but target_time={self.target_time}. Adaptive pooling would "
                    f"upsample (interpolate), not compress. Either reduce target_time "
                    f"to ≤ {compressed_T}, use a shorter kernel/pool, or use a "
                    f"longer window."
                )

            x = self.adaptive_pools[sig_name](x)  # (batch, n_filters, target_time)
            branch_outputs.append(x)

        # Fuse branches
        # Concatenate along feature/channel axis
        fused = torch.cat(
            branch_outputs, dim=1
        )  # (batch, total_cnn_features, target_time)

        # TEMP scalar feature
        if self.use_temp and self.temp_signal_name in signals:
            temp = signals[self.temp_signal_name]  # (batch, 1, T_temp)
            # Compute mean temperature across the window → scalar
            temp_mean = temp.mean(dim=2, keepdim=True)  # (batch, 1, 1)
            # Expand to match temporal dimension and concatenate
            temp_expanded = temp_mean.expand(
                -1, 1, self.target_time
            )  # (batch, 1, target_time)
            fused = torch.cat(
                [fused, temp_expanded], dim=1
            )  # (batch, total_features, target_time)

        # Transpose for LSTM: (batch, time, features)
        fused = fused.permute(0, 2, 1)  # (batch, target_time, total_features)

        #  LSTM
        lstm_out, (h_n, c_n) = self.lstm(fused)
        # Use the hidden state from the last timestep of the last layer
        last_hidden = h_n[-1]  # (batch, lstm_hidden)

        # BatchNorm on final hidden state
        last_hidden = self.lstm_bn(last_hidden)

        # Classification
        logits = self.classifier(last_hidden)  # (batch, n_classes)

        return logits


#  Factory functions for chest and wrist models


def build_chest_model(
    n_classes: int = 2,
    lstm_hidden: int = 32,
    lstm_layers: int = 2,
    cnn_dropout: float = 0.5,
    fc_dropout: float = 0.5,
    n_conv_layers: int = 1,
    n_filters_override: int | None = None,
    target_time: int = 5,
) -> CNNLSTM:
    """Build the chest CNN-LSTM model (upper-bound baseline).

    Uses ECG, EMG, chest EDA, RESP, chest ACC branches.
    TEMP is included as a scalar feature (not a CNN branch).

    Default target_time=5 is constrained by the RESP branch (kernel=48,
    pool=64).  Safety table:
      16s window: safe across 1–3 CNN layers (RESP min = 5 @ 3-layer) ✓
      30s window: safe across 1–3 CNN layers (RESP min = 12) ✓
      10s window: NOT safe — RESP produces 4 @ 1-layer, 3 @ 2-layer,
        2 @ 3-layer.  The 10s ablation is not recommended for the chest
        model.  We will tun the 10s window ablation on the wrist model only.

    This is lower than the wrist model's target_time because RESP is
    a chest-only signal.  The chest model is the upper-bound comparison,
    not the primary FL contribution.
    """
    configs = {k: dict(v) for k, v in CHEST_BRANCH_CONFIGS.items()}

    if n_filters_override is not None:
        for cfg in configs.values():
            cfg["n_filters"] = n_filters_override

    return CNNLSTM(
        branch_configs=configs,
        n_classes=n_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        cnn_dropout=cnn_dropout,
        fc_dropout=fc_dropout,
        n_conv_layers=n_conv_layers,
        target_time=target_time,
        use_temp=True,
        temp_signal_name="c_temp",
    )


def build_wrist_model(
    n_classes: int = 2,
    lstm_hidden: int = 32,
    lstm_layers: int = 2,
    cnn_dropout: float = 0.5,
    fc_dropout: float = 0.5,
    n_conv_layers: int = 1,
    n_filters_override: int | None = None,
    target_time: int = 8,
) -> CNNLSTM:
    """Build the wrist CNN-LSTM model (FL candidate).

    Uses BVP, wrist EDA, wrist ACC branches.
    TEMP is included as a scalar feature (not a CNN branch).

    Default target_time=8 is constrained by w_eda (kernel=8, pool=4).
    Safety table:
      16s window: safe across 1–3 CNN layers (w_eda min = 10 @ 3-layer) ✓
      30s window: safe across 1–3 CNN layers (w_eda min = 25) ✓
      10s window / 1 CNN layer: safe (w_eda = 8) ✓
      10s window / 2–3 CNN layers: NOT safe — w_eda produces 6 @ 2-layer,
        4 @ 3-layer.  Reduce target_time to 4 for 10s multi-layer ablation.

    This is higher than the chest model's target_time because the
    wrist model has no RESP branch (RESP is the chest bottleneck).
    The wrist model is the primary FL contribution and benefits from
    the higher temporal resolution.
    """
    configs = {k: dict(v) for k, v in WRIST_BRANCH_CONFIGS.items()}

    if n_filters_override is not None:
        for cfg in configs.values():
            cfg["n_filters"] = n_filters_override

    return CNNLSTM(
        branch_configs=configs,
        n_classes=n_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        cnn_dropout=cnn_dropout,
        fc_dropout=fc_dropout,
        n_conv_layers=n_conv_layers,
        target_time=target_time,
        use_temp=True,
        temp_signal_name="w_temp",
    )


#  Model summary utility


def print_model_summary(model: CNNLSTM, device_type: str = "wrist"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  CNN-LSTM Model Summary ({device_type})")
    print("\n")
    print(f"  Branches: {model.branch_names}")
    print(f"  TEMP scalar: {model.use_temp} ({model.temp_signal_name})")
    print(f"  Adaptive pool target: {model.target_time} timesteps")
    print(f"  LSTM: {model.lstm.num_layers} layers × {model.lstm.hidden_size} hidden")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

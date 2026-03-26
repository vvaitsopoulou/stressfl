# Biosignal embedding network for PeFLL.
#
# Takes a batch of multi-modal time-series windows from a client and
# produces a fixed-size embedding vector d ∈ R^embedding_dim that
# represents the client's physiological profile.
#
# Adapts from:
#   - PeFLL models.py MLPEmbed/CNNEmbed  : the permutation-invariant
#     embedding pattern (mean-aggregation over samples)
#
# DESIGN DECISIONS
#
# 1. PERMUTATION INVARIANCE (critical for PeFLL's theory):
#    f(batch) = f(shuffled_batch).  Achieved by mean-pooling over the
#    sample dimension.  We cannot use LSTM or attention over the batch
#    dimension because that would impose ordering and violate PeFLL's
#    theoretical guarantees, that the order of the samples does not matter (Scott et al. 2024).
#
# 2. ARCHITECTURE:
#    The embedding network processes multi-modal biosignal windows
#    using lightweight 1D CNNs (one per modality, similar to but
#    smaller than the main model's branches), fuses them, and
#    mean-pools across the batch to produce a client descriptor.
#
#    Flow: signals_dict → per-modality CNN → adaptive pool → concat
#          → per-sample MLP → mean over batch → embedding vector
#
#    This is a signal-based embedding.  For biosignals, signal-based
#    embedding was chosen over the label embedding done in the original paper
#    because the physiological profile (heart rate variability, EDA reactivity etc) is
#    encoded in the signal morphology and not the label distribution.
#
# 3. PARAMETER EFFICIENCY:
#    The embedding network is deliberately small (fewer filters,
#    single conv layer per branch) to keep the total parameter count
#    manageable.  It doesn't need to be as powerful as the main model
#    it only needs to produce a discriminative client descriptor.

from __future__ import annotations

import torch
import torch.nn as nn


class WESADEmbedNet(nn.Module):
    """Permutation-invariant biosignal embedding network.

    Takes a batch of multi-modal signal windows and produces a single
    embedding vector describing the client's physiological profile.

    Parameters:
    branch_configs : dict[str, dict]
        Per-signal configuration.  Keys = signal names, values = dict
        with 'in_channels' and 'n_filters'.  Kernel/pool sizes are
        simplified from the main model (fixed small values for the
        lightweight encoder).
    embedding_dim : int
        Dimensionality of the output embedding vector.
    encoder_hidden : int
        Hidden dimension of the per-sample MLP after branch fusion.
    target_time : int
        Adaptive pooling target (should match the main model).
    use_temp : bool
        Whether temperature is present in the signal dict.
    temp_signal_name : str
        Name of the temperature signal.
    """

    def __init__(
        self,
        branch_configs: dict[str, dict] | None = None,
        embedding_dim: int = 64,  # Scott et al. 2024 use 64–128 depending on task complexity
        encoder_hidden: int = 128,  # can be ablated later, chosen because it lies between 64 dims (embedded) and fused CNN 244 feats
        target_time: int = 8,  # same as baseline
        use_temp: bool = True,
        temp_signal_name: str = "w_temp",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.target_time = target_time
        self.use_temp = use_temp
        self.temp_signal_name = temp_signal_name

        # Default wrist branch configs (lightweight versions)
        # The ratio roughly follows the main model's ratio (50:32:32 ≈ 16:8:8 div down by 4) but this is a hyperparameter that should be validated TODO!! Ablate
        # but acc to paper filter counts are not the primary experimental variable.
        # The primary ablation axis for PeFLL is embedding_dim ∈ {32, 64, 128} whiole this is internal encoder architecture.
        if branch_configs is None:
            branch_configs = {
                "bvp": {"in_channels": 1, "n_filters": 16},
                "w_eda": {"in_channels": 1, "n_filters": 8},
                "w_acc": {"in_channels": 3, "n_filters": 8},
            }

        # Build lightweight CNN branches (single conv + pool per modality)
        self.branches = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        self.branch_names = []
        total_features = 0

        for sig_name, cfg in branch_configs.items():
            if sig_name in ("w_temp", "c_temp"):
                continue
            self.branches[sig_name] = nn.Sequential(
                nn.Conv1d(
                    cfg["in_channels"], cfg["n_filters"], kernel_size=7, padding=3
                ),
                nn.BatchNorm1d(cfg["n_filters"]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4),
            )
            self.pools[sig_name] = nn.AdaptiveAvgPool1d(target_time)
            self.branch_names.append(sig_name)
            total_features += cfg["n_filters"]

        if use_temp:
            total_features += 1

        # Per-sample MLP: fused features → sample encoding
        fused_dim = total_features * target_time
        self.sample_encoder = nn.Sequential(
            nn.Linear(fused_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, embedding_dim),
        )

    def forward(
        self,
        signals: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute client embedding from a batch of signal windows.

        Parameters:
        signals : dict[str, torch.Tensor]
            signal_name → (batch, channels, time_steps)

        Returns:
        torch.Tensor
            (embedding_dim,) — single embedding vector for this client.
            Mean-pooled over the batch dimension for permutation invariance.
        """
        branch_outputs = []

        for sig_name in self.branch_names:
            if sig_name not in signals:
                raise ValueError(f"Signal '{sig_name}' not found in input")
            x = signals[sig_name]  # (batch, C, T)
            x = self.branches[sig_name](x)  # (batch, n_filters, T')
            x = self.pools[sig_name](x)  # (batch, n_filters, target_time)
            branch_outputs.append(x)

        # Fuse along feature axis
        fused = torch.cat(branch_outputs, dim=1)  # (batch, total_features, target_time)

        # Temperature scalar
        if self.use_temp and self.temp_signal_name in signals:
            temp = signals[self.temp_signal_name]  # (batch, 1, T)
            temp_mean = temp.mean(dim=2, keepdim=True)  # (batch, 1, 1)
            temp_expanded = temp_mean.expand(-1, 1, self.target_time)
            fused = torch.cat([fused, temp_expanded], dim=1)

        # Flatten spatial dims for MLP
        batch_size = fused.shape[0]
        fused_flat = fused.reshape(
            batch_size, -1
        )  # (batch, total_features * target_time)

        # Per-sample encoding
        sample_encodings = self.sample_encoder(fused_flat)  # (batch, embedding_dim)

        # Mean-pool over batch dimension → permutation invariance
        client_embedding = sample_encodings.mean(dim=0)  # (embedding_dim,)

        return client_embedding

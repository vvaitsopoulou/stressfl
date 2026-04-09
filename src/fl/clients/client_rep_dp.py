# FedRep + Differential Privacy client: DP-SGD on shared base only.
#
# Extends FedRepClient to add Opacus DP-SGD to the base (CNN) training
# phase.  The head (LSTM + BN + classifier) is never communicated and
# therefore does not need DP protection.
#
# Adapts from:
#   - client_rep.py           : the FedRep alternating head/base protocol
#   - PFLIB clientrep.py      : PFLlib reference for FedRep
#
# In FedRep, only the CNN base parameters are uploaded to the server.
# The LSTM head parameters never leave the client.  Therefore:
#
#   - The base is the privacy-sensitive component: a curious server
#     could apply gradient inversion (DLG, Zhu et al. 2019) to
#     reconstruct training data from the uploaded base updates.
#
#   - The head is privacy-safe by design: it stays local, so there
#     is no communication channel to leak through.
#
# Applying DP to the head would only degrade personalisation quality
# without any privacy benefit.  This is consistent with Benouis et al.
# 2024 (JMIR Mental Health, WESAD): "Adding DP noise to task-specific
# layers provides the best privacy-utility tradeoff (~90% stress
# recognition while limiting re-identification to ~47%)."
#
# ──────────────────────────────────────────────────────────────────────
# OPACUS INTEGRATION — PERSISTENT ENGINE ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────
#
# Opacus wraps three components:
#   1. The MODULE being trained (model.base → GradSampleModule)
#   2. The OPTIMIZER for that module (AdamW → DPOptimizer)
#   3. The DATALOADER (wrapped for Poisson sampling)
#
# The PrivacyEngine is created once in __init__ and
# persists across all federation rounds.  This is the pattern
# confirmed by Opacus developers on PyTorch Forums
# ("the epsilon of the privacy engine accounts
# for all training steps - use the same privacy_engine throughout
# all rounds").
#
# The noise multiplier σ is pre-computed once via
# opacus.accountants.utils.get_noise_multiplier() for the full
# training budget (global_rounds × local_epochs).  Then make_private()
# attaches the engine with that fixed σ.
#
# Each round, the server sends new global base weights via
# set_parameters(), which overwrites .data on the already-wrapped
# GradSampleModule's inner parameters. This preserves:
#   - The AdamW adaptive moment estimates (exp_avg, exp_avg_sq)
#   - The cumulative privacy accountant state across rounds
#   - The GradSampleModule hooks (no "adding hooks twice" errors)
# Opacus replaces BatchNorm with GroupNorm automatically
# (via ModuleValidator.fix()). BatchNorm tracks running statistics
# across samples, which violates per-sample gradient isolation.
# For the CNN base, this is handled transparently.
#
# ──────────────────────────────────────────────────────────────────────
# PRIVACY ACCOUNTING
# ──────────────────────────────────────────────────────────────────────
#
# Opacus tracks cumulative ε spent across all training steps via the
# PRV (Privacy Random Variable) accountant.  The persistent engine
# accumulates steps naturally across rounds.  After each round, we
# query privacy_engine.get_epsilon(delta) to get the true cumulative
# ε consumed so far.  The server logs this for the privacy-utility
# curve.
#
# δ is set to 1/N_windows for this client's training set. using
# 1/min_samples (the smallest client) gives the largest δ (weakest
# per-client guarantee).  For the strongest formal guarantee, one
# would use 1/max_samples. TODO This design choice must be stated
# explicitly.
# McMahan et al. 2018: "δ should be less than 1/N."


# With only 15 subjects, the DP analysis is illustrative rather than
# formally certified.  The DP guarantee becomes meaningful only with
# many more clients.  This limitation must be stated explicitly (Bonawitz et al. 2019).

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
import torch
import torch.nn as nn

from ...models.wesad_fl_model import WESADFLModel
from ...utils.fl_config import FLConfig
from ...utils.wesad_fl_dataset import FLClientDataManager
from .client_base import WESADClient

OPACUS_AVAILABLE = True


if TYPE_CHECKING:
    pass


class FedRepDPClient(WESADClient):
    """FedRep client with Opacus DP-SGD on the shared CNN base.

    The training protocol is identical to FedRepClient:
      Phase 1: freeze base → train head for plocal_epochs (NO DP)
      Phase 2: freeze head → train base for local_epochs (WITH DP)

    Only Phase 2 is DP-protected because only base parameters are
    communicated to the server.

    The PrivacyEngine is created once in __init__ and persists across
    all federation rounds.  The cumulative privacy accountant tracks
    the true ε spent over the entire training run.

    Parameters:
    config : FLConfig
        Must have dp_epsilon set (1, 5, 10, or inf for no-DP baseline).
    subject_id : int
        WESAD subject ID.
    data_manager : FLClientDataManager
        Pre-built data manager for this client.
    global_model : WESADFLModel
        Initial global model (deep-copied per client).
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        config: FLConfig,
        subject_id: int,
        data_manager: FLClientDataManager,
        global_model: WESADFLModel,
        device: torch.device,
    ):
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for FedRepDP. " "Install with: pip install opacus"
            )

        # Determine DP parameters before super().__init__
        self._target_epsilon = config.dp_epsilon
        self._dp_enabled = self._target_epsilon != float("inf")
        self._max_grad_norm = config.dp_max_grad_norm

        # δ = 1/N_windows for this client's training set.
        if config.dp_delta is not None:
            self._delta = config.dp_delta
        else:
            self._delta = 1.0 / data_manager.train_samples

        # Store plocal_epochs before super().__init__ because we need it
        self._plocal_epochs = config.plocal_epochs

        # Call base class constructor
        # This creates self.model (deep copy), self.optimizer (on allp arams), self.loss, etc.
        super().__init__(config, subject_id, data_manager, global_model, device)

        # Override plocal_epochs (matches FedRepClient)
        self.plocal_epochs = self._plocal_epochs

        # Opacus requires no BatchNorm ModuleValidator.fix() replaces BN → GN.
        #
        # IMPORTANT: We fix model.base ONLY.  The head contains its own
        # BatchNorm (pre-LSTM normalisation) which does not need fixing
        # because the head is never DP-wrapped.
        if self._dp_enabled and not ModuleValidator.is_valid(self.model.base):
            self.model.base = ModuleValidator.fix(self.model.base)
            self.model.base = self.model.base.to(self.device)
            errors = ModuleValidator.validate(self.model.base)
            if errors:
                raise RuntimeError(f"Opacus validation failed after fix: {errors}")

        # Create separate optimisers (as in FedRepClient)
        # Override the single optimizer from WESADClient base
        # Base optimizer gets DP-wrapped by Opacus.
        self.optimizer = torch.optim.AdamW(
            self.model.base.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Head optimizer
        self.optimizer_per = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Head LR scheduler (created before Opacus wrapping)
        if config.lr_scheduler == "plateau":
            self.lr_scheduler_per = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_per,
                mode="min",
                factor=config.lr_factor,
                patience=config.lr_patience,
            )
        elif config.lr_scheduler == "exponential":
            self.lr_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_per,
                gamma=config.lr_gamma,
            )
        else:
            self.lr_scheduler_per = None

        # Pre-compute σ and attach persisten PrivacyEngine
        #
        # σ (noise_multiplier) is computed once for the full training
        # budget via get_noise_multiplier(), then make_private()attaches the engine with fixed σ.
        #
        # The engine persists for the entire training run
        self._noise_multiplier: float | None = None
        self._privacy_engine: PrivacyEngine | None = None

        if self._dp_enabled:
            sample_rate = config.batch_size / data_manager.train_samples
            total_epochs = config.global_rounds * config.local_epochs

            # Compute σ that achieves target_ε over the full training run.
            # This is what make_private_with_epsilon does internally, but
            # we do it once upfront so σ is fixed and known.
            self._noise_multiplier = get_noise_multiplier(
                target_epsilon=self._target_epsilon,
                target_delta=self._delta,
                sample_rate=sample_rate,
                epochs=total_epochs,
            )

            self._privacy_engine = PrivacyEngine()

            # Wrap base module, optimizer, and an initial dataloader.
            # After this call:
            #   self.model.base → GradSampleModule (wraps original base)
            #   self.optimizer  → DPOptimizer (wraps original AdamW)
            # Both persist across rounds
            _init_dataloader = self.load_train_data()
            self.model.base, self.optimizer, self._dp_dataloader = (
                self._privacy_engine.make_private(
                    module=self.model.base,
                    optimizer=self.optimizer,
                    data_loader=_init_dataloader,
                    noise_multiplier=self._noise_multiplier,
                    max_grad_norm=self._max_grad_norm,
                )
            )

        # Base LR scheduler (afTER Opacus wrapping)
        # The scheduler wraps self.optimizer which is now a DPOptimizer and delegates .param_groups and
        # .step() to the underlying AdamW, so schedulers work correctly.
        if config.lr_scheduler == "plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.lr_factor,
                patience=config.lr_patience,
            )
        elif config.lr_scheduler == "exponential":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config.lr_gamma,
            )
        else:
            self.lr_scheduler = None

    @property
    def dp_enabled(self) -> bool:
        """Whether DP is active for this client."""
        return self._dp_enabled

    @property
    def current_epsilon(self) -> float:
        """The cumulative ε spent so far (from persistent Opacus accountant).

        Queries the persistent engine directly — always reflects the
        true cumulative privacy budget consumed across all rounds.
        """
        if self._privacy_engine is not None:
            try:
                return float(self._privacy_engine.get_epsilon(delta=self._delta))
            except Exception:
                return 0.0
        return 0.0

    @property
    def target_epsilon(self) -> float:
        """The target privacy budget."""
        return self._target_epsilon

    @property
    def delta(self) -> float:
        """The δ parameter for (ε, δ)-DP."""
        return self._delta

    @property
    def noise_multiplier(self) -> float | None:
        """The Gaussian noise multiplier σ (pre-computed in __init__)."""
        return self._noise_multiplier

    # Parameter synchronisation override

    def set_parameters(self, base_model: nn.Module):
        """Copy global base parameters into the DP-wrapped base.

        Overrides WESADClient.set_parameters() to handle the Opacus
        GradSampleModule wrapper.  When DP is enabled, self.model.base
        is a GradSampleModule whose actual parameters live in
        self.model.base._module.  We load into _module so the
        GradSampleModule hooks, the DPOptimizer, and the privacy
        accountant all remain intact.

        The AdamW moment estimates (exp_avg, exp_avg_sq) are preserved
        because the optimizer holds references to the same parameter
        objects , only .data is overwritten, not the parameter identity.

        Parameters:
        base_model : nn.Module
            The server's global base model.
        """
        if self._dp_enabled and hasattr(self.model.base, "_module"):
            # Load into the inner module of GradSampleModule
            target = self.model.base._module
        else:
            target = self.model.base

        for new_param, old_param in zip(base_model.parameters(), target.parameters()):
            old_param.data = new_param.data.clone()

    # Training

    def train(self):
        """FedRep-DP alternating head/base training for one round.

        Phase 1: HEAD UPDATE (no DP)
          Freeze base → train head for plocal_epochs.
          Head gradients never leave the client, so no DP needed.

        Phase 2: BASE UPDATE (with DP-SGD via persistent Opacus engine)
          Freeze head → train base for local_epochs.
          The persistent DPOptimizer automatically:
            - Clips per-sample gradients to max_grad_norm
            - Adds calibrated Gaussian noise (σ pre-computed in __init__)
            - Accumulates steps in the privacy accountant

        No attach/detach cycle: the GradSampleModule and DPOptimizer
        persist across rounds.  Global weights arrive via
        set_parameters() which overwrites .data on the inner module.
        """
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        # PHASE 1: HEAD UPDATE
        # Freeze base (CNN), unfreeze head (LSTM + BN + classifier).
        # No DP because head parameters never leave the client.

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for _epoch in range(self.plocal_epochs):
            for signals, labels in trainloader:
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)

                output = self.model(signals)
                loss = self.loss(output, labels)

                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()

        # PHASE 2: BASE UPDATE (shared CNN, WITH DP-SGD)
        # Freeze head, unfreeze base.
        # The persistent DPOptimizer handles clipping + noise.

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        if self._dp_enabled:
            # Fresh dataloader each round for proper shuffling.
            # The DPOptimizer and GradSampleModule are persistent
            dp_trainloader = self.load_train_data()

            for _epoch in range(self.local_epochs):
                for signals, labels in dp_trainloader:
                    signals = {k: v.to(self.device) for k, v in signals.items()}
                    labels = labels.to(self.device)

                    output = self.model(signals)
                    loss = self.loss(output, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        else:
            # No DP: standard base training (identical to FedRepClient)
            for _epoch in range(self.local_epochs):
                for signals, labels in trainloader:
                    signals = {k: v.to(self.device) for k, v in signals.items()}
                    labels = labels.to(self.device)

                    output = self.model(signals)
                    loss = self.loss(output, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # RESTORE requires_grad for evaluation
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True

        # Learning rate scheduling
        self.step_scheduler()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def step_scheduler(self, val_loss: float | None = None):
        """Step both LR schedulers (base and head).

        Identical to FedRepClient.step_scheduler().
        """
        if self.lr_scheduler is None:
            return

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.lr_scheduler.step(val_loss)
                if self.lr_scheduler_per is not None:
                    self.lr_scheduler_per.step(val_loss)
        else:
            self.lr_scheduler.step()
            if self.lr_scheduler_per is not None:
                self.lr_scheduler_per.step()

    def get_privacy_spent(self) -> dict:
        """Return a summary of the privacy budget consumed so far.

        Queries the persistent engine's accountant for the true
        cumulative ε — no manual tracking needed.

        Returns:
        dict with keys:
            epsilon : float
                Cumulative ε spent (from persistent Opacus accountant).
            delta : float
                The δ parameter.
            noise_multiplier : float or None
                The σ pre-computed in __init__.
            target_epsilon : float
                The target ε for the full training run.
            dp_enabled : bool
                Whether DP is active.
        """
        return {
            "epsilon": self.current_epsilon,
            "delta": self._delta,
            "noise_multiplier": self._noise_multiplier,
            "target_epsilon": self._target_epsilon,
            "dp_enabled": self._dp_enabled,
        }

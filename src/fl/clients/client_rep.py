# FedRep client: alternating head/base training for personalised FL.
#
# This is the primary PFL method in the dissertation.  The alternating
# optimisation protocol is the algorithm's core and must be exact.
#
# Adapts from:
#   - PFLIB clientrep.py    : the alternating freeze/unfreeze pattern,
#                              two optimisers, plocal_epochs
#   - client_base.py        : our WESADClient base class
#
# ──────────────────────────────────────────────────────────────────────
# THE FEDREP PROTOCOL (Collins et al. 2021, Action Plan §4.1)
# ──────────────────────────────────────────────────────────────────────
#
# Each federation round, each client executes two phases IN THIS ORDER:
#
#   Phase 1: HEAD UPDATE (plocal_epochs epochs):
#     Freeze model.base (CNN branches).
#     Update model.head (LSTM + BN + classifier) only.
#     Uses optimizer_per (AdamW on head parameters).
#     Purpose: train the personal temporal reasoner to classify using the current global CNN features.
#
#   Phase 2: BASE UPDATE (local_epochs epochs):
#     Freeze model.head (LSTM + BN + classifier).
#     Update model.base (CNN branches) only.
#     Uses optimizer (AdamW on base parameters).
#     Purpose: train the shared CNN to produce features useful for the local classification task.
#
# After both phases, only model.base is sent to the server.
# model.head is never communicated as it stays local permanently.

#   Collins et al. 2021 Theorem 1 proves convergence under the
#   assumption that the head update approaches the optimal head
#   for the current representation before the representation is
#   updated.  Updating the head first ensures the base gradient
#   is computed with respect to a well-optimised head, making
#   the representation update meaningful.
#
#   This is why plocal_epochs ≥ local_epochs in practice , as the
#   head must be well-optimised before the base gradient is computed.
#
# OPTIMIZER CHOICE
#   AdamW for both optimisers, not SGD as in PFLlib to match the standalone training baseline (wesad_cnnlstm_train.py).
#
# HEAD INITIALISATION
#   Default: COLD-START (random head initialisation).
#   This matches Collins et al. 2021's experimental setup and
#   isolates the algorithm's contribution from any pre-training
#   benefit.  It also makes the FedBABU comparison clean (both
#   start with random heads).
#
#   Warm-start (from pre-trained centralised model) is supported
#   via wrap_pretrained_model() from wesad_fl_model.py and should
#   be reported as an ABLATION with direct implications for
#   new-client deployment (FedHealth paradigm, Chen et al. 2020).

from __future__ import annotations

import time

import torch

from ...models.wesad_fl_model import WESADFLModel
from ...utils.fl_config import FLConfig
from ...utils.wesad_fl_dataset import FLClientDataManager
from .client_base import WESADClient


class FedRepClient(WESADClient):
    """FedRep client: alternating head/base training per round.

    Overrides __init__ to create two separate optimisers (base + head)
    and overrides train() to implement the alternating protocol.

    Parameters:
    same as WESADClient
    """

    def __init__(
        self,
        config: FLConfig,
        subject_id: int,
        data_manager: FLClientDataManager,
        global_model: WESADFLModel,
        device: torch.device,
    ):
        super().__init__(config, subject_id, data_manager, global_model, device)

        self.plocal_epochs = config.plocal_epochs

        # Override the base class optimizer
        # FedRep needs two separate optimisers: one for base (shared CNN)
        # and one for head (personal LSTM).  The base class creates a single
        # optimizer over all parameters so we mustreplace it.
        #
        # PFLlib uses SGD for both.  We use AdamW for both, matching the
        # standalone training baseline.

        # Base optimizer (shared CNN branches)
        self.optimizer = torch.optim.AdamW(
            self.model.base.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Head optimizer (personal LSTM + BN + classifier)
        self.optimizer_per = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Independent LR schedulers for each optimizer
        # (ReduceLROnPlateau on validation loss, patience=5, factor=0.5)"
        if config.lr_scheduler == "plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.lr_factor,
                patience=config.lr_patience,
            )
            self.lr_scheduler_per = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_per,
                mode="min",
                factor=config.lr_factor,
                patience=config.lr_patience,
            )
        elif config.lr_scheduler == "exponential":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config.lr_gamma,
            )
            self.lr_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_per,
                gamma=config.lr_gamma,
            )
        else:
            self.lr_scheduler = None
            self.lr_scheduler_per = None

    def train(self):
        """FedRep alternating head/base training for one federation round.

        Implements the exact protocol from Collins et al. 2021 and
        PFLIB clientrep.py:

          Phase 1: freeze base → train head for plocal_epochs
          Phase 2: freeze head → train base for local_epochs

        The full model forward pass runs in both phases (base produces
        features, head classifies), but gradients only flow to the
        unfrozen component.
        """
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        # PHASE 1: HEAD UPDATE (personal LSTM training)

        # Freeze base (CNN), unfreeze head (LSTM + BN + classifier).
        # Train the personal temporal reasoner to classify using the
        # current global CNN features.
        #
        # Collins et al. 2021: "the head update should approach
        # convergence before the base update."

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

        # PHASE 2: BASE UPDATE (shared CNN training)
        # Freeze head (LSTM + BN + classifier), unfreeze base (CNN).
        # Train the shared CNN to produce features that are useful
        # for the local classification task.
        #
        # After this phase, only model.base is sent to the server.
        # model.head stays local permanently.

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for _epoch in range(self.local_epochs):
            for signals, labels in trainloader:
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)

                output = self.model(signals)
                loss = self.loss(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # RESTORE requires_grad for both
        # After training, restore requires_grad on all parameters.
        # This is needed because test_metrics() runs a forward pass
        # that should not be affected by the training-phase freeze.
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

        Overrides the base class to step both schedulers.
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

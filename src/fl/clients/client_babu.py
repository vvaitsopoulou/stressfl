# FedBABU client: frozen head during federation, local fine-tuning after.
#
# FedBABU answers the question does training the head during FL rounds
# (as FedRep does) help or hurt?  FedBABU's answer is that freezing the
# head during FL and fine-tuning it only at evaluation time is better,
# because training the head during FL can cause it to overfit to the
# global distribution rather than the local one.
#
# Adapts from:
#   - PFLIB clientbabu.py   : head frozen in __init__, train() only
#                              updates base, fine_tune() unfreezes head
#   - client_base.py        : our WESADClient base class
#
# THE FEDBABU PROTOCOL (Oh et al. 2022)
#
# During ALL FL rounds:
#   - model.base (CNN) is updated via standard local training
#   - model.head (LSTM + BN + classifier) is frozen
#     (requires_grad = False for all head parameters)
#   - Only model.base is sent to/from the server
#
# After federation completes:
#   - Each client fine-tunes model.head locally for fine_tuning_epochs
#   - model.base is frozen during fine-tuning
#   - The head is initialised randomly and only ever trained on local
#     data so it never sees the global objective
#
# KEY INSIGHT (Oh et al. 2022):
#   "Training the head during FL can degrade personalisation because
#   the head adapts to the global distribution rather than the local
#   one."
#
# ABLATION Plan:
#   fine_tuning_epochs ∈ {1, 2, 5, 10, 20} and plot macro-F1
#   vs fine_tuning_epochs. Thn we compare this curve against FedRep's best
#   result.
#
# TODO: Ditto, implement if time permits)
#   - adapt PFLib client_ditto.py and server_ditto.py to wesad dataset specifics

from __future__ import annotations

import time

import torch

from ...models.wesad_fl_model import WESADFLModel
from ...utils.fl_config import FLConfig
from ...utils.wesad_fl_dataset import FLClientDataManager
from .client_base import WESADClient


class FedBABUClient(WESADClient):
    """FedBABU client: frozen head during FL, local fine-tuning after.

    Overrides __init__ to freeze head parameters and create a base-only
    optimizer.  Overrides train() to update only the base.  Adds
    fine_tune() for post-federation local head training.

    Parameters: same as WESADClient
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

        self.fine_tuning_epochs = config.fine_tuning_epochs

        # The head remains frozen throughout all FL rounds.
        # It is only unfrozen during fine_tune() after federation.
        for param in self.model.head.parameters():
            param.requires_grad = False

        # Since head is frozen on fedBabu, override optimizer to target base only
        self.optimizer = torch.optim.AdamW(
            self.model.base.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler for base optimizer
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

    def train(self):
        """FedBABU local training: base-only update, head frozen.

        Identical to standard FedAvg training except head parameters
        have requires_grad=False, so only CNN base receives gradients.
        The full forward pass still runs but gradients only flow to the base.

        Mirrors PFLIB clientbabu.py train().
        """
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        for _epoch in range(self.local_epochs):
            for signals, labels in trainloader:
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)

                output = self.model(signals)
                loss = self.loss(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # LR scheduling (base only)
        self.step_scheduler()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def fine_tune(self, which_module: list[str] | None = None):
        """Post-federation local fine-tuning of the head.

        Called by the server after all FL rounds complete.
        Unfreezes the head and trains it locally for fine_tuning_epochs,
        with the base frozen.  This is FedBABU's personalisation step.

        The default (which_module=['head']) matches the standard FedBABU
        protocol: only the head is fine-tuned, the base stays frozen.
        Passing which_module=['base', 'head'] fine-tunes both

        Parameters:
        which_module : list[str] | None
            Which modules to fine-tune.  Default ['head'] = head only.
            Options: 'base', 'head'.

        Mirrors PFLIB clientbabu.py fine_tune().
        """
        if which_module is None:
            which_module = ["head"]

        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        # Unfreeze requested modules
        if "head" in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = True

        # Freeze modules NOT requested
        if "base" not in which_module:
            for param in self.model.base.parameters():
                param.requires_grad = False

        # Create a fine-tuning optimizer over all unfrozen parameters
        ft_params = [p for p in self.model.parameters() if p.requires_grad]
        ft_optimizer = torch.optim.AdamW(
            ft_params,
            lr=self.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        for _epoch in range(self.fine_tuning_epochs):
            for signals, labels in trainloader:
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)

                output = self.model(signals)
                loss = self.loss(output, labels)

                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()

        # Restore requires_grad for all parameters (for evaluation)
        for param in self.model.parameters():
            param.requires_grad = True

        self.train_time_cost["total_cost"] += time.time() - start_time

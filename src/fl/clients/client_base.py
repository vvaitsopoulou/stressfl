# Base client class for all federated learning algorithms on WESAD.
#
# Every algorithm-specific client (FedAvg, FedRep, FedBABU, Ditto, etc.)
# inherits from WESADClient.
# Adapts from:
#   - PFLIB clientbase.py
#
# Key differences from PFLlib:
#   - Model is a WESADFLModel (with .base and .head), not a flat CNN
#   - Data loading via FLClientDataManager instead of read_client_data()
#   - Loss: CrossEntropyLoss with inverse-frequency class weights
#     (Prajod & André, 2022)
#   - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
#   - Metrics: macro-F1, stress recall, per-subject scores via
#     metrics.py evaluate_client() returning ClientMetrics dataclass
#   - Two parameter-setting methods:
#       set_parameters(base_model): copies base only (FedRep/FedBABU)
#       set_full_parameters(model): copies everything (FedAvg)

from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn as nn

from ...models.wesad_fl_model import WESADFLModel
from ...utils.fl_config import FLConfig
from ...utils.metrics import ClientMetrics, evaluate_client
from ...utils.wesad_fl_dataset import FLClientDataManager

# Custom collate for dict-of-tensors dataset
# This collate function is identical to wesad_cnnlstm_train.py's
# _collate_fn, duplicated here to avoid cross-module dependency.


def _collate_fn(batch):
    """Collate a batch of (signal_dict, label) tuples."""
    signal_dicts, labels = zip(*batch)
    signal_names = signal_dicts[0].keys()

    batched_signals = {}
    for name in signal_names:
        batched_signals[name] = torch.stack([d[name] for d in signal_dicts], dim=0)

    batched_labels = torch.stack(labels, dim=0)
    return batched_signals, batched_labels


# Base Client


class WESADClient:
    """Base class for all FL clients on WESAD.

    Adapts PFLIB clientbase.py to the WESAD-specific context.
    Algorithm-specific clients (FedAvgClient, FedRepClient, etc.)
    inherit from this and override train().

    Parameters:
    config : FLConfig
        Experiment configuration.
    subject_id : int
        This client's WESAD subject ID.
    data_manager : FLClientDataManager
        Pre-built data manager for this client (from build_fl_fold_datasets).
    global_model : WESADFLModel
        The initial global model (deep-copied per client, as in PFLlib).
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
        self.config = config
        self.id = subject_id
        self.device = device
        self.data_manager = data_manager

        # Model
        # Deep copy so each client has independent parameters.
        # This matches PFLIB clientbase.py: self.model = copy.deepcopy(args.model)
        self.model = copy.deepcopy(global_model).to(self.device)

        # Training hyperparameters
        self.num_classes = config.n_classes
        self.local_epochs = config.local_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate

        # Sample counts (for server-side weighted aggregation)
        # PFLlib uses client.train_samples for weighting in receive_models().
        self.train_samples = data_manager.train_samples
        self.test_samples = data_manager.test_samples

        # Loss function
        # CrossEntropyLoss with inverse-frequency class weights.
        # Weights computed from this client's training data, not global.
        class_weights = data_manager.get_class_weights().to(self.device)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        # AdamW on ALL model parameters (base + head).
        # Algorithm-specific clients override this:
        #   - FedRep creates two optimisers (base + head)
        #   - Ditto creates an additional optimizer_per (TODO: IF time permits)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )

        #  Learning rate schedule
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

        # timing
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}

    # Data loading
    # Mirrors PFLIB clientbase.py load_train_data() / load_test_data().
    # Delegates to FLClientDataManager but wraps with collate_fn.

    def load_train_data(
        self, batch_size: int | None = None
    ) -> torch.utils.data.DataLoader:
        bs = batch_size or self.batch_size
        return torch.utils.data.DataLoader(
            self.data_manager._train_dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def load_test_data(
        self, batch_size: int | None = None
    ) -> torch.utils.data.DataLoader:
        bs = batch_size or self.batch_size
        return torch.utils.data.DataLoader(
            self.data_manager._test_dataset,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def load_val_data(
        self, batch_size: int | None = None
    ) -> torch.utils.data.DataLoader:
        bs = batch_size or self.batch_size
        return torch.utils.data.DataLoader(
            self.data_manager._val_dataset,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Parameter synchronisation
    def set_parameters(self, base_model: nn.Module):
        """Copy base (shared) parameters from the server's global model.

        This is the FedRep/FedBABU interface: only the CNN base is
        synchronised from the server, while he LSTM head stays local.

        Mirrors PFLIB clientrep.py set_parameters() and
        PFLIB_clientbabu.py set_parameters().

        Parameters:
        base_model : nn.Module
            The server's global base model (server.global_model.base).
        """
        for new_param, old_param in zip(
            base_model.parameters(), self.model.base.parameters()
        ):
            old_param.data = new_param.data.clone()

    def set_full_parameters(self, full_model: WESADFLModel):
        """Copy ALL parameters (base + head) from the server's global model.

        This is the FedAvg interface: the entire model is
        synchronised from the server each round.

        Mirrors PFLIB clientbase.py set_parameters()

        Parameters:
        full_model : WESADFLModel
            The server's global model (base + head).
        """
        for new_param, old_param in zip(
            full_model.parameters(), self.model.parameters()
        ):
            old_param.data = new_param.data.clone()

    def clone_model(self, source: nn.Module, target: nn.Module):
        """Clone parameters from source to target.

        Utility method preserved from PFLIB clientbase.py.
        """
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data = src_param.data.clone()

    def step_scheduler(self, val_loss: float | None = None):
        """Step the learning rate scheduler.

        Handles both epoch-based schedulers (ExponentialLR) and
        metric-based schedulers (ReduceLROnPlateau).

        For ReduceLROnPlateau: call with val_loss from the server's
        evaluate() method.  Without val_loss, plateau scheduler is
        not stepped.

        For epoch-based schedulers: call without arguments after each
        training round.

        Parameters:
        val_loss : float | None
            Validation loss for ReduceLROnPlateau.  Ignored by
            epoch-based schedulers.
        """
        if self.lr_scheduler is None:
            return

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.lr_scheduler.step(val_loss)
        else:
            self.lr_scheduler.step()

    # Training (to be overridden by algorithm-specific clients)

    def train(self):
        """Local training for one federation round.

        The base implementation trains the FULL model (base + head) for
        local_epochs epochs ---> the FedAvg training loop.

        Algorithm-specific clients override this method:
          - FedRepClient: alternating head/base training
          - FedBABUClient: base-only training (head frozen)
          - DittoClient: additional personal model training (potential future extension??)
        """
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        for _epoch in range(self.local_epochs):
            for signals, labels in trainloader:
                # Move to device
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)

                # Forward
                output = self.model(signals)
                loss = self.loss(output, labels)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Learning rate scheduling (epoch-based only bc plateau is
        # handled via step_scheduler() called by the server after eval)
        self.step_scheduler()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    # Evaluation

    def test_metrics(self) -> ClientMetrics:
        """Evaluate this client's model on their test data.

        Returns a ClientMetrics dataclass with macro-F1, accuracy,
        per-class recall/precision/F1, and stress-specific metrics.

        This replaces PFLIB clientbase.py test_metrics() which only
        returned (test_acc, test_num, auc).

        !! The server's evaluate() calls this for each client.
        For the test subject, this evaluates on their held-out test
        windows.  For training clients, this evaluates on their local
        non-overlapped windows
        """
        testloader = self.load_test_data()
        return evaluate_client(
            model=self.model,
            dataloader=testloader,
            device=self.device,
            subject_id=self.id,
            num_classes=self.num_classes,
        )

    def train_metrics(self) -> tuple[float, int]:
        """Compute training loss (for logging convergence).

        Returns:
        (total_loss, num_samples) : tuple[float, int]
            Total loss (sum) and number of samples.
            The server divides to get mean loss.

        Adapted from PFLIB clientbase.py train_metrics().
        """
        trainloader = self.load_train_data()
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for signals, labels in trainloader:
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)

                output = self.model(signals)
                loss = self.loss(output, labels)

                num_samples += labels.shape[0]
                total_loss += loss.item() * labels.shape[0]

        return total_loss, num_samples

    # Persistence (kept from PFLlib for checkpoint compatibility)

    def save_item(self, item, item_name: str, item_path: str | None = None):
        """Save a PyTorch object to disk.

        Mirrors PFLIB_clientbase.py save_item().
        """
        if item_path is None:
            item_path = self.config.experiment_name
        os.makedirs(item_path, exist_ok=True)
        torch.save(
            item,
            os.path.join(item_path, f"client_{self.id}_{item_name}.pt"),
        )

    def load_item(self, item_name: str, item_path: str | None = None):
        """Load a PyTorch object from disk.

        Mirrors PFLIB_clientbase.py load_item().
        """
        if item_path is None:
            item_path = self.config.experiment_name
        return torch.load(
            os.path.join(item_path, f"client_{self.id}_{item_name}.pt"),
            map_location=self.device,
            weights_only=True,
        )

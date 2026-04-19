# Base server class for all federated learning algorithms on WESAD.
#
# Orchestrates the federation round loop, client selection, model
# broadcast, gradient collection, aggregation, evaluation, and result
# saving.  Algorithm-specific servers (FedAvg, FedRep, etc.) inherit
# from WESADServer and override only what differs.
#
# Adapts from:
#   - PFLIB serverbase.py
#
# Adaptaions done to PFLlib:
#   - Clients are instantiated from FLFoldConfig + FLClientDataManager,
#     not from read_client_data() with integer IDs
#   - Global model is a WESADFLModel with .base and .head
#   - Full participation every round (no subsampling since we only have 15 clients)
#   - No slow-client / drop-out simulation
#   - Aggregation is on BASE parameters only (CNN branches) by default.
#     FedAvg servers override to aggregate full model.
#   - Weighting by window count
#   - Evaluation via metrics.py → ClientMetrics → FoldMetrics
#   - Results stored via result_store.py → HDF5

from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...models.wesad_fl_model import WESADFLModel, build_fl_model_from_config
from ...utils.fl_config import FLConfig
from ...utils.metrics import (
    ClientMetrics,
    FoldMetrics,
    compute_fold_metrics,
    evaluate_client,
    print_fold_metrics,
)
from ...utils.result_store import FoldResultAccumulator
from ...utils.wesad_fl_dataset import (
    FLClientDataManager,
    FLFoldConfig,
    build_fl_fold_datasets,
    build_test_subject_splits,
)
from ..clients.client_base import WESADClient


class WESADServer:
    """Base server for federated learning on WESAD.

    Manages the global model, the client pool, and the federation round
    loop.  Algorithm-specific servers inherit and override:
      - receive_models(): what to collect from clients (base only vs full)
      - aggregate_parameters(): how to aggregate
      - The train() loop if the round structure differs ( PeFLL)

    Parameters:
    config : FLConfig
        Experiment configuration.
    fold_config : FLFoldConfig
        LOSO fold configuration (test subject, val subject, train subjects).
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        config: FLConfig,
        fold_config: FLFoldConfig,
        device: torch.device | None = None,
    ):
        self.config = config
        self.fold_config = fold_config
        self.device = device or torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # Global model
        self.global_model = build_fl_model_from_config(config).to(self.device)

        # Client pool
        self.clients: list[WESADClient] = []
        self.selected_clients: list[WESADClient] = []

        # Aggregation buffers
        # NOTE toself Filled by receive_models() and consumed by aggregate_parameters()
        self.uploaded_weights: list[float] = []
        self.uploaded_ids: list[int] = []
        self.uploaded_models: list[nn.Module] = []

        # Result accumulator
        # val_subject is guaranteed non-None from FLFoldConfig
        assert (
            fold_config.val_subject is not None
        ), "val_subject is None - FLFoldConfig.__post_init__ should have set it"
        self.results = FoldResultAccumulator(
            algorithm=config.algorithm,
            fold_id=fold_config.test_subject,
            test_subject=fold_config.test_subject,
            val_subject=fold_config.val_subject,
            hyperparams={
                "global_rounds": config.global_rounds,
                "local_epochs": config.local_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        )

    # Client management

    def set_clients(
        self,
        client_class: type[WESADClient] = WESADClient,
        data_managers: dict[int, FLClientDataManager] | None = None,
    ):
        """Instantiate one client per participating subject.

        Adapts PFLIB serverbase.py set_clients().  Key difference:
        clients are constructed from FLFoldConfig + FLClientDataManager,
        not from integer IDs + read_client_data(). (because wesad structure is already
        compatible with fl clients requirement)

        Parameters
        client_class : Type[WESADClient]
            Client class to instantiate.  Default is WESADClient (FedAvg
            behaviour).  Algorithm-specific servers pass their own class
            (e.g. FedRepClient).
        data_managers : dict[int, FLClientDataManager] | None
            Pre-built data managers, one per subject.  If None, builds
            them via build_fl_fold_datasets().
        """
        if data_managers is None:
            data_managers = build_fl_fold_datasets(
                fold_config=self.fold_config,
                windowed_dir=self.config.windowed_dir,
                device_type=self.config.device_type,
                window_sec=self.config.window_sec,
                label_map=self.config.label_map,
                batch_size=self.config.batch_size,
                noise_std=self.config.noise_std,
                scale_range=self.config.scale_range,
                signals_to_load=self.config.signals_to_load,
                num_workers=self.config.num_workers,
            )

        self.clients = []
        for sid in self.fold_config.all_participating_subjects:
            if sid not in data_managers:
                raise ValueError(
                    f"No data manager for subject {sid}. "
                    f"Expected managers for: {self.fold_config.all_participating_subjects}"
                )

            client = client_class(
                config=self.config,
                subject_id=sid,
                data_manager=data_managers[sid],
                global_model=self.global_model,
                device=self.device,
            )
            self.clients.append(client)

        # Record subject IDs in result accumulator
        self.results.subject_ids = [c.id for c in self.clients]

        print(
            f"  Server: {len(self.clients)} clients instantiated "
            f"({[c.id for c in self.clients]})"
        )

    def select_clients(self) -> list[WESADClient]:
        """Select clients for this round.

        With 15 subjects and full participation this always meanbs
        ALL clients. method signature is preserved for PFLlib
        compatibility but algorithm-specific servers can override if needed.

        Returns:
        list[WESADClient]
            All participating clients.
        """
        self.selected_clients = list(self.clients)
        return self.selected_clients

    def _get_client_by_id(self, subject_id: int) -> WESADClient | None:
        """Find a client by subject ID."""
        for c in self.clients:
            if c.id == subject_id:
                return c
        return None

    # Model broadcast

    def send_models(self):
        """Broadcast the global BASE model to all clients.

        Each client receives a copy of the server's CNN base parameters.
        The LSTM head is NOT overwritten, as it stays local.

        This is the default for FedRep/FedBABU.  FedAvg servers
        override this to send the full model via set_full_parameters().

        Mirrors PFLIB serverbase.py send_models().
        """
        assert len(self.clients) > 0, "No clients! call set_clients() first"

        for client in self.clients:
            client.set_parameters(self.global_model.base)

    def send_full_models(self):
        """Broadcast the FULL global model (base + head) to all clients.

        Used by FedAvg where the entire model is synchronised.
        """
        assert len(self.clients) > 0, "No clients! call set_clients() first"

        for client in self.clients:
            client.set_full_parameters(self.global_model)

    # Model collection

    def receive_models(self):
        """Collect BASE models from all selected clients.

        Weights by number of training windows (client.train_samples),
        not uniform. window counts vary across WESAD subjects because
        subjects have different session durations, so we have to deviate from the PFLib default

        Deep-copies each client's base model to decouple the aggregation
        buffer from live client state.  Without this, self.uploaded_models
        holds live references, and any in-place mutationcould corrupt client models.

        Stores results in self.uploaded_weights, self.uploaded_ids,
        self.uploaded_models for use by aggregate_parameters().
        """
        assert len(self.selected_clients) > 0

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        total_samples = 0

        for client in self.selected_clients:
            total_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(copy.deepcopy(client.model.base))

        # Normalise weights to sum to 1
        for i in range(len(self.uploaded_weights)):
            self.uploaded_weights[i] /= total_samples

    def receive_full_models(self):
        """Collect FULL models (base + head) from all selected clients.

        Used by FedAvg where the entire model is aggregated.
        Deep-copies to decouple aggregation buffer from live client state.
        """
        assert len(self.selected_clients) > 0

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        total_samples = 0

        for client in self.selected_clients:
            total_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(copy.deepcopy(client.model))

        for i in range(len(self.uploaded_weights)):
            self.uploaded_weights[i] /= total_samples

    # Aggregation

    def aggregate_parameters(self):
        """Weighted average of uploaded models into the global model.

        By default, aggregates BASE parameters only (from receive_models).
        FedAvg servers call receive_full_models() first, then
        this method aggregates the full model.

        The target for aggregation is determined by what receive_models()
        stored: if it stored client.model.base, then self.global_model.base
        is what we aggregate into.  If it stored client.model (full), then
        self.global_model (full) is the target.

        Mirrors PFLIB serverbase.py aggregate_parameters().
        """
        assert (
            len(self.uploaded_models) > 0
        ), "No models uploaded! call receive_models() first"

        # Determine aggregation target: if uploaded models are WESADBase
        # modules, aggregate into global_model.base.  If they're full
        # WESADFLModel instances, aggregate into global_model.
        first_uploaded = self.uploaded_models[0]
        if isinstance(first_uploaded, WESADFLModel):
            target = self.global_model
        else:
            # Assume it's a base module (WESADBase or nn.Module)
            target = self.global_model.base

        # Zero out target parameters
        for param in target.parameters():
            param.data.zero_()

        # Weighted sum
        for weight, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(
                target.parameters(), client_model.parameters()
            ):
                server_param.data += client_param.data.clone() * weight

    # Evaluation
    def evaluate(self, round_num: int | None = None) -> FoldMetrics:
        """Evaluate all clients and aggregate into fold-level metrics.

        For each client, evaluates their CURRENT model (which may be
        personalised) on their LOCAL test data.  The test subject's
        result is the LOSO fold metric.

        Stores results in self.results accumulator.

        Adapts PFLIB serverbase.py evaluate() but except we use metrics.py
        for macro-F1, stress recall, and per-subject tracking.

        Parameters:
        round_num : int | None
            Current round number (for logging).

        Returns:
        FoldMetrics
            Aggregated metrics across all clients.
        """
        client_metrics_list = []

        for client in self.clients:
            cm = client.test_metrics()
            client_metrics_list.append(cm)

        fold_metrics = compute_fold_metrics(
            client_metrics_list=client_metrics_list,
            test_subject_id=self.fold_config.test_subject,
        )

        # Compute train loss
        total_train_loss = 0.0
        total_train_samples = 0
        for client in self.clients:
            loss, n = client.train_metrics()
            total_train_loss += loss
            total_train_samples += n
        mean_train_loss = total_train_loss / max(total_train_samples, 1)

        # Append to result accumulator
        per_subject_f1 = {cm.subject_id: cm.macro_f1 for cm in client_metrics_list}
        self.results.append_round(
            macro_f1=fold_metrics.mean_macro_f1,
            accuracy=fold_metrics.mean_accuracy,
            stress_recall=fold_metrics.mean_stress_recall,
            train_loss=mean_train_loss,
            per_subject_f1=per_subject_f1,
        )

        # Print summary
        print_fold_metrics(fold_metrics, round_num=round_num)

        return fold_metrics

    # DLG attack hook IF TIME PERMITS IMPLEMENT

    # def call_dlg(self, round_num: int):
    #     """Run Deep Leakage from Gradients attack.

    #     Attempts to reconstruct training samples from gradient updates.
    #     Reports PSNR (higher = better reconstruction = worse privacy).

    #     Adapts PFLIB serverbase.py call_dlg() with minimal changes.
    #     The DLG implementation itself is imported from PFLIB dlg.py
    #     (direct copy, no modifications needed).

    #     Only runs when config.dlg_eval is True and at the DLG gap interval.
    #     """
    #     # Import DLG here to avoid hard dependency when not used
    #     try:
    #         from ...utils.dlg import DLG
    #     except ImportError:
    #         print("  DLG module not available — skipping")
    #         return

    #     cnt = 0
    #     psnr_val = 0.0

    #     for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
    #         client_model.eval()

    #         # Compute approximate gradient: global - client
    #         origin_grad = []
    #         for gp, pp in zip(
    #             self.global_model.base.parameters(), client_model.parameters()
    #         ):
    #             origin_grad.append(gp.data - pp.data)

    #         # Get target inputs for reconstruction
    #         client = self._get_client_by_id(cid)
    #         if client is None:
    #             continue

    #         target_inputs = []
    #         trainloader = client.load_train_data()
    #         with torch.no_grad():
    #             for i, (signals, labels) in enumerate(trainloader):
    #                 if i >= self.config.dlg_batch_num_per_client:
    #                     break
    #                 signals = {k: v.to(self.device) for k, v in signals.items()}
    #                 labels = labels.to(self.device)
    #                 output = client_model(signals)
    #                 target_inputs.append((signals, output))

    #         d = DLG(client_model, origin_grad, target_inputs)
    #         if d is not None:
    #             psnr_val += d
    #             cnt += 1

    #     if cnt > 0:
    #         print(f"  DLG PSNR: {psnr_val / cnt:.2f} dB (round {round_num})")
    #     else:
    #         print(f"  DLG: no valid reconstructions (round {round_num})")

    # ------------------------------------------------------------------
    # Held-out test subject evaluation with calibration
    # ------------------------------------------------------------------
    #
    # These three methods implement the post-federation evaluation of
    # the held-out test subject under the held-out-with-calibration
    # protocol (Collins et al. 2021 §5.3 "Generalization to new clients";
    # Chen et al. 2020 FedHealth; Cross_Modal_KD plan).
    #
    # Protocol:
    #   1. Build per-condition-block 20/80 temporal stratified split of
    #      the test subject's non-overlapped windows.
    #   2. Deep-copy the final global model (keeps self.global_model
    #      unmodified so save_global_model() saves the federated model).
    #   3. Dispatch by algorithm:
    #        FedAvg  -> no calibration; evaluate global on eval set.
    #        FedRep  -> fresh random head; K-epoch head-only training
    #                   on calibration set with base frozen; AdamW +
    #                   CE loss with class weights from calibration set.
    #        FedBABU -> same as FedRep (by design).
    #        PeFLL   -> hypernet generates head from embedding
    #                   (PeFLLServer overrides this method).
    #   4. Evaluate on the 80% evaluation set.  Return ClientMetrics.

    def evaluate_test_subject_with_calibration(self) -> ClientMetrics:
        """Post-federation LOSO evaluation of the held-out test subject.

        Returns
        -------
        ClientMetrics
            Metrics on the 80% evaluation split, after the algorithm-
            appropriate calibration step (if any).

        Raises
        ------
        ValueError
            If config.algorithm is unknown to this dispatcher.
        NotImplementedError
            If config.algorithm requires a subclass override (PeFLL).
        """
        test_subject = self.fold_config.test_subject
        algorithm = self.config.algorithm

        print(
            f"\n  ══ Held-out calibration + evaluation "
            f"(S{test_subject}, {algorithm}) ══"
        )

        # Step 1: build calibration / evaluation splits.
        calibration_ds, evaluation_ds = build_test_subject_splits(
            test_subject=test_subject,
            windowed_dir=self.config.windowed_dir,
            device_type=self.config.device_type,
            window_sec=self.config.window_sec,
            label_map=self.config.label_map,
            signals_to_load=self.config.signals_to_load,
            calibration_fraction=self.config.calibration_fraction,
            verbose=True,
        )

        if len(evaluation_ds) == 0:
            raise ValueError(
                f"Evaluation set for S{test_subject} is empty; "
                f"cannot compute LOSO metric."
            )

        # DataLoaders.  Eval is deterministic; calibration shuffles.
        # Neither uses augmentation (datasets were built with mode="test").
        pin = torch.cuda.is_available()
        cal_batch = min(self.config.batch_size, max(len(calibration_ds), 1))
        calibration_loader = DataLoader(
            calibration_ds,
            batch_size=cal_batch,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=pin,
        )
        evaluation_loader = DataLoader(
            evaluation_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=pin,
        )

        # Step 2: deep-copy the global model so calibration does not
        # mutate self.global_model (which save_global_model() saves).
        eval_model = copy.deepcopy(self.global_model).to(self.device)

        # Step 3: algorithm dispatch.
        if algorithm in ("FedAvg", "FedProx"):
            print(
                "  No calibration (FedAvg/FedProx): evaluating global "
                "model directly on the 80% evaluation set."
            )

        elif algorithm in ("FedRep", "FedBABU", "FedRepDP"):
            print(
                f"  Calibration: {self.config.calibration_epochs} epochs "
                f"on {len(calibration_ds)} windows (head only, base "
                f"frozen, lr={self.config.calibration_lr})"
            )
            self._reset_head_weights(eval_model)
            self._calibrate_head(
                model=eval_model,
                calibration_loader=calibration_loader,
                calibration_dataset=calibration_ds,
                epochs=self.config.calibration_epochs,
            )

        elif algorithm == "PeFLL":
            raise NotImplementedError(
                "PeFLLServer must override "
                "evaluate_test_subject_with_calibration() to use the "
                "hypernetwork-based head generation."
            )

        else:
            raise ValueError(
                f"Unknown algorithm {algorithm!r} for held-out "
                f"calibration.  Expected: FedAvg, FedProx, FedRep, "
                f"FedBABU, FedRepDP, PeFLL."
            )

        # Step 4: evaluate on the 80% evaluation set.
        test_metrics = evaluate_client(
            model=eval_model,
            dataloader=evaluation_loader,
            device=self.device,
            subject_id=test_subject,
            num_classes=self.config.n_classes,
        )

        print(f"  S{test_subject} post-calibration metrics:")
        print(f"    macro-F1:      {test_metrics.macro_f1:.4f}")
        print(f"    accuracy:      {test_metrics.accuracy:.4f}")
        print(f"    stress recall: {test_metrics.stress_recall:.4f}")
        print(f"    stress F1:     {test_metrics.stress_f1:.4f}")
        print(f"    n_samples:     {test_metrics.n_samples}")

        return test_metrics

    def _reset_head_weights(self, model: WESADFLModel) -> None:
        """Re-initialise the head with fresh random weights.

        For the held-out-with-calibration protocol, FedRep/FedBABU
        evaluate on a *new* client that had no role in federation, so
        the head cannot inherit any training client's parameters.
        Collins et al. 2021 handles this by solving for the optimal
        head given the representation and the new client's calibration
        samples; our proxy is K epochs of gradient descent on a random
        initialisation, which is sufficient for a small LSTM head.

        Walks all submodules of model.head and calls reset_parameters()
        (covers Linear, Conv, LSTM, GRU, BatchNorm, LayerNorm).
        """
        reset_count = 0
        for module in model.head.modules():
            if hasattr(module, "reset_parameters") and callable(
                module.reset_parameters
            ):
                module.reset_parameters()
                reset_count += 1
        print(
            f"    [head reset] re-initialised {reset_count} submodules "
            f"in model.head"
        )

    def _calibrate_head(
        self,
        model: WESADFLModel,
        calibration_loader: DataLoader,
        calibration_dataset,
        epochs: int,
    ) -> None:
        """Train model.head on the calibration loader with base frozen.

        Uses AdamW with class weights computed from the CALIBRATION
        dataset (not the federation pool), because the held-out test
        subject has their own class balance and using federation-level
        weights would bias the decision threshold.

        BatchNorm handling: model.base contains BatchNorm1d layers.
        Running the base in train() mode during calibration would
        update its BN running statistics from the test subject's data,
        which is a subtle leakage source.  We therefore put the base
        in eval() mode while leaving the head in train() mode — base
        BN statistics stay fixed at their federation-final values.
        """
        # Freeze base parameters; unfreeze head parameters.
        for p in model.base.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

        class_weights = calibration_dataset.get_class_weights().to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Use calibration_lr (smaller than federation LR) because the
        # calibration set is tiny (~28 windows) and a larger step
        # would overshoot.  Matches Cross_Modal_KD plan's 1e-4.
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=self.config.calibration_lr,
            weight_decay=self.config.weight_decay,
        )

        # Base in eval() mode freezes its BN running statistics so
        # they cannot be updated by test subject's data.  Head in
        # train() mode enables dropout/BN training-mode behaviour
        # for the head layers.
        model.base.eval()
        model.head.train()

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            n_samples_seen = 0
            for signals, labels in calibration_loader:
                signals = {
                    k: v.to(self.device, non_blocking=True) for k, v in signals.items()
                }
                labels = labels.to(self.device, non_blocking=True)

                logits = model(signals)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                n_samples_seen += labels.size(0)

            if n_batches > 0:
                mean_loss = total_loss / n_batches
                print(
                    f"    [calibration epoch {epoch + 1}/{epochs}] "
                    f"mean_loss={mean_loss:.4f}  batches={n_batches}  "
                    f"samples={n_samples_seen}"
                )

        # Restore requires_grad and put model fully in eval mode for
        # the subsequent evaluation pass.
        for p in model.base.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True

        model.eval()

    # Result saving

    def save_results(self, result_dir: str = "results/"):
        """Save all accumulated results to HDF5.

        Also stores final per-subject metrics for the results tables.

        Mirrors PFLIB serverbase.py save_results().
        """
        # Collect final per-subject metrics
        final_f1 = {}
        final_acc = {}
        final_stress_recall = {}

        for client in self.clients:
            cm = client.test_metrics()
            final_f1[cm.subject_id] = cm.macro_f1
            final_acc[cm.subject_id] = cm.accuracy
            final_stress_recall[cm.subject_id] = cm.stress_recall

        self.results.set_final_metrics(
            per_subject_f1=final_f1,
            per_subject_acc=final_acc,
            per_subject_stress_recall=final_stress_recall,
        )

        return self.results.save(result_dir=result_dir)

    def save_global_model(self, path: str | None = None):
        """Save the global model checkpoint."""
        if path is None:
            path = os.path.join(
                "models",
                f"{self.config.algorithm}_fold{self.fold_config.test_subject}_global.pt",
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.global_model.state_dict(), path)
        print(f"  Global model saved: {path}")

    # Main training loop

    def train(self):
        """Run the full federation training loop.

        Default implementation: standard FedAvg-style loop with
        base-only aggregation.  Algorithm-specific servers override
        this if the loop structure differs (e.g. PeFLL has a
        fundamentally different training paradigm).

        The loop follows PFLIB serveravg.py / PFLIB serverrep.py:
          1. Select clients (all, for WESAD)
          2. Send global model to clients
          3. Evaluate (at eval_gap intervals)
          4. Each client trains locally
          5. Collect client models
          6. Optionally run DLG attack
          7. Aggregate into global model
          8. Repeat

        !!!!!!!!!!!!!!!EVALUATION ORDERING NOTE FOR SUBCLASSES:
        This base loop evaluates AFTER send_models() overwrites client
        base parameters with the global model.  For FedAvg this is
        correct (the global model is what matters).  For PFL methods
        where clients have personalised models (FedRep), this
        means evaluation sees the new global base, NOT the client's
        locally-trained model from the previous round.

        FedRep's server MUST override train() to reorder:
          evaluate → send → train → receive → aggregate
        so that evaluation captures each client's personalised model
        (local head + previous round's base) before the new global
        base overwrites it.
        """
        print(
            f"\n  ══ Training: {self.config.algorithm} "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds: {self.config.global_rounds}")
        print(f"  Clients: {len(self.clients)}")

        budget = []

        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # 1. Select clients
            self.select_clients()

            # 2. Broadcast global model
            self.send_models()

            # 3. Evaluate at eval_gap intervals
            if round_num % self.config.eval_gap == 0:
                print(f"\n  ─── Round {round_num} ───")
                self.evaluate(round_num=round_num)

            # 4. Local training
            for client in self.selected_clients:
                client.train()

            # 5. Collect models
            self.receive_models()

            # 6. DLG attack (optional)
            if self.config.dlg_eval and round_num % self.config.dlg_gap == 0:
                self.call_dlg(round_num)

            # 7. Aggregate
            self.aggregate_parameters()

            budget.append(time.time() - round_start)

        # Final training-client monitoring evaluation.  Under the
        # held-out-with-calibration protocol, the test subject is NOT
        # in self.clients, so final_metrics.test_subject_metrics is
        # None by construction.  The true LOSO fold metric comes from
        # evaluate_test_subject_with_calibration() below.
        print(
            f"\n  -> Final monitoring evaluation "
            f"(after round {self.config.global_rounds - 1}) <-"
        )
        final_metrics = self.evaluate(round_num=self.config.global_rounds)

        # Held-out test subject evaluation with calibration — the LOSO
        # fold metric.  Uses self.global_model (deep-copied inside the
        # utility so the saved checkpoint is unaffected).
        test_subject_metrics = self.evaluate_test_subject_with_calibration()
        self.results.set_test_subject_metrics(test_subject_metrics)

        # Summary
        print(f"\n  Final test subject F1: {test_subject_metrics.macro_f1:.4f}")
        print(
            f"  Mean training-client F1: "
            f"{final_metrics.mean_macro_f1:.4f} ± "
            f"{final_metrics.std_macro_f1:.4f}"
        )
        if budget:
            avg_time = sum(budget) / len(budget)
            print(f"  Average round time: {avg_time:.2f}s")

        # Save
        self.save_results()
        self.save_global_model()

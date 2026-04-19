# PeFLL server: joint training of embedding network + hypernetwork.
#
# The server jointly trains two networks:
#   - EmbedNet: maps client data -> client embedding
#   - HyperNet: maps client embedding -> personalised head parameters
#
# The training loop is fundamentally different from FedAvg/FedRep:
# instead of clients training locally and uploading model updates,
# the server generates personalised parameters for each client,
# evaluates them on client data, and backpropagates through the
# generated parameters into the hypernetwork and embedding network.
#
# Adapts from:
#   - PeFLL train.py                : the joint training loop
#   - PeFLL models.py EmbedHyper    : the combined embed+hyper module
#   - server_base.py                : our WESADServer base class + the
#                                     evaluate_test_subject_with_calibration
#                                     entry point (overridden here for
#                                     hypernet-based head generation)
#
# THE PEFLL TRAINING LOOP (Scott et al. 2024)
#
# Each round:
#   1. Sample a batch of data from each client's local DataLoader
#   2. Compute client embeddings: d_i = EmbedNet(batch_i)
#   3. Generate personalised head params: theta_i = HyperNet(d_i)
#   4. For each client: load theta_i into client's head, compute loss
#      on a SEPARATE batch (not the embedding batch)
#   5. Sum losses across clients, backpropagate through:
#      loss -> head params -> HyperNet -> EmbedNet
#   6. Update EmbedNet + HyperNet parameters via optimizer
#   7. The CNN base is trained separately (FedAvg-style aggregation)
#
# HELD-OUT TEST SUBJECT EVALUATION (LOSO metric)
#
# PeFLL's personalisation mechanism is the hypernetwork itself -
# there is NO gradient-based calibration step on the held-out subject.
# Instead:
#   1. Build the same 20/80 per-condition-block temporal split used
#      for all other methods.
#   2. Use ALL calibration windows as the embedding batch (single
#      forward pass, mean-pooled over the sample dimension for
#      permutation invariance per Scott et al. 2024). Using all
#      windows minimises embedding variance vs the 32-sample stochastic
#      estimate used during training.
#   3. Feed the embedding through the hypernetwork to generate the
#      held-out subject's personalised head parameters.
#   4. Load those params into a deep-copy of the final global model
#      (leaves self.global_model untouched for save_global_model).
#   5. Evaluate on the 80% evaluation split.
#
# This is fair vs FedAvg/FedRep/FedBABU: all four methods are evaluated
# on the SAME 80% partition of the held-out subject, using whatever
# calibration mechanism is native to each method (none for FedAvg, K
# gradient steps for FedRep/FedBABU, hypernet-generation for PeFLL).
#
# SCIENTIFIC CONTEXT
#
# Scott et al. 2024 (ICLR): "Personalized Federated Learning by
# Learning to Learn"
#   - Achieves state-of-the-art in low-data regimes (well suited to
#     the WESAD dataset's short-duration biosignal windows)
#   - Generalises to unseen future clients (PAC-Bayesian guarantees)

from __future__ import annotations

import copy
import time

import torch
from torch.utils.data import DataLoader

from ...utils.fl_config import FLConfig
from ...utils.metrics import ClientMetrics, evaluate_client
from ...utils.wesad_fl_dataset import (
    FLClientDataManager,
    FLFoldConfig,
    build_test_subject_splits,
)
from ..servers.server_base import WESADServer
from .client_pefll import PeFLLClient
from .wesad_embednet import WESADEmbedNet
from .wesad_hypernet import (
    WESADHyperNet,
    get_head_shapes_from_model,
)


class PeFLLServer(WESADServer):
    """PeFLL server: joint EmbedNet + HyperNet training.

    The server generates personalised head parameters for each client
    and trains the generation mechanism end-to-end.

    The CNN base is still trained via FedAvg-style aggregation (clients
    train base locally, server aggregates), and only the head generation
    is done by the hypernetwork.

    Parameters:
    config : FLConfig
        Must have algorithm="PeFLL".
    fold_config : FLFoldConfig
        LOSO fold configuration.
    device : torch.device | None
        Computation device.
    """

    def __init__(
        self,
        config: FLConfig,
        fold_config: FLFoldConfig,
        device: torch.device | None = None,
    ):
        super().__init__(config, fold_config, device)

        # PeFLL-specific networks

        # Get head parameter shapes from the global model
        head_shapes = get_head_shapes_from_model(self.global_model.head)

        # Embedding network
        self.embed_net = WESADEmbedNet(
            embedding_dim=config.embedding_dim,
            target_time=self._get_target_time(),
            use_temp=(config.device_type == "wrist"),
            temp_signal_name="w_temp" if config.device_type == "wrist" else "c_temp",
        ).to(self.device)

        # Hypernetwork
        self.hyper_net = WESADHyperNet(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hypernet_hidden,
            head_param_shapes=head_shapes,
        ).to(self.device)

        # Server-side optimizer for EmbedNet + HyperNet
        # Both networks are trained jointly on the server.
        self.pefll_optimizer = torch.optim.AdamW(
            list(self.embed_net.parameters()) + list(self.hyper_net.parameters()),
            lr=config.pefll_lr,
            weight_decay=config.weight_decay,
        )

        self.pefll_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pefll_optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
        )

    def _get_target_time(self) -> int:
        """Get the adaptive pooling target time from config or defaults."""
        if hasattr(self.config, "target_time") and self.config.target_time is not None:
            return self.config.target_time
        return 8 if self.config.device_type == "wrist" else 5

    def train(self):
        """PeFLL training loop.

        Two training mechanisms per round:
          1) HyperNet training: server generates head params, evaluates
             on client data, backprops through HyperNet + EmbedNet.
          2) Base training: clients train CNN base locally like FedRep,
             server aggregates base parameters.

        The CNN base and HyperNet are trained in the same round but
        with separate optimizers and separate gradient flows.
        """
        print(
            f"\n  ══ Training: {self.config.algorithm} "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds:    {self.config.global_rounds}")
        print(f"  Local epochs:     {self.config.local_epochs} (base)")
        print(f"  Embedding dim:    {self.config.embedding_dim}")
        print(f"  HyperNet hidden:  {self.config.hypernet_hidden}")
        print(
            f"  Clients:          {len(self.clients)} (training only; "
            f"test subject S{self.fold_config.test_subject} excluded)"
        )

        budget = []

        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # 1. EVALUATE (before parameter generation)
            if round_num % self.config.eval_gap == 0:
                print(f"\n  ─── Round {round_num} ───")
                # Generate personalised params for all clients before eval
                self._generate_and_load_heads()
                self.evaluate(round_num=round_num)

            # 2. HYPERNETWORK TRAINING STEP
            # For each client:
            #   a) Get embedding batch -> compute embedding
            #   b) Generate head params from embedding
            #   c) Compute loss using functional_call with generated params
            # Sum losses, backprop through HyperNet + EmbedNet

            self.embed_net.train()
            self.hyper_net.train()

            total_hyper_loss = torch.tensor(0.0, device=self.device)
            n_clients_trained = 0

            for client in self.clients:
                client_loss = self._compute_client_hyper_loss(client)
                if client_loss is not None:
                    total_hyper_loss = total_hyper_loss + client_loss
                    n_clients_trained += 1

            if n_clients_trained > 0:
                mean_hyper_loss = total_hyper_loss / n_clients_trained

                self.pefll_optimizer.zero_grad()
                mean_hyper_loss.backward()
                self.pefll_optimizer.step()

                self.pefll_scheduler.step(mean_hyper_loss.item())

            # 3. BASE TRAINING (FedAvg-style)
            # Broadcast global base, clients train locally collect and aggregate.
            #
            # Generate current heads first so base training computes
            # gradients against a meaningful head (not stale).
            self._generate_and_load_heads()

            self.select_clients()
            self.send_models()  # base only

            for client in self.selected_clients:
                self._train_client_base(client)

            self.receive_models()  # base only (deep-copy)
            self.aggregate_parameters()

            budget.append(time.time() - round_start)

        # Final training-client monitoring evaluation.
        # Regenerate heads on the final global base so the 13 training
        # clients are evaluated with their up-to-date personalised models.
        self._generate_and_load_heads()
        print(
            f"\n  ─── Final monitoring evaluation "
            f"(after round {self.config.global_rounds - 1}) ───"
        )
        final_metrics = self.evaluate(round_num=self.config.global_rounds)

        # LOSO FOLD METRIC - held-out test subject evaluation via hypernet.
        # Calls the PeFLL override of evaluate_test_subject_with_calibration
        # (below), which:
        #   1. Builds the 20/80 per-condition-block split of S{test}.
        #   2. Uses ALL calibration windows as the embedding batch.
        #   3. Generates head params via embed_net -> hyper_net.
        #   4. Loads them into a deep-copy of self.global_model.
        #   5. Evaluates on the 80% evaluation split.
        # No gradient calibration - the hypernet IS the calibration
        # mechanism.  Same 80% partition as FedAvg/FedRep/FedBABU for
        # fair cross-method comparison.
        test_subject_metrics = self.evaluate_test_subject_with_calibration()
        self.results.set_test_subject_metrics(test_subject_metrics)

        # Summary
        print(
            f"\n  ═══ PeFLL Final Results "
            f"(fold S{self.fold_config.test_subject}) ═══"
        )
        print("  Held-out test subject (LOSO fold metric):")
        print(f"    macro-F1:       {test_subject_metrics.macro_f1:.4f}")
        print(f"    accuracy:       {test_subject_metrics.accuracy:.4f}")
        print(f"    stress recall:  {test_subject_metrics.stress_recall:.4f}")
        print(f"    stress F1:      {test_subject_metrics.stress_f1:.4f}")
        print(f"    n_eval samples: {test_subject_metrics.n_samples}")
        print("  Training-client monitoring (13 clients):")
        print(
            f"    mean macro-F1:  {final_metrics.mean_macro_f1:.4f} "
            f"± {final_metrics.std_macro_f1:.4f}"
        )

        if budget:
            avg_time = sum(budget) / len(budget)
            total_time = sum(budget)
            print(f"  Average round time:  {avg_time:.2f}s")
            print(f"  Total training time: {total_time:.1f}s")

        self.save_results()
        self.save_global_model()

        return final_metrics

    def _compute_client_hyper_loss(self, client: PeFLLClient) -> torch.Tensor | None:
        """Compute the differentiable loss for one client.

        Gradient flows: loss -> model output -> generated head params ->
        HyperNet -> EmbedNet.

        Uses torch.func.functional_call to apply the generated head
        parameters differentiably.

        Parameters:
        client : PeFLLClient

        Returns:
        torch.Tensor | None
            Scalar loss (with grad_fn), or None if the client has no data.
        """
        try:
            # Get two separate batches: one for embedding, one for loss
            embed_signals = client.get_embedding_batch(n_samples=32)
            train_loader = client.load_train_data(batch_size=self.config.batch_size)
            train_signals, train_labels = next(iter(train_loader))
            train_signals = {k: v.to(self.device) for k, v in train_signals.items()}
            train_labels = train_labels.to(self.device)
        except StopIteration:
            return None

        # 1. Compute client embedding
        client_embedding = self.embed_net(embed_signals)  # (embedding_dim,)

        # 2. Generate head parameters
        generated_params = self.hyper_net(client_embedding)  # OrderedDict

        # 3. Compute loss using functional_call
        #
        # The base forward is run normally (shared CNN features),
        # then the head forward uses the generated parameters.

        # Run base forward (detached from hyper gradients; the base has
        # its own training via FedAvg-style aggregation).
        with torch.no_grad():
            fused = client.model.base(train_signals)

        # cuDNN's fused LSTM kernel requires .train() mode for its
        # backward pass.  self.evaluate() (called earlier in the round)
        # left the head in eval mode.  Flip to train mode for the
        # differentiable forward, then restore eval mode for safety.
        was_training = client.model.head.training
        client.model.head.train()
        try:
            head_output = torch.func.functional_call(
                client.model.head,
                generated_params,
                (fused,),
            )
            loss = client.loss(head_output, train_labels)
        finally:
            if not was_training:
                client.model.head.eval()

        return loss

    def _generate_and_load_heads(self):
        """Generate personalised head params for all clients and load them.

        Called before evaluation to ensure each client has their
        personalised model.
        """
        self.embed_net.eval()
        self.hyper_net.eval()

        with torch.no_grad():
            for client in self.clients:
                try:
                    embed_signals = client.get_embedding_batch(n_samples=32)
                    client_embedding = self.embed_net(embed_signals)
                    generated_params = self.hyper_net(client_embedding)
                    client.set_head_parameters(generated_params)
                except StopIteration:
                    pass  # client has no data

    def _train_client_base(self, client: PeFLLClient):
        """Train a client's CNN base locally (FedAvg-style)."""
        trainloader = client.load_train_data()
        client.model.train()

        # Freeze head, train base only
        for param in client.model.head.parameters():
            param.requires_grad = False
        for param in client.model.base.parameters():
            param.requires_grad = True

        for _epoch in range(client.local_epochs):
            for signals, labels in trainloader:
                signals = {k: v.to(client.device) for k, v in signals.items()}
                labels = labels.to(client.device)

                output = client.model(signals)
                loss = client.loss(output, labels)

                client.base_optimizer.zero_grad()
                loss.backward()
                client.base_optimizer.step()

        # Restore requires_grad
        for param in client.model.head.parameters():
            param.requires_grad = True

    # Held-out test subject evaluation (LOSO)

    def evaluate_test_subject_with_calibration(self) -> ClientMetrics:
        """PeFLL's held-out LOSO evaluation: hypernet-based head generation.

        Overrides WESADServer.evaluate_test_subject_with_calibration().
        Because PeFLL personalises via the hypernetwork rather than
        gradient descent on the head, this method does NOT call
        _calibrate_head() or _reset_head_weights().  Instead it:

        1. Builds the 20/80 per-condition-block temporal split of the
           held-out subject's non-overlapped windows (same helper used
           for all methods).
        2. Assembles a SINGLE batch containing ALL calibration windows
           (~28 windows at 16s non-overlapped).  Using the full set
           rather than a 32-sample stochastic subset minimises the
           embedding's estimator variance; the EmbedNet is permutation-
           invariant via mean-pooling (Scott et al. 2024), so the
           embedding is well-defined for any batch size.
        3. Passes the calibration batch through embed_net to compute
           the subject's embedding, then through hyper_net to generate
           personalised head parameters.
        4. Deep-copies self.global_model (so save_global_model writes
           the federated model, not the personalised copy), loads the
           generated parameters into the copy's head, puts the copy in
           eval mode.
        5. Evaluates on the 80% evaluation split using evaluate_client.

        Protocol note: the 80% evaluation split is the SAME partition
        used by FedAvg/FedRep/FedBABU for this subject, so the four
        methods are directly comparable on a per-subject basis.

        Returns:
        ClientMetrics
            Metrics on the 80% evaluation split, using the hypernet-
            generated personalised head.

        Raises:
        ValueError
            If the evaluation set is empty.
        """
        test_subject = self.fold_config.test_subject

        print(
            f"\n  ══ Held-out calibration + evaluation "
            f"(S{test_subject}, {self.config.algorithm}) ══"
        )

        # Step 1: build the same 20/80 per-condition-block split used
        # by every other method.
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
        if len(calibration_ds) == 0:
            raise ValueError(
                f"Calibration set for S{test_subject} is empty; "
                f"cannot compute embedding for hypernet."
            )

        print(
            f"  PeFLL calibration: all {len(calibration_ds)} cal windows used "
            f"as the embedding batch (single forward pass, no gradient "
            f"descent on the head - hypernet IS the calibration mechanism)."
        )

        # Step 2: build a single batch containing ALL calibration windows.
        # batch_size=len(calibration_ds) + drop_last=False guarantees
        # one batch of all N cal samples.
        pin = torch.cuda.is_available()
        cal_loader = DataLoader(
            calibration_ds,
            batch_size=len(calibration_ds),
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=pin,
        )
        # One iteration gives the full calibration batch
        embed_signals, _embed_labels = next(iter(cal_loader))
        embed_signals = {
            k: v.to(self.device, non_blocking=True) for k, v in embed_signals.items()
        }

        # Step 3: generate the subject's personalised head parameters.
        # EmbedNet + HyperNet in eval mode, no grad.
        self.embed_net.eval()
        self.hyper_net.eval()
        with torch.no_grad():
            client_embedding = self.embed_net(embed_signals)  # (embedding_dim,)
            generated_params = self.hyper_net(client_embedding)  # OrderedDict

        # Step 4: deep-copy the global model, load generated head params.
        # Deep-copy preserves self.global_model for save_global_model
        # and avoids mutating it during evaluation.
        eval_model = copy.deepcopy(self.global_model).to(self.device)
        self._load_head_params_into_model(eval_model, generated_params)
        eval_model.eval()

        # Step 5: evaluate on the 80% evaluation split.
        evaluation_loader = DataLoader(
            evaluation_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=pin,
        )

        test_metrics = evaluate_client(
            model=eval_model,
            dataloader=evaluation_loader,
            device=self.device,
            subject_id=test_subject,
            num_classes=self.config.n_classes,
        )

        print(f"  S{test_subject} hypernet-personalised metrics:")
        print(f"    macro-F1:      {test_metrics.macro_f1:.4f}")
        print(f"    accuracy:      {test_metrics.accuracy:.4f}")
        print(f"    stress recall: {test_metrics.stress_recall:.4f}")
        print(f"    stress F1:     {test_metrics.stress_f1:.4f}")
        print(f"    n_samples:     {test_metrics.n_samples}")

        return test_metrics

    @staticmethod
    def _load_head_params_into_model(model, head_params) -> None:
        """Load hypernetwork-generated parameters into model.head.

        Mirrors PeFLLClient.set_head_parameters() exactly: writes the
        generated tensors into model.head.state_dict() and loads with
        strict=False.  strict=False is required because model.head
        contains BatchNorm running_mean / running_var / num_batches_tracked
        buffers that the hypernetwork does NOT generate (those are
        learned locally during federation on the base-only side and
        copied from self.global_model by the deep-copy above).

        Parameters:
        model : WESADFLModel
            Target model (deep-copied from self.global_model).
        head_params : OrderedDict[str, torch.Tensor]
            Parameter name -> tensor, as produced by WESADHyperNet.
        """
        head_state = model.head.state_dict()

        learnable_names = {name for name, _ in model.head.named_parameters()}
        for name in head_params:
            if name not in learnable_names:
                raise KeyError(
                    f"Generated parameter '{name}' not found in head's "
                    f"learnable parameters. Available: {sorted(learnable_names)}"
                )

        for name, param in head_params.items():
            head_state[name] = param

        model.head.load_state_dict(head_state, strict=False)

    # Unseen client experiment (legacy; not on the main LOSO path)

    def evaluate_unseen_client(
        self,
        data_manager: FLClientDataManager,
        subject_id: int,
    ) -> dict:
        """Evaluate PeFLL on a completely unseen client.

        DEPRECATED for the main LOSO path: the primary held-out
        evaluation entry point is now evaluate_test_subject_with_calibration(),
        which uses the standard 20/80 per-condition-block split shared
        with FedAvg/FedRep/FedBABU.  This method is retained for the
        optional "unseen client" experiment where PeFLL is evaluated on
        subjects that were completely excluded from federation (not
        just held out as a LOSO fold).

        generate personalised parameters for a client
        not seen during training, using only a small batch of their data.

        No fine-tuning, as the hypernetwork should produce a ready-to-use model.

        NOTE: The unseen client receives the globally aggregated CNN base.
        This is correct per PeFLL's design:
        the base is shared universally, and all personalisation comes
        through the hypernetwork-generated head parameters.  The embedding
        network captures the new client's physiological profile from their
        data, and the hypernetwork translates that into personalised LSTM
        weights.  This is fundamentally different from FedRep where the
        test client must participate in federation to train their local head.

        Parameters:
        data_manager : FLClientDataManager
            Data manager for the unseen subject.
        subject_id : int
            The unseen subject's ID.

        Returns:
        dict
            subject_id, macro_f1, accuracy, stress_recall
        """

        # Create a temporary client
        temp_client = PeFLLClient(
            config=self.config,
            subject_id=subject_id,
            data_manager=data_manager,
            global_model=self.global_model,
            device=self.device,
        )

        # Generate personalised head from a small data batch
        self.embed_net.eval()
        self.hyper_net.eval()
        with torch.no_grad():
            embed_signals = temp_client.get_embedding_batch(n_samples=32)
            client_embedding = self.embed_net(embed_signals)
            generated_params = self.hyper_net(client_embedding)
            temp_client.set_head_parameters(generated_params)

        # Evaluate
        testloader = temp_client.load_test_data()
        metrics = evaluate_client(
            model=temp_client.model,
            dataloader=testloader,
            device=self.device,
            subject_id=subject_id,
            num_classes=self.config.n_classes,
        )

        print(
            f"  Unseen client S{subject_id}: "
            f"F1={metrics.macro_f1:.4f}  Acc={metrics.accuracy:.4f}  "
            f"StressRecall={metrics.stress_recall:.4f}"
        )

        return {
            "subject_id": subject_id,
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "stress_recall": metrics.stress_recall,
        }

    def save_global_model(self, path: str | None = None):
        """Save global model + EmbedNet + HyperNet."""
        import os

        if path is None:
            path = os.path.join(
                "models",
                f"{self.config.algorithm}_fold{self.fold_config.test_subject}_global.pt",
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "global_model": self.global_model.state_dict(),
                "embed_net": self.embed_net.state_dict(),
                "hyper_net": self.hyper_net.state_dict(),
            },
            path,
        )
        print(f"  PeFLL model saved: {path}")

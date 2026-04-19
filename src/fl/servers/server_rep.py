# FedRep server: base-only aggregation with personalised evaluation.
#
# The critical differences from FedAvg:
#   1. Only model.base (CNN) is broadcast and aggregated: the LSTM head
#      stays local on each client permanently.
#   2. Evaluation captures each client's PERSONALISED model (local head +
#      current base), so eval must happen BEFORE send_models() overwrites
#      the base with the new global model.
#   3. Communication cost is ~80% lower than FedAvg (only CNN parameters
#      are transmitted, not the LSTM).
#
# Adapts from:
#   - PFLIB serverrep.py    : base-only receive_models(), the training
#                              loop structure
#   - server_base.py        : our WESADServer base class
#
# EVALUATION ORDERING
#
# FedAvg:  send_full → eval → train → receive → aggregate
#   all clients hold the same global model after send.
#
# FedRep:  eval → send_base → train → receive_base → aggregate
#   Eval MUST come BEFORE send_models() because:
#     - Each client has a personalised model: their LOCAL head (trained
#       in the previous round) + the BASE they received last round.
#     - send_models() overwrites the base with the newly-aggregated
#       global base, destroying the base the head was trained against.
#     - Evaluating after send would measure the head's performance
#       on features it hasn't been adapted to yet, giving misleadingly
#       poor results in early rounds.
#
#   On round 0 (before any training), eval still runs first.  Clients
#   hold their initial models (cold-start: random base + random head,
#   or warm-start: pre-trained base + pre-trained head).  This gives
#   the untrained baseline measurement.
#
# SCIENTIFIC CONTEXT:
#
# Collins et al. 2021 (ICML): "Exploiting Shared Representations for
# Personalized Federated Learning"
#   - Proves linear convergence to ground-truth representations
#   - The CNN/LSTM split maps exactly onto their framework:
#     representation = CNN branches (universal physiological features)
#     head = LSTM (subject-specific temporal dynamics)
#
# Our CNN/LSTM split has biological justification:
#   - CNN branches learn modality-specific features (BVP cardiac rhythm,
#     EDA electrodermal response, ACC motion) that are universal
#   - LSTM learns how a specific individual transitions into/out of stress
#
# Expected results:
#   - FedRep should improve over FedAvg (personalisation helps)
#   - FedRep should approach or exceed centralised on some subjects
#   - Per-subject variance should be lower than FedAvg (each client
#     has a model adapted to their physiology)
#   - If FedRep ≈ FedAvg, the split is wrong or WESAD subjects are
#     too similar -> debug before proceeding !!!!!!!!!

from __future__ import annotations

import time

from .server_base import WESADServer


class FedRepServer(WESADServer):
    """FedRep server: base-only broadcast, collection, and aggregation.

    Overrides the base server's train() loop to:
      1. Evaluate BEFORE sending the new global base (captures
         personalised models)
      2. Use send_models() / receive_models() which operate on
         base only (inherited from WESADServer and in that class this is already
         the default)

    Usage
    -----
    >>> config = FLConfig(algorithm="FedRep", fold_id=2, plocal_epochs=5)
    >>> fold_config = FLFoldConfig(test_subject=2)
    >>> server = FedRepServer(config, fold_config)
    >>> server.set_clients(FedRepClient)
    >>> server.train()
    """

    def train(self):
        """FedRep training loop with corrected evaluation ordering.

        The loop structure is:
          For each round:
            1. EVALUATE (at eval_gap intervals): captures personalised
               models BEFORE the base is overwritten
            2. Select clients (all 15)
            3. Send global BASE to all clients (overwrites client bases)
            4. Each client trains: head first, then base (alternating)
            5. Collect base models from all clients (deep-copy)
            6. Optionally run DLG attack (TODO)
            7. Aggregate base models into new global base

        This ordering ensures that evaluation always sees each client's
        personalised model: their local head + the base it was trained
        against.  This is the scientifically correct and literature-baqcked measurement for PFL.
        """
        print(
            f"\n  ══ Training: {self.config.algorithm} "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds:    {self.config.global_rounds}")
        print(f"  Local epochs:     {self.config.local_epochs} (base)")
        print(f"  plocal_epochs:    {self.config.plocal_epochs} (head)")
        print(f"  Clients:          {len(self.clients)}")
        print("  Aggregation:      BASE ONLY (CNN branches)")
        print("  Head:             LOCAL (LSTM + BN + classifier, never communicated)")

        budget = []

        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # 1. EVALUATE (before send to get personalised models)

            # Each client currently holds:
            #   base: the global base from last round's aggregation (or initial model on round 0)
            #   head: their locally-trained personal head from last round's Phase 1 (or initial on round 0)
            # This combination is the personalised model.
            if round_num % self.config.eval_gap == 0:
                print(f"\n  ─── Round {round_num} ───")
                print("  Evaluating personalised models (before base broadcast)")
                self.evaluate(round_num=round_num)

            # 2. Select clients
            self.select_clients()

            # 3. Broadcast global BASE to all clients
            # This overwrites each client's base with the latest
            # aggregated global base.  Heads are not touched.
            self.send_models()

            # 4. Local training (alternating head/base per client)
            # FedRepClient.train() executes:
            #   Phase 1: freeze base, train head for plocal_epochs
            #   Phase 2: freeze head, train base for local_epochs
            for client in self.selected_clients:
                client.train()

            # 5. Collect BASE models from all clients (deep-copy)
            # Only base (CNN) parameters are collected.
            # Head (LSTM) stays local and thuis never sent to server.
            self.receive_models()

            # 6. DLG attack (optional)
            if self.config.dlg_eval and round_num % self.config.dlg_gap == 0:
                self.call_dlg(round_num)

            # 7. Aggregate base models into global base
            # Weighted average of CNN base parameters.
            # The global head is not updated bc it's never used.
            # each client maintains their own head).
            self.aggregate_parameters()

            budget.append(time.time() - round_start)

        # Final training-client monitoring evaluation.  After the last
        # round's aggregation, each of the 13 TRAINING clients holds:
        #   base: their locally-trained base from Phase 2 of the last
        #         round (NOT yet overwritten by the new global)
        #   head: their locally-trained head from Phase 1
        # Under the held-out-with-calibration protocol, the TEST
        # subject is NOT an FL client, so this evaluation tracks only
        # the 13 training subjects as a convergence-monitoring metric.
        print(
            f"\n  ─── Final monitoring evaluation "
            f"(after round {self.config.global_rounds - 1}) ───"
        )
        print("  Evaluating training-clients' personalised models")
        final_metrics = self.evaluate(round_num=self.config.global_rounds)

        # Held-out test subject evaluation WITH calibration.
        # Utility:
        #   1. Builds the 20/80 per-block calibration/evaluation split.
        #   2. Deep-copies self.global_model (leaves it untouched for
        #      save_global_model()).
        #   3. Re-initialises the head with random weights.
        #   4. Trains the head for calibration_epochs epochs on the
        #      calibration set with base frozen and base BN statistics
        #      frozen.
        #   5. Evaluates on the 80% evaluation split.
        # THIS IS THE LOSO FOLD METRIC for FedRep.
        test_subject_metrics = self.evaluate_test_subject_with_calibration()
        self.results.set_test_subject_metrics(test_subject_metrics)

        # Summary
        print(
            f"\n  ═══ FedRep Final Results "
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
            print(f"  Average round time: {avg_time:.2f}s")
            print(f"  Total training time: {total_time:.1f}s")

        # Save results and model
        self.save_results()
        self.save_global_model()

        return final_metrics

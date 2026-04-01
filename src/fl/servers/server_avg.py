# FedAvg server: standard federated averaging with full-model aggregation.
#
# The critical difference from the base server (which defaults to
# base-only aggregation for FedRep/FedBABU) is that FedAvg aggregates
# the FULL model (base + head).  This means:
#   - send_full_models() broadcasts base + head to all clients
#   - receive_full_models() collects full models from all clients
#   - aggregate_parameters() averages the full model
#
# Adapts from:
#   - PFLIB serveravg.py    : the simplest server in PFLlib, just calls
#                              aggregate_parameters() each round
#   - server_base.py        : our WESADServer base class
#
# FedAvg protocol (McMahan et al., 2017):
#   Each round:
#     1. Server broadcasts full global model → all clients
#     2. Clients train locally for E epochs
#     3. Server collects full models from all clients
#     4. Server computes weighted average (by window count)
#     5. Result becomes the new global model
#
# Evaluation:
#   Each client is evaluated using the GLOBAL model (since FedAvg has
#   no personalisation, all clients share the same model after each
#   round).  The eval ordering in the base server's train() loop
#   (send → eval → train) is therefore correct for FedAvg: after
#   send_full_models(), every client holds the latest global model,
#   which is exactly what we want to evaluate.
#
# Literature:
#   McMahan et al. 2017: "Communication-Efficient Learning of Deep
#   Networks from Decentralized Data"
# ──────────────────────────────────────────────────────────────────────
# TODO (remaining baselines, implement if time permits):
#   - fl/servers/server_prox.py   : FedProx (proximal term)
#   - fl/servers/server_per.py    : FedAvg+FT (fine-tuning after)
#   - fl/servers/server_local.py  : Local-only (no federation)
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time

from .server_base import WESADServer


# inherits WESADSServer and overrides train() to use full-model send/receive
class FedAvgServer(WESADServer):
    """FedAvg server: full-model broadcast, collection, and aggregation.

    Overrides the base server's train() loop to use full-model methods
    (send_full_models / receive_full_models) instead of base-only.

    Usage
    -----
    >>> config = FLConfig(algorithm="FedAvg", fold_id=2)
    >>> fold_config = FLFoldConfig(test_subject=2)
    >>> server = FedAvgServer(config, fold_config)
    >>> server.set_clients(FedAvgClient)
    >>> server.train()
    """

    def train(self):
        """FedAvg training loop with full-model aggregation.

        The key difference from the base server's train() is:
          - send_full_models() instead of send_models()
          - receive_full_models() instead of receive_models()

        This ensures the ENTIRE model (CNN base + LSTM head) is
        synchronised and aggregated each round, which is standard
        FedAvg behaviour.

        Loop structure (same as PFLIB serveravg.py):
          For each round:
            1. Select clients (all 15)
            2. Broadcast full global model
            3. Evaluate at eval_gap intervals
            4. Each client trains locally for local_epochs
            5. Collect full models from all clients
            6. Optionally run DLG attack
            7. Aggregate full models into new global model

        The eval-before-train ordering is correct for FedAvg because
        evaluation uses the global model (which all clients receive
        in step 2).
        """
        print(
            f"\n  ══ Training: {self.config.algorithm} "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds: {self.config.global_rounds}")
        print(f"  Local epochs:  {self.config.local_epochs}")
        print(f"  Clients: {len(self.clients)}")
        print("  Aggregation: FULL MODEL (base + head)")

        budget = []

        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # 1. Select clients (all, for WESAD)
            self.select_clients()

            # 2. Broadcast FULL global model to all clients
            self.send_full_models()

            # 3. Evaluate at eval_gap intervals
            #    After send_full_models(), all clients hold the global model.
            #    For FedAvg this is the correct model to evaluate.
            if round_num % self.config.eval_gap == 0:
                print(f"\n  ─── Round {round_num} ───")
                self.evaluate(round_num=round_num)

            # 4. Local training (each client trains full model)
            for client in self.selected_clients:
                client.train()

            # 5. Collect FULL models from all clients
            self.receive_full_models()

            # 6. DLG attack (optional)
            if self.config.dlg_eval and round_num % self.config.dlg_gap == 0:
                self.call_dlg(round_num)

            # 7. Aggregate full models into global model
            self.aggregate_parameters()

            budget.append(time.time() - round_start)

        # Before final evaluation, broadcast the aggregated global model
        # so clients are evaluated on the global model
        self.send_full_models()

        # Final evaluation: clients hold their post-training parameters
        # (the global model after aggregation of the last round's uploads).
        # For FedAvg, all clients should produce identical results since
        # they all received the same global model and trained independently.
        # Any per-client variance comes from the eval being on different
        # subjects' test data, not from model differences.
        print(
            f"\n  ─── Final evaluation (after round {self.config.global_rounds - 1}) ───"
        )
        final_metrics = self.evaluate(round_num=self.config.global_rounds)

        # Summary
        if final_metrics.test_subject_metrics is not None:
            test_f1 = final_metrics.test_subject_metrics.macro_f1
            test_acc = final_metrics.test_subject_metrics.accuracy
            test_sr = final_metrics.test_subject_metrics.stress_recall
            print(
                f"\n  ═══ FedAvg Final Results (fold S{self.fold_config.test_subject}) ═══"
            )
            print(f"  Test subject F1:            {test_f1:.4f}")
            print(f"  Test subject Accuracy:      {test_acc:.4f}")
            print(f"  Test subject Stress Recall: {test_sr:.4f}")
            print(
                f"  Mean client F1:             {final_metrics.mean_macro_f1:.4f} "
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

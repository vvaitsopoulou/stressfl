# FedBABU server: base-only aggregation + post-federation head fine-tuning.
#
# The FedBABU server has two phases:
#   Phase A (FL rounds): identical to FedRep server: base-only broadcast,
#     base-only collection, base-only aggregation.  Heads are frozen.
#   Phase B (post-federation): calls client.fine_tune() for every client,
#     then evaluates the personalised models.
#
# Adapts from:
#   - PFLIB serverbabu.py   : the two-phase structure (FL then fine-tune)
#   - server_base.py        : our WESADServer base class
#
# SCIENTIFIC CONTEXT (Oh et al. 2022)
#
# FedBABU = Federated Averaging with Body Aggregation and Body Update.
#
# The ablation answers: does training the head during FL rounds (FedRep)
# help or hurt compared to freezing it (FedBABU)?
#
# Expected ablation results:
#   - Plot macro-F1 vs fine_tuning_epochs ∈ {1, 2, 5, 10, 20}
#   - Compare FedBABU curve against FedRep's best result
#   - If FedBABU > FedRep: the LSTM head overfits to the global
#     distribution during FL (Oh et al.'s hypothesis holds for biosignals)
#   - If FedRep > FedBABU: alternating training helps the head adapt
#     to evolving features (domain-specific evidence)
#
# Evaluation ordering:
#   During FL rounds, the same eval-before-send ordering as FedRep is
#   used (clients hold their current base + frozen random head).
#   After fine-tuning, a final evaluation captures the personalised
#   models (trained base + fine-tuned head).
#
# TODO: Ditto, implement if time permits:
#   - PFLib client_ditto.py  : dual-model client (global + personal)
#   - PFLib server_ditto.py  : Ditto server

from __future__ import annotations

import time

from .server_base import WESADServer


class FedBABUServer(WESADServer):
    """FedBABU server: FL rounds with frozen heads, then local fine-tuning.

    Two-phase training:
      Phase A: standard base-only FL (same as FedRep server)
      Phase B: each client fine-tunes their head locally

    Usage
    -----
    >>> config = FLConfig(algorithm="FedBABU", fold_id=2, fine_tuning_epochs=10)
    >>> fold_config = FLFoldConfig(test_subject=2)
    >>> server = FedBABUServer(config, fold_config)
    >>> server.set_clients(FedBABUClient)
    >>> server.train()
    """

    def train(self):
        """FedBABU two-phase training.

        Phase A Federated base training (global_rounds rounds):
          Same loop as FedRep: eval → send_base → train → receive → aggregate.
          During this phase, every client's head is frozen (random init).
          Only the CNN base is trained and aggregated.

        Phase B Local head fine-tuning (fine_tuning_epochs per client):
          After all FL rounds complete, each client unfreezes their head
          and fine-tunes it on their local data with the base frozen.
          This is the personalisation step.

        Final evaluation captures the personalised models (trained base +
        fine-tuned head).
        """
        print(
            f"\n  ══ Training: {self.config.algorithm} "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds:        {self.config.global_rounds}")
        print(f"  Local epochs (base):  {self.config.local_epochs}")
        print(
            f"  Fine-tuning epochs:   {self.config.fine_tuning_epochs} (head, post-federation)"
        )
        print(f"  Clients:              {len(self.clients)}")
        print("  Head during FL:       FROZEN (random initialisation)")

        budget = []

        # PHASE A: Federated base training (heads frozen)
        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # Evaluate before send
            # NOTE TOSELF PLEASE REMEMBER THIS: Phase A metrics (trained base + frozen
            # random head) are NOT directly comparable to FedRep's per-round
            # metrics (trained base + trained head).  They measure different
            # things.  The only fair comparison between FedBABU and FedRep is
            # the final post-fine-tuning F1.
            if round_num % self.config.eval_gap == 0:
                print(
                    f"\n  ─── Round {round_num} (Phase A: base training, head frozen) ───"
                )
                self.evaluate(round_num=round_num)

            self.select_clients()
            self.send_models()  # base only

            for client in self.selected_clients:
                client.train()  # base only (head frozen)

            self.receive_models()  # base only (deep-copy)

            if self.config.dlg_eval and round_num % self.config.dlg_gap == 0:
                self.call_dlg(round_num)

            self.aggregate_parameters()

            budget.append(time.time() - round_start)

        # Evaluate after Phase A (before fine-tuning)
        # This shows what the model achieves with a frozen random head
        # expected poor, because the head hasn't been trained.
        print("\n  ─── Post-Phase A evaluation (frozen head) ───")
        pre_ft_metrics = self.evaluate(round_num=self.config.global_rounds)

        if pre_ft_metrics.test_subject_metrics is not None:
            print(
                f"  Pre-fine-tuning test F1: "
                f"{pre_ft_metrics.test_subject_metrics.macro_f1:.4f}"
            )

        # PHASE B: Local head fine-tuning (base frozen)
        # Each client unfreezes their head and trains it locally.
        # The base is frozen during fine-tuning.
        print(
            f"\n  ─── Phase B: Fine-tuning heads "
            f"({self.config.fine_tuning_epochs} epochs per client) ───"
        )

        ft_start = time.time()
        for client in self.clients:
            client.fine_tune(which_module=["head"])
        ft_time = time.time() - ft_start

        print(f"  Fine-tuning completed in {ft_time:.1f}s")

        # Final evaluation (personalised: trained base + fine-tuned head)
        print("\n  ─── Final evaluation (after fine-tuning) ───")
        final_metrics = self.evaluate(round_num=self.config.global_rounds + 1)

        # Summary
        if final_metrics.test_subject_metrics is not None:
            test_f1 = final_metrics.test_subject_metrics.macro_f1
            test_acc = final_metrics.test_subject_metrics.accuracy
            test_sr = final_metrics.test_subject_metrics.stress_recall
            print(
                f"\n  ═══ FedBABU Final Results (fold S{self.fold_config.test_subject}) ═══"
            )
            print(f"  Test subject F1:            {test_f1:.4f}")
            print(f"  Test subject Accuracy:      {test_acc:.4f}")
            print(f"  Test subject Stress Recall: {test_sr:.4f}")
            print(
                f"  Mean client F1:             {final_metrics.mean_macro_f1:.4f} "
                f"± {final_metrics.std_macro_f1:.4f}"
            )

            # Show improvement from fine-tuning
            if pre_ft_metrics.test_subject_metrics is not None:
                pre_f1 = pre_ft_metrics.test_subject_metrics.macro_f1
                delta = test_f1 - pre_f1
                print(
                    f"  F1 gain from fine-tuning:   {delta:+.4f} "
                    f"({pre_f1:.4f} → {test_f1:.4f})"
                )

        if budget:
            avg_time = sum(budget) / len(budget)
            total_time = sum(budget) + ft_time
            print(f"  Average FL round time:  {avg_time:.2f}s")
            print(f"  Fine-tuning time:       {ft_time:.1f}s")
            print(f"  Total training time:    {total_time:.1f}s")

        # Save results and model
        self.save_results()
        self.save_global_model()

        return final_metrics

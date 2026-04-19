# FedBABU server: base-only aggregation + post-federation head fine-tuning.
#
# The FedBABU server has two phases:
#   Phase A (FL rounds): identical to FedRep server: base-only broadcast,
#     base-only collection, base-only aggregation.  Heads are frozen
#     (random initialisation maintained throughout).
#   Phase B (post-federation monitoring): calls client.fine_tune() for
#     each of the 13 TRAINING clients so their locally-fine-tuned
#     heads can be evaluated as a monitoring signal.  This is NOT the
#     LOSO metric.
#
# LOSO metric: evaluate_test_subject_with_calibration() is called after
# Phase B.  It dispatches to the FedRep/FedBABU branch in
# server_base._calibrate_head(): fresh random head + K-epoch head-only
# training on the held-out test subject's 20% calibration split, then
# evaluation on the 80% split.  This is the scientifically defensible
# held-out-client protocol (Collins et al. 2021 §5.3; Chen et al. 2020
# FedHealth) and is identical to the calibration step used for FedRep,
# so the four methods (FedAvg, FedRep, FedBABU, PeFLL) are directly
# comparable on the same 80% evaluation split per subject.
#
# Adapts from:
#   - PFLIB serverbabu.py   : the two-phase structure (FL then fine-tune)
#   - server_base.py        : our WESADServer base class + the shared
#                              evaluate_test_subject_with_calibration()
#                              utility
#
# SCIENTIFIC CONTEXT (Oh et al. 2022 ICLR)
#
# FedBABU = Federated Averaging with Body Aggregation and Body Update.
# The ablation answers: does training the head during FL rounds (FedRep)
# help or hurt compared to freezing it (FedBABU)?
#
# Expected ablation results:
#   - Plot macro-F1 vs fine_tuning_epochs in {1, 2, 5, 10, 20}
#   - Compare FedBABU curve against FedRep's best result
#   - If FedBABU > FedRep: the LSTM head overfits to the global
#     distribution during FL (Oh et al.'s hypothesis holds for biosignals)
#   - If FedRep > FedBABU: alternating training helps the head adapt
#     to evolving features (domain-specific evidence)
#
# Protocol under the held-out-with-calibration paradigm:
#   - Phase A pre-FT monitoring: 13 training clients with frozen random
#     heads (weak, as expected)
#   - Phase B post-FT monitoring: 13 training clients with their own
#     locally-fine-tuned heads (the within-sample personalisation baseline)
#   - LOSO held-out metric: the test subject's 80% eval split, after
#     fresh head + K-epoch calibration on the 20% split (the fair,
#     leakage-free measurement used in all methods)
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
      Phase B: each TRAINING client fine-tunes their head locally
               (monitoring only - the LOSO metric comes from the
                held-out calibration utility, below).

    Usage
    -----
    >>> config = FLConfig(algorithm="FedBABU", fold_id=2, fine_tuning_epochs=10)
    >>> fold_config = FLFoldConfig(test_subject=2)
    >>> server = FedBABUServer(config, fold_config)
    >>> server.set_clients(FedBABUClient)
    >>> server.train()
    """

    def train(self):
        """FedBABU two-phase training + held-out calibration evaluation.

        Phase A - Federated base training (global_rounds rounds):
          Same loop as FedRep: eval -> send_base -> train -> receive -> aggregate.
          During this phase, every client's head is frozen (random init).
          Only the CNN base is trained and aggregated.

        Phase B - Local head fine-tuning (fine_tuning_epochs per client):
          After all FL rounds complete, each of the 13 TRAINING clients
          unfreezes their head and fine-tunes it on their local data
          with the base frozen.  This is a monitoring metric - it shows
          how a FedBABU-trained base + locally-tuned head performs on
          the training subjects.  The held-out test subject is NOT in
          self.clients, so this phase does NOT produce the LOSO metric.

        LOSO evaluation:
          After Phase B, call evaluate_test_subject_with_calibration()
          from the base class.  It dispatches to the FedRep/FedBABU
          branch: fresh random head + K-epoch head-only training on the
          test subject's 20% calibration split, evaluation on the 80%
          split.  THIS IS THE LOSO FOLD METRIC.
        """
        print(
            f"\n  ══ Training: {self.config.algorithm} "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds:        {self.config.global_rounds}")
        print(f"  Local epochs (base):  {self.config.local_epochs}")
        print(
            f"  Fine-tuning epochs:   {self.config.fine_tuning_epochs} "
            f"(head, post-federation, training-client monitoring only)"
        )
        print(
            f"  Calibration epochs:   {self.config.calibration_epochs} "
            f"(held-out subject LOSO evaluation)"
        )
        print(
            f"  Clients:              {len(self.clients)} (training only; "
            f"test subject S{self.fold_config.test_subject} excluded)"
        )
        print("  Head during FL:       FROZEN (random initialisation)")

        budget = []

        # PHASE A: Federated base training (heads frozen)
        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # Evaluate before send
            # Phase A monitoring metrics (trained base + frozen random
            # head) measure base-only convergence on the 13 training
            # clients.  These are NOT directly comparable to FedRep's
            # per-round metrics (trained base + trained head).  The fair
            # cross-method comparison is the LOSO fold metric (below).
            if round_num % self.config.eval_gap == 0:
                print(
                    f"\n  ─── Round {round_num} "
                    f"(Phase A: base training, head frozen) ───"
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

        # Post-Phase A monitoring (13 training clients, frozen heads).
        # Expected to be weak (LSTM head is still random init), but
        # documents the base-only convergence point.
        print(
            "\n  ─── Post-Phase A monitoring " "(13 training clients, frozen heads) ───"
        )
        pre_ft_metrics = self.evaluate(round_num=self.config.global_rounds)
        print(
            f"  Pre-fine-tuning mean training-client F1: "
            f"{pre_ft_metrics.mean_macro_f1:.4f} "
            f"± {pre_ft_metrics.std_macro_f1:.4f}"
        )

        # PHASE B: Local head fine-tuning (base frozen)
        # Each of the 13 training clients unfreezes their head and
        # fine-tunes it locally with the base frozen.  Monitoring only.
        print(
            f"\n  ─── Phase B: Fine-tuning heads "
            f"({self.config.fine_tuning_epochs} epochs per training client) ───"
        )

        ft_start = time.time()
        for client in self.clients:
            client.fine_tune(which_module=["head"])
        ft_time = time.time() - ft_start

        print(f"  Fine-tuning completed in {ft_time:.1f}s")

        # Post-Phase B monitoring (13 training clients, locally-tuned heads).
        # This is the within-sample personalisation baseline - shows
        # what FedBABU achieves on subjects that participated in
        # federation.
        print(
            "\n  ─── Post-Phase B monitoring "
            "(13 training clients, fine-tuned heads) ───"
        )
        post_ft_metrics = self.evaluate(round_num=self.config.global_rounds + 1)
        print(
            f"  Post-fine-tuning mean training-client F1: "
            f"{post_ft_metrics.mean_macro_f1:.4f} "
            f"± {post_ft_metrics.std_macro_f1:.4f}"
        )
        print(
            f"  Training-client F1 gain from Phase B:     "
            f"{post_ft_metrics.mean_macro_f1 - pre_ft_metrics.mean_macro_f1:+.4f}"
        )

        # LOSO FOLD METRIC - held-out test subject evaluation with calibration.
        # The utility (server_base.evaluate_test_subject_with_calibration)
        # dispatches to the FedRep/FedBABU branch:
        #   1. Builds the 20/80 per-condition-block temporal split of
        #      the held-out subject's non-overlapped windows.
        #   2. Deep-copies self.global_model (preserves it for
        #      save_global_model).
        #   3. Re-initialises the head with fresh random weights.
        #   4. Trains the head for calibration_epochs epochs on the
        #      20% calibration split, with base frozen and base BN
        #      statistics frozen (eval mode) to prevent leakage via
        #      running statistics.
        #   5. Evaluates on the 80% evaluation split.
        # This is the fair, leakage-free LOSO metric used for all
        # methods (FedAvg, FedRep, FedBABU, PeFLL) on the same 80%
        # evaluation partition per subject.
        test_subject_metrics = self.evaluate_test_subject_with_calibration()
        self.results.set_test_subject_metrics(test_subject_metrics)

        # Summary
        print(
            f"\n  ═══ FedBABU Final Results "
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
            f"    pre-FT  mean macro-F1:  {pre_ft_metrics.mean_macro_f1:.4f} "
            f"± {pre_ft_metrics.std_macro_f1:.4f}"
        )
        print(
            f"    post-FT mean macro-F1:  {post_ft_metrics.mean_macro_f1:.4f} "
            f"± {post_ft_metrics.std_macro_f1:.4f}"
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

        return post_ft_metrics

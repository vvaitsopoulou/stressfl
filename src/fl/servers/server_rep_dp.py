# FedRep-DP server with Differential Privacy accounting.
#
# Extends FedRepServer with:
#   1. Privacy budget tracking across all clients per round
#   2. Epsilon sweep (ε ∈ {1, 5, 10, ∞})
#   3. Selective noise experiment : full-base DP vs no-DP
#   4. DP metadata storage in HDF5 results
#
# Adapts from:
#   - server_rep.py             : the FedRep server training loop
#   - PFLIB serverrep.py        : PFLlib reference
#
# PRIVACY ACCOUNTING AT THE SERVER
#
# Each FedRepDPClient has a persistent Opacus PrivacyEngine whose
# accountant accumulates steps across all rounds.  The server queries
# each client's current_epsilon property to get the true cumulative ε.
#
# The server logs:
#   - max_epsilon: the largest ε across all clients (worst-case privacy)
#   - mean_epsilon: average ε across clients (typical privacy)
#   - per-client ε values
#
# The worst-case (max) ε is the correct metric for formal DP claims
# because the guarantee must hold for all participants.  With
# heterogeneous dataset sizes (WESAD subjects have different window
# counts), clients with fewer windows consume ε faster because the
# sample rate (batch_size / N_windows) is larger, leading to more
# noise per step but also more privacy budget consumed per step.
#
# McMahan et al. 2018 (DP-FedAvg): the moments accountant (which
# Opacus implements via PRV) provides tighter ε bounds than basic
# composition, especially over many rounds.
#
# EPSILON SWEEP
#
# The primary experimental contribution of the DLG module is a 4-point
# privacy-utility curve:
#
#   ε = 1   : strong privacy, we expect significant utility drop
#   ε = 5   : moderate privacy, the primary comparison target
#              (Benouis et al. 2024 achieved ~90% stress at ε=15)
#   ε = 10  : mild privacy, weexpect small utility drop
#   ε = ∞   : no DP (standard FedRep result, baseline reference)
#
# SELECTIVE NOISE EXPERIMENT
#
# At ε = 5, compare:
#   1. Full-base DP: noise on all CNN base parameters (including GN,
#      since Opacus replaces BN → GN)
#   2. No DP: standard FedRep (ε = ∞)

# EVALUATION ORDERING (same as FedRepServer)
#
# eval → send_base → train (head then base+DP) → receive_base → aggregate
#
# Privacy metrics are collected after client training not
# before, so that the ε values reflect the current round's training.

from __future__ import annotations

import time

from ...data.wesad_dl_dataset import ALL_SUBJECT_IDS
from ...utils.fl_config import FLConfig, make_fedrep_dp_config
from ...utils.wesad_fl_dataset import FLFoldConfig
from ..clients.client_base import WESADClient
from ..clients.client_rep_dp import FedRepDPClient
from .server_base import WESADServer


class FedRepDPServer(WESADServer):
    """FedRep server with Differential Privacy accounting.

    Inherits the FedRep evaluation ordering (eval before send) and
    base-only aggregation from WESADServer.  Adds per-round privacy
    budget tracking and DP metadata logging.

    Usage
    -----
    >>> config = make_fedrep_dp_config(fold_id=2, dp_epsilon=5.0)
    >>> fold_config = FLFoldConfig(test_subject=2)
    >>> server = FedRepDPServer(config, fold_config)
    >>> server.set_clients(FedRepDPClient)
    >>> final_metrics = server.train()
    >>> print(server.privacy_history)
    """

    def __init__(
        self,
        config: FLConfig,
        fold_config: FLFoldConfig,
        device=None,
    ):
        super().__init__(config, fold_config, device)

        # Privacy tracking: one entry per evaluation point
        self.privacy_history: list[dict] = []

        # Store the global δ (set after set_clients()).
        self._global_delta: float | None = None

    def set_clients(
        self,
        client_class: type[WESADClient] = FedRepDPClient,
        data_managers=None,
    ):
        """Instantiate FedRepDP clients and compute global δ.

        After instantiation, determines the global δ from the smallest
        client's training set: δ = 1/min(N_windows).

        Note!!! this gives the largest δ (weakest privacy)
        because the smallest N produces the largest 1/N.  For the most
        conservative (smallest) δ, one would use 1/max(N_windows).
        This is a deliberate design choice TODO remember to check again
        """
        super().set_clients(client_class, data_managers)

        dp_clients = [c for c in self.clients if isinstance(c, FedRepDPClient)]
        if dp_clients:
            min_samples = min(c.data_manager.train_samples for c in dp_clients)
            self._global_delta = 1.0 / min_samples

            # Log DP configuration.
            # σ is available immediately because it was pre-computed in
            # FedRepDPClient.__init__
            sample = dp_clients[0]
            print("\n  DP Configuration:")
            print(f"    Target ε:       {sample.target_epsilon}")
            print(f"    Global δ:       {self._global_delta:.6f}")
            print(f"    Max grad norm:  {sample._max_grad_norm}")
            print(f"    DP enabled:     {sample.dp_enabled}")
            if sample.noise_multiplier is not None:
                print(f"    Noise σ:        {sample.noise_multiplier:.4f}")
            else:
                print("    Noise σ:        N/A (DP disabled)")

    def _collect_privacy_metrics(self, round_num: int) -> dict:
        """Query all DP clients for their current ε consumption.

        Each client's current_epsilon property queries the persistent
        Opacus engine's accountant, so the returned values reflect the
        true cumulative ε after all training steps completed so far.

        Returns
        dict with:
            round : int
            max_epsilon : float  (worst-case across clients)
            mean_epsilon : float (average across clients)
            min_epsilon : float  (best-case across clients)
            per_client : dict[int, float]  (subject_id → ε)
            noise_multiplier : float or None
            delta : float
        """
        per_client = {}
        epsilons = []

        for client in self.clients:
            if isinstance(client, FedRepDPClient) and client.dp_enabled:
                privacy = client.get_privacy_spent()
                per_client[client.id] = privacy["epsilon"]
                epsilons.append(privacy["epsilon"])

        if not epsilons:
            return {
                "round": round_num,
                "max_epsilon": 0.0,
                "mean_epsilon": 0.0,
                "min_epsilon": 0.0,
                "per_client": {},
                "noise_multiplier": None,
                "delta": self._global_delta or 0.0,
            }

        # Get noise multiplier from first DP client (same σ for all
        # clients with the same target ε, though sample_rate differs).
        noise_mult = None
        for client in self.clients:
            if isinstance(client, FedRepDPClient) and client.dp_enabled:
                noise_mult = client.noise_multiplier
                break

        return {
            "round": round_num,
            "max_epsilon": max(epsilons),
            "mean_epsilon": sum(epsilons) / len(epsilons),
            "min_epsilon": min(epsilons),
            "per_client": per_client,
            "noise_multiplier": noise_mult,
            "delta": self._global_delta or 0.0,
        }

    def train(self):
        """FedRep-DP training loop with privacy accounting.

        Loop structure (FedRep eval ordering):
          For each round:
            1. EVALUATE (before send → captures personalised models)
            2. Select clients
            3. Send global BASE to all clients
            4. Each client trains: head (no DP), then base (with DP)
            5. Collect privacy metrics (AFTER training, not before)
            6. Collect base models from all clients
            7. Optionally run DLG attack
            8. Aggregate base models

        Privacy metrics are collected at step 5 (after training) so
        that the ε values reflect the current round's DP steps.
        """
        dp_label = (
            f"ε={self.config.dp_epsilon}"
            if self.config.dp_epsilon != float("inf")
            else "no DP"
        )
        print(
            f"\n  ══ Training: {self.config.algorithm} [{dp_label}] "
            f"(fold S{self.fold_config.test_subject}) ══"
        )
        print(f"  Global rounds:    {self.config.global_rounds}")
        print(f"  Local epochs:     {self.config.local_epochs} (base, DP-protected)")
        print(f"  plocal_epochs:    {self.config.plocal_epochs} (head, no DP)")
        print(f"  Clients:          {len(self.clients)}")
        print("  Aggregation:      BASE ONLY (CNN branches)")
        print(f"  Privacy:          {dp_label}")

        budget = []

        for round_num in range(self.config.global_rounds):
            round_start = time.time()

            # 1. EVALUATE (before send → personalised models)
            if round_num % self.config.eval_gap == 0:
                print(f"\n  ─── Round {round_num} ───")
                print("  Evaluating personalised models (before base broadcast)")
                self.evaluate(round_num=round_num)

            # 2. Select clients
            self.select_clients()

            # 3. Broadcast global BASE to all clients
            self.send_models()

            # 4. Local training (head then base+DP per client)
            for client in self.selected_clients:
                client.train()

            # 5. Collect privacy metrics after training
            if round_num % self.config.eval_gap == 0:
                privacy_metrics = self._collect_privacy_metrics(round_num)
                self.privacy_history.append(privacy_metrics)

                if privacy_metrics["max_epsilon"] > 0:
                    print(
                        f"  Privacy (after round {round_num} training): "
                        f"ε_max={privacy_metrics['max_epsilon']:.4f}, "
                        f"ε_mean={privacy_metrics['mean_epsilon']:.4f}, "
                        f"σ={privacy_metrics['noise_multiplier']}"
                    )

            # 6. Collect BASE models from all clients
            self.receive_models()

            # 7. DLG attack (optional if it is ready by the end)
            if self.config.dlg_eval and round_num % self.config.dlg_gap == 0:
                self.call_dlg(round_num)

            # 8. Aggregate base models
            self.aggregate_parameters()

            budget.append(time.time() - round_start)

        # Final evaluation
        print(
            f"\n  ─── Final evaluation (after round {self.config.global_rounds - 1}) ───"
        )
        print("  Evaluating final personalised models")
        final_metrics = self.evaluate(round_num=self.config.global_rounds)

        # Final privacy metrics (after all training is complete)
        final_privacy = self._collect_privacy_metrics(self.config.global_rounds)
        self.privacy_history.append(final_privacy)

        # Summary
        if final_metrics.test_subject_metrics is not None:
            test_f1 = final_metrics.test_subject_metrics.macro_f1
            test_acc = final_metrics.test_subject_metrics.accuracy
            test_sr = final_metrics.test_subject_metrics.stress_recall
            print(
                f"\n  ═══ FedRep-DP Final Results (fold S{self.fold_config.test_subject}) ═══"
            )
            print(f"  Target ε:               {self.config.dp_epsilon}")
            print(f"  Achieved ε (max):       {final_privacy['max_epsilon']:.4f}")
            print(f"  Achieved ε (mean):      {final_privacy['mean_epsilon']:.4f}")
            print(f"  δ:                      {final_privacy['delta']:.6f}")
            print(f"  Noise multiplier σ:     {final_privacy['noise_multiplier']}")
            print(f"  Test subject F1:        {test_f1:.4f}")
            print(f"  Test subject Accuracy:  {test_acc:.4f}")
            print(f"  Test subject Stress R:  {test_sr:.4f}")
            print(
                f"  Mean client F1:         {final_metrics.mean_macro_f1:.4f} "
                f"± {final_metrics.std_macro_f1:.4f}"
            )

        if budget:
            avg_time = sum(budget) / len(budget)
            total_time = sum(budget)
            print(f"  Average round time:     {avg_time:.2f}s")
            print(f"  Total training time:    {total_time:.1f}s")

        # Save results with DP metadata
        self.save_results()
        self.save_global_model()

        return final_metrics

    def save_results(self, result_dir: str = "results/"):
        """Save results with additional DP-specific metadata.

        Extends WESADServer.save_results() to include:
          - per-round epsilon values (privacy_history)
          - noise_multiplier σ
          - final achieved epsilon
          - delta
          - target epsilon

        These are stored as additional datasets in the HDF5 file
        under the algorithm group.
        """
        # Add DP hyperparameters to the result accumulator
        self.results.hyperparams.update(
            {
                "dp_epsilon_target": self.config.dp_epsilon,
                "dp_delta": self._global_delta or self.config.dp_delta,
                "dp_max_grad_norm": self.config.dp_max_grad_norm,
            }
        )

        # Add final privacy metrics
        if self.privacy_history:
            final = self.privacy_history[-1]
            self.results.hyperparams.update(
                {
                    "dp_epsilon_achieved_max": final["max_epsilon"],
                    "dp_epsilon_achieved_mean": final["mean_epsilon"],
                    "dp_noise_multiplier": final["noise_multiplier"],
                }
            )

        # Call parent save (writes HDF5 with standard metrics)
        result_path = super().save_results(result_dir)

        # Append DP-specific datasets to the HDF5 file
        if result_path and self.privacy_history:
            try:
                import h5py
                import numpy as np

                with h5py.File(result_path, "a") as f:
                    group_name = (
                        f"{self.config.algorithm}_{self.fold_config.test_subject}"
                    )
                    if group_name in f:
                        grp = f[group_name]

                        # Per-round epsilon tracking
                        rounds = [p["round"] for p in self.privacy_history]
                        max_eps = [p["max_epsilon"] for p in self.privacy_history]
                        mean_eps = [p["mean_epsilon"] for p in self.privacy_history]

                        # Create or overwrite DP datasets
                        for name, data in [
                            ("dp_rounds", rounds),
                            ("dp_max_epsilon", max_eps),
                            ("dp_mean_epsilon", mean_eps),
                        ]:
                            if name in grp:
                                del grp[name]
                            grp.create_dataset(name, data=np.array(data))

                        # Store noise multiplier as attribute
                        if self.privacy_history[-1]["noise_multiplier"] is not None:
                            grp.attrs["dp_noise_multiplier"] = self.privacy_history[-1][
                                "noise_multiplier"
                            ]

            except Exception as e:
                print(f"  Warning: could not append DP metadata to HDF5: {e}")

        return result_path


# EPSILON sweep
#
# Run FedRep with DP at ε ∈ {1, 5, 10, ∞}.
# For each ε, run all 15 LOSO folds.
# Produces the 4-point privacy-utility curve


def run_epsilon_sweep(
    fold_id: int,
    epsilons: list[float] | None = None,
    device: str = "cuda",
    **config_overrides,
) -> dict[float, dict]:
    """Run FedRep-DP at multiple ε values for one LOSO fold.

    For each ε value, trains FedRep-DP and records the
    final macro-F1, stress recall, accuracy, and achieved ε.

    Parameters:
    fold_id : int
        The WESAD subject ID for the LOSO test fold.
    epsilons : list[float] | None
        Privacy budgets to sweep.  Default: [1.0, 5.0, 10.0, inf].
        ε = inf means no DP (standard FedRep, the reference baseline).
    device : str
        "cuda" or "cpu".
    **config_overrides
        Additional FLConfig overrides (e.g., global_rounds, batch_size).

    Returns:
    dict[float, dict]
        Mapping from ε → {
            "macro_f1": float,
            "accuracy": float,
            "stress_recall": float,
            "epsilon_achieved": float,
            "noise_multiplier": float or None,
            "delta": float,
        }

    Example
    -------
    >>> results = run_epsilon_sweep(fold_id=2)
    >>> for eps, metrics in results.items():
    ...     print(f"ε={eps}: F1={metrics['macro_f1']:.4f}")
    """
    import torch

    if epsilons is None:
        epsilons = [1.0, 5.0, 10.0, float("inf")]

    device_obj = torch.device(
        device if device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    sweep_results = {}

    for eps in epsilons:
        dp_label = f"ε={eps}" if eps != float("inf") else "no DP"
        print(f"\n{'='*60}")
        print(f"  EPSILON SWEEP: {dp_label} (fold S{fold_id})")
        print(f"{'='*60}")

        # Create config for this epsilon value
        config = make_fedrep_dp_config(
            fold_id=fold_id,
            dp_epsilon=eps,
            device=device,
            **config_overrides,
        )

        # Build fold config
        fold_config = FLFoldConfig(test_subject=fold_id)

        # Instantiate server
        server = FedRepDPServer(config, fold_config, device_obj)
        server.set_clients(FedRepDPClient)

        # Train
        final_metrics = server.train()

        # Collect results
        test_metrics = final_metrics.test_subject_metrics
        final_privacy = server.privacy_history[-1] if server.privacy_history else {}

        sweep_results[eps] = {
            "macro_f1": test_metrics.macro_f1 if test_metrics else 0.0,
            "accuracy": test_metrics.accuracy if test_metrics else 0.0,
            "stress_recall": test_metrics.stress_recall if test_metrics else 0.0,
            "epsilon_achieved": final_privacy.get("max_epsilon", 0.0),
            "noise_multiplier": final_privacy.get("noise_multiplier", None),
            "delta": final_privacy.get("delta", 0.0),
            "mean_client_f1": final_metrics.mean_macro_f1,
            "std_client_f1": final_metrics.std_macro_f1,
        }

    # Print summary table
    print(f"  EPSILON SWEEP SUMMARY (fold S{fold_id})")
    print("\n")
    print(
        f"  {'ε target':>10} {'ε achieved':>12} {'σ':>8} "
        f"{'F1':>8} {'Acc':>8} {'Stress R':>10}"
    )
    print("\n")

    for eps in epsilons:
        r = sweep_results[eps]
        eps_str = "∞" if eps == float("inf") else f"{eps:.1f}"
        ach_str = f"{r['epsilon_achieved']:.4f}" if r["epsilon_achieved"] > 0 else "N/A"
        sigma_str = f"{r['noise_multiplier']:.4f}" if r["noise_multiplier"] else "N/A"
        print(
            f"  {eps_str:>10} {ach_str:>12} {sigma_str:>8} "
            f"{r['macro_f1']:>8.4f} {r['accuracy']:>8.4f} "
            f"{r['stress_recall']:>10.4f}"
        )

    return sweep_results


def run_full_epsilon_sweep(
    epsilons: list[float] | None = None,
    device: str = "cuda",
    **config_overrides,
) -> dict[float, list[dict]]:
    """Run epsilon sweep across all 15 LOSO folds

    For each ε, runs all 15 folds and aggregates statistics.

    Parameters:
    epsilons : list[float] | None
        Default: [1.0, 5.0, 10.0, inf].
    device : str
        "cuda" or "cpu".
    **config_overrides
        Additional FLConfig overrides.

    Returns:
    dict[float, list[dict]]
        ε → list of 15 per-fold result dicts.
    """
    if epsilons is None:
        epsilons = [1.0, 5.0, 10.0, float("inf")]

    all_results: dict[float, list[dict]] = {eps: [] for eps in epsilons}

    for fold_id in ALL_SUBJECT_IDS:
        fold_results = run_epsilon_sweep(
            fold_id=fold_id,
            epsilons=epsilons,
            device=device,
            **config_overrides,
        )
        for eps in epsilons:
            all_results[eps].append(fold_results[eps])

    # Print cross-fold summary
    import numpy as np

    print("  FULL EPSILON SWEEP: ALL 15 LOSO FOLDS")
    print("\n")
    print(
        f"  {'ε':>6} {'Mean F1':>10} {'Std F1':>10} "
        f"{'Mean Acc':>10} {'Mean SR':>10}"
    )
    print("\n")

    for eps in epsilons:
        f1s = [r["macro_f1"] for r in all_results[eps]]
        accs = [r["accuracy"] for r in all_results[eps]]
        srs = [r["stress_recall"] for r in all_results[eps]]
        eps_str = "∞" if eps == float("inf") else f"{eps:.1f}"
        print(
            f"  {eps_str:>6} {np.mean(f1s):>10.4f} {np.std(f1s):>10.4f} "
            f"{np.mean(accs):>10.4f} {np.mean(srs):>10.4f}"
        )

    return all_results


# SELECTIVE NOISE EXPERIMENT
#
# At ε = 5, compare:
#   1. Full-base DP (noise on all base params including GN)
#   2. No DP (ε = ∞, standard FedRep)
#
# The two-way comparison demonstrates the privacy-utility
# tradeoff.


def run_selective_noise_experiment(
    fold_id: int,
    epsilon: float = 5.0,
    device: str = "cuda",
    **config_overrides,
) -> dict[str, dict]:
    """Run the selective noise experiment at a fixed ε:

    Compares full-base DP vs no-DP at the given epsilon.

    Parameters:
    fold_id : int
        LOSO fold.
    epsilon : float
        Privacy budget for the DP variant.  Default: 5.0.
    device : str
        "cuda" or "cpu".

    Returns:
    dict[str, dict]
        "full_base_dp" → metrics dict
        "no_dp" → metrics dict
    """
    results = run_epsilon_sweep(
        fold_id=fold_id,
        epsilons=[epsilon, float("inf")],
        device=device,
        **config_overrides,
    )

    return {
        "full_base_dp": results[epsilon],
        "no_dp": results[float("inf")],
    }

# Hyperparameter dataclass for all FL experiments.
#
# Modifies PFLlib's main.py into a
# cleaner dataclass tailored to the WESAD FL use case.  Stores one
# JSON config file per experiment so results are always reproducible
# by re-running with the same config file.
#
# Adapts from:
#   - PFLIB main.py   : complete parameter list across all algorithms we will be using
#
# Key modifications from PFLlib:
#   - num_clients is fixed at 15 (one per WESAD subject 14 train + 1 test-as-client)
#   - Optimizer is AdamW (not SGD) for consistency of comparison with wesad_cnnlstm_train.py (centralised baseline uses
#      AdamW ), a selection which was based on Moser et al. 2024
#   - No client subsampling (join_ratio=1.0, all clients participate
#     every round): with only 15 subjects, subsampling adds noise
#     without benefit (this is is a scalability technique for large federations, for 15 lcients full participation does not add significant
#     overhead
#   - No slow client / drop-out simulation,  these exist to model real-world deployment where devices go offline or have inconsistent compute
#     therefore not relevant for this dataset and this project scope

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os


@dataclass
class FLConfig:
    """Complete hyperparameter configuration for one FL experiment.

    Create one instance per experiment.  Save to JSON for reproducibility.

    Attributes:
    # Experiment identity
    algorithm : str
        Algorithm name: "FedAvg", "FedProx", "FedRep", "FedBABU",
        "Ditto", "PeFLL", "FedAvgFT", "Local", "Centralised",
        "FedRepDP"
    fold_id : int
        Test subject ID for the current LOSO fold.
    experiment_name : str
        Human-readable experiment identifier for logging.
    seed : int
        Random seed for reproducibility.

    # Data configuration
    device_type : str
        "wrist" (primary) or "chest" (upper bound).
    window_sec : float
        Window length in seconds.
    label_mode : str
        "binary" or "3class".
    windowed_dir : str
        Root directory for windowed data.
    signals_to_load : list[str] | None
        Specific signals to load.  None = all available.

    # Federation parameters
    num_clients : int
        Total number of FL clients.  Fixed at 15 for WESAD, this includes the test subject.
    global_rounds : int
        Number of federation rounds.
    local_epochs : int
        Number of local training epochs per round.
    batch_size : int
        Batch size for local training.
    eval_gap : int
        Evaluate every eval_gap rounds.

    # Optimizer parameters
    # AdamW is used for all methods
    learning_rate : float
        Learning rate for local training.
    weight_decay : float
        Weight decay for AdamW.
    lr_scheduler : str
        Learning rate scheduler type.
    lr_patience : int
        Patience for ReduceLROnPlateau.
    lr_factor : float
        Factor for ReduceLROnPlateau.

    # Algorithm-specific parameters
    # FedProx
    mu : float
        Proximal term weight for FedProx.  Also used as regularisation
        strength for Ditto's PerturbedGradientDescent.
        (from PFLIB main.py --mu)

    # FedRep / FedBABU
    plocal_epochs : int
        Number of personal (head) update epochs per round for FedRep.
        (from PFLIB main.py --plocal_epochs)

    # FedBABU
    fine_tuning_epochs : int
        Number of fine-tuning epochs after federation for FedBABU.
        (from PFLIB main.py --fine_tuning_epochs)

    # Held-out calibration (all personalised methods)
    calibration_epochs : int
        Number of epochs to train the test subject's local head on
        their calibration set after federation completes.  Used by
        FedRep, FedBABU, and PeFLL under the held-out-with-calibration
        protocol.  Default 10.
    calibration_fraction : float
        Fraction of the test subject's non-overlapped windows used for
        calibration; the remainder is the evaluation set.  Default 0.20.
    calibration_lr : float
        Learning rate for the post-federation head calibration step.
        Default 1e-4 (smaller than the federation-time learning_rate to
        avoid overshooting on the small calibration set).  Used by
        FedRep / FedBABU / FedRepDP only.

    # Ditto
    ditto_lambda : float
        Personalisation regularisation weight for Ditto.
        λ = 0 corresponds local-only, λ → ∞ FedAvg.
        (from PFLIB main.py --lamda)

    # PeFLL
    embedding_dim : int
        Dimensionality of client embedding vector.
    hypernet_hidden : int
        Hidden layer size of the hypernetwork MLP.
    pefll_lr : float
        Learning rate for hypernetwork + embedding network.

    # DP-FL
    dp_epsilon : float
        Target privacy budget (ε).  float('inf') = no DP.
    dp_delta : float | None
        Privacy parameter δ.  If None, set to 1/N_min_windows.
    dp_max_grad_norm : float
        Per-sample gradient clipping norm for Opacus.

    # DLG
    dlg_eval : bool
        Whether to run DLG attack during training.
    dlg_gap : int
        Run DLG every dlg_gap rounds.
    dlg_batch_num_per_client : int
        Number of batches per client for DLG.

    #  Hardware
    device : str
        "cuda" or "cpu".
    num_workers : int
        DataLoader workers.

    # Augmentation
    noise_std : float
        Gaussian noise std for training augmentation.
    scale_range_low : float
        Amplitude scaling lower bound.
    scale_range_high : float
        Amplitude scaling upper bound.
    """

    # Experiment identity
    algorithm: str = "FedAvg"
    fold_id: int = 2  # default to first subject
    experiment_name: str = ""
    seed: int = 42

    # Data
    device_type: str = "wrist"
    window_sec: float = 16.0
    label_mode: str = "binary"
    windowed_dir: str = "data/processed/wesad/dl_windowed/"
    signals_to_load: list[str] | None = None

    # Federation parameters

    num_clients: int = 15
    global_rounds: int = 100
    # Collins et al. 2021 (FedRep): T=100 for CIFAR, T=200 for FEMNIST.
    # McMahan et al. 2017 (FedAvg): convergence within 100 rounds on
    # most tasks.Sweep: {50, 100, 200}

    local_epochs: int = 5
    # Collins et al. 2021: "FedRep executes ten local epochs of SGD
    # with momentum to train the local head, followed by one or five
    # epochs for the representation."  Their base-update uses 1 or 5.
    # McMahan et al. 2017: E=5 is the default across most experiments
    # ands Li et al. 2020 (FedProx) also use E=5 as baseline.
    # Sweep: {1, 3, 5, 10}

    batch_size: int = 64
    # matches wesad_cnnlstm_train.py standalone training config.
    # Ordóñez & Roggen 2016 had 100 for DeepConvLSTM, Prajod & André had
    # 128 for WESAD CNN, anf Smith & Le, 2018 found that
    # smaller batches provide implicit regularisation .

    eval_gap: int = 5
    # With 100 global rounds, this gives 20
    # evaluation points for convergence. PFLlib default is 1, but their eval is computationally
    # cheaper (accuracy only), and ours is not so this is a compromise

    # Optimizer (AdamW)

    learning_rate: float = 1e-3
    # Matches wesad_cnnlstm_train.py.
    # Ordóñez & Roggen 2016 use lr=1e-3 for DeepConvLSTM,
    # Prajod & André 2022 use lr=1e-3 for their WESAD CNN.

    weight_decay: float = 1e-4
    #  1e-4 is the standard default for adamw
    # (Loshchilov & Hutter, 2019).  also matches wesad_cnnlstm_train.py.

    lr_scheduler: str = "plateau"
    # matching wesad_cnnlstm_train.py base don Moser et al. 2024: "terminate once
    # validation loss stops improving" which matches plateau scheduler

    lr_patience: int = 5
    # Matches wesad_cnnlstm_train.py: patience=5

    lr_factor: float = 0.5
    # Halve the learning rate on plateau

    lr_gamma: float = 0.99  # for exponential scheduler (PFLlib default)

    # FedProx / Ditto
    mu: float = 0.01
    # per Li et al. 2020, then sweep μ ∈ {0.001, 0.01, 0.1, 1.0}.

    # FedRep
    plocal_epochs: int = 5
    # Collins et al. 2021: "ten local epochs of SGD to train the local
    # head" , reduce because we are using adamw
    # Sweep: {1, 5, 10}

    # FedBABU
    fine_tuning_epochs: int = 10
    # Oh et al. 2022 (FedBABU): Their experiments show most gains within 5-10 epochs of fine-tuning.
    # Sweep: {1, 2, 5, 10, 20}

    # Held-out calibration (all methods that use personalisation)
    calibration_epochs: int = 10
    # After federation completes, the test subject receives the final
    # global base, initialises a fresh random head (or, for PeFLL, a
    # hypernetwork-generated head), and trains the head on the
    # calibration set for `calibration_epochs` epochs with the base
    # frozen.  This is the same mechanism as FedBABU's post-federation
    # fine-tuning.
    #
    # Default 10 matches the FedBABU default and sits in the middle of
    # Oh et al. 2022's 5-10-epoch recommendation.  The FedBABU sweep on
    # fold S2 validates this choice empirically.
    # Sweep: {1, 2, 5, 10, 20}
    #
    # Scientific basis: Collins et al. 2021 Section 5.3 ("Generalization
    # to new clients"); Collins et al. 2022 (FedAvg + fine-tuning); the
    # FedHealth two-step paradigm (Chen et al. 2020).

    calibration_fraction: float = 0.20
    # Fraction of the test subject's non-overlapped windows used for
    # calibration.  The remaining 1 - calibration_fraction is the
    # evaluation set on which the LOSO fold metric is computed.  The
    # split is stratified per contiguous condition block to preserve
    # class proportions.  Matches the KD plan's 20/80 split.

    calibration_lr: float = 1e-4
    # Learning rate for the post-federation head calibration step.
    # Smaller than the federation-time learning_rate (1e-3) because the
    # calibration set is tiny (~28 windows) and a larger step would
    # overshoot.  Matches the Cross_Modal_KD plan's
    # "K gradient steps with lr=1e-4, CE loss only" for personalisation.
    # Used by server_base._calibrate_head() for FedRep / FedBABU /
    # FedRepDP.  Not used by FedAvg (no calibration step) or PeFLL
    # (head generated by the hypernetwork, no gradient updates).

    calibration_init: str = "cold_start"
    # Head initialisation strategy for the post-federation calibration
    # step (FedRep / FedBABU / FedRepDP only; FedAvg has no calibration
    # and PeFLL generates the head from the hypernet).
    #
    #   "cold_start" (default, Collins et al. 2021 original protocol):
    #       Re-initialise model.head from scratch via reset_parameters()
    #       before the K-epoch calibration loop.  This is the pure PFL
    #       setting where the held-out subject is treated as a fully
    #       new client with no prior over the head.
    #
    #   "warm_start" (modified protocol, this project):
    #       Initialise model.head from the weighted mean of all
    #       training-client heads at end of federation (weighted by
    #       client sample count, same scheme as base aggregation).  BN
    #       running statistics on the mean head are reset to neutral
    #       values (mean=0, var=1, num_batches_tracked=0) so the
    #       calibration loop re-estimates them on the test subject's
    #       own calibration data.  Rationale: with only ~28 calibration
    #       windows a freshly-randomised head cannot converge in K
    #       epochs; warm-starting from a trained head gives the
    #       calibration step a much better prior and dramatically
    #       reduces per-subject variance.  This is aligned with the
    #       FedHealth two-step paradigm (Chen et al. 2020) and the
    #       FedBABU-style deployment scenario (Oh et al. 2022) where
    #       new users receive both a base and a "starter" head.
    #
    # Requires save_global_model() to have persisted the training-client
    # mean head alongside the base (format: dict with keys
    # "global_model_state_dict" and "federated_mean_head_state_dict").
    # If the checkpoint only contains a raw state dict (legacy format),
    # warm_start falls back to cold_start with a warning.

    #  Ditto
    ditto_lambda: float = 0.01
    # Li et al. 2021 (Ditto): λ controls personalisation strength. Li et al. note optimal λ depends on
    # heterogeneity level.
    # Sweep: {0.001, 0.01, 0.1, 0.5, 1.0}

    #  PeFLL
    embedding_dim: int = 64
    # Scott et al. 2024 (PeFLL): use 64–128 dim embeddings depending on
    # task complexity. wesad is on the low end.
    # Sweep: {32, 64, 128}

    hypernet_hidden: int = 512
    # Scott et al. 2024: 512 is their default for CIFAR experiments.

    pefll_lr: float = 1e-3
    # Matches the client learning rate.  same LR as local training

    # DP-FL
    dp_epsilon: float = float("inf")  # no DP by default
    #  When DP is enabled,
    # sweep ε ∈ {1, 5, 10}
    # Benouis et al. 2024 (WESAD DP-FL) achieved ~90% stress recognition
    # at moderate ε and we might use thus as the comparison target.

    dp_delta: float | None = None
    # TODO: resolve to 1/N_min_windows in client_rep_dp.py __init__.
    # McMahan et al. 2018 (DP-FedAvg): δ should be ≪ 1/N where N is the
    # dataset size.

    dp_max_grad_norm: float = 1.0
    # for Opacus 1.0 is the standard
    # default (Abadi et al. 2016, McMahan et al. 2018).

    # DLG
    dlg_eval: bool = False
    # Disabled by default since DLG is expensive

    dlg_gap: int = 20
    # PFLlib default is 100 but with only 100 global rounds 20 gives
    # 5 DLG evaluation points.

    dlg_batch_num_per_client: int = 2
    # PFLlib default (PFLIB main.py --batch_num_per_client=2).

    # Hardware
    device: str = "cuda"
    num_workers: int = 8
    # Was 0 (single-threaded data loading).  With num_workers=0 and
    # per-window augmentation/normalisation in the Dataset, the main
    # Python thread becomes the bottleneck and starves the GPU
    # (observed: 2% GPU utilisation, 29 MB allocated on Vast.ai RTX 4080).
    # num_workers=4 + persistent_workers=True (see client_base.py
    # _make_loader) keeps the workers alive across rounds and allows
    # several windows to be loaded/augmented in parallel, feeding the
    # GPU continuously.  On CPU-only runs this also helps slightly by
    # overlapping I/O with compute.  Set lower if running on a cloud
    # instance with very few vCPUs.

    # Augmentation
    # These match wesad_cnnlstm_train.py and wesad_dl_dataset.py exactly.

    noise_std: float = 0.05

    scale_range_low: float = 0.9
    scale_range_high: float = 1.1

    # Model architecture
    # These match wesad_cnnlstm_model.py defaults.

    lstm_hidden: int = 32
    lstm_layers: int = 2

    cnn_dropout: float = 0.5

    fc_dropout: float = 0.5

    n_conv_layers: int = 1

    target_time: int | None = None
    # Temporal dimension after adaptive pooling (the sequence length the
    # LSTM reads).  If None, uses the per-device factory default:
    #   wrist = 8  (constrained by w_eda branch)
    #   chest = 5  (constrained by RESP branch)
    # Must be set explicitly for window-length ablations (e.g. 10s window
    # needs a lower target_time to avoid upsampling in adaptive pooling).
    # See wesad_cnnlstm_model.py for valid values.

    n_classes: int = 2

    def __post_init__(self):
        # Auto-generate experiment name if not provided
        if not self.experiment_name:
            self.experiment_name = (
                f"{self.algorithm}_{self.device_type}_fold{self.fold_id}"
            )

        # Set n_classes from label_mode
        if self.label_mode == "binary":
            self.n_classes = 2
        elif self.label_mode == "3class":
            self.n_classes = 3

        # Validate
        if self.device_type not in ("wrist", "chest"):
            raise ValueError(f"Invalid device_type: {self.device_type}")
        if self.label_mode not in ("binary", "3class"):
            raise ValueError(f"Invalid label_mode: {self.label_mode}")
        if self.fold_id not in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
            raise ValueError(
                f"Invalid fold_id: {self.fold_id} (must be a WESAD subject ID)"
            )

    @property
    def scale_range(self) -> tuple[float, float]:
        """Amplitude scaling range as a tuple."""
        return (self.scale_range_low, self.scale_range_high)

    @property
    def label_map(self) -> dict:
        """Get the label mapping dict based on label_mode."""
        # Import here to avoid circular imports
        from ..data.wesad_dl_dataset import LABEL_MAP_3CLASS, LABEL_MAP_BINARY

        return LABEL_MAP_BINARY if self.label_mode == "binary" else LABEL_MAP_3CLASS

    def save_json(self, path: str):
        """Save configuration to a JSON file for reproducibility.

        Store one JSON per experiment so results are
        always reproducible by re-running with the same config file.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        # handling None and special types
        d = asdict(self)
        # Handle float('inf') for JSON serialization
        for k, v in d.items():
            if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
                d[k] = str(v)

        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)

        print(f"  Config saved: {path}")

    @classmethod
    def load_json(cls, path: str) -> FLConfig:
        """Load configuration from a JSON file.

        Parameters:
        path : str
            Path to the JSON config file.

        Returns:
        FLConfig
        """
        with open(path) as f:
            d = json.load(f)

        # Handle serialized float('inf')
        for k, v in d.items():
            if v == "inf":
                d[k] = float("inf")
            elif v == "-inf":
                d[k] = float("-inf")

        # Remove any keys not in the dataclass (forward compatibility)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid_keys}

        return cls(**d)

    def print_summary(self):
        """Print a readable summary of the configuration."""
        print(f"\n  ══ FL Config: {self.experiment_name} ══")
        print(f"  Algorithm:      {self.algorithm}")
        print(f"  Fold (test):    S{self.fold_id}")
        print(f"  Device type:    {self.device_type}")
        print(f"  Label mode:     {self.label_mode} ({self.n_classes} classes)")
        print(f"  Global rounds:  {self.global_rounds}")
        print(f"  Local epochs:   {self.local_epochs}")
        print(f"  Batch size:     {self.batch_size}")
        print(f"  Learning rate:  {self.learning_rate}")
        print(f"  Weight decay:   {self.weight_decay}")
        print(f"  Eval gap:       {self.eval_gap}")

        # Algorithm-specific
        if self.algorithm in ("FedProx",):
            print(f"  μ (proximal):   {self.mu}")
        if self.algorithm in ("FedRep", "FedBABU"):
            print(f"  plocal_epochs:  {self.plocal_epochs}")
        if self.algorithm in ("FedBABU",):
            print(f"  FT epochs:      {self.fine_tuning_epochs}")
        if self.algorithm in ("Ditto",):
            print(f"  λ (Ditto):      {self.ditto_lambda}")
            print(f"  μ (proximal):   {self.mu}")
        if self.algorithm in ("PeFLL",):
            print(f"  Embedding dim:  {self.embedding_dim}")
            print(f"  HyperNet hidden:{self.hypernet_hidden}")
        if self.algorithm in ("FedRepDP",):
            print(f"  DP ε:           {self.dp_epsilon}")
            print(f"  DP δ:           {self.dp_delta}")
            print(f"  Max grad norm:  {self.dp_max_grad_norm}")
        # Calibration (FedRep/FedBABU/PeFLL only; FedAvg has no calibration step)
        if self.algorithm in ("FedRep", "FedBABU", "PeFLL", "FedRepDP"):
            print(f"  Calib epochs:   {self.calibration_epochs}")
            print(f"  Calib fraction: {self.calibration_fraction:.2f}")
        # calibration_lr only applies to gradient-based calibration
        # (FedRep / FedBABU / FedRepDP).  PeFLL generates the head via
        # the hypernetwork with no gradient steps, so it does not use it.
        if self.algorithm in ("FedRep", "FedBABU", "FedRepDP"):
            print(f"  Calib LR:       {self.calibration_lr}")
            print(f"  Calib init:     {self.calibration_init}")
        if self.dlg_eval:
            print(f"  DLG eval:       every {self.dlg_gap} rounds")


# Convenience constructors for common experiments


def make_fedavg_config(fold_id: int, **overrides) -> FLConfig:
    """Create a FedAvg configuration."""
    return FLConfig(algorithm="FedAvg", fold_id=fold_id, **overrides)


def make_fedavgft_config(
    fold_id: int, fine_tuning_epochs: int = 10, **overrides
) -> FLConfig:
    """Create a FedAvg + Fine-Tuning configuration"""
    return FLConfig(
        algorithm="FedAvgFT",
        fold_id=fold_id,
        fine_tuning_epochs=fine_tuning_epochs,
        **overrides,
    )


def make_local_config(fold_id: int, **overrides) -> FLConfig:
    """Create a Local-only (no federation) configuration."""
    return FLConfig(algorithm="Local", fold_id=fold_id, global_rounds=0, **overrides)


def make_centralised_config(fold_id: int, **overrides) -> FLConfig:
    """Create a Centralisedconfiguration."""
    return FLConfig(algorithm="Centralised", fold_id=fold_id, **overrides)


def make_fedprox_config(fold_id: int, mu: float = 0.01, **overrides) -> FLConfig:
    """Create a FedProx configuration."""
    return FLConfig(algorithm="FedProx", fold_id=fold_id, mu=mu, **overrides)


def make_fedrep_config(fold_id: int, plocal_epochs: int = 5, **overrides) -> FLConfig:
    """Create a FedRep configuration."""
    return FLConfig(
        algorithm="FedRep", fold_id=fold_id, plocal_epochs=plocal_epochs, **overrides
    )


def make_fedbabu_config(
    fold_id: int, fine_tuning_epochs: int = 10, **overrides
) -> FLConfig:
    """Create a FedBABU configuration."""
    return FLConfig(
        algorithm="FedBABU",
        fold_id=fold_id,
        fine_tuning_epochs=fine_tuning_epochs,
        **overrides,
    )


def make_ditto_config(
    fold_id: int, ditto_lambda: float = 0.01, mu: float = 0.01, **overrides
) -> FLConfig:
    """Create a Ditto configuration."""
    return FLConfig(
        algorithm="Ditto",
        fold_id=fold_id,
        ditto_lambda=ditto_lambda,
        mu=mu,
        **overrides,
    )


def make_pefll_config(
    fold_id: int,
    embedding_dim: int = 64,
    hypernet_hidden: int = 512,
    **overrides,
) -> FLConfig:
    """Create a PeFLL configuration."""
    return FLConfig(
        algorithm="PeFLL",
        fold_id=fold_id,
        embedding_dim=embedding_dim,
        hypernet_hidden=hypernet_hidden,
        **overrides,
    )


def make_fedrep_dp_config(
    fold_id: int,
    dp_epsilon: float = 5.0,
    dp_delta: float | None = None,
    **overrides,
) -> FLConfig:
    """Create a FedRep+DP configuration."""
    return FLConfig(
        algorithm="FedRepDP",
        fold_id=fold_id,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
        **overrides,
    )


def generate_all_configs_for_algorithm(algorithm: str, **kwargs) -> list[FLConfig]:
    """Generate configs for all 15 LOSO folds for one algorithm.

    Parameters:
    algorithm : str
        Algorithm name.
    **kwargs
        Algorithm-specific parameters forwarded to FLConfig.

    Returns:
    list[FLConfig]
        15 configs, one per fold.
    """
    from ..data.wesad_dl_dataset import ALL_SUBJECT_IDS

    return [
        FLConfig(algorithm=algorithm, fold_id=sid, **kwargs) for sid in ALL_SUBJECT_IDS
    ]

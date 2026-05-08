# Cross-Modal Knowledge Distillation Centralised Configuration

# Single source of truth for all hyperparameters, paths, feature
# definitions, and experimental settings used in the KD pipeline.
# every downstream module imports
# from here.

# Design :
# This file mirrors the role that fl_config.py plays for the federated
# learning pipeline: a self-containedconfiguration that
# makes every experiment reproducible.  Unlike fl_config.py which uses
# a dataclass because it varies per fold kd_config.py exposes module-
# level constants and a frozen dataclass for the mutable KD hyper-
# parameters because the data  constants (subject IDs, signal
# rates, feature lists) are fixed across all experiments.

# Notes

# * Window size = 16 s to matche the CNNLSTM teacher's training
#   (wesad_dl_windowing.py default, wesad_dl_dataset.py default,
#   fl_config.py default, also in demo.ipynb window_sec=16.0).
# * BVP @ 32 Hz to match DL_TARGET_HZ['bvp'] in wesad_dl_preprocess.py
#   and the ('bvp', 32, 1) entry in wesad_dl_windowing.WRIST_SIGNALS.
# * Label map {1.0:0, 2.0:1, 3.0:0} just like LABEL_MAP_BINARY in
#   wesad_dl_dataset.py
# * LSTM hidden = 32 : matches wesad_cnnlstm_model.py default and the
#   build_fl_wrist_model() in wesad_fl_model.py.

# References
# ---------------------------------------------------------------
# Hinton et al. 2015, arXiv:1503.02531 for the KD framework
# Castaldo et al. 2019, BMC Med Inform valiudation for ultra-short HRV
# Baek et al. 2015 validation that time-domain HRV is reliable at 10 s
# Task Force 1996, Eur Heart J —> HRV standards
# Fuller et al. 2022 —> Fitbit HR accuracy meta-analysis
# Nissen et al. 2022 for wearable HR noise
# Dalmeida & Masala 2021, Sensors 21(8):2873 —> HRV ranking
# Collins et al. 2021, ICML —> FedRep base/head split
# Peng et al. 2018, CBM —> 1 Hz HR to HRV limitations
# Makowski et al. 2021, DOI:10.3758/s13428-020-01516-y —> NK2

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os

# PATHS

# WESAD raw and preprocessed data
WESAD_RAW_DIR: str = "data/raw/wesad/"
WESAD_PREPROCESSED_DIR: str = "data/processed/wesad/"
WESAD_WRIST_PREPROCESSED: str = "data/processed/wesad/wrist_preprocessed.pkl"

# DL-windowed data (input for teacher inference)
# The teacher was trained on windows produced by wesad_dl_windowing.py,
# stored under this directory tree.  The paired_data_generator reads
# signal arrays from here to guarantee byte-identical teacher inputs.
WESAD_WINDOWED_DIR: str = "data/processed/wesad/dl_windowed/"

# KD-specific outputs
KD_OUTPUT_DIR: str = "data/kd/"
KD_PAIRED_DIR: str = os.path.join(KD_OUTPUT_DIR, "paired/")
KD_FIGURES_DIR: str = os.path.join(KD_OUTPUT_DIR, "figures/")

# Legacy generic result/checkpoint dirs (kept for any code that still
# imports them).  New code should use the _FED or _CEN variants below.
KD_CHECKPOINT_DIR: str = os.path.join(KD_OUTPUT_DIR, "checkpoints/")
KD_RESULTS_DIR: str = os.path.join(KD_OUTPUT_DIR, "results/")

# Per-teacher-source result and checkpoint directories.  Keeping these
# separate prevents the ablation run from overwriting primary results.
KD_CHECKPOINT_DIR_FED: str = os.path.join(KD_OUTPUT_DIR, "checkpoints_fed/")
KD_CHECKPOINT_DIR_CEN: str = os.path.join(KD_OUTPUT_DIR, "checkpoints_cen/")
KD_RESULTS_DIR_FED: str = os.path.join(KD_OUTPUT_DIR, "results_fed/")
KD_RESULTS_DIR_CEN: str = os.path.join(KD_OUTPUT_DIR, "results_cen/")

# Fitbit pilot data
FITBIT_DATA_DIR: str = "data/fitbit/"

# Teacher checkpoints — two sources for the dual KD experiment.
#
# KD-Fed (primary, privacy-preserving):
#   Per-fold FedRep warm-start checkpoints.  File layout:
#     {FEDREP_TEACHER_DIR}/FedRep_fold{sid}_global.pt
#   Each file is a dict with keys:
#     - 'global_model_state_dict'          : full WESADFLModel state
#     - 'federated_mean_head_state_dict'   : (unused here)
#     - 'algorithm'                        : 'FedRep'
#   LOSO-honest by construction: each .pt was produced by a federation
#   run in which the fold's test subject was never a client.
FEDREP_TEACHER_DIR: str = "production_downloads/fedrep/models_warmstart/"

# KD-Cen (ablation, quantifies privacy tax):
#   Per-fold centralised CNN-LSTM checkpoints.  File layout:
#     {CENTRALISED_TEACHER_DIR}/fold_S{sid}/best_model.pt
#   Each file is a raw state_dict of a CNNLSTM.  LOSO-honest by
#   construction: wesad_cnnlstm_train.train_loso_cv holds out the fold's
#   test subject from the training data.
#
#   NOTE: the directory name is "wrist_baseline" (not "centralised")
#   because this is the wrist-signal variant of the centralised
#   baseline from Iteration 1.  The chest variant lives under
#   chest_baseline/ but is not used as a KD teacher.
CENTRALISED_TEACHER_DIR: str = "results/dl/wrist_baseline/"

# Deprecated alias retained for backwards compatibility with any code
# still importing the old name.  New code should use
# CENTRALISED_TEACHER_DIR explicitly.
TEACHER_CHECKPOINT_DIR: str = CENTRALISED_TEACHER_DIR

# Source selector strings.  Pass one of these as the `source` argument
# to kd_teacher_loading.get_teacher_loader().
TEACHER_SOURCE_FED: str = "fedrep_warmstart"
TEACHER_SOURCE_CEN: str = "centralised_cnnlstm"
VALID_TEACHER_SOURCES: tuple[str, ...] = (TEACHER_SOURCE_FED, TEACHER_SOURCE_CEN)

# WESAD SUBJECT IDS

ALL_SUBJECT_IDS: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
NUM_SUBJECTS: int = len(ALL_SUBJECT_IDS)  # 15

#  3. SIGNAL DEFINITIONS
# replicated from wesad_dl_windowing.py to avoid an import dependency on the DL pipeline.
# Format: (signal_name, sampling_rate_hz, n_channels)
#
# CRITICAL: BVP is 32 Hz (post DL preprocessing downsampling from
# native 64 Hz).  All sample-count calculations use 32 Hz.

WRIST_SIGNALS: list[tuple[str, int, int]] = [
    ("bvp", 32, 1),
    ("w_eda", 4, 1),
    ("w_acc", 32, 3),
    ("w_temp", 4, 1),
]

# Convenience lookup: signal_name → sampling rate
SIGNAL_RATES: dict[str, int] = {name: hz for name, hz, _ in WRIST_SIGNALS}

# WINDOWING PARAMETERS
# Window size = 16 s to match the teacher's training configuration.
#
#   The CNN-LSTM teacher was trained on 16 s windows and feeding 60 s
#   signals to a 16 s-trained teacher would produce soft labels from a
#   model operating outside its training distribution thereforee the CNN
#   temporal dimensions would differ from what the model expects.
#   bY using 16s the same model serves as centralised baseline, FL participant, and KD teacher.
#
# Trade-off acknowledged here!: frequency-domain HRV features (LF, HF,
# LF/HF ratio) are unreliable at 16 s (TaskForce96 recommends ≥50 s
# for LF/HF).  The primary feature set is therefore restricted to
# time-domain features.  A 60 s supplementary experiment with a
# re-trained teacher is planned as future work.

WINDOW_SEC: float = 16.0
WINDOW_OVERLAP: float = 0.5  # 50 % overlap → 8 s step
SLIDE_SEC: float = WINDOW_SEC * (1.0 - WINDOW_OVERLAP)  # 8.0 s

# Per-signal sample counts for one 16 s window
# (used by paired_data_generator.py for slicing)
BVP_SAMPLES_PER_WINDOW: int = int(WINDOW_SEC * SIGNAL_RATES["bvp"])  # 512
EDA_SAMPLES_PER_WINDOW: int = int(WINDOW_SEC * SIGNAL_RATES["w_eda"])  # 64
ACC_SAMPLES_PER_WINDOW: int = int(WINDOW_SEC * SIGNAL_RATES["w_acc"])  # 512
TEMP_SAMPLES_PER_WINDOW: int = int(WINDOW_SEC * SIGNAL_RATES["w_temp"])  # 64


# LABEL MAPPING

LABEL_MAP_BINARY: dict[float, int] = {1.0: 0, 2.0: 1, 3.0: 0}
N_CLASSES: int = 2

# Labels considered valid for inclusion (discard transient / meditation)
ACTIVE_LABELS: set[float] = {1.0, 2.0, 3.0}

# Approximate class ratio for reference (from demo.ipynb label dist:
# baseline=3354, stress=1826, amusement=944 → non-stress≈4298, stress=1826
# → ratio ≈ 2.35:1).  Exact weights will be computed dynamically per fold.
APPROX_CLASS_RATIO: float = 2.35

# HRV FEATURE DEFINITIONS

# Primary feature set (6 time-domain features, reliable at 16 s)
#
# Justification:
#   Castaldo et al. (2019, BMC Medical Informatics) validated ultra-short
#   HRV from 10 s–3 min, confirming that mean_rr, sdnn, rmssd, and
#   pnn50 are reliable at windows as short as 10 s.  Baek et al. (2015)
#   corroborated time-domain reliability at 10 s.
#
#   mean_hr and std_hr are simple 60000/RR transformations that add no
#   new information in theory, but provide numerically scaled features
#   that can improve gradient flow in shallow MLPs (Dahal et al. 2023
#   used 8 HRV features including mean_hr; Dalmeida & Masala 2021
#   confirmed mean_rr and rmssd as top-ranked discriminators).
#
# Feature descriptions:
#   mean_rr: Mean RR interval (ms) that reflects average cardiac cycle
#   sdnn : Std of NN intervals, overall HRV, sympatho-vagal balance
#   rmssd: Root mean square of successive differences, vagal tone
#   pnn50: % successive intervals differing > 50 ms, parasympathetic
#   mean_hr: Mean heart rate (BPM), inverse of mean_rr
#   std_hr: Std of heart rate (BPM), HR variability in BPM domain

PRIMARY_HRV_FEATURES: list[str] = [
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "std_hr",
]
N_PRIMARY_FEATURES: int = len(PRIMARY_HRV_FEATURES)  # 6

# Fitbit-recoverable subset (5 features)
#
# These features can be approximated from Fitbit Charge 6's 1 Hz
# integer HR data (via pseudo-RR conversion: rr_ms = 60000 / hr_bpm)
#
# NOTE We include hr_range (max − min HR in window) because it is trivially
# recoverable from 1 Hz HR, captures sympathetic activation amplitude,
# and costs nothing computationally.
# Caveat (Peng et al. 2018): at 1 Hz sampling, pseudo-SDNN and
# pseudo-RMSSD are approximations, not true beat-to-beat HRV.
# LF error is 3.7 % but HF error reaches 15.3 %, so no
# frequency-domain features in this subset.

FITBIT_HRV_FEATURES: list[str] = [
    "mean_hr",
    "std_hr",
    "pseudo_sdnn",
    "pseudo_rmssd",
    "hr_range",
]
N_FITBIT_FEATURES: int = len(FITBIT_HRV_FEATURES)  # 5

# Minimal feature set (ablation: single feature)
MINIMAL_HRV_FEATURES: list[str] = ["mean_hr"]

# Extended feature set (12 features for 60 s future-work ablation)
#
# Frequency-domain features require ≥ 50 s for LF/HF reliability
# (TaskForce96).

EXTENDED_HRV_FEATURES: list[str] = [
    # Time-domain (same 6 as primary)
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "std_hr",
    # Frequency-domain
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "lf_norm",
    # Nonlinear
    "sd2",  # Poincaré plot long-term variability
    "sample_entropy",  # Signal complexity / irregularity
]
N_EXTENDED_FEATURES: int = len(EXTENDED_HRV_FEATURES)  # 12

# Feature-set for ablation loops
FEATURE_SETS: dict[str, list[str]] = {
    "primary_6": PRIMARY_HRV_FEATURES,
    "fitbit_5": FITBIT_HRV_FEATURES,
    "minimal_1": MINIMAL_HRV_FEATURES,
    "extended_12": EXTENDED_HRV_FEATURES,
}

# PEAK DETECTION & IBI PARAMETERS
# BVP peak detection is performed with NeuroKit2's ppg_findpeaks(),
# which implements the Elgendi (2013) algorithm.

BVP_SAMPLING_RATE: int = 32  # Hz (post DL-preprocessing)

# Physiologically plausible IBI range
# Lower bound: 375 ms → 160 BPM (maximum physiological sinus HR)
# Upper bound: 2000 ms → 30 BPM (extreme bradycardia)
IBI_MIN_MS: float = 375.0
IBI_MAX_MS: float = 2000.0

# Minimum number of valid IBIs required to compute reliable features
# from a single window.  With 16 s at resting HR (~70 BPM, ~857 ms
# per beat), expect ~18–19 beats → ~17–18 IBIs.  During stress
# (~100 BPM), expect ~26 beats → ~25 IBIs.  A threshold of 10
# retains most windows while excluding those with severe signal
# dropout.  Windows below this threshold return NaN and are excluded.
MIN_VALID_IBIS: int = 10

# Frequency-domain resampling rate for extended features (60 s ablation).
# Resample IBI to 4 Hz with cubic spline before PSD estimation,
# per Task Force (1996) guidelines.
IBI_RESAMPLE_HZ: float = 4.0

# HRV frequency bands, standard definitions (Task Force 1996)
LF_BAND: tuple[float, float] = (0.04, 0.15)
HF_BAND: tuple[float, float] = (0.15, 0.40)

# NOISE SIMULATION - FITBIT DEGRADATION
# Simulates the accuracy profile of consumer-grade wrist HR sensors
# to train a noise-robust student.
#
# hr_bias: Fitbit systematically underestimates HR.  Fuller et al.
#   (2022) meta-analysis of wrist-worn wearables found mean absolute
#   error of 3–5 BPM with negative bias during activity.  We use −3
#   BPM as a conservative central estimate.
#
# hr_noise_std: Nissen et al. (2022, DOI:10.2196/33635) reported
#   SD ≈ 5 BPM for Fitbit Charge 4 during ambulatory monitoring.
#   This captures both measurement noise and motion artefact.
#
# physiological_bounds: hard clip to [40, 200] BPM.  Values outside
#   this range are physiologically implausible in ambulatory settings
#   and likely sensor artefacts.

HR_BIAS_BPM: float = -3.0
HR_NOISE_STD_BPM: float = 5.0
HR_CLIP_LOW: int = 40
HR_CLIP_HIGH: int = 200

# INSTANCE NORMALISATION (TEACHER INPUT)
# The teacher was trained via WESADDLDataset, which applies per-window
# instance Z-score in __getitem__() BEFORE the forward pass:
#
#   1-channel: x_norm = (x - x.mean()) / (x.std() + eps)
#   3-channel: per-channel mean/std
#
# This MUST be replicated when caching teacher logits.  Failure to
# do so feeds the teacher a different input distribution than it was
# trained on, degrading the soft labels.
#
# The epsilon value matches wesad_dl_dataset.py (default 1e-8).

INSTANCE_NORM_EPS: float = 1e-8

# TEACHER ARCHITECTURE CONSTANTS (READ-ONLY reference)
# =====================================================================
# These are not configurable they describe the pre-trained teacher.
# They exist here so downstream modules can assert shape consistency
# without importing the model code.
#
# IMPORTANT TODO: These values reflect build_fl_wrist_model() DEFAULTS.
# They cannot be fully validated at import time because the actual
# checkpoint may have been trained with non-default settings (e.g. a
# 10 s ablation with target_time=4).  The kd_trainer.py MUST add
# runtime assertions when loading the checkpoint:
#
#   assert fl_model.head.lstm.hidden_size == TEACHER_LSTM_HIDDEN
#   assert fl_model.base.target_time == TEACHER_TARGET_TIME
#   assert fl_model.head.classifier[-1].out_features == TEACHER_N_CLASSES

TEACHER_LSTM_HIDDEN: int = 32  # wesad_cnnlstm_model.py default
TEACHER_LSTM_LAYERS: int = 2  # wesad_cnnlstm_model.py default
TEACHER_TARGET_TIME: int = 8  # build_fl_wrist_model() default
TEACHER_N_CLASSES: int = N_CLASSES  # binary

# Expected teacher input tensor shapes for a batch of B windows:
#   bvp:    (B, 1, 512)
#   w_eda:  (B, 1,  64)
#   w_acc:  (B, 3, 512)
#   w_temp: (B, 1,  64)
TEACHER_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "bvp": (1, BVP_SAMPLES_PER_WINDOW),  # (1, 512)
    "w_eda": (1, EDA_SAMPLES_PER_WINDOW),  # (1,  64)
    "w_acc": (3, ACC_SAMPLES_PER_WINDOW),  # (3, 512)
    "w_temp": (1, TEMP_SAMPLES_PER_WINDOW),  # (1,  64)
}

# Dimension of the teacher's intermediate representation used for
# optional feature-matching loss.  This is the post-BatchNorm LSTM
# hidden state extracted by KDTeacherWrapper, verified to be dim=32
# (matching TEACHER_LSTM_HIDDEN).  No projection layer is needed
# because the student's intermediate dim also = 32 by design.
TEACHER_FEATURE_DIM: int = TEACHER_LSTM_HIDDEN  # 32

# STUDENT ARCHITECTURE DEFAULTS
# Lightweight MLP with FedRep-compatible base/head split.
# Total ~3,500–5,000 parameters (vs teacher's ~50K–100K = 10–20× compression).
#
# Design decisions
#   - MLP over LSTM: input is a flat feature vector per window, not a
#     sequence.  Ay et al. (2022, IJCNN) demonstrated comparable
#     performance with MLPs on pre-extracted HRV features.
#   - BatchNorm: stabilises training with small batches typical in
#     LOSO (Ioffe & Szegedy, 2015).
#   - Dropout 0.2: lighter than teacher's 0.5 because the student is
#     already small and thus less prone to overfitting (Holda et al.
#     2024).
#   - Base/head split at the 32→16 boundary: base (64→32) learns
#     universal HRV-to-stress mappings (shared in FedRep), head
#     (16→2) learns personal decision boundaries (Collins et al. 2021).

STUDENT_BASE_HIDDEN_1: int = 64
STUDENT_BASE_HIDDEN_2: int = 32  # intermediate dim = TEACHER_FEATURE_DIM
STUDENT_HEAD_HIDDEN: int = 16
STUDENT_DROPOUT: float = 0.2
STUDENT_FEATURE_DIM: int = STUDENT_BASE_HIDDEN_2  # 32, matches teacher


# KD HYPERPARAMETERS (defaults)


@dataclass(frozen=True)
class KDHyperparameters:
    """Immutable hyperparameters for one KD experiment.

    Frozen (immutable) after creation to prevent accidental mutation
    of shared defaults during training.  To create a variant for grid
    search, use ``dataclasses.replace(KD_DEFAULTS, temperature=4.0)``.
    Serialisable to JSON for reproducibility.

    Default values follow Hinton et al. (2015) recommendations
    (alpha=0.5–0.9, T=2–5) and are tuned via grid search during
    training phase.
    """

    # Temperature
    # Controls softness of the teacher's probability distribution.
    # Higher T → softer distributions → more inter-class relationship
    # information transferred.
    # Search range: {1, 2, 3, 4, 5}.
    temperature: float = 3.0

    # Distillation weight (alpha)
    # Balances KD loss (KL divergence from teacher) vs hard-label CE.
    # alpha=0 → pure hard-label training (no KD, ablation baseline).
    # alpha=1 → pure distillation (no ground truth signal).
    # Hinton et al. (2015) recommend 0.5–0.9.
    # Search range: {0.1, 0.3, 0.5, 0.7, 0.9}
    alpha: float = 0.5

    # Feature-matching weight (beta)
    # Weight of the optional feature-matching loss:
    #   L_feat = beta * MSE(student_features, teacher_features)
    # beta=0 → response-only KD (default, simpler, standard).
    # MemKD (2025) suggests feature matching adds value for binary
    # tasks where soft output carries limited information.
    # Search range: {0.0, 0.1, 0.5, 1.0}
    beta: float = 0.0

    #  Optimiser
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # AdamW default (Loshchilov & Hutter 2019)
    max_grad_norm: float = 1.0  # gradient clipping

    # Training schedule
    epochs: int = 100
    batch_size: int = 64  # matches fl_config.py and teacher training
    early_stopping_patience: int = 10

    # Cosine annealing
    # The trainer should use:
    #   CosineAnnealingLR(optimizer, T_max=config.epochs)
    # Preferred over ReduceLROnPlateau for KD because the loss landscape
    # changes as the student progressively captures teacher knowledge.

    # Reproducibility
    seed: int = 42

    #  Grid search ranges (for kd_trainer.py Step 4)
    # 4 × 5 = 20 combinations × 15 LOSO folds = 300 runs.
    # Each run: small MLP, ~100 epochs, ~thousands of windows → ~2–4 h.
    temperature_search: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    alpha_search: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"  KD config saved: {path}")

    @classmethod
    def load_json(cls, path: str) -> KDHyperparameters:
        with open(path) as f:
            d = json.load(f)
        # Drop keys not in the dataclass (forward compatibility)
        valid = {fld.name for fld in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid}
        for key in ("temperature_search", "alpha_search"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)


# Default instance (import as `from kd.kd_config import KD_DEFAULTS`)
KD_DEFAULTS = KDHyperparameters()


# PERSONALISATION FINE-TUNING HYPERPARAMETERS
# After global KD training, freeze the student base and fine-tune
# the head on a calibration portion of the held-out subject's data.
#
# IMPLEMENTATION NOTE :
#   finetune_fraction and fitbit_calibration_minutes are two
#   fundamentally different protocols.  WESAD personalisation uses
#   a percentage of test subject windows and the Fitbit pilot uses
#   a fixed temporal duration.  Need to remember to keep them kept as separate, clearly
#   documented parameters.

# WESAD personalisation
FINETUNE_LR: float = 1e-4  # smaller than KD training (head-only, few samples)
FINETUNE_STEPS_DEFAULT: int = 10
FINETUNE_STEPS_SEARCH: tuple[int, ...] = (5, 10, 20, 50)
FINETUNE_FRACTION: float = 0.20  # 20 % of test subject's windows for calibration

# Fitbit pilot personalisation
FITBIT_CALIBRATION_MINUTES: float = 8.0  # first ~8 min of structured session

# FITBIT DATA QUALITY PARAMETERS
# The Fitbit API "1sec detail level" does NOT guarantee uniform 1 Hz.
# The raw data is an irregularly sampled time series.

FITBIT_MAX_GAP_SEC: float = 5.0  # gaps > 5 s flagged (Bottcher et al. 2022)
FITBIT_UNRELIABLE_FRAC: float = 0.20  # exclude window if > 20 % flagged

# EVALUATION SETTINGS
PRIMARY_METRIC: str = "macro_f1"

# Statistical testing
# Wilcoxon signed-rank: non-parametric, paired, 15 observations per test.
# Bonferroni correction for 3 comparisons: (c) vs (b), (d) vs (c), (d) vs (b).
STAT_ALPHA: float = 0.05
N_COMPARISONS: int = 3
BONFERRONI_ALPHA: float = STAT_ALPHA / N_COMPARISONS  # 0.0167

# ABLATION STUDIES

ABLATION_TEMPERATURES: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
ABLATION_ALPHAS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
ABLATION_FEATURE_SETS: tuple[str, ...] = ("primary_6", "fitbit_5", "minimal_1")
ABLATION_WINDOW_SIZES: tuple[float, ...] = (10.0, 16.0)  # 60.0 is future work
ABLATION_BETAS: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
ABLATION_FINETUNE_STEPS: tuple[int, ...] = (0, 5, 10, 20, 50)

# HARDWARE / RUNTIME

try:
    import torch
except ModuleNotFoundError:
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 0


# UTILITY FUNCTIONS
def get_paired_data_path(subject_id: int) -> str:
    """Return the path to a subject's paired .npz file."""
    return os.path.join(KD_PAIRED_DIR, f"S{subject_id}_paired.npz")


def get_teacher_checkpoint_path(
    fold_id: int,
    source: str = TEACHER_SOURCE_CEN,
) -> str:
    """Return the expected teacher checkpoint path for a LOSO fold.

    Parameters
    ----------
    fold_id : int
        Test subject ID.
    source : str
        One of TEACHER_SOURCE_FED or TEACHER_SOURCE_CEN.  Controls
        which directory and filename convention is used.

    Returns
    -------
    str
        Full path to the expected checkpoint file.  Does NOT check that
        the file exists; callers should handle missing files.
    """
    if source == TEACHER_SOURCE_FED:
        return os.path.join(FEDREP_TEACHER_DIR, f"FedRep_fold{fold_id}_global.pt")
    if source == TEACHER_SOURCE_CEN:
        return os.path.join(
            CENTRALISED_TEACHER_DIR, f"fold_S{fold_id}", "best_model.pt"
        )
    raise ValueError(
        f"Unknown teacher source: {source!r}. "
        f"Expected one of: {TEACHER_SOURCE_FED!r}, {TEACHER_SOURCE_CEN!r}."
    )


def get_windowed_dir(window_sec: float = WINDOW_SEC) -> str:
    """Return the windowed data directory for a given window length."""
    return os.path.join(WESAD_WINDOWED_DIR, f"{window_sec:.0f}s")


def ensure_dirs() -> None:
    """Create all KD output directories if they don't exist."""
    for d in (
        KD_OUTPUT_DIR,
        KD_PAIRED_DIR,
        KD_CHECKPOINT_DIR,  # legacy
        KD_CHECKPOINT_DIR_FED,
        KD_CHECKPOINT_DIR_CEN,
        KD_RESULTS_DIR,  # legacy
        KD_RESULTS_DIR_FED,
        KD_RESULTS_DIR_CEN,
        KD_FIGURES_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def validate_subject_id(subject_id: int) -> None:
    """Raise ValueError if subject_id is not a valid WESAD subject."""
    if subject_id not in ALL_SUBJECT_IDS:
        raise ValueError(
            f"Invalid subject ID: {subject_id}. " f"Valid IDs: {ALL_SUBJECT_IDS}"
        )


def get_loso_splits(test_subject: int) -> dict[str, list[int]]:
    """Generate LOSO train/val/test split for a given test subject.

    Parameters
    ----------
    test_subject : int
        Subject ID to hold out for testing.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each mapping to a list
    of subject IDs.
    """
    validate_subject_id(test_subject)
    remaining = [s for s in ALL_SUBJECT_IDS if s != test_subject]

    # Rotating validation subject replicates wesad_dl_dataset.py logix exactky
    test_idx = ALL_SUBJECT_IDS.index(test_subject)
    val_idx = (test_idx + 1) % len(ALL_SUBJECT_IDS)
    val_subject = ALL_SUBJECT_IDS[val_idx]

    assert val_subject != test_subject, (
        f"Rotation picked test subject {test_subject} as val — "
        f"check ALL_SUBJECT_IDS length ({len(ALL_SUBJECT_IDS)})"
    )
    assert (
        val_subject in remaining
    ), f"val_subject {val_subject} not in remaining subjects"

    train_subjects = [s for s in remaining if s != val_subject]

    return {
        "train": train_subjects,  # 13 subjects
        "val": [val_subject],  # 1 subject
        "test": [test_subject],  # 1 subject
    }


def print_config_summary() -> None:
    """Print a human-readable summary of the KD configuration."""
    print("\n")
    print("Cross-Modal KD Configuration Summary")
    print("-" * 60)
    print(f"  Subjects:           {NUM_SUBJECTS} ({ALL_SUBJECT_IDS})")
    print(
        f"  Window:             {WINDOW_SEC} s, {WINDOW_OVERLAP*100:.0f}% overlap "
        f"(slide={SLIDE_SEC} s)"
    )
    print(f"  Label map:          binary — {LABEL_MAP_BINARY}")
    print(f"  Primary features:   {N_PRIMARY_FEATURES} time-domain HRV")
    print(f"  BVP rate:           {BVP_SAMPLING_RATE} Hz")
    print(f"  BVP samples/window: {BVP_SAMPLES_PER_WINDOW}")
    print(f"  Device:             {DEVICE}")
    print("\n  KD defaults:")
    print(
        f"    T={KD_DEFAULTS.temperature}, "
        f"alpha={KD_DEFAULTS.alpha}, "
        f"beta={KD_DEFAULTS.beta}"
    )
    print(
        f"    lr={KD_DEFAULTS.learning_rate}, "
        f"epochs={KD_DEFAULTS.epochs}, "
        f"batch_size={KD_DEFAULTS.batch_size}"
    )
    print(f"    patience={KD_DEFAULTS.early_stopping_patience}")
    print("\n  Teacher (frozen):")
    print(
        f"    LSTM hidden={TEACHER_LSTM_HIDDEN}, "
        f"layers={TEACHER_LSTM_LAYERS}, "
        f"feature_dim={TEACHER_FEATURE_DIM}"
    )
    print("\n  Student:")
    print(
        f"    Base: {N_PRIMARY_FEATURES}→{STUDENT_BASE_HIDDEN_1}"
        f"→{STUDENT_BASE_HIDDEN_2}"
    )
    print(f"    Head: {STUDENT_HEAD_HIDDEN}→{N_CLASSES}")
    print(f"    Dropout: {STUDENT_DROPOUT}")
    print("\n  Noise simulation:")
    print(
        f"    bias={HR_BIAS_BPM} BPM, std={HR_NOISE_STD_BPM} BPM, "
        f"clip=[{HR_CLIP_LOW}, {HR_CLIP_HIGH}]"
    )
    print("\n  Personalisation:")
    print(
        f"    WESAD: lr={FINETUNE_LR}, fraction={FINETUNE_FRACTION}, "
        f"steps={FINETUNE_STEPS_DEFAULT}"
    )
    print(f"    Fitbit: {FITBIT_CALIBRATION_MINUTES} min calibration")


# Catch configuration inconsistencies at import time.

assert BVP_SAMPLES_PER_WINDOW == 512, (
    f"BVP sample count mismatch: expected 512, got {BVP_SAMPLES_PER_WINDOW}. "
    f"Check WINDOW_SEC ({WINDOW_SEC}) and BVP rate ({SIGNAL_RATES['bvp']})."
)
assert (
    EDA_SAMPLES_PER_WINDOW == 64
), f"EDA sample count mismatch: expected 64, got {EDA_SAMPLES_PER_WINDOW}."
assert STUDENT_FEATURE_DIM == TEACHER_FEATURE_DIM, (
    f"Student-teacher feature dim mismatch: student={STUDENT_FEATURE_DIM}, "
    f"teacher={TEACHER_FEATURE_DIM}.  Feature-matching requires equal dims."
)
assert len(ALL_SUBJECT_IDS) == 15, f"Expected 15 subjects, got {len(ALL_SUBJECT_IDS)}."
assert (
    SLIDE_SEC == 8.0
), f"Slide should be 8.0 s for 50% overlap of 16 s window, got {SLIDE_SEC}."

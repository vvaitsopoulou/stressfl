# DL PyTorch Dataset
# Loads the windowed .npz arrays produced by wesad_dl_windowing.py and
# serves them to the DataLoader as PyTorch tensors, with
# instance normalization and training-time augmentation.

# Design Decisions

# 1. Instance (per-window) Z-score normalization
#    Each window is normalised independently: x' = (x − μ) / (σ + ε)
#    where μ and σ are computed over that window's time axis only.
#    This forces the CNN to learn the shape of the signal, not the
#    absolute amplitude, which varies wildly across subjects.

#    This design has been made keeping in mind the need to avoid data leakage:
#      Instance Z-score uses only the current window's statistics.
#      No information from other windows, other subjects, or the test
#      set is involved.

#    Citation: Moser et al. (2024): "standardized by subtracting the
#    mean μ and dividing by σ to have mean=0 and std=1 for each signal.
#    This removes bias related to differing value ranges."

# 2. On-the-fly augmentation (training only)
#    Two augmentations are applied stochastically per window per epoch:

#    a) Gaussian noise injection (Jitter):
#       x_aug = x + N(0, σ_noise)  where σ_noise = noise_std × σ_window
#       Applied independently per channel as using the same noise vector
#       across channels would create cross-channel correlations.

#    b) Amplitude scaling:
#       x_aug = x × α  where α ~ U(scale_range[0], scale_range[1])
#       One scalar α per signal per window in orderto preserve relative channel
#       relationships within multi-axis signals (e.g. ACC x/y/z).

#    Both are applied on;ly during training (mode="train").
#    Validation and test windows receive normalization only.

#    Citation: Moser et al. (2024): "we added random Gaussian noise to
#    the stress moments and scaled the amplitude randomly, ultimately
#    arriving at new samples for the training dataset."

# 3. Asymmetric overlap filtering
#    The windowing module saves ALL windows with an is_overlapped flag.
#    This Dataset filters them based on mode:
#      - mode="train":  ALL windows (overlapped + non-overlapped)
#      - mode="val":    non-overlapped windows ONLY
#      - mode="test":   non-overlapped windows ONLY

# 4. Label mapping
#    Raw WESAD labels (1.0=baseline, 2.0=stress, 3.0=amusement) are
#    mapped to integer classes at load time.  Two mappings are supported:
#      - Binary:  baseline+amusement → 0, stress → 1
#      - 3-class: baseline → 0, stress → 1, amusement → 2

# 5. Multi-signal output
#    __getitem__ returns a dict of signal tensors (one per modality)
#    plus the label.  This matches the multi-branch CNN-LSTM architecture
#    where each modality has its own CNN branch.

# References:
#   Moser et al. (2024) - Z-score, Gaussian noise, amplitude scaling
#   Ordóñez & Roggen (2016) - high-overlap training as augmentation
#   Prajod & André (2022) - LOSO evaluation, class weighting

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

#  Label mappings

LABEL_MAP_BINARY = {1.0: 0, 2.0: 1, 3.0: 0}
LABEL_MAP_3CLASS = {1.0: 0, 2.0: 1, 3.0: 2}

#  Dataset


class WESADDLDataset(Dataset):
    # Loads windowed arrays from one or more subjects, applies instance
    # Z-score normalization, and optionally augmentation
    # during training.

    # Parameters
    # subject_ids : list[int]
    #     Which subjects to include (e.g. [2, 3, 4] for training fold).
    # windowed_dir : str
    #     Root of the windowed data (e.g. "data/dl_windowed/").
    # device_type : str
    #     "chest" or "wrist"
    # window_sec : float
    #     Window length that was used during windowing (for directory lookup).
    # mode : str
    #     "train", "val", or "test".
    #     - "train": uses ALL windows (overlapped + independent), applies augmentation.
    #     - "val" / "test": uses ONLY non-overlapped (independent) windows, no augmentation.
    # label_map : dict
    #     Label mapping (LABEL_MAP_BINARY or LABEL_MAP_3CLASS).
    # noise_std : float
    #     Gaussian noise standard deviation as a fraction of the window's σ.
    #     Default 0.05 (5% of window std).  Set to 0.0 to disable noise.
    # scale_range : tuple[float, float]
    #     Amplitude scaling range.  Default (0.9, 1.1).
    #     Set to (1.0, 1.0) to disable scaling.
    # signals_to_load : list[str] | None
    #     Specific signal names to load (e.g. ["ecg", "resp"]).
    #     If None, loads all signals present in the .npz file.
    # epsilon : float
    #     Small constant added to σ during Z-score to avoid division by zero.
    #     Default 1e-8.

    def __init__(
        self,
        subject_ids: list[int],
        windowed_dir: str,
        device_type: str,
        window_sec: float = 16.0,
        mode: str = "train",
        label_map: dict | None = None,
        noise_std: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
        signals_to_load: list[str] | None = None,
        epsilon: float = 1e-8,
        exclude_nonoverlapped: bool = False,
    ):
        assert mode in ("train", "val", "test"), f"Invalid mode: {mode}"
        assert device_type in ("chest", "wrist"), f"Invalid device: {device_type}"

        self.mode = mode
        self.device_type = device_type
        self.label_map = label_map or LABEL_MAP_3CLASS
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.signals_to_load = signals_to_load
        self.epsilon = epsilon
        # When mode="train", if exclude_nonoverlapped=True we keep ONLY
        # overlapped windows and drop the non-overlapped ones.  This is
        # used in FL for the test subject's own training dataset, so that
        # their local training windows are disjoint from their test set
        # (which consists of non-overlapped windows).  Without this flag,
        # personalised FL methods (FedRep, FedBABU, PeFLL) can trivially
        # memorise the test windows via the local head (observed: F1=1.00
        # on S2 after 5 rounds of FedRep).  No-op unless mode="train".
        self.exclude_nonoverlapped = exclude_nonoverlapped

        # Load and concatenate windows from all requested subjects
        self.signal_names: list[str] = []  # populated on first subject load
        self._signal_arrays: dict[str, np.ndarray] = {}  # name → (N, T[, C])
        self._labels: np.ndarray = np.array([], dtype=np.int64)
        self._subject_ids_per_window: np.ndarray = np.array([], dtype=np.int64)

        for sid in subject_ids:
            self._load_subject(sid, windowed_dir, window_sec, device_type)

        self.n_windows = len(self._labels)

        if self.n_windows == 0:
            print(
                f"  WESADDLDataset: 0 windows loaded for {device_type} "
                f"mode={mode} subjects={subject_ids}"
            )

    def _load_subject(
        self,
        subject_id: int,
        windowed_dir: str,
        window_sec: float,
        device_type: str,
    ):
        subj_dir = os.path.join(
            windowed_dir, f"{window_sec:.0f}s", f"S{subject_id}", device_type
        )
        npz_path = os.path.join(subj_dir, "windows.npz")
        meta_path = os.path.join(subj_dir, "metadata.csv")

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            print(f"  Missing data for S{subject_id}/{device_type} — skipping")
            return

        # Load metadata
        labels = []
        is_overlapped = []

        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(float(row["label"]))
                is_overlapped.append(row["is_overlapped"] == "True")

        labels = np.array(labels)
        is_overlapped = np.array(is_overlapped)

        # Filter by mode
        # Train: keep ALL windows (overlapped + independent), UNLESS
        #   exclude_nonoverlapped=True (used for the test-subject-as-FL-client
        #   whose non-overlapped windows are the test set and must not
        #   appear in their local training data).
        # Val/Test: keep ONLY non-overlapped (independent) windows
        if self.mode in ("val", "test"):
            keep_mask = ~is_overlapped
        elif self.mode == "train" and self.exclude_nonoverlapped:
            keep_mask = is_overlapped.astype(bool)
        else:
            keep_mask = np.ones(len(labels), dtype=bool)

        # Map labels to integer classes
        mapped_labels = np.array(
            [self.label_map[lbl] for lbl in labels[keep_mask]],
            dtype=np.int64,
        )

        # Load signal arrays
        npz = np.load(npz_path)
        available_signals = list(npz.keys())

        # Determine which signals to use
        if not self.signal_names:
            # First subject:establish the signal list
            if self.signals_to_load is not None:
                self.signal_names = [
                    s for s in self.signals_to_load if s in available_signals
                ]
            else:
                self.signal_names = sorted(
                    available_signals
                )  # to have determinittic order regardless of how npz was written

        # Extract and filter windows per signal
        # If any signal is missing, skip this subject entirely since3
        # the multi-branch architecture needs all signals aligned
        # by window index.
        missing = [s for s in self.signal_names if s not in npz]
        if missing:
            print(
                f"    S{subject_id}: missing signals {missing} -> "
                f"skipping entire subject to prevent array misalignment"
            )
            return

        # Second pass: extract data (all signals guaranteed present)
        for sig_name in self.signal_names:
            sig_data = npz[sig_name][keep_mask]  # (n_kept, T[, C])

            if sig_name not in self._signal_arrays:
                self._signal_arrays[sig_name] = sig_data
            else:
                self._signal_arrays[sig_name] = np.concatenate(
                    [self._signal_arrays[sig_name], sig_data], axis=0
                )

        # Append labels and subject IDs
        self._labels = np.concatenate([self._labels, mapped_labels])
        sid_array = np.full(len(mapped_labels), subject_id, dtype=np.int64)
        self._subject_ids_per_window = np.concatenate(
            [self._subject_ids_per_window, sid_array]
        )

    #  Normalization

    def _instance_znorm(self, x: np.ndarray) -> np.ndarray:
        """Per-window, per-channel Z-score normalization.

        Parameters
        x : np.ndarray
            Shape (T,) for 1-channel signals or (T, C) for multi-channel.

        Returns
        np.ndarray
            Normalised array, same shape as input.

        Uses only this window's own μ and σ so zero data leakage.
        ε is added to σ to avoid division by zero for constant windows
        (e.g. a TEMP window where temperature didn't change).
        """
        if x.ndim == 1:
            mu = x.mean()
            sigma = x.std()
            return (x - mu) / (sigma + self.epsilon)
        else:
            # Per-channel: compute μ and σ along the time axis (axis=0)
            mu = x.mean(axis=0, keepdims=True)  # (1, C)
            sigma = x.std(axis=0, keepdims=True)  # (1, C)
            return (x - mu) / (sigma + self.epsilon)

    #  Augmentation (training only)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise injection and amplitude scaling.

        Parameters
        x : np.ndarray
            Shape (T,) or (T, C).  Already Z-normalised.

        Augmentation is applied after normalization so that
        noise_std is relative to the normalised signal's std (~1.0),
        making the noise magnitude consistent across signals and
        subjects regardless of original amplitude.

        Noise is generated independently per channel to avoid
        creating cross-channel correlations.
        """
        # Amplitude scaling
        # One scalar per signal (not per channel) to preserves
        # relative channel relationships (e.g. ACC x/y/z ratios)
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        x = x * alpha

        # Gaussian noise injection
        if self.noise_std > 0:
            # noise_std is relative to the window's post-norm std (~1.0)
            # so noise_std=0.05 means noise amplitude is ~5% of signal
            noise = np.random.normal(0, self.noise_std, size=x.shape)
            x = x + noise

        return x

    #  PyTorch Dataset interface

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Return one sample: dict of signal tensors + label.

        Returns
        signals : dict[str, torch.Tensor]
            signal_name → tensor of shape (C, T) for CNN input.
            1-channel signals: (1, T).  3-channel signals: (3, T).
            The channel-first layout matches nn.Conv1d's expected input.
        label : torch.Tensor
            Scalar integer label.
        """
        signals = {}

        for sig_name in self.signal_names:
            x = self._signal_arrays[sig_name][idx].copy()  # (T,) or (T, C)

            # Normalize
            x = self._instance_znorm(x)

            # Augment
            if self.mode == "train":
                x = self._augment(x)

            # Convert to tensor
            x = x.astype(np.float32)

            if x.ndim == 1:
                # (T,) → (1, T) for Conv1d
                tensor = torch.from_numpy(x).unsqueeze(0)
            else:
                # (T, C) → (C, T) for Conv1d
                tensor = torch.from_numpy(x).permute(1, 0)

            signals[sig_name] = tensor

        label = torch.tensor(self._labels[idx], dtype=torch.long)

        return signals, label

    #  Utility methods

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss function.

        Returns a tensor of shape (n_classes,) where each weight is
        proportional to 1 / class_count.  This handles the WESAD label
        imbalance (subjects spend more time at baseline than stress).

        Citation: Prajod & André (2022): "we use a weighted loss to
        tackle class imbalance in the training dataset."
        """
        classes, counts = np.unique(self._labels, return_counts=True)
        weights = 1.0 / counts.astype(np.float64)
        weights = weights / weights.sum() * len(classes)  # normalise so mean = 1

        # Ensure weights tensor has an entry for every class index
        # Default to 1.0 (neutral weight) for classes not present in this
        # dataset split eg if a held-out subject
        # has no amusement windows. Neutral weight means the loss function treats any
        # prediction of that class normally instead of crashing
        n_classes = max(self.label_map.values()) + 1
        full_weights = np.ones(n_classes, dtype=np.float64)
        for cls, w in zip(classes, weights):
            full_weights[cls] = w

        return torch.tensor(full_weights, dtype=torch.float32)

    def get_label_distribution(self) -> dict[int, int]:
        """Return {class_id: count} for the loaded windows."""
        classes, counts = np.unique(self._labels, return_counts=True)
        return dict(zip(classes.tolist(), counts.tolist()))

    def get_subject_ids(self) -> np.ndarray:
        """Return the subject ID for each window (for stratification etc)."""
        return self._subject_ids_per_window

    def get_signal_shapes(self) -> dict[str, tuple]:
        """Return {signal_name: (n_windows, T[, C])} for inspection."""
        return {name: arr.shape for name, arr in self._signal_arrays.items()}


# build train/val/test datasets for one LOSO fold

ALL_SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


def build_loso_datasets(
    test_subject: int,
    windowed_dir: str = "data/dl_windowed/",
    device_type: str = "wrist",
    window_sec: float = 16.0,
    label_map: dict | None = None,
    val_subject: int | None = None,
    noise_std: float = 0.05,
    scale_range: tuple[float, float] = (0.9, 1.1),
    signals_to_load: list[str] | None = None,
) -> tuple["WESADDLDataset", "WESADDLDataset", "WESADDLDataset"]:
    """Build train/val/test datasets for one LOSO fold.

    Parameters
    ----------
    test_subject : int
        Subject ID held out for testing.
    windowed_dir : str
        Root of the windowed data.
    device_type : str
        "chest" or "wrist".
    window_sec : float
        Window length used during windowing.
    label_map : dict
        Label mapping.
    val_subject : int | None
        Subject ID held out for validation (from the remaining 14).
        If None, the validation subject is selected by rotation:
        the subject immediately after the test subject in the sorted
        ID list (wrapping around).  This ensures every subject serves
        as validation in exactly one fold and appears in training in
        all other folds so no subject is systematically underrepresented.
    noise_std : float
        Gaussian noise std for augmentation.
    scale_range : tuple
        Amplitude scaling range for augmentation.
    signals_to_load : list[str] | None
        Which signals to load.

    Returns
    -------
    train_ds, val_ds, test_ds : WESADDLDataset
        Ready for torch.utils.data.DataLoader.

    Notes
    -----
    Validation subject selection:
      In LOSO with 15 subjects, the test subject is fixed per fold.
      We hold out one additional subject from the remaining 14 as a
      validation set for early stopping.  This is a nested LOSO design:
      the validation subject's data is never seen during training,
      preventing the early stopping criterion from overfitting to
      training data.

      Using a deterministic rotation rather than random split ensures
      reproducibility across runs.  The validation subject is the next
      subject after the test subject in the sorted ID list (wrapping
      around).  This guarantees every subject serves as validation in
      exactly one fold and appears in training in all other 13 folds so
      no subject is systematically underrepresented.

      The training set therefore contains 13 subjects, validation has 1,
      and test has 1.
    """
    if label_map is None:
        label_map = LABEL_MAP_BINARY

    remaining = [s for s in ALL_SUBJECT_IDS if s != test_subject]

    if val_subject is None:
        # Rotate: pick the subject immediately after test_subject
        # in the sorted full list, wrapping around.
        test_idx = ALL_SUBJECT_IDS.index(test_subject)
        val_idx = (test_idx + 1) % len(ALL_SUBJECT_IDS)
        val_candidate = ALL_SUBJECT_IDS[val_idx]
        # If val_candidate == test_subject (impossible with len>1), fallback
        val_subject = val_candidate if val_candidate != test_subject else remaining[0]
    else:
        assert (
            val_subject in remaining
        ), f"val_subject {val_subject} is not in remaining subjects"

    train_subjects = [s for s in remaining if s != val_subject]

    print(
        f"\n  LOSO fold: test=S{test_subject}, val=S{val_subject}, "
        f"train=S{train_subjects}"
    )

    train_ds = WESADDLDataset(
        subject_ids=train_subjects,
        windowed_dir=windowed_dir,
        device_type=device_type,
        window_sec=window_sec,
        mode="train",
        label_map=label_map,
        noise_std=noise_std,
        scale_range=scale_range,
        signals_to_load=signals_to_load,
    )

    val_ds = WESADDLDataset(
        subject_ids=[val_subject],
        windowed_dir=windowed_dir,
        device_type=device_type,
        window_sec=window_sec,
        mode="val",
        label_map=label_map,
        noise_std=0.0,  # no augmentation for validation
        scale_range=(1.0, 1.0),
        signals_to_load=signals_to_load,
    )

    test_ds = WESADDLDataset(
        subject_ids=[test_subject],
        windowed_dir=windowed_dir,
        device_type=device_type,
        window_sec=window_sec,
        mode="test",
        label_map=label_map,
        noise_std=0.0,  # no augmentation for test
        scale_range=(1.0, 1.0),
        signals_to_load=signals_to_load,
    )

    print(f"    Train: {len(train_ds)} windows from {len(train_subjects)} subjects")
    print(f"    Val:   {len(val_ds)} windows from S{val_subject} (non-overlapped only)")
    print(
        f"    Test:  {len(test_ds)} windows from S{test_subject} (non-overlapped only)"
    )
    print(f"    Labels: {train_ds.get_label_distribution()}")
    print(f"    Class weights: {train_ds.get_class_weights().tolist()}")

    return train_ds, val_ds, test_ds

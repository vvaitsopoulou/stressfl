# LOSO-aware DataLoader adapter for Federated Learning clients.
#
# This module wraps the existing WESADDLDataset
# into the client-facing interface that FL clients expect.  Given a
# subject ID and a fold configuration (train subjects, val subject,
# test subject), it returns train/val/test DataLoaders that FL clients
# can call without knowing about LOSO internals.
#
# We adapt from:
#   - already existing wesad_dl_dataset.py          : WESADDLDataset, build_loso_datasets(), ALL_SUBJECT_IDS
#   - adapting from PFLIB_clientbase.py (apache license)          : load_train_data() / load_test_data() interface contract
#
# DESIGN DECISIONS
#
# 1. Validation Strategy: ROTATING VALIDATION (one per fold)
#
#    For each LOSO fold with test subject S_t, the validation subject is
#    selected by deterministic rotation: the subject immediately after S_t
#    in the sorted subject list (wrapping around).  This means every
#    subject serves as validation in exactly one fold, so no subject is
#    systematically underrepresented in training.
#
# 2. Federation participation: TEST SUBJECT IS HELD OUT
#
#    The test subject S_t does NOT participate in federated training.
#    Federation proceeds across the remaining 14 subjects.  After
#    federation completes, S_t's data is split into a calibration set
#    (20%, used to personalise the local head) and an evaluation set
#    (80%, used to report the LOSO fold metric).  The split is stratified
#    by contiguous condition block so both sets preserve class balance.
#
#    This held-out-with-calibration protocol is established in both the
#    FL literature (Collins et al. 2021 "Generalization to new clients",
#    Chen et al. 2020 FedHealth two-step paradigm) and the WESAD-specific
#    personalisation literature.  It prevents the local-head memorisation
#    that occurs when the test subject's overlapped training windows
#    share signal with their non-overlapped test windows.
#
#    All FL methods in this codebase use this unified protocol:
#      - FedAvg   : no calibration; evaluate global model on eval set
#      - FedRep   : fresh random head, train on calibration set for K
#                   epochs with base frozen, evaluate on eval set
#      - FedBABU  : identical to FedRep (post-fed fine-tuning is its design)
#      - PeFLL    : hypernet generates head from calibration embedding,
#                   evaluate on eval set
#


from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch.utils.data import DataLoader

from ..data.wesad_dl_dataset import (
    ALL_SUBJECT_IDS,
    LABEL_MAP_BINARY,
    # LABEL_MAP_3CLASS,
    WESADDLDataset,
)


class _WESADCommonKwargs(TypedDict):  # fix mypy complaining
    windowed_dir: str
    device_type: str
    window_sec: float
    label_map: dict | None
    signals_to_load: list[str] | None


# Fold Configuration


@dataclass
class FLFoldConfig:
    """Configuration for one LOSO fold in federated learning.

    Parameters:
    test_subject : int
        Subject ID held out for final evaluation.
    val_subject : int | None
        Subject ID used for validation / early stopping.
        If None, determined by rotating selection
    train_subjects : list[int] | None
        Subjects participating in federated training.
        If None, automatically computed as ALL - {test, val}.
    """

    test_subject: int
    val_subject: int | None = None
    train_subjects: list[int] | None = None

    def __post_init__(self):
        """Compute train_subjects and val_subject if not provided.

        NEW SEMANTICS (held-out-with-calibration protocol):
          - test_subject is NOT included in train_subjects.
          - val_subject is NOT included in train_subjects either; they
            are held out as a validation-only source.  This matches the
            centralised training design (wesad_cnnlstm_train.py) and
            avoids the minor leakage that would arise from using the same
            subject's overlapped windows in training while their
            non-overlapped windows drive LR scheduling / early-stopping.
          - Federation therefore runs across 13 training subjects.

        Participation counts per fold:
          - 1 test subject (held out, calibration + evaluation after fed)
          - 1 val subject  (held out for LR scheduling / eval monitoring)
          - 13 training subjects (federation participants)
          total: 15 subjects, each serving each role in exactly one fold.
        """
        # Guard: val_subject must not equal test_subject (data leakage)
        if self.val_subject is not None and self.val_subject == self.test_subject:
            raise ValueError(
                f"val_subject ({self.val_subject}) cannot equal "
                f"test_subject ({self.test_subject})! this would cause "
                f"data leakage (same subject used for validation and testing)."
            )

        remaining = [s for s in ALL_SUBJECT_IDS if s != self.test_subject]

        if self.val_subject is None:
            # Rotating validation: next subject after test in sorted list
            test_idx = ALL_SUBJECT_IDS.index(self.test_subject)
            val_idx = (test_idx + 1) % len(ALL_SUBJECT_IDS)
            self.val_subject = ALL_SUBJECT_IDS[val_idx]
            if self.val_subject == self.test_subject:
                self.val_subject = remaining[0]

        if self.train_subjects is None:
            # 13 training participants: all subjects EXCEPT test and val.
            self.train_subjects = [s for s in remaining if s != self.val_subject]

        # Sanity: test subject must never appear in train_subjects
        # under the held-out protocol.
        if self.test_subject in self.train_subjects:
            raise ValueError(
                f"test_subject ({self.test_subject}) must not appear in "
                f"train_subjects under the held-out-with-calibration "
                f"protocol.  Got train_subjects={self.train_subjects}."
            )
        if self.val_subject in self.train_subjects:
            raise ValueError(
                f"val_subject ({self.val_subject}) must not appear in "
                f"train_subjects under the held-out-with-calibration "
                f"protocol.  Got train_subjects={self.train_subjects}."
            )

    @property
    def num_clients(self) -> int:
        """Number of clients participating in federated training."""
        assert self.train_subjects is not None  # fix mypy error
        return len(self.train_subjects)

    @property
    def all_participating_subjects(self) -> list[int]:
        """All subjects that participate as FL clients (= train_subjects)."""
        assert self.train_subjects is not None  # fix mypy eror
        return list(self.train_subjects)


def generate_all_fold_configs() -> list[FLFoldConfig]:
    """Generate FLFoldConfig for all 15 LOSO folds.

    Returns a list of 15 configs, one per test subject.
    each fold has 14 training clients
    (13 pure-train + 1 test-subject-as-client) and 1 validation subject.
    """
    return [FLFoldConfig(test_subject=s) for s in ALL_SUBJECT_IDS]


# Per-Client Dataset Builder


class FLClientDataManager:
    """Manages datasets and DataLoaders for a single FL client.

    Each client (subject) has:
      - train_loader: their LOCAL training data (windows from this subject only)
      - val_loader:   validation data (from the fold's val subject, shared across
                      clients for early stopping or from the client's own data
                      if this client IS the val subject)
      - test_loader:  test data (only meaningful for the test subject)

    This mirrors PFLIB_clientbase.py's load_train_data() / load_test_data()
    interface, but adapted for WESAD's LOSO structure where each client issone subject.

    Parameters:
    subject_id : int
        This client's subject ID.
    fold_config : FLFoldConfig
        The LOSO fold configuration.
    windowed_dir : str
        Root of the windowed data.
    device_type : str
        "chest" or "wrist".
    window_sec : float
        Window length used during windowing.
    label_map : dict
        Label mapping (LABEL_MAP_BINARY or LABEL_MAP_3CLASS).
    batch_size : int
        Batch size for DataLoaders.
    noise_std : float
        Gaussian noise std for training augmentation.
    scale_range : tuple
        Amplitude scaling range for training augmentation.
    signals_to_load : list[str] | None
        Specific signals to load.  None = all.
    num_workers : int
        DataLoader workers.
    """

    def __init__(
        self,
        subject_id: int,
        fold_config: FLFoldConfig,
        windowed_dir: str = "data/dl_windowed/",
        device_type: str = "wrist",
        window_sec: float = 16.0,
        label_map: dict | None = None,
        batch_size: int = 64,
        noise_std: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
        signals_to_load: list[str] | None = None,
        num_workers: int = 0,
    ):
        self.subject_id = subject_id
        self.fold_config = fold_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_map = label_map or LABEL_MAP_BINARY
        self._pin_memory = torch.cuda.is_available()

        common_kwargs: _WESADCommonKwargs = {
            "windowed_dir": windowed_dir,
            "device_type": device_type,
            "window_sec": window_sec,
            "label_map": self.label_map,
            "signals_to_load": signals_to_load,
        }

        # Local training data (this subject's windows only).
        # Under the held-out protocol, the test subject is NOT an FL
        # client, so this code path is only reached for training
        # subjects.  Their full set of overlapped + non-overlapped
        # windows is used for local training (standard Collins et al.
        # 2021 FedRep setup).
        self._train_dataset = WESADDLDataset(
            subject_ids=[subject_id],
            mode="train",
            noise_std=noise_std,
            scale_range=scale_range,
            **common_kwargs,
        )

        # Validation data
        # val_subject is held out of federation under the new protocol,
        # so it's never the same as subject_id (this client).  Every
        # training client uses the same shared val subject for
        # LR scheduling / early-stopping metrics.
        assert fold_config.val_subject is not None
        assert subject_id != fold_config.val_subject, (
            f"val_subject {fold_config.val_subject} should never be "
            f"instantiated as an FL client under the held-out protocol"
        )

        self._val_dataset = WESADDLDataset(
            subject_ids=[fold_config.val_subject],
            mode="val",
            noise_std=0.0,
            scale_range=(1.0, 1.0),
            **common_kwargs,
        )

        # Test data is removed from the per-client interface because
        # under the held-out protocol, no FL client is the test subject.
        # Test-subject evaluation is handled separately via
        # build_test_subject_splits() + the server's calibration step.
        # Keeping an empty placeholder for legacy compatibility.
        self._test_dataset = WESADDLDataset(
            subject_ids=[subject_id],
            mode="test",
            noise_std=0.0,
            scale_range=(1.0, 1.0),
            **common_kwargs,
        )

    @property
    def train_samples(self) -> int:
        """Number of training windows for this client.

        Used by the server for weighted aggregation
        (referece PFLIB_serverbase.py receive_models()).
        """
        return len(self._train_dataset)

    @property
    def test_samples(self) -> int:
        return len(self._test_dataset)

    def get_class_weights(self) -> torch.Tensor:
        """Get inverse-frequency class weights from this client's training data.

        Returns a tensor of shape (n_classes,) for CrossEntropyLoss.
        equiv to WESADDLDataset.get_class_weights().
        """
        return self._train_dataset.get_class_weights()

    def load_train_data(self, batch_size: int | None = None) -> DataLoader:
        """Return a DataLoader over this client's local training windows.
        we mirror PFLIB_clientbase.py load_train_data().

        Parameters:
        batch_size : int | None
            None uses the default from __init__.
        """
        bs = batch_size or self.batch_size
        return DataLoader(
            self._train_dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory,
        )

    def load_val_data(self, batch_size: int | None = None) -> DataLoader:
        """Return a DataLoader over validation data for early stopping."""
        bs = batch_size or self.batch_size
        return DataLoader(
            self._val_dataset,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory,
        )

    def load_test_data(self, batch_size: int | None = None) -> DataLoader:
        """Return a DataLoader over this client's test windows.
        Mirroring PFLIB_clientbase.py's load_test_data().
        """
        bs = batch_size or self.batch_size
        return DataLoader(
            self._test_dataset,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory,
        )

    def get_label_distribution(
        self,
    ) -> dict[
        int, int
    ]:  # expect return {class_id: count} for this client's training data
        return self._train_dataset.get_label_distribution()


# Fold-Level Dataset Builder


def build_fl_fold_datasets(
    fold_config: FLFoldConfig,
    windowed_dir: str = "data/dl_windowed/",
    device_type: str = "wrist",
    window_sec: float = 16.0,
    label_map: dict | None = None,
    batch_size: int = 64,
    noise_std: float = 0.05,
    scale_range: tuple[float, float] = (0.9, 1.1),
    signals_to_load: list[str] | None = None,
    num_workers: int = 0,
) -> dict[int, FLClientDataManager]:
    """Build FLClientDataManager for every participating subject in a fold.

    Parameters:
    fold_config : FLFoldConfig
        LOSO fold configuration.

    Returns:
    dict[int, FLClientDataManager]
        subject_id → data manager.  Includes ALL participating clients
    """
    managers = {}

    for sid in fold_config.all_participating_subjects:
        managers[sid] = FLClientDataManager(
            subject_id=sid,
            fold_config=fold_config,
            windowed_dir=windowed_dir,
            device_type=device_type,
            window_sec=window_sec,
            label_map=label_map,
            batch_size=batch_size,
            noise_std=noise_std,
            scale_range=scale_range,
            signals_to_load=signals_to_load,
            num_workers=num_workers,
        )

    # Summary
    print(
        f"\n  FL Fold Setup (test=S{fold_config.test_subject}, "
        f"val=S{fold_config.val_subject})"
    )
    print(
        f"    Clients: {fold_config.num_clients} "
        f"({fold_config.all_participating_subjects})"
    )
    for sid, mgr in managers.items():
        # Test subject is not an FL client under held-out protocol; only
        # the val_subject tag appears here now.
        role = " [VAL-source]" if sid == fold_config.val_subject else ""
        print(
            f"      S{sid}{role}: {mgr.train_samples} train windows, "
            f"dist={mgr.get_label_distribution()}"
        )

    return managers


# Test-subject calibration / evaluation split


def build_test_subject_splits(
    test_subject: int,
    windowed_dir: str = "data/dl_windowed/",
    device_type: str = "wrist",
    window_sec: float = 16.0,
    label_map: dict | None = None,
    signals_to_load: list[str] | None = None,
    calibration_fraction: float = 0.20,
    verbose: bool = True,
) -> tuple[WESADDLDataset, WESADDLDataset]:
    """Split the test subject's non-overlapped windows into calibration
    and evaluation sets, stratified by contiguous condition block.

    Protocol (Collins et al. 2021 held-out client evaluation; see
    FIX_PFL_PLAN):
      - Load only the test subject's non-overlapped windows (mode=test).
      - Group consecutive windows into "condition blocks": a block is a
        maximal contiguous run of windows with the same label (in the
        order they were recorded, which is the WESAD protocol order
        baseline -> stress -> amusement -> ... ).
      - From each block, take the first calibration_fraction (default
        20%) of windows as calibration, and the remaining
        (1 - calibration_fraction) as evaluation.
      - Return (calibration_dataset, evaluation_dataset).

    The split is TEMPORAL (chronological) within each block and
    STRATIFIED across blocks, so both sets preserve class proportions
    while also preserving the temporal structure (calibration windows
    come from the first moments of each phase, evaluation windows from
    later moments).  Because we use non-overlapped windows only, the
    two sets have no signal overlap: each 16-s window's samples are
    disjoint from every other window's samples in the same subject.

    Parameters:
    test_subject : int
        Held-out subject's ID.
    windowed_dir : str
        Root of the windowed data tree.
    device_type : str
        "chest" or "wrist".
    window_sec : float
        Window length used during windowing.
    label_map : dict | None
        Label mapping.  Defaults to LABEL_MAP_BINARY.
    signals_to_load : list[str] | None
        Specific signals.  None = all.
    calibration_fraction : float
        Fraction of each condition block to allocate to the calibration
        set.  Default 0.20 (matches the KD plan and Cross_Modal_KD plan).
    verbose : bool
        If True, print a summary of the split.

    Returns:
    (calibration_dataset, evaluation_dataset) : tuple[WESADDLDataset, WESADDLDataset]
        Two datasets over disjoint subsets of the test subject's
        non-overlapped windows.  Both use mode="test" (no augmentation,
        no noise; only instance Z-score normalisation).
    """
    import numpy as np

    from ..data.wesad_dl_dataset import WESADDLDataset

    if label_map is None:
        label_map = LABEL_MAP_BINARY

    # Step 1: load the test subject's non-overlapped windows in
    # recording order, so we can extract per-window timing and labels
    # to plan the block-stratified split.
    full = WESADDLDataset(
        subject_ids=[test_subject],
        windowed_dir=windowed_dir,
        device_type=device_type,
        window_sec=window_sec,
        mode="test",
        label_map=label_map,
        noise_std=0.0,
        scale_range=(1.0, 1.0),
        signals_to_load=signals_to_load,
    )

    if len(full) == 0:
        raise ValueError(
            f"build_test_subject_splits: no non-overlapped windows "
            f"found for S{test_subject}"
        )

    labels = full._labels  # mapped integer labels
    start_secs = full.get_start_sec()
    orig_idx = full.get_original_indices()

    # Step 2: sort windows chronologically (they should already be,
    # but make it explicit for safety).
    order = np.argsort(start_secs, kind="stable")
    labels = labels[order]
    start_secs = start_secs[order]
    orig_idx = orig_idx[order]

    # Step 3: identify contiguous condition blocks.  A block ends when
    # the label changes between consecutive windows.
    block_boundaries = np.concatenate(
        [[0], np.where(np.diff(labels) != 0)[0] + 1, [len(labels)]]
    )  # e.g. [0, 34, 79, 131] for three blocks of sizes 34, 45, 52

    calib_orig: list[int] = []
    eval_orig: list[int] = []
    block_report: list[tuple[int, int, int, int]] = []  # (label, start, cal_n, eval_n)

    for bi in range(len(block_boundaries) - 1):
        lo = block_boundaries[bi]
        hi = block_boundaries[bi + 1]
        block_len = hi - lo
        block_label = int(labels[lo])
        # Ceil so small blocks still contribute at least 1 calibration window
        n_calib = max(1, int(np.ceil(block_len * calibration_fraction)))
        n_calib = min(n_calib, block_len - 1) if block_len > 1 else block_len
        n_eval = block_len - n_calib

        # First n_calib chronologically -> calibration
        calib_orig.extend(orig_idx[lo : lo + n_calib].tolist())
        # Remaining -> evaluation
        if n_eval > 0:
            eval_orig.extend(orig_idx[lo + n_calib : hi].tolist())

        block_report.append((block_label, block_len, n_calib, n_eval))

    calib_orig_arr = np.array(sorted(calib_orig), dtype=np.int64)
    eval_orig_arr = np.array(sorted(eval_orig), dtype=np.int64)

    # Sanity: no overlap between the two sets
    overlap = np.intersect1d(calib_orig_arr, eval_orig_arr)
    if overlap.size > 0:
        raise RuntimeError(
            f"build_test_subject_splits: calibration and evaluation "
            f"sets overlap on window indices {overlap.tolist()[:5]}..."
        )

    # Step 4: construct the two datasets using window_indices filtering.
    calibration_dataset = WESADDLDataset(
        subject_ids=[test_subject],
        windowed_dir=windowed_dir,
        device_type=device_type,
        window_sec=window_sec,
        mode="test",  # no augmentation
        label_map=label_map,
        noise_std=0.0,
        scale_range=(1.0, 1.0),
        signals_to_load=signals_to_load,
        window_indices={test_subject: calib_orig_arr},
    )
    evaluation_dataset = WESADDLDataset(
        subject_ids=[test_subject],
        windowed_dir=windowed_dir,
        device_type=device_type,
        window_sec=window_sec,
        mode="test",
        label_map=label_map,
        noise_std=0.0,
        scale_range=(1.0, 1.0),
        signals_to_load=signals_to_load,
        window_indices={test_subject: eval_orig_arr},
    )

    if verbose:
        print(
            f"\n  Test-subject split (S{test_subject}, "
            f"calibration_fraction={calibration_fraction:.2f})"
        )
        print(f"    Total non-overlapped windows: {len(full)}")
        print(
            f"    Calibration windows: {len(calibration_dataset)} "
            f"({100*len(calibration_dataset)/max(len(full),1):.1f}%)"
        )
        print(
            f"    Evaluation windows:  {len(evaluation_dataset)} "
            f"({100*len(evaluation_dataset)/max(len(full),1):.1f}%)"
        )
        print(
            f"    Calibration class distribution: "
            f"{calibration_dataset.get_label_distribution()}"
        )
        print(
            f"    Evaluation  class distribution: "
            f"{evaluation_dataset.get_label_distribution()}"
        )
        print("    Condition blocks (label, length, cal, eval):")
        for lbl, blen, ncal, neval in block_report:
            print(f"      label={lbl}: n={blen}  cal={ncal}  eval={neval}")

    return calibration_dataset, evaluation_dataset

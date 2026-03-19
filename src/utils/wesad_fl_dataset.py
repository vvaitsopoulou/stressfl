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
#    For each LOSO fold with test subject S, the validation subject is
#    selected by deterministic rotation: the subject immediately after S
#    in the sorted subject list (wrapping around).  This means every
#    subject serves as validation in exactly one fold, so no subject is
#    systematically underrepresented in training.
#
#    This is already implemented in wesad_dl_dataset.build_loso_datasets()
#
# 2. Federation Participation  : TEST SUBJECT PARTICIPATES
#
#    The test subject participates in federated training as a regular
#    client (receives global base, trains locally, uploads base updates).
#    At evaluation time, the test subject is evaluated using their
#    personalised model (global base + local head for FedRep, personal
#    model for Ditto, hypernetwork-generated params for PeFLL, etc.).
#
#    This is the proper PFL evaluation regime (Collins et al. 2021,
#    Li et al. 2021) and is used uniformly for ALL methods, including
#    FedAvg and FedProx baselines.


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
        """Compute train_subjects and val_subject if not provided."""
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
            self.train_subjects = [s for s in remaining if s != self.val_subject]

        # as per our decisiontest subject is INCLUDED in train_subjects
        # (participates in federation, evaluates on local personalised model)
        if self.test_subject not in self.train_subjects:
            self.train_subjects.append(self.test_subject)
            self.train_subjects.sort()

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

        # Local training data (this subject's windows only)
        self._train_dataset = WESADDLDataset(
            subject_ids=[subject_id],
            mode="train",
            noise_std=noise_std,
            scale_range=scale_range,
            **common_kwargs,
        )

        # Validation data
        # If this client IS the validation subject, their val data is their
        # own non-overlapped windows.  Otherwise, val data comes from the
        # fold's designated val subject.
        assert fold_config.val_subject is not None
        val_subject_for_this_client: int = (
            subject_id
            if subject_id == fold_config.val_subject
            else fold_config.val_subject
        )

        self._val_dataset = WESADDLDataset(
            subject_ids=[val_subject_for_this_client],
            mode="val",
            noise_std=0.0,
            scale_range=(1.0, 1.0),
            **common_kwargs,
        )

        # ── Test data (this subject's non-overlapped windows) ──
        # Only meaningful when subject_id == fold_config.test_subject,
        # but we always create it for uniform interface.
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
        role = ""
        if sid == fold_config.test_subject:
            role = " [TEST]"
        elif sid == fold_config.val_subject:
            role = " [VAL-source]"
        print(
            f"      S{sid}{role}: {mgr.train_samples} train windows, "
            f"dist={mgr.get_label_distribution()}"
        )

    return managers

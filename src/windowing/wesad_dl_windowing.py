# DL Windowing Pipeline

# Segments the per-subject .npy arrays produced by wesad_dl_preprocess.py into
# fixed-length windows for the CNN-LSTM pipeline.

# Key differences from the RF windowing module (wesad_windowing.py):
#   • Multi-rate windowing: each modality is at a different sampling rate.
#     A single temporal window (e.g. 16 s) produces arrays of different
#     lengths per signal (ECG: 4096 @ 256 Hz, BVP: 512 @ 32 Hz, etc.).
#   • Per-subject input/output: no merged dataframe.  Each subject's
#     signals are loaded and windowed independently.
#   • Asymmetric overlap: high-overlap windows for training (0.25 s stride),
#     non-overlapping windows for validation/test.  This is the single most
#     impactful anti-overfitting decision after early stopping.
#   • No normalization or augmentation: windows are stored as raw signal
#     slices.  Instance Z-score normalization and on-the-fly augmentation
#     are applied downstream in wesad_dl_dataset.py.  This preserves
#     flexibility for ablation (e.g. window-level vs recording-level
#     Z-score) without re-running windowing.
#   • Strict label purity: windows spanning label transitions are rejected.
#     No majority vote: ambiguous windows introduce contradictory training
#     signal on a 15-subject dataset where every clean sample matters.

# Window Length Decision (from literature review):
#   • 16 s (default): fits one complete EDA SCR cycle (latency + rise +
#     recovery).  Moser et al. (2024) used 16 s for EDA stress detection.
#   • 10 s (ablation): matches Prajod & André (2022) and Hwang et al. (2018)
#     who demonstrated DL stress classification from 10 s ECG windows.
#   • 30 s (ablation): tests whether longer temporal context improves
#     classification at the cost of fewer training samples.

# Overlap / Stride:
#   • Training: 0.25 s stride.  With a 16 s window this produces ~64×
#     more windows than non-overlapping.  This is the primary data
#     augmentation strategy and does NOT require a GAN.
#     Citation: Ordóñez & Roggen (2016) used dense windowing for DL training.
#   • Test: non-overlapping (stride = window length).  Overlapping test
#     windows create correlated samples and inflate performance metrics.

# Active Labels:
#   1 = baseline, 2 = stress, 3 = amusement
#   Labels 0, 4, 5, 6, 7 (not defined / transient / meditation) are
#   filtered out

# Output Structure (per subject, per window length):
#     data/dl_windowed/{window_len}s/
#         S{id}/
#             chest/
#                 windows.npz     — dict of signal_name → (n_windows, n_samples[, channels])
#                 metadata.csv    — per-window: window_idx, label, start_s, end_s, is_overlapped
#             wrist/
#                 windows.npz
#                 metadata.csv

# References:
#   Moser et al. (2024) — 16 s window for EDA stress detection
#   Prajod & André (2022) — 10 s window for DL ECG
#   Hwang et al. (2018) — 10 s ultra-short-term ECG DL
#   Ordóñez & Roggen (2016) — high-overlap training as data augmentation
#   Schmidt et al. (2018) — WESAD dataset, label definitions

import csv
from dataclasses import dataclass
import os

import numpy as np

#  Signal definitions:
# Each entry: (filename_without_.npy, target_hz, n_channels)
# n_channels = 1 for scalar signals, 3 for ACC axes

CHEST_SIGNALS = [
    ("ecg", 256, 1),
    ("emg", 64, 1),
    ("c_eda", 8, 1),
    ("resp", 32, 1),
    ("c_acc", 32, 3),
    ("c_temp", 32, 1),  # scalar feature stored for optional use at fusion layer
]

WRIST_SIGNALS = [
    ("bvp", 32, 1),
    ("w_eda", 4, 1),
    ("w_acc", 32, 3),
    ("w_temp", 4, 1),  # scalar feature stored for optional use at fusion layer
]

# Labels considered valid for stress detection
ACTIVE_LABELS = {1.0, 2.0, 3.0}  # baseline, stress, amusement

#  Data class for window metadata


@dataclass
class WindowMeta:
    """Metadata for a single temporal window."""

    window_idx: int
    label: float
    start_sample: int  # index into the label array at its rate
    end_sample: int
    start_sec: float  # wall-clock seconds from recording start
    end_sec: float
    is_overlapped: bool  # True if this window overlaps with a non-overlapped window


#  Label run detection (carried over from RF windowing)


def _get_label_runs(labels: np.ndarray) -> list[tuple[float, int, int]]:
    # Identify contiguous runs of the same label value.

    # Returns list of (label_value, start_idx, end_idx) tuples.
    # end_idx is exclusive.

    # This is the same optimisation used in the RF windowing module:
    # instead of sliding blindly and checking label purity after,
    # we identify contiguous segments and only create windows within them.
    # This avoids most rejected windows and is much faster.
    if len(labels) == 0:
        return []

    runs = []
    current_label = labels[0]
    run_start = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            runs.append((float(current_label), run_start, i))
            current_label = labels[i]
            run_start = i

    runs.append((float(current_label), run_start, len(labels)))
    return runs


#  Core: window one subject for one device (chest or wrist)


def _window_subject_device(
    subject_dir: str,
    signal_defs: list[tuple[str, int, int]],
    window_sec: float,
    slide_sec: float,
) -> tuple[dict[str, np.ndarray], list[WindowMeta]]:
    # Window all signals for one subject and one device (chest or wrist).

    # Parameters
    # subject_dir : str
    #     Path to the subject's directory containing .npy files.
    # signal_defs : list of (name, hz, n_channels)
    #     Signal definitions for this device.
    # window_sec : float
    #     Window length in seconds.
    # slide_sec : float
    #     Stride in seconds.

    # Returns
    # signal_windows : dict[str, np.ndarray]
    #     signal_name → array of shape (n_windows, window_samples) for 1-ch
    #     or (n_windows, window_samples, 3) for 3-ch signals.
    # metadata : list[WindowMeta]
    #     One entry per accepted window.

    # Load all signals and their label arrays
    signals = {}
    label_arrays = {}

    for sig_name, _hz, _n_ch in signal_defs:
        sig_path = os.path.join(subject_dir, f"{sig_name}.npy")
        lbl_path = os.path.join(subject_dir, f"label_{sig_name}.npy")

        if not os.path.exists(sig_path):
            print(f"      Missing {sig_path} — skipping signal")
            continue
        if not os.path.exists(lbl_path):
            print(f"      Missing {lbl_path} — skipping signal")
            continue

        signals[sig_name] = np.load(sig_path)
        # Cast labels to float32 to guarantee consistent dtype.
        # WESAD raw labels are integers, ACTIVE_LABELS uses floats.
        # Without this cast, checks could silently fail
        # if a future refactor removes the float() cast in _get_label_runs.
        label_arrays[sig_name] = np.load(lbl_path).astype(np.float32)

    if not signals:
        return {}, []

    #  Use the first signal's label array as the reference for
    #    window placement.  All label arrays represent the same
    #    recording, they differ only in sampling rate.  We pick the
    #    one with the highest rate for label boundary detection.
    ref_name, ref_hz, _ = max(
        [(n, h, c) for n, h, c in signal_defs if n in signals],
        key=lambda x: x[1],
    )
    ref_labels = label_arrays[ref_name]

    # Window/slide in samples at the reference rate
    window_samples_ref = int(window_sec * ref_hz)
    slide_samples_ref = int(slide_sec * ref_hz)

    # Identify contiguous label runs at the reference rate
    label_runs = _get_label_runs(ref_labels)

    # slide through runs and collect valid window positions
    accepted_positions = []  # list of (start_idx_ref, label, is_overlapped)

    rejected = {"inactive_label": 0, "mixed_label": 0, "short_tail": 0}

    for label_val, run_start, run_end in label_runs:
        if label_val not in ACTIVE_LABELS:
            rejected["inactive_label"] += run_end - run_start
            continue

        pos = run_start
        while pos + window_samples_ref <= run_end:
            # Strict label purity check (sshould always pass
            # because we're inside a contiguous run)
            window_labels = ref_labels[pos : pos + window_samples_ref]
            if len(set(window_labels.tolist())) != 1:
                rejected["mixed_label"] += 1
                pos += slide_samples_ref
                continue

            # Determine if this window is overlapped (non-independent)
            relative_pos = pos - run_start
            is_overlapped = (relative_pos % window_samples_ref) != 0

            accepted_positions.append((pos, label_val, is_overlapped))
            pos += slide_samples_ref

        # Count short tails
        if run_end - pos > 0 and run_end - pos < window_samples_ref:
            rejected["short_tail"] += 1

    if not accepted_positions:
        return {}, []

    # Extract signal windows at each accepted position
    signal_windows: dict[str, list[np.ndarray]] = {name: [] for name in signals}
    metadata: list[WindowMeta] = []

    for win_idx, (ref_start, label_val, is_overlapped) in enumerate(accepted_positions):
        # Convert reference position to wall-clock seconds
        start_sec = ref_start / ref_hz
        end_sec = start_sec + window_sec

        for sig_name, hz, _n_ch in signal_defs:
            if sig_name not in signals:
                continue

            sig_data = signals[sig_name]
            expected_samples = int(window_sec * hz)

            # Convert from reference time to this signal's sample space
            sig_start = int(start_sec * hz)
            sig_end = sig_start + expected_samples

            # Bounds check
            if sig_end > len(sig_data):
                break

            chunk = sig_data[sig_start:sig_end]

            # Verify exact length (rounding guard)
            if len(chunk) != expected_samples:
                break

            signal_windows[sig_name].append(chunk)
        else:
            # Only add metadata if ALL signals were successfully extracted
            metadata.append(
                WindowMeta(
                    window_idx=win_idx,
                    label=label_val,
                    start_sample=ref_start,
                    end_sample=ref_start + window_samples_ref,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    is_overlapped=is_overlapped,
                )
            )
            continue

        # If we broke out of the signal loop, remove partial entries
        for sig_name in signal_windows:
            if len(signal_windows[sig_name]) > len(metadata):
                signal_windows[sig_name].pop()

    # Stack into arrays
    stacked: dict[str, np.ndarray] = {}
    for sig_name in signal_windows:
        if signal_windows[sig_name]:
            stacked[sig_name] = np.stack(signal_windows[sig_name], axis=0).astype(
                np.float32
            )

    n_final = len(metadata)
    n_overlapped = sum(1 for m in metadata if m.is_overlapped)
    n_independent = n_final - n_overlapped

    # Label distribution
    label_counts: dict[float, int] = {}
    for m in metadata:
        label_counts[m.label] = label_counts.get(m.label, 0) + 1

    print(
        f"      Windows: {n_final} total ({n_independent} independent, "
        f"{n_overlapped} overlapped)"
    )
    print(f"      Labels: {label_counts}")
    print(
        f"      Rejected — inactive: {rejected['inactive_label']} samples, "
        f"mixed: {rejected['mixed_label']}, short: {rejected['short_tail']}"
    )

    return stacked, metadata


#  Save windowed data + metadata


def _save_windows(
    output_dir: str,
    signal_windows: dict[str, np.ndarray],
    metadata: list[WindowMeta],
):
    """Save windowed arrays as .npz and metadata as .csv."""
    os.makedirs(output_dir, exist_ok=True)

    # Save signal arrays
    np.savez_compressed(
        os.path.join(output_dir, "windows.npz"),
        **signal_windows,
    )

    # Save metadata CSV
    meta_path = os.path.join(output_dir, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_idx",
                "label",
                "start_sample",
                "end_sample",
                "start_sec",
                "end_sec",
                "is_overlapped",
            ]
        )
        for m in metadata:
            writer.writerow(
                [
                    m.window_idx,
                    m.label,
                    m.start_sample,
                    m.end_sample,
                    f"{m.start_sec:.4f}",
                    f"{m.end_sec:.4f}",
                    m.is_overlapped,
                ]
            )


# window one subject


def window_subject(
    subject_id: int,
    prepared_dir: str,
    output_dir: str,
    window_sec: float = 16.0,
    slide_sec: float = 0.25,
):
    """Window all signals for one subject (both chest and wrist).

    Parameters
    subject_id : int
        WESAD subject ID (e.g. 2, 3, ..., 17).
    prepared_dir : str
        Root of the DL-prepared data (output of wesad_dl_prepare.py).
    output_dir : str
        Root output directory for windowed data.
    window_sec : float
        Window length in seconds.
    slide_sec : float
        Stride in seconds.
    """
    subject_dir = os.path.join(prepared_dir, f"S{subject_id}")
    if not os.path.isdir(subject_dir):
        print(f"  ⚠  S{subject_id} not found in {prepared_dir} — skipping")
        return

    print(f"\n  S{subject_id} — window={window_sec}s, stride={slide_sec}s")

    for device, signal_defs in [("chest", CHEST_SIGNALS), ("wrist", WRIST_SIGNALS)]:
        print(f"    {device}:")
        out_dir = os.path.join(
            output_dir, f"{window_sec:.0f}s", f"S{subject_id}", device
        )

        signal_windows, metadata = _window_subject_device(
            subject_dir=subject_dir,
            signal_defs=signal_defs,
            window_sec=window_sec,
            slide_sec=slide_sec,
        )

        if metadata:
            _save_windows(out_dir, signal_windows, metadata)
            print(f"      Saved → {out_dir}")
        else:
            print("      No valid windows — nothing saved")


# window all subjects
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


def window_all(
    prepared_dir: str = "data/dl_prepared/",
    output_dir: str = "data/dl_windowed/",
    window_sec: float = 16.0,
    slide_sec: float = 0.25,
    subject_ids: list[int] | None = None,
):
    """Window all subjects for the DL pipeline.

    Parameters
    prepared_dir : str
        Root of the DL-prepared data.
    output_dir : str
        Root output directory for windowed data.
    window_sec : float
        Window length in seconds.
    slide_sec : float
        Stride in seconds.
    subject_ids : list[int], optional
        Subjects to process.  Defaults to all 15.
    """
    if subject_ids is None:
        subject_ids = SUBJECT_IDS

    overlap_pct = max(0, (1 - slide_sec / window_sec) * 100)

    print("  WESAD DL Windowing\n")
    print(f"  Input:      {prepared_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Window:     {window_sec} s")
    print(f"  Stride:     {slide_sec} s ({overlap_pct:.1f}% overlap)")
    print(f"  Subjects:   {subject_ids}")
    print(f"  Labels:     {sorted(ACTIVE_LABELS)}")
    print("  Label rule: strict purity (reject mixed windows)")
    print()

    # Print expected samples per signal for this window length
    print("  Samples per window:")
    for device, signal_defs in [("chest", CHEST_SIGNALS), ("wrist", WRIST_SIGNALS)]:
        print(f"    {device}:")
        for name, hz, n_ch in signal_defs:
            n_samples = int(window_sec * hz)
            ch_str = f" × {n_ch}" if n_ch > 1 else ""
            print(f"      {name:10s}  {n_samples:>6,} samples ({hz} Hz{ch_str})")
    print()

    for sid in subject_ids:
        window_subject(sid, prepared_dir, output_dir, window_sec, slide_sec)

    print("\n")
    print("  Windowing complete.")
    print(f"  Output: {output_dir}")


#  Window length ablation


def run_window_ablation(
    prepared_dir: str = "data/dl_prepared/",
    output_dir: str = "data/dl_windowed/",
    window_lengths: list[float] | None = None,
    slide_sec: float = 0.25,
    subject_ids: list[int] | None = None,
):
    """Run windowing at multiple window lengths for ablation study.

    Parameters
    window_lengths : list[float]
        Window durations to test (in seconds).
        Default: [10.0, 16.0, 30.0]
    """
    if window_lengths is None:
        window_lengths = [10.0, 16.0, 30.0]

    print("  WINDOW LENGTH ABLATION")
    print(f"  Testing: {window_lengths} seconds")

    for wl in window_lengths:
        print(f"\n{'─' * 60}")
        print(f"  Window length: {wl} s")
        print(f"{'─' * 60}")
        window_all(
            prepared_dir=prepared_dir,
            output_dir=output_dir,
            window_sec=wl,
            slide_sec=slide_sec,
            subject_ids=subject_ids,
        )

    print("\n")
    print("  Ablation complete.")
    print(f"  Results in: {output_dir}")
    print("  Subdirectories: " + ", ".join(f"{wl:.0f}s/" for wl in window_lengths))
    print("=")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WESAD DL Windowing Pipeline")
    parser.add_argument(
        "--prepared-dir",
        default="data/dl_prepared/",
        help="Path to DL-prepared per-subject arrays",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dl_windowed/",
        help="Output directory for windowed data",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=16.0,
        help="Window length in seconds (default: 16)",
    )
    parser.add_argument(
        "--slide",
        type=float,
        default=0.25,
        help="Stride in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run window length ablation (10s, 16s, 30s)",
    )

    args = parser.parse_args()

    if args.ablation:
        run_window_ablation(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            slide_sec=args.slide,
        )
    else:
        window_all(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            window_sec=args.window,
            slide_sec=args.slide,
        )

# Paired Data Generator for Cross-Modal Knowledge Distillation

# Purpose:

# Process all 15 WESAD subjects to create aligned (teacher_input,
# student_features, label) triplets for KD training.  For each 16 s
# window, this module produces:

#     Teacher signals: raw wrist signals (BVP, EDA, ACC, TEMP) at their DL-target rates, stored as arrays for later teacher inference.
#     Student features: 6 time-domain HRV features extracted from BVP-derived IBIs (the student's input modality).
#     Student features (noisy): same features extracted after simulating Fitbit HR degradation (noise + bias + quantisation).
#     Labels: binary stress labels (majority vote per window).

# The paired data follows the same-source-different-processing paradigm
# used in PULSE (2025, arXiv:2510.24058): the same BVP window is processed
# two ways: through the teacher's CNN pathway (raw signal) and through the student's HRV pathway (peak detection → IBI → features).

# Data Source: Reads from the DL-windowed .npz files produced by wesad_dl_windowing.py which are the same files the CNN-LSTM teacher
# was trained on.  This guarantees byte-identical teacher inputs.

# IMPORTANT: This module does not cache teacher logits.  Teacher inference
# is performed in next phase (kd_trainer.py) because different
# LOSO folds use different teacher checkpoints.  The .npz files
# produced here include a placeholder key teacher_logits that is
# filled during training phase3.

# Output Format:
# Per-subject .npz files in KD_PAIRED_DIR containing::

#     bvp:                   (N, 512)       -> raw BVP for teacher
#     w_eda:                 (N, 64)        -> raw EDA for teacher
#     w_acc:                 (N, 512, 3)    -> raw ACC for teacher (stored T,C)
#     w_temp:                (N, 64)        -> raw TEMP for teacher
#     student_features:      (N, 6)         -> clean HRV features
#     student_features_noisy:(N, 6)         -> noise-degraded HRV features
#     labels:                (N,)           -> binary integer labels
#     window_is_overlapped:  (N,)           -> overlap flag from windowing
#     peak_counts:           (N,)           -> peaks per window (diagnostics)
#     valid_ibi_counts:      (N,)           -> valid IBIs per window

# References
# ----------
# [PULSE25] arXiv:2510.24058 -> same-source-different-processing
# Elgendi 2013, PLoS ONE -> PPG peak detection
# Castaldo et al. 2019, BMC Med Inform -> ultra-short HRV
# Fuller et al. 2022 -> Fitbit HR accuracy meta-analysis
# Nissen et al. 2022, DOI:10.2196/33635 -> wearable HR noise

from __future__ import annotations

import csv
import os
import time
import warnings

import neurokit2 as nk
import numpy as np

from src.kd.hrv_feature_extractor import (
    clean_ibi_series,
    extract_primary_features,
    features_to_array,
    is_valid_window,
)
from src.kd.kd_config import (
    ACC_SAMPLES_PER_WINDOW,
    ACTIVE_LABELS,
    ALL_SUBJECT_IDS,
    BVP_SAMPLES_PER_WINDOW,
    BVP_SAMPLING_RATE,
    EDA_SAMPLES_PER_WINDOW,
    HR_BIAS_BPM,
    HR_CLIP_HIGH,
    HR_CLIP_LOW,
    HR_NOISE_STD_BPM,
    KD_PAIRED_DIR,
    LABEL_MAP_BINARY,
    N_PRIMARY_FEATURES,
    PRIMARY_HRV_FEATURES,
    TEMP_SAMPLES_PER_WINDOW,
    WESAD_WINDOWED_DIR,
    WINDOW_SEC,
    WRIST_SIGNALS,
    ensure_dirs,
    get_paired_data_path,
    validate_subject_id,
)

# SIGNAL LOADING


def _load_windowed_subject(
    subject_id: int,
    windowed_dir: str = WESAD_WINDOWED_DIR,
    window_sec: float = WINDOW_SEC,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Load all windowed wrist signals and metadata for one subject.

    Reads from the exact same .npz + metadata.csv files that the
    CNN-LSTM teacher was trained on, guaranteeing identical inputs.

    Parameters:
    subject_id : int
        WESAD subject ID.
    windowed_dir : str
        Root of the DL-windowed data tree.
    window_sec : float
        Window length used during windowing.

    Returns:
    signals : dict[str, np.ndarray]
        signal_name → (N_windows, T[, C]) array.
    metadata : list[dict]
        Per-window metadata dicts with keys: window_idx, label,
        is_overlapped, start_sec, end_sec.
    """
    subj_dir = os.path.join(
        windowed_dir, f"{window_sec:.0f}s", f"S{subject_id}", "wrist"
    )
    npz_path = os.path.join(subj_dir, "windows.npz")
    meta_path = os.path.join(subj_dir, "metadata.csv")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Windowed data not found: {npz_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    # Load signals
    npz = np.load(npz_path)
    signals = {name: npz[name] for name in npz.files}

    # Load metadata
    metadata = []
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(
                {
                    "window_idx": int(row["window_idx"]),
                    "label": float(row["label"]),
                    "is_overlapped": row["is_overlapped"] == "True",
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                }
            )

    # Validate shapes
    n_windows = len(metadata)
    for sig_name, hz, n_ch in WRIST_SIGNALS:
        if sig_name not in signals:
            raise KeyError(
                f"S{subject_id}: signal '{sig_name}' missing from {npz_path}. "
                f"Available: {list(signals.keys())}"
            )
        expected_T = int(window_sec * hz)
        actual_shape = signals[sig_name].shape
        if actual_shape[0] != n_windows:
            raise ValueError(
                f"S{subject_id} {sig_name}: expected {n_windows} windows, "
                f"got {actual_shape[0]}"
            )
        if n_ch == 1 and actual_shape[1] != expected_T:
            raise ValueError(
                f"S{subject_id} {sig_name}: expected {expected_T} samples, "
                f"got {actual_shape[1]}"
            )
        elif n_ch > 1 and (
            len(actual_shape) < 3
            or actual_shape[1] != expected_T
            or actual_shape[2] != n_ch
        ):
            raise ValueError(
                f"S{subject_id} {sig_name}: expected "
                f"({n_windows}, {expected_T}, {n_ch}), got {actual_shape}"
            )

    return signals, metadata


# PEAK DETECTION & IBI EXTRACTION


def _detect_peaks_and_extract_ibi(
    bvp_window: np.ndarray,
    sampling_rate: int = BVP_SAMPLING_RATE,
) -> tuple[np.ndarray, int]:
    """Detect peaks in a BVP window and compute IBI series.

    Uses NeuroKit2's ppg_findpeaks() with the Elgendi (2013) algorithm,
    which is the standard for photoplethysmographic peak detection.

    Parameters
    ----------
    bvp_window : np.ndarray
        Raw BVP signal, shape (512,) for a 16 s window at 32 Hz.
    sampling_rate : int
        Sampling rate in Hz (default 32).

    Returns
    -------
    ibi_ms : np.ndarray
        Inter-beat intervals in milliseconds.  May be empty if peak
        detection fails.
    n_peaks : int
        Number of detected peaks (for diagnostics).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = nk.ppg_findpeaks(
                bvp_window.astype(np.float64),
                sampling_rate=sampling_rate,
            )
        peak_indices = info["PPG_Peaks"]
    except Exception:
        return np.array([]), 0

    n_peaks = len(peak_indices)
    if n_peaks < 2:
        return np.array([]), n_peaks

    # Compute IBIs: time between successive peaks in ms
    ibi_ms = np.diff(peak_indices) / sampling_rate * 1000.0

    return ibi_ms, n_peaks


# FITBIT HR DEGRADATION SIMULATION


def _simulate_fitbit_hr(
    ibi_ms: np.ndarray,
    window_sec: float = WINDOW_SEC,
    # sampling_rate: int = BVP_SAMPLING_RATE,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Fitbit-style HR from BVP-derived IBIs.

      1. Compute instantaneous HR from IBIs.
      2. Average into 1-second bins (align to window clock).
      3. Round to integer BPM (Fitbit reports integers).
      4. Optionally add Gaussian noise N(μ=−3, σ=5) and clip [40,200].

    Returns BOTH clean and noisy versions for ablation.

    Parameters:
    ibi_ms : np.ndarray
        Cleaned IBI series in milliseconds.
    window_sec : float
        Window duration in seconds.
    sampling_rate : int
        BVP sampling rate (for computing beat times).
    add_noise : bool
        Whether to apply Fitbit noise simulation.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns:
    hr_clean : np.ndarray
        Clean 1 Hz integer HR array, shape (n_seconds,).
    hr_noisy : np.ndarray
        Noise-degraded 1 Hz integer HR array, shape (n_seconds,).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_seconds = int(window_sec)

    if len(ibi_ms) < 2:
        return np.full(n_seconds, np.nan), np.full(n_seconds, np.nan)

    # Instantaneous HR from each IBI
    hr_instantaneous = 60000.0 / ibi_ms  # BPM

    # Map each IBI to its 1-second time bin.
    # Beat times are the cumulative sum of IBIs (in seconds).
    # Each IBI is assigned to the bin of its terminating beat.
    beat_times_s = np.cumsum(ibi_ms) / 1000.0
    # IBI[i] spans from beat_times_s[i-1] to beat_times_s[i]
    # (with beat_times_s[-1] = ibi_ms[0]/1000 for the first)

    hr_1sec = np.full(n_seconds, np.nan)
    for sec in range(n_seconds):
        # Find IBIs whose terminating beat falls within this second
        t_start = sec
        t_end = sec + 1
        mask = (beat_times_s >= t_start) & (beat_times_s < t_end)
        if mask.any():
            hr_1sec[sec] = np.mean(hr_instantaneous[mask])

    # Forward-fill NaN bins (Fitbit holds last known value)
    for i in range(1, len(hr_1sec)):
        if np.isnan(hr_1sec[i]) and not np.isnan(hr_1sec[i - 1]):
            hr_1sec[i] = hr_1sec[i - 1]

    # Backward-fill leading NaNs.  At resting HR (~60 BPM), the first
    # beat terminates at ~1.0 s, leaving second 0 empty.  Real Fitbit
    # devices report the current HR from the moment the sensor starts,
    # so backward-fill is the correct simulation.
    for i in range(len(hr_1sec) - 2, -1, -1):
        if np.isnan(hr_1sec[i]) and not np.isnan(hr_1sec[i + 1]):
            hr_1sec[i] = hr_1sec[i + 1]

    # Round to integer BPM (Fitbit reports integers)
    hr_clean = np.round(hr_1sec).astype(np.float64)

    # Optionally add noise
    if add_noise:
        noise = rng.normal(HR_BIAS_BPM, HR_NOISE_STD_BPM, size=hr_clean.shape)
        hr_noisy = np.round(hr_clean + noise)
        hr_noisy = np.clip(hr_noisy, HR_CLIP_LOW, HR_CLIP_HIGH)
    else:
        hr_noisy = hr_clean.copy()

    return hr_clean, hr_noisy


# PROCESS ONE SUBJECT


def process_subject(
    subject_id: int,
    windowed_dir: str = WESAD_WINDOWED_DIR,
    output_dir: str = KD_PAIRED_DIR,
    window_sec: float = WINDOW_SEC,
    seed: int = 42,
) -> dict:
    """Process one WESAD subject into paired KD training data.

    For each valid window:
      1. Extract raw teacher signals (BVP, EDA, ACC, TEMP).
      2. Detect BVP peaks → compute IBIs → extract HRV features.
      3. Simulate Fitbit HR degradation → extract noisy HRV features.
      4. Assign binary label by majority vote.

    Parameters
    ----------
    subject_id : int
        WESAD subject ID.
    windowed_dir : str
        Root of the DL-windowed data tree.
    output_dir : str
        Directory for output .npz files.
    window_sec : float
        Window length (default 16 s).
    seed : int
        Random seed for noise simulation reproducibility.

    Returns
    -------
    dict
        Processing statistics for this subject.
    """
    validate_subject_id(subject_id)
    rng = np.random.default_rng(seed + subject_id)  # per-subject seed

    print(f"\n  S{subject_id} — loading windowed data...")
    signals, metadata = _load_windowed_subject(subject_id, windowed_dir, window_sec)

    n_total = len(metadata)
    print(f"    {n_total} windows loaded")

    # Pre-allocate output arrays
    # Teacher signals stored at their native shapes
    out_bvp = []
    out_eda = []
    out_acc = []
    out_temp = []
    out_student_feats = []
    out_student_feats_noisy = []
    out_labels = []
    out_is_overlapped = []
    out_peak_counts = []
    out_valid_ibi_counts = []

    # Counters
    n_inactive_label = 0
    n_peak_fail = 0
    n_ibi_insufficient = 0
    n_hrv_invalid = 0
    n_accepted = 0

    label_dist = {0: 0, 1: 0}

    for i in range(n_total):
        meta = metadata[i]

        # Label check
        # Only keep windows where the raw label is in ACTIVE_LABELS
        raw_label = meta["label"]
        if raw_label not in ACTIVE_LABELS:
            n_inactive_label += 1
            continue

        # Map to binary
        mapped_label = LABEL_MAP_BINARY[raw_label]

        # Extract raw signals
        bvp_window = signals["bvp"][i]  # (512,)
        eda_window = signals["w_eda"][i]  # (64,)
        acc_window = signals["w_acc"][i]  # (512, 3)
        temp_window = signals["w_temp"][i]  # (64,)

        # Peak detection + IBI extraction
        ibi_raw, n_peaks = _detect_peaks_and_extract_ibi(bvp_window)

        if len(ibi_raw) < 2:
            n_peak_fail += 1
            continue

        # Clean IBIs (range filter + Malik correction)
        ibi_clean = clean_ibi_series(ibi_raw)

        if len(ibi_clean) < 10:  # MIN_VALID_IBIS
            n_ibi_insufficient += 1
            continue

        # Extract HRV features from clean IBIs
        features_clean = extract_primary_features(ibi_clean, clean=False)

        if not is_valid_window(features_clean):
            n_hrv_invalid += 1
            continue

        # Simulate Fitbit HR + extract noisy features
        hr_clean_1hz, hr_noisy_1hz = _simulate_fitbit_hr(
            ibi_clean,
            window_sec=window_sec,
            rng=rng,
        )

        # Derive pseudo-IBIs from noisy HR and extract features.
        #
        # DESIGN NOTE: We intentionally pass pseudo-RR intervals (one
        # per second) to extract_primary_features to produce the samr
        # 6 features as the clean path.  This is an approximation:
        # pseudo-RR from 1 Hz HR are not true beat-to-beat IBIs, so
        # rmssd/pnn50 reflect second-to-second RR variation, not beat-
        # to-beat.  We accept this because:
        #   (a) The noise ablation compares clean vs degraded inputs
        #       to the SAME 6-feature student so using a different
        #       feature set (5 Fitbit features) would break that
        #       controlled comparison.
        #   (b) The numeric values of sdnn and rmssd on pseudo-RR are
        #       identical to pseudo_sdnn and pseudo_rmssd from
        #       compute_fitbit_features (same formula, same input).
        #
        # The Fitbit-recoverable 5-feature subset (compute_fitbit_features)
        # is used separately in the feature-set ablation
        valid_noisy = hr_noisy_1hz[np.isfinite(hr_noisy_1hz) & (hr_noisy_1hz > 0)]
        if len(valid_noisy) >= 4:
            pseudo_rr_noisy = 60000.0 / valid_noisy
            features_noisy = extract_primary_features(pseudo_rr_noisy, clean=False)
        else:
            features_noisy = dict.fromkeys(PRIMARY_HRV_FEATURES, np.nan)

        # Store window
        out_bvp.append(bvp_window)
        out_eda.append(eda_window)
        out_acc.append(acc_window)
        out_temp.append(temp_window)
        out_student_feats.append(
            features_to_array(features_clean, PRIMARY_HRV_FEATURES)
        )
        out_student_feats_noisy.append(
            features_to_array(features_noisy, PRIMARY_HRV_FEATURES)
        )
        out_labels.append(mapped_label)
        out_is_overlapped.append(meta["is_overlapped"])
        out_peak_counts.append(n_peaks)
        out_valid_ibi_counts.append(len(ibi_clean))

        label_dist[mapped_label] += 1
        n_accepted += 1

    if n_accepted == 0:
        print(f"    WARNING: S{subject_id} — 0 valid windows!")
        return {
            "subject_id": subject_id,
            "n_total": n_total,
            "n_accepted": 0,
            "n_inactive_label": n_inactive_label,
            "n_peak_fail": n_peak_fail,
            "n_ibi_insufficient": n_ibi_insufficient,
            "n_hrv_invalid": n_hrv_invalid,
        }

    # Save .npz
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"S{subject_id}_paired.npz")

    np.savez_compressed(
        output_path,
        # Teacher signals (raw, un-normalised — normalisation applied at teacher inference time in kd_trainer.py to match
        # WESADDLDataset's per-window instance Z-score)
        bvp=np.stack(out_bvp, axis=0),  # (N, 512)
        w_eda=np.stack(out_eda, axis=0),  # (N, 64)
        w_acc=np.stack(out_acc, axis=0),  # (N, 512, 3)
        w_temp=np.stack(out_temp, axis=0),  # (N, 64)
        # Student features
        student_features=np.stack(out_student_feats, axis=0),  # (N, 6)
        student_features_noisy=np.stack(out_student_feats_noisy, axis=0),  # (N, 6)
        # Labels
        labels=np.array(out_labels, dtype=np.int64),  # (N,)
        # Metadata
        window_is_overlapped=np.array(out_is_overlapped, dtype=bool),  # (N,)
        peak_counts=np.array(out_peak_counts, dtype=np.int32),  # (N,)
        valid_ibi_counts=np.array(out_valid_ibi_counts, dtype=np.int32),  # (N,)
    )

    stats = {
        "subject_id": subject_id,
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_inactive_label": n_inactive_label,
        "n_peak_fail": n_peak_fail,
        "n_ibi_insufficient": n_ibi_insufficient,
        "n_hrv_invalid": n_hrv_invalid,
        "label_dist": label_dist,
        "output_path": output_path,
    }

    print(
        f"    Accepted: {n_accepted}/{n_total} windows "
        f"(inactive={n_inactive_label}, peak_fail={n_peak_fail}, "
        f"ibi_short={n_ibi_insufficient}, hrv_invalid={n_hrv_invalid})"
    )
    print(f"    Labels: non-stress={label_dist[0]}, stress={label_dist[1]}")
    print(f"    Saved: {output_path}")

    return stats


# PROCESS ALL SUBJECTS


def generate_all_paired_data(
    windowed_dir: str = WESAD_WINDOWED_DIR,
    output_dir: str = KD_PAIRED_DIR,
    window_sec: float = WINDOW_SEC,
    subject_ids: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Process all WESAD subjects into paired KD training data.

    Parameters:
    windowed_dir : str
        Root of the DL-windowed data tree.
    output_dir : str
        Directory for output .npz files.
    window_sec : float
        Window length (default 16 s).
    subject_ids : list[int], optional
        Subjects to process.  Defaults to all 15.
    seed : int
        Base random seed.

    Returns:
    list[dict]
        Per-subject processing statistics.
    """
    if subject_ids is None:
        subject_ids = ALL_SUBJECT_IDS

    ensure_dirs()

    print("\n")
    print("Paired Data Generation for Cross-Modal KD")
    print("\n")
    print(f"  Source:     {windowed_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Window:     {window_sec} s")
    print(f"  Subjects:   {subject_ids}")
    print(f"  Features:   {N_PRIMARY_FEATURES} primary HRV")
    print(f"  Noise sim:  bias={HR_BIAS_BPM} BPM, std={HR_NOISE_STD_BPM} BPM")
    print()

    t_start = time.time()
    all_stats = []

    for sid in subject_ids:
        try:
            stats = process_subject(
                subject_id=sid,
                windowed_dir=windowed_dir,
                output_dir=output_dir,
                window_sec=window_sec,
                seed=seed,
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"\n  S{sid} — ERROR: {e}")
            all_stats.append({"subject_id": sid, "error": str(e)})

    elapsed = time.time() - t_start

    # Summary
    total_accepted = sum(s.get("n_accepted", 0) for s in all_stats)
    total_windows = sum(s.get("n_total", 0) for s in all_stats)
    total_stress = sum(s.get("label_dist", {}).get(1, 0) for s in all_stats)
    total_nonstress = sum(s.get("label_dist", {}).get(0, 0) for s in all_stats)

    print("\n")
    print(f"Generation complete in {elapsed:.1f} s")
    print(f"  Total accepted: {total_accepted}/{total_windows} windows")
    print(f"  Labels: non-stress={total_nonstress}, stress={total_stress}")
    if total_nonstress > 0:
        print(f"  Class ratio: {total_nonstress / max(total_stress, 1):.2f}:1")
    print(f"  Output dir: {output_dir}")
    print("\n")

    return all_stats


# VERIFICATION UTILITY


def verify_paired_data(
    subject_id: int,
    # output_dir: str = KD_PAIRED_DIR,
) -> dict:
    """Load and verify a paired data .npz file.

    Checks shapes, dtypes, NaN counts, and label distribution.

    Parameters:
    subject_id : int
        WESAD subject ID.
    output_dir : str
        Directory containing .npz files.

    Returns:
    dict
        Verification results.
    """
    path = get_paired_data_path(subject_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Paired data not found: {path}")

    data = np.load(path)

    n_windows = len(data["labels"])
    results = {
        "subject_id": subject_id,
        "n_windows": n_windows,
        "shapes": {key: data[key].shape for key in data.files},
    }

    # Shape checks
    assert data["bvp"].shape == (
        n_windows,
        BVP_SAMPLES_PER_WINDOW,
    ), f"BVP shape mismatch: {data['bvp'].shape}"
    assert data["w_eda"].shape == (
        n_windows,
        EDA_SAMPLES_PER_WINDOW,
    ), f"EDA shape mismatch: {data['w_eda'].shape}"
    assert data["w_acc"].shape == (
        n_windows,
        ACC_SAMPLES_PER_WINDOW,
        3,
    ), f"ACC shape mismatch: {data['w_acc'].shape}"
    assert data["w_temp"].shape == (
        n_windows,
        TEMP_SAMPLES_PER_WINDOW,
    ), f"TEMP shape mismatch: {data['w_temp'].shape}"
    assert data["student_features"].shape == (
        n_windows,
        N_PRIMARY_FEATURES,
    ), f"Student features shape mismatch: {data['student_features'].shape}"
    assert data["student_features_noisy"].shape == (
        n_windows,
        N_PRIMARY_FEATURES,
    ), f"Noisy features shape mismatch: {data['student_features_noisy'].shape}"

    # NaN counts
    clean_nans = np.isnan(data["student_features"]).sum()
    noisy_nans = np.isnan(data["student_features_noisy"]).sum()
    results["clean_nan_count"] = int(clean_nans)
    results["noisy_nan_count"] = int(noisy_nans)

    # Label distribution
    labels = data["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    results["label_dist"] = dict(zip(unique.tolist(), counts.tolist()))

    # Feature statistics (clean)
    feat = data["student_features"]
    results["feature_means"] = {
        name: float(np.nanmean(feat[:, i]))
        for i, name in enumerate(PRIMARY_HRV_FEATURES)
    }
    results["feature_stds"] = {
        name: float(np.nanstd(feat[:, i]))
        for i, name in enumerate(PRIMARY_HRV_FEATURES)
    }

    print(
        f"  S{subject_id}: {n_windows} windows, "
        f"labels={results['label_dist']}, "
        f"clean_nans={clean_nans}, noisy_nans={noisy_nans}"
    )

    return results


#  CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paired KD training data from WESAD."
    )
    parser.add_argument(
        "--windowed-dir",
        type=str,
        default=WESAD_WINDOWED_DIR,
        help="Root of DL-windowed data tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=KD_PAIRED_DIR,
        help="Output directory for .npz files.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject IDs to process (default: all 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise simulation.",
    )

    args = parser.parse_args()
    generate_all_paired_data(
        windowed_dir=args.windowed_dir,
        output_dir=args.output_dir,
        subject_ids=args.subjects,
        seed=args.seed,
    )

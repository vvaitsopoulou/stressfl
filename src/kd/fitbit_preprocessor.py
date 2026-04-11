# Fitbit HR Data Preprocessor for Cross-Modal KD Pilot Evaluation

# Purpose:

# Clean and prepare personal Fitbit Charge 6 HR data for the single-participant deployment evaluation.
# This module handles the full pipeline from raw Fitbit Web API JSON to windowed HRV features
# ready for student inference.

# Critical Assumption:
# The Fitbit API "1sec detail level" does not guarantee uniform 1 Hz
# output.  The raw data is an irregularly sampledtime series, as intervals of 2–5+ seconds are
# observed, and this fact is also confirmed by
# Fitbit documentation and Bottcher et al. 2022.  This module treats the input as irregularly sampled throughout.

# Pipeline:
# 1. Parse —>Read Fitbit Web API JSON, extract (timestamp, HR) pairs.
# 2. Interpolate —> Cubic spline on actual timestamps to uniform 1 Hz.
# 3. Quality flag —> Mark gaps > 5 s as unreliable (Bottcher et al. 2022).
# 4. Window + extract —> 16 s windows at 8 s step, extract 6 HRV features.
# 5. Label —> Map windows to pilot protocol phases.

# Feature Normalisation:
# Features are Z-score normalised using WESAD training-set statistics
# (not Fitbit data statistics) to match the student's training
# distribution.  The scaler is passed in by fitbit_evaluator.py.

# References
# ----------
# Bottcher et al. 2022, Scientific Reports for Fitbit data quality
# Fuller et al. 2022 for Fitbit HR accuracy meta-analysis
# Ko et al. 2025, Psychophysiology for Fitbit HR accuracy

from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import CubicSpline

from kd.hrv_feature_extractor import (
    extract_primary_features,
    features_to_array,
)
from kd.kd_config import (
    FITBIT_CALIBRATION_MINUTES,
    FITBIT_MAX_GAP_SEC,
    FITBIT_UNRELIABLE_FRAC,
    HR_CLIP_HIGH,
    HR_CLIP_LOW,
    N_PRIMARY_FEATURES,
    PRIMARY_HRV_FEATURES,
    SLIDE_SEC,
    WINDOW_SEC,
)

# PARSE RAW FITBIT JSON


def parse_fitbit_json(
    json_path: str,
    session_start: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse Fitbit Web API intraday HR JSON into (timestamp, HR) pairs.

    The Fitbit intraday HR response has the structure::

        {
          "activities-heart-intraday": {
            "dataset": [
              {"time": "HH:MM:SS", "value": <int BPM>},
              ...
            ],
            "datasetInterval": 1,
            "datasetType": "second"
          }
        }

    Parameters:
    json_path : str
        Path to the Fitbit JSON file.
    session_start : str, optional
        Timestamp (HH:MM:SS) marking the beginning of the pilot session.
        If provided, all timestamps are relative to this point.  If
        None, timestamps are relative to the first data point.

    Returns
    -------
    timestamps_s : np.ndarray
        Time in seconds from session start.
    hr_bpm : np.ndarray
        Heart rate in BPM (integer values as recorded by Fitbit).
    sampling_report : dict
        Statistics about the raw sampling: n_samples, duration_s,
        median_interval, max_gap, pct_1sec (% of intervals ≤ 1.5 s).
    """
    with open(json_path) as f:
        data = json.load(f)

    # Navigate to intraday dataset, handle both nested and flat formats
    if "activities-heart-intraday" in data:
        dataset = data["activities-heart-intraday"]["dataset"]
    elif "dataset" in data:
        # Flat format (already extracted)
        dataset = data["dataset"]
    else:
        raise ValueError(
            f"Unrecognised Fitbit JSON structure. Expected key "
            f"'activities-heart-intraday' or 'dataset'. "
            f"Found: {list(data.keys())}"
        )

    if len(dataset) == 0:
        raise ValueError(f"Empty dataset in {json_path}")

    # Parse timestamps
    times_str = [entry["time"] for entry in dataset]
    hr_values = [int(entry["value"]) for entry in dataset]

    # Convert HH:MM:SS to seconds from midnight
    def _time_to_seconds(t_str: str) -> float:
        parts = t_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 3600 + m * 60 + s

    times_sec = np.array([_time_to_seconds(t) for t in times_str])
    hr_bpm = np.array(hr_values, dtype=np.float64)

    # Handle midnight wraparound (session crossing midnight)
    for i in range(1, len(times_sec)):
        if times_sec[i] < times_sec[i - 1]:
            times_sec[i:] += 86400  # add 24 hours

    # Rebase to session start
    if session_start is not None:
        start_sec = _time_to_seconds(session_start)
        times_sec = times_sec - start_sec
    else:
        times_sec = times_sec - times_sec[0]

    # Sampling statistics
    intervals = np.diff(times_sec)
    sampling_report = {
        "n_samples": len(hr_bpm),
        "duration_s": float(times_sec[-1] - times_sec[0]),
        "median_interval_s": float(np.median(intervals)) if len(intervals) > 0 else 0.0,
        "max_gap_s": float(np.max(intervals)) if len(intervals) > 0 else 0.0,
        "mean_interval_s": float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
        "pct_1sec": (
            float(100.0 * np.mean(intervals <= 1.5)) if len(intervals) > 0 else 0.0
        ),
    }

    return times_sec, hr_bpm, sampling_report


# INTERPOLATION TO UNIFORM 1 Hz


def interpolate_to_uniform(
    timestamps_s: np.ndarray,
    hr_bpm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline interpolation to uniform 1 Hz grid.

    Operates on actual timestamps (not assumed 1 Hz spacing),

    After interpolation, values are rounded to integer BPM (Fitbit
    reports integers) and clipped to [HR_CLIP_LOW, HR_CLIP_HIGH].

    Parameters:
    timestamps_s : np.ndarray
        Raw timestamps in seconds from session start.
    hr_bpm : np.ndarray
        Raw HR values in BPM.

    Returns:
    t_uniform : np.ndarray
        Uniform 1 Hz timestamps (integer seconds).
    hr_uniform : np.ndarray
        Interpolated HR at each uniform timestamp.
    """
    if len(timestamps_s) < 4:
        raise ValueError(
            f"Too few data points for cubic spline interpolation: "
            f"{len(timestamps_s)} (need ≥ 4)"
        )

    # Remove duplicate timestamps if oresent
    _, unique_idx = np.unique(timestamps_s, return_index=True)
    unique_idx = np.sort(unique_idx)
    ts_unique = timestamps_s[unique_idx]
    hr_unique = hr_bpm[unique_idx]

    # Cubic spline on actual timestamps
    cs = CubicSpline(ts_unique, hr_unique)

    # Uniform 1 Hz grid from first to last whole second
    t_start = int(np.ceil(ts_unique[0]))
    t_end = int(np.floor(ts_unique[-1]))
    t_uniform = np.arange(t_start, t_end + 1, dtype=np.float64)

    # Warn if the grid doesn't cover the expected session start.
    # This catches cases where the Fitbit started recording late which would cause the protocol
    # phase labels to misalign with the available windows.
    if t_start > 5:
        import warnings

        warnings.warn(
            f"Fitbit uniform grid starts at t={t_start}s — the first "
            f"{t_start}s of the session have no HR data.  Protocol "
            f"phase labels assume t=0 is session start.  Verify that "
            f"session_start was set correctly in parse_fitbit_json().",
            UserWarning,
            stacklevel=2,
        )

    hr_uniform = cs(t_uniform)

    # Round to integer BPM and clip to physiological range
    hr_uniform = np.round(hr_uniform)
    hr_uniform = np.clip(hr_uniform, HR_CLIP_LOW, HR_CLIP_HIGH)

    return t_uniform, hr_uniform


# QUALITY FLAGGING


def flag_gaps(
    raw_timestamps_s: np.ndarray,
    uniform_timestamps_s: np.ndarray,
    max_gap_sec: float = FITBIT_MAX_GAP_SEC,
) -> np.ndarray:
    """Flag uniform-grid samples that fall within large raw-data gaps.

    Any uniform timestamp that lies within a raw-data gap exceeding
    max_gap_sec is flagged as unreliable, because the interpolated
    HR at that point is extrapolated across a period with no real
    measurements.

    Parameters
    raw_timestamps_s : np.ndarray
        Original irregular timestamps.
    uniform_timestamps_s : np.ndarray
        Uniform 1 Hz grid timestamps.
    max_gap_sec : float
        Gap threshold in seconds (default 5.0, Bottcher et al. 2022).

    Returns
    flags : np.ndarray of bool
        True = unreliable (within a gap), False = reliable.
    """
    flags = np.zeros(len(uniform_timestamps_s), dtype=bool)

    intervals = np.diff(raw_timestamps_s)
    gap_mask = intervals > max_gap_sec

    if not gap_mask.any():
        return flags

    # For each large gap flag all uniform timestamps within it
    gap_indices = np.where(gap_mask)[0]
    for idx in gap_indices:
        gap_start = raw_timestamps_s[idx]
        gap_end = raw_timestamps_s[idx + 1]
        in_gap = (uniform_timestamps_s > gap_start) & (uniform_timestamps_s < gap_end)
        flags[in_gap] = True

    return flags


# WINDOWING + FEATURE EXTRACTION


def extract_windowed_features(
    hr_uniform: np.ndarray,
    t_uniform: np.ndarray,
    gap_flags: np.ndarray,
    window_sec: float = WINDOW_SEC,
    slide_sec: float = SLIDE_SEC,
    unreliable_frac: float = FITBIT_UNRELIABLE_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract HRV features from sliding windows over uniform 1 Hz HR.

    For each 16 s window (8 s step):
      1. Check quality: if > 20% of samples are gap-flagged, skip.
      2. Convert HR → pseudo-RR (60000 / hr_bpm).
      3. Extract 6 primary HRV features via extract_primary_features.

    normalisation with WESAD training statistics is applied downstream

    Parameters
    ----------
    hr_uniform : np.ndarray
        Uniform 1 Hz HR in BPM.
    t_uniform : np.ndarray
        Uniform 1 Hz timestamps in seconds.
    gap_flags : np.ndarray of bool
        Per-sample gap flags from flag_gaps().
    window_sec : float
        Window length in seconds (default 16).
    slide_sec : float
        Slide/step in seconds (default 8).
    unreliable_frac : float
        Max fraction of flagged samples before excluding window.

    Returns
    -------
    features : np.ndarray, shape (N, 6)
        HRV feature matrix.
    timestamps : np.ndarray, shape (N,)
        Window start times in seconds.
    quality : np.ndarray of bool, shape (N,)
        True = window passed quality check, False = excluded.
    window_indices : np.ndarray, shape (N, 2)
        Start and end indices into the uniform array for each window.
    """
    window_samples = int(window_sec)  # 1 Hz → 1 sample per second
    slide_samples = int(slide_sec)
    n_total = len(hr_uniform)

    all_features = []
    all_timestamps = []
    all_quality = []
    all_indices = []

    start = 0
    while start + window_samples <= n_total:
        end = start + window_samples
        hr_window = hr_uniform[start:end]
        flags_window = gap_flags[start:end]
        t_start = t_uniform[start]

        # Quality check: fraction of unreliable samples
        unreliable_count = flags_window.sum()
        is_reliable = (unreliable_count / window_samples) <= unreliable_frac

        if is_reliable:
            # Convert to pseudo-RR and extract features.
            #
            # TRAINING/SERVING CONSISTENCY: This uses extract_primary_features on pseudo-RR (1 Hz HR → 60000/HR), which is the same operation
            # used in paired_data_generator.py for the noisy training features (student_features_noisy).  Both paths produce 6 features with
            # identical semantics. Rmssd/pnn50 reflect second-to-second variability, not true beat-to-beat HRV acording to literature
            # however, this is an intentional approximation documented in paired_data_generator.py . KEY design choice: whatever the
            # student was trained on must match what it receives at inference.
            valid_hr = hr_window[hr_window > 0]
            if len(valid_hr) >= 4:
                pseudo_rr = 60000.0 / valid_hr
                feats = extract_primary_features(pseudo_rr, clean=False)
                feat_array = features_to_array(feats, PRIMARY_HRV_FEATURES)
            else:
                feat_array = np.full(N_PRIMARY_FEATURES, np.nan)
        else:
            feat_array = np.full(N_PRIMARY_FEATURES, np.nan)

        all_features.append(feat_array)
        all_timestamps.append(t_start)
        all_quality.append(is_reliable and not np.any(np.isnan(feat_array)))
        all_indices.append([start, end])

        start += slide_samples

    features = np.stack(all_features, axis=0)
    timestamps = np.array(all_timestamps)
    quality = np.array(all_quality, dtype=bool)
    window_indices = np.array(all_indices)

    return features, timestamps, quality, window_indices


# LABEL MAPPING (PILOT PROTOCOL)

# Default pilot protocol
#   Calibration: 0–5 min baseline, 5–10 min stress A, 10–15 min recovery
#   Evaluation:  15–20 min baseline, 20–25 min stress B, 25–30 min recovery,
#                30–35 min journaling, 35–40 min recovery, 40–50 min baseline

DEFAULT_PROTOCOL = [
    # (start_min, end_min, label, phase_name)
    (0, 5, 0, "calibration_baseline"),
    (5, 10, 1, "calibration_stress_A"),
    (10, 15, 0, "calibration_recovery"),
    (15, 20, 0, "eval_baseline"),
    (20, 25, 1, "eval_stress_B"),
    (25, 30, 0, "eval_recovery_1"),
    (30, 35, -1, "eval_journaling_stressor"),  # TODO REMEMBER To modify this
    # physiological response is weaker and more variable
    # than TSST-style cognitive stress.  Excluded from primary evaluation
    # to avoid contaminating metrics.  Can be analysed separately.
    (35, 40, 0, "eval_recovery_2"),
    (40, 50, 0, "eval_final_baseline"),
]


def assign_labels(
    window_timestamps_s: np.ndarray,
    protocol: list[tuple] | None = None,
    window_sec: float = WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign binary stress labels to windows based on protocol timing.

    A window's label is determined by which protocol phase its CENTRE
    falls in.  Windows spanning phase boundaries get the label of the
    phase containing the window centre.  Windows outside all protocol
    phases get label -1 (unlabelled).

    Parameters
    ----------
    window_timestamps_s : np.ndarray
        Window start times in seconds from session start.
    protocol : list of (start_min, end_min, label, name), optional
        Protocol phase definitions.  Defaults to DEFAULT_PROTOCOL.
    window_sec : float
        Window duration (for computing centre time).

    Returns
    -------
    labels : np.ndarray of int
        Per-window labels: 0 (non-stress), 1 (stress), -1 (unlabelled).
    phase_names : np.ndarray of str
        Per-window phase name (or "unlabelled").
    """
    if protocol is None:
        protocol = DEFAULT_PROTOCOL

    labels = np.full(len(window_timestamps_s), -1, dtype=np.int64)
    phase_names = np.full(len(window_timestamps_s), "unlabelled", dtype=object)

    for i, t_start in enumerate(window_timestamps_s):
        t_centre = t_start + window_sec / 2.0
        t_centre_min = t_centre / 60.0

        for p_start, p_end, p_label, p_name in protocol:
            if p_start <= t_centre_min < p_end:
                labels[i] = p_label
                phase_names[i] = p_name
                break

    return labels, phase_names


def split_calibration_evaluation(
    window_timestamps_s: np.ndarray,
    # labels: np.ndarray,
    # phase_names: np.ndarray,
    calibration_minutes: float = FITBIT_CALIBRATION_MINUTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into calibration and evaluation sets.

    The calibration set uses the first calibration_minutes of the
    session (for head fine-tuning).  The evaluation set uses everything
    after (for metric computation).

    This is a temporal split (not random).

    Parameters
    ----------
    window_timestamps_s : np.ndarray
        Window start times in seconds.
    labels : np.ndarray
        Per-window labels.
    phase_names : np.ndarray
        Per-window phase names.
    calibration_minutes : float
        Duration of calibration portion (default 15 min).

    Returns
    -------
    cal_mask : np.ndarray of bool
        True for calibration windows.
    eval_mask : np.ndarray of bool
        True for evaluation windows.
    """
    cal_cutoff_s = calibration_minutes * 60.0

    # Window centre determines assignment
    window_centres = window_timestamps_s + WINDOW_SEC / 2.0
    cal_mask = window_centres < cal_cutoff_s
    eval_mask = ~cal_mask

    return cal_mask, eval_mask


# FULL PREPROCESSING PIPELINE


def preprocess_fitbit_session(
    json_path: str,
    session_start: str | None = None,
    protocol: list[tuple] | None = None,
    window_sec: float = WINDOW_SEC,
    slide_sec: float = SLIDE_SEC,
) -> dict:
    """Full Fitbit preprocessing pipeline: JSON → windowed features.

    Parameters
    ----------
    json_path : str
        Path to the Fitbit intraday HR JSON file.
    session_start : str, optional
        Session start time as HH:MM:SS.
    protocol : list of (start_min, end_min, label, name), optional
        Pilot protocol definition.
    window_sec : float
        Window length in seconds.
    slide_sec : float
        Slide/step in seconds.

    Returns
    -------
    dict with keys:
        features : np.ndarray (N, 6) — HRV feature matrix
        labels : np.ndarray (N,) — binary labels (-1 for unlabelled)
        phase_names : np.ndarray (N,) — protocol phase names
        quality : np.ndarray (N,) bool — quality flags
        timestamps : np.ndarray (N,) — window start times (seconds)
        cal_mask : np.ndarray (N,) bool — calibration window mask
        eval_mask : np.ndarray (N,) bool — evaluation window mask
        gap_report : dict — data quality statistics
        sampling_report : dict — raw sampling statistics
        hr_uniform : np.ndarray — interpolated 1 Hz HR
        t_uniform : np.ndarray — uniform timestamps
        gap_flags : np.ndarray — per-sample gap flags
    """
    # Step 1: Parse
    print(f"  Parsing: {json_path}")
    raw_ts, raw_hr, sampling_report = parse_fitbit_json(json_path, session_start)
    print(
        f"    {sampling_report['n_samples']} samples, "
        f"{sampling_report['duration_s']:.0f} s, "
        f"median interval={sampling_report['median_interval_s']:.2f} s, "
        f"max gap={sampling_report['max_gap_s']:.1f} s, "
        f"{sampling_report['pct_1sec']:.1f}% at ~1 Hz"
    )

    # Step 2: Interpolate
    t_uniform, hr_uniform = interpolate_to_uniform(raw_ts, raw_hr)
    print(
        f"    Interpolated: {len(hr_uniform)} uniform 1 Hz samples "
        f"({len(hr_uniform)/60:.1f} min)"
    )

    # Step 3: Quality flagging
    gap_flags = flag_gaps(raw_ts, t_uniform)
    n_flagged = gap_flags.sum()
    pct_flagged = 100.0 * n_flagged / len(gap_flags) if len(gap_flags) > 0 else 0
    print(
        f"    Gap flags: {n_flagged}/{len(gap_flags)} samples "
        f"({pct_flagged:.1f}%) in gaps > {FITBIT_MAX_GAP_SEC} s"
    )

    gap_report = {
        "n_total_samples": len(hr_uniform),
        "n_flagged_samples": int(n_flagged),
        "pct_flagged": float(pct_flagged),
        "n_gaps_over_threshold": int(np.sum(np.diff(raw_ts) > FITBIT_MAX_GAP_SEC)),
    }

    # Step 4: Window + extract features
    features, timestamps, quality, window_indices = extract_windowed_features(
        hr_uniform,
        t_uniform,
        gap_flags,
        window_sec,
        slide_sec,
    )
    n_valid = quality.sum()
    print(
        f"    Windows: {len(features)} total, {n_valid} valid "
        f"({len(features) - n_valid} excluded by quality)"
    )

    # Step 5: Label mapping
    labels, phase_names = assign_labels(timestamps, protocol, window_sec)
    n_labelled = (labels >= 0).sum()
    n_stress = (labels == 1).sum()
    n_nonstress = (labels == 0).sum()
    print(
        f"    Labels: {n_labelled} labelled "
        f"(stress={n_stress}, non-stress={n_nonstress}), "
        f"{(labels < 0).sum()} unlabelled"
    )

    # Calibration / evaluation split
    cal_mask, eval_mask = split_calibration_evaluation(
        timestamps,
        # labels,
        # phase_names,
    )
    print(f"    Split: {cal_mask.sum()} calibration, " f"{eval_mask.sum()} evaluation")

    return {
        "features": features,
        "labels": labels,
        "phase_names": phase_names,
        "quality": quality,
        "timestamps": timestamps,
        "cal_mask": cal_mask,
        "eval_mask": eval_mask,
        "gap_report": gap_report,
        "sampling_report": sampling_report,
        "hr_uniform": hr_uniform,
        "t_uniform": t_uniform,
        "gap_flags": gap_flags,
    }


# REPORT GENERATION


def generate_quality_report(result: dict) -> str:
    """Generate a human-readable data quality report.

    Parameters
    ----------
    result : dict
        Output of preprocess_fitbit_session().

    Returns
    -------
    str
        Formatted report string.
    """
    sr = result["sampling_report"]
    gr = result["gap_report"]
    q = result["quality"]
    labels = result["labels"]
    cal = result["cal_mask"]
    evl = result["eval_mask"]

    lines = [
        "-" * 67,
        "Fitbit Pilot Data Quality Report",
        "-" * 67,
        "",
        "Raw Sampling:",
        f"  Total samples:      {sr['n_samples']}",
        f"  Duration:           {sr['duration_s']:.0f} s ({sr['duration_s']/60:.1f} min)",
        f"  Median interval:    {sr['median_interval_s']:.3f} s",
        f"  Mean interval:      {sr['mean_interval_s']:.3f} s",
        f"  Max gap:            {sr['max_gap_s']:.1f} s",
        f"  Samples at ~1 Hz:   {sr['pct_1sec']:.1f}%",
        "",
        "Quality:",
        f"  Flagged samples:    {gr['n_flagged_samples']}/{gr['n_total_samples']} "
        f"({gr['pct_flagged']:.1f}%)",
        f"  Gaps > {FITBIT_MAX_GAP_SEC}s:        {gr['n_gaps_over_threshold']}",
        f"  Valid windows:      {q.sum()}/{len(q)}",
        "",
        f"Calibration ({FITBIT_CALIBRATION_MINUTES:.0f} min):",
        f"  Total windows:      {cal.sum()}",
        f"  Valid windows:      {(cal & q).sum()}",
        f"  Stress windows:     {((labels == 1) & cal & q).sum()}",
        f"  Non-stress windows: {((labels == 0) & cal & q).sum()}",
        "",
        "Evaluation:",
        f"  Total windows:      {evl.sum()}",
        f"  Valid windows:      {(evl & q).sum()}",
        f"  Stress windows:     {((labels == 1) & evl & q).sum()}",
        f"  Non-stress windows: {((labels == 0) & evl & q).sum()}",
        "=" * 60,
    ]

    return "\n".join(lines)

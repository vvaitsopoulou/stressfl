# HRV Feature Extractor for Cross-Modal Knowledge Distillation

# Given an IBI (inter-beat interval) series in milliseconds, compute HRV features for one window.  This module is called by:

#   1. paired_data_generator.py which extracts features from BVP-derived
#      IBIs (32 Hz peak detection → real beat-to-beat intervals).
#   2. fitbit_preprocessor.py whid extracts features from pseudo-IBIs
#      derived from Fitbit 1 Hz integer HR (rr_ms = 60000 / hr_bpm).

# The module does not perform peak detection or IBI computation. It receives cleaned IBI arrays and
# returns named feature dicts or NaN-filled dicts for invalid windows.

# Feature Sets:
#   Primary (6 time-domain): reliable at 16 s windows.
#   Validated by Castaldo et al. (2019) and Baek et al. (2015) for ultra-short HRV from windows as short as 10 s.

#   Extended (12 features): for 60 s future-work ablation only. Adds frequency-domain (LF, HF, LF/HF, LFn) and nonlinear (SD2,
#   SampEn) features.  Frequency-domain requires ≥50 s per Task Force (1996) meaning it is unreliable at 16 s.

#   Fitbit-recoverable (5 features): subset computable from 1 Hz integer HR.  Pseudo-SDNN and pseudo-RMSSD are approximations, not
#   true beat-to-beat HRV (Peng et al. 2018).

# IBI Cleaning:
# Before feature computation, the IBI array undergoes:
#   1. Physiological range filtering: reject IBIs outside [375, 2000] ms
#      (HR 30–160 BPM).
#   2. Malik's correction for ectopic beats: any IBI differing from the
#      local median by >20% is replaced with the local median.  This is
#      the standard ectopic correction recommended by Malik et al. (1993)
#      and implemented in most HRV toolboxes.
#   3. Minimum count check: windows with fewer than MIN_VALID_IBIS (10)
#      valid intervals after cleaning return NaN features.

# Normalisation:
# This module does not normalise features.  Z-score normalisation is
# applied downstream in kd_trainer.py using training-set statistics only. We do this to avoid leakage, following the same pattern as the
# RF pipeline (wesad_fe.py + ml_classifier.py).

# References:
# Castaldo et al. 2019, BMC Med Inform -> ultra-short HRV
# Baek et al. 2015 -> time-domain HRV reliable at 10 s
# Task Force 1996, Eur Heart J -> HRV measurement standards
# Malik et al. 1993, Eur Heart J -> ectopic correction
# Makowski et al. for NK-2 kit
# Peng et al. 2018, CBM -> 1 Hz HR to HRV limitations
# Dahal et al. 2023 -> HRV stress
# Dalmeida & Masala 2021, Sensors 21(8):2873 -> HRV ranking
# Brennan et al. 2001, IEEE Trans BME -> Poincaré SD1/SD2
# Richman & Moorman 2000, Am J Physiol -> Sample Entropy
# Lake et al. 2002, Am J Physiol -> SampEn for short HRV

from __future__ import annotations

import warnings

import numpy as np

from kd.kd_config import (
    EXTENDED_HRV_FEATURES,
    FITBIT_HRV_FEATURES,
    HF_BAND,
    HR_CLIP_HIGH,
    HR_CLIP_LOW,
    IBI_MAX_MS,
    IBI_MIN_MS,
    IBI_RESAMPLE_HZ,
    LF_BAND,
    MIN_VALID_IBIS,
    PRIMARY_HRV_FEATURES,
)

# IBI CLEANING


def filter_ibi_range(
    ibi_ms: np.ndarray,
    ibi_min: float = IBI_MIN_MS,
    ibi_max: float = IBI_MAX_MS,
) -> np.ndarray:
    """Remove physiologically implausible IBIs.

    Parameters:
    ibi_ms : np.ndarray
        Raw IBI values in milliseconds.
    ibi_min : float
        Minimum plausible IBI (default 375 ms → 160 BPM).
    ibi_max : float
        Maximum plausible IBI (default 2000 ms → 30 BPM).

    Returns:
    np.ndarray
        Filtered IBI array with out-of-range values removed.
    """
    mask = (ibi_ms >= ibi_min) & (ibi_ms <= ibi_max)
    return ibi_ms[mask]


def malik_ectopic_correction(
    ibi_ms: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """Apply Malik's ectopic beat correction.

    Any IBI that differs from the local median (computed over a sliding
    window of the surrounding intervals) by more than ``threshold``
    fraction is replaced with that local median.

    This is the standard correction recommended by Malik et al. (1993,
    Eur Heart J) and is equivalent to NeuroKit2's "Malik" correction
    method.  We implement it manually to maintain full control over the
    IBI array (NeuroKit2's HRV functions expect peak indices or a
    processed DataFrame, not raw IBI arrays).

    Parameters:
    ibi_ms : np.ndarray
        IBI values in milliseconds (already range-filtered).
    threshold : float
        Fractional deviation threshold.  Default 0.20 (20%), the
        standard value from Malik et al. (1993).

    Returns:
    np.ndarray
        Corrected IBI array (same length ectopic values replaced).
    """
    if len(ibi_ms) < 3:
        return ibi_ms.copy()

    original = ibi_ms.copy()
    corrected = ibi_ms.copy()

    for i in range(len(original)):
        # Local context: up to 2 neighbours on each side (5-beat window), excluding the current beat.  This is robust to consecutive
        # ectopics while keeping the window physiologically local.
        #
        # Neighbours are read from the ORIGINAL (unmodified) array,
        # not from corrected, to prevent cascading corrections where
        # an already-corrected value shifts its neighbour's local
        # median.  This is the standard convention in HRV toolboxes (pyHRV, hrvanalysis).
        start = max(0, i - 2)
        end = min(len(original), i + 3)
        neighbours = np.concatenate([original[start:i], original[i + 1 : end]])

        if len(neighbours) == 0:
            continue

        local_median = np.median(neighbours)

        # Replace if deviation exceeds threshold
        if (
            local_median > 0
            and abs(original[i] - local_median) / local_median > threshold
        ):
            corrected[i] = local_median

    return corrected


def clean_ibi_series(
    ibi_ms: np.ndarray,
    ibi_min: float = IBI_MIN_MS,
    ibi_max: float = IBI_MAX_MS,
    malik_threshold: float = 0.20,
) -> np.ndarray:
    """Full IBI cleaning pipeline: range filter → ectopic correction.

    Parameters:
    ibi_ms : np.ndarray
        Raw IBI values in milliseconds.
    ibi_min, ibi_max : float
        Physiological range bounds.
    malik_threshold : float
        Ectopic correction threshold (default 0.20).

    Returns:
    np.ndarray
        Cleaned IBI array.  May be shorter than input (range-filtered
        IBIs are removed, not replaced).
    """
    # remove physiologically impossible values
    filtered = filter_ibi_range(ibi_ms, ibi_min, ibi_max)

    if len(filtered) < MIN_VALID_IBIS:
        return filtered  # too few for meaningful correction

    # correct ectopic beats among remaining valid IBIs
    corrected = malik_ectopic_correction(filtered, malik_threshold)

    return corrected


# PRIMARY FEATURES (6 time-domain reliable at 16 s)


def _compute_time_domain_features(ibi_ms: np.ndarray) -> dict[str, float]:
    """Compute 6 time-domain HRV features from a cleaned IBI series.

    These are computed manually rather than through NeuroKit2's
    hrv_time() for three reasons:
      1. We operate on raw IBI arrays, not peak-indexed DataFrames.
      2. We need exactly the 6 features defined in kd_config.py, with
         our exact naming convention (not NK2's HRV_MeanNN etc.).
      3. Manual computation is transparent and auditable

    The formulas follow the Task Force (1996) standards exactly:

    mean_rr : Mean of all NN intervals (ms).
        Reflects average cardiac cycle duration.

    sdnn : Standard deviation of NN intervals (ms).
        Reflects overall HRV including both sympathetic and
        parasympathetic contributions.  Task Force (1996)

    rmssd : Root mean square of successive differences (ms).
        Primarily reflects parasympathetic (vagal) modulation.
        Most reliable short-term HRV metric (Castaldo et al. 2019).

    pnn50 : Percentage of successive NN intervals differing by > 50 ms.
        Parasympathetic marker.  Correlated with rmssd but uses a
        threshold-based measure.  Task Force (1996)

    mean_hr : Mean heart rate (BPM).
        Computed as mean(60000 / ibi_ms), not as 60000 / mean(ibi_ms).
        The former is the arithmetic mean of instantaneous HR values,
        the latter would be the harmonic mean, which underestimates
        mean HR when variability is high.

    std_hr : Standard deviation of heart rate (BPM).
        Computed from the instantaneous HR series.  Provides HRV
        information in the BPM domain, which may help gradient flow
        in the MLP student (Dahal et al. 2023).

    Parameters:
    ibi_ms : np.ndarray
        Cleaned IBI series in milliseconds.  Must have length ≥
        MIN_VALID_IBIS.

    Returns:
    dict[str, float]
        Feature name → value mapping.
    """
    # NN intervals = cleaned IBIs (ectopics already corrected)
    nn = ibi_ms

    # mean_rr
    mean_rr = np.mean(nn)

    # sdnn
    # ddof=1 for sample standard deviation (N-1 denominator), matching Task Force (1996) definition and NeuroKit2's implementation.
    sdnn = np.std(nn, ddof=1)

    # rmssd
    successive_diffs = np.diff(nn)
    rmssd = np.sqrt(np.mean(successive_diffs**2))

    # pnn50
    # NOTE on denominator convention: we divide by len(successive_diffs)
    # (N−1 successive pairs), which is the convention used by pyHRV,
    # hrvanalysis, and most toolboxes.  NeuroKit2 divides by len(nn)
    # (N beats), producing slightly lower values.  Task Force (1996)
    # is ambiguous and both appear in the literature so this is an explicit choice here
    n_successive = len(successive_diffs)
    if n_successive > 0:
        pnn50 = 100.0 * np.sum(np.abs(successive_diffs) > 50.0) / n_successive
    else:
        pnn50 = 0.0

    # mean_hr and std_hr
    # Instantaneous HR from each IBI
    hr_bpm = 60000.0 / nn
    mean_hr = np.mean(hr_bpm)
    std_hr = np.std(hr_bpm, ddof=1)

    return {
        "mean_rr": float(mean_rr),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "pnn50": float(pnn50),
        "mean_hr": float(mean_hr),
        "std_hr": float(std_hr),
    }


# EXTENDED FEATURES (12 features for 60 s future-work ablation)


def _compute_frequency_domain_features(
    ibi_ms: np.ndarray,
    resample_hz: float = IBI_RESAMPLE_HZ,
    lf_band: tuple[float, float] = LF_BAND,
    hf_band: tuple[float, float] = HF_BAND,
) -> dict[str, float]:
    """Compute frequency-domain HRV features via Welch's PSD.

    Follows Task Force (1996) guidelines:
      1. Resample the unevenly spaced IBI series to a uniform grid
         at ``resample_hz`` using cubic spline interpolation.
      2. Estimate PSD via Welch's method.
      3. Integrate power in LF (0.04–0.15 Hz) and HF (0.15–0.40 Hz).

    NOTE: These features are UNRELIABLE at 16 s windows.  The LF band
    requires at least one full cycle at 0.04 Hz = 25 s, and the Task
    Force recommends ≥ 2 min (ideally 5 min) for short-term recordings.
    At 16 s, LF power is borderline and HF power captures at most ~6
    cycles (0.4 Hz × 16 s).  These features are included ONLY for the
    60 s future-work ablation experiment with a re-trained teacher.

    HRV resampling note:
      The 4 Hz cubic spline resampling is correct for real BVP-derived
      IBIs (from 32 Hz peak detection).  For Fitbit's 1 Hz integer HR
      converted to pseudo-RR, resampling to 4 Hz does NOT recover
      information it only interpolates between 1 Hz samples.  Peng
      et al. (2018) showed LF error is 3.7% at 1 Hz but HF error is
      15.3%.

    Parameters:
    ibi_ms : np.ndarray
        Cleaned IBI series in milliseconds.
    resample_hz : float
        Uniform resampling rate for PSD estimation (default 4 Hz).
    lf_band, hf_band : tuple[float, float]
        Frequency band boundaries in Hz.

    Returns:
    dict[str, float]
        Keys: lf_power, hf_power, lf_hf_ratio, lf_norm.
        Returns NaN values if computation fails (e.g. too few samples
        for meaningful PSD).
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import welch

    nan_result = {
        "lf_power": np.nan,
        "hf_power": np.nan,
        "lf_hf_ratio": np.nan,
        "lf_norm": np.nan,
    }

    if len(ibi_ms) < MIN_VALID_IBIS:
        return nan_result

    try:
        # Construct beat occurrence times (cumulative sum of IBIs).
        # Each IBI value is attributed to the time of the R-peak that
        # terminates that interval as is the standard convention used by
        # the Task Force (1996), NeuroKit2, and PhysioNet.
        beat_times_ms = np.cumsum(ibi_ms)
        ibi_times_s = beat_times_ms / 1000.0

        # Cubic spline interpolation to uniform grid
        cs = CubicSpline(ibi_times_s, ibi_ms)
        t_uniform = np.arange(
            ibi_times_s[0],
            ibi_times_s[-1],
            1.0 / resample_hz,
        )

        if len(t_uniform) < 8:
            # Too few resampled points for meaningful PSD
            return nan_result

        ibi_resampled = cs(t_uniform)

        # Welch PSD estimation
        # nperseg: use the full segment length (capped at 256) for
        # maximum frequency resolution on short windows.
        nperseg = min(len(ibi_resampled), 256)
        freqs, psd = welch(
            ibi_resampled,
            fs=resample_hz,
            nperseg=nperseg,
            noverlap=nperseg // 2,
        )

        # Band power integration via trapezoidal rule
        lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

        lf_power = (
            float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        )
        hf_power = (
            float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
        )

        # Derived ratios
        total_lf_hf = lf_power + hf_power
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
        lf_norm = 100.0 * lf_power / total_lf_hf if total_lf_hf > 0 else np.nan

        return {
            "lf_power": lf_power,
            "hf_power": hf_power,
            "lf_hf_ratio": float(lf_hf_ratio),
            "lf_norm": float(lf_norm),
        }

    except Exception:
        return nan_result


def _compute_nonlinear_features(ibi_ms: np.ndarray) -> dict[str, float]:
    """Compute nonlinear HRV features: Poincaré SD2 and Sample Entropy.

    SD2 (Poincaré plot long-axis standard deviation):
        Quantifies long-term HRV.  Computed from the successive-difference
        formula: SD2² = 2·SDNN² − 0.5·SDSD², where SDSD is the standard
        deviation of successive differences.  This avoids the need for
        explicit Poincaré plot construction and matches NeuroKit2's
        computation (Brennan et al. 2001, IEEE Trans BME).

    Sample Entropy (SampEn):
        Measures signal complexity/irregularity.  Lower SampEn during
        stress reflects reduced autonomic flexibility (Richman & Moorman
        2000).  Computed via NeuroKit2's entropy_sample() which
        implements the Richman-Moorman algorithm.

        Standard parameters: m=2, r=0.2×SD (Lake et al. 2002 recommend
        these for short HRV series).

    Parameters:
    ibi_ms : np.ndarray
        Cleaned IBI series in milliseconds.

    Returns:
    dict[str, float]
        Keys: sd2, sample_entropy.
    """
    nan_result = {
        "sd2": np.nan,
        "sample_entropy": np.nan,
    }

    if len(ibi_ms) < MIN_VALID_IBIS:
        return nan_result

    features: dict[str, float] = {}

    # SD2 from Poincaré analysis
    # Formula: SD2² = 2·SDNN² − ½·SDSD²
    # (Brennan et al. 2001, equivalent to NeuroKit2's implementation)
    try:
        sdnn = np.std(ibi_ms, ddof=1)
        successive_diffs = np.diff(ibi_ms)
        sdsd = np.std(successive_diffs, ddof=1)

        sd2_squared = 2.0 * sdnn**2 - 0.5 * sdsd**2
        # SD2² can be negative if SDSD is very large relative to SDNN (pathological) so we clamp
        # to zero before taking sqrt.
        features["sd2"] = float(np.sqrt(max(0.0, sd2_squared)))
    except Exception:
        features["sd2"] = np.nan

    # Sample Entropy
    try:
        import neurokit2 as nk

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # (Lake et al. 2002; Richman & Moorman 2000 for parameters)
            sampen_result = nk.entropy_sample(
                ibi_ms,
                dimension=2,
                tolerance=0.2 * np.std(ibi_ms, ddof=1),
            )
            # entropy_sample returns a tuple (value, info_dict) or just
            # a float depending on NeuroKit2 version
            if isinstance(sampen_result, tuple):
                sampen_val = float(sampen_result[0])
            else:
                sampen_val = float(sampen_result)
            # SampEn can return -inf or inf on degenerate series so we map non-finite values to NaN
            # so downstream code handles them uniformly.
            features["sample_entropy"] = (
                sampen_val if np.isfinite(sampen_val) else np.nan
            )
    except Exception:
        features["sample_entropy"] = np.nan

    return features


# FITBIT-RECOVERABLE FEATURES (5 features from 1 Hz HR)


def compute_fitbit_features(hr_bpm: np.ndarray) -> dict[str, float]:
    """Compute HRV features from a 1 Hz integer HR array (Fitbit-style).

    These features approximate true HRV from consumer-grade HR data.
    The input is not an IBI series it is a sequence of HR values in
    BPM at 1 Hz (one value per second).  Features are derived via
    pseudo-RR conversion: rr_ms = 60000 / hr_bpm.

    IMPORTANT CAVEAT (Peng et al. 2018):
      Pseudo-SDNN and pseudo-RMSSD computed from 1 Hz HR are
      approximations of true beat-to-beat HRV.  The 1 Hz sampling
      averages out beat-to-beat variability.  These features capture
      the trend of HRV but underestimate its true magnitude.

    Features:
    mean_hr : Mean heart rate (BPM).
    std_hr : Standard deviation of HR (BPM).
    pseudo_sdnn : SD of pseudo-RR intervals (ms).  Approximation of
        true SDNN, systematic underestimate due to 1 Hz averaging.
    pseudo_rmssd : RMSSD of pseudo-RR intervals (ms).  Approximation
        of true RMSSD.
    hr_range : Max HR − Min HR within window (BPM).  Captures
        sympathetic activation amplitude.  Trivially recoverable from
        any HR sensor regardless of sampling rate.

    Parameters:
    hr_bpm : np.ndarray
        Heart rate values in BPM (1 Hz, integer or float).  Expected
        length: 16 values for a 16 s window, though gaps may reduce
        this.

    Returns:
    dict[str, float]
        Feature name → value.  Returns NaN-filled dict if input is
        too short (< 4 valid HR values).
    """
    nan_result = dict.fromkeys(FITBIT_HRV_FEATURES, np.nan)

    # Remove invalid values (0, NaN, out of deployment-plausible range).
    # Uses HR_CLIP_LOW/HIGH (40/200 BPM) from kd_config, which are the
    # operational bounds for consumer wearable HR, consistent with the
    # noise simulation clip in paired_data_generator.py.
    valid_mask = (
        np.isfinite(hr_bpm) & (hr_bpm >= HR_CLIP_LOW) & (hr_bpm <= HR_CLIP_HIGH)
    )
    hr_valid = hr_bpm[valid_mask]

    if len(hr_valid) < 4:
        return nan_result

    # Convert to pseudo-RR intervals
    pseudo_rr = 60000.0 / hr_valid

    return {
        "mean_hr": float(np.mean(hr_valid)),
        "std_hr": float(np.std(hr_valid, ddof=1)) if len(hr_valid) > 1 else 0.0,
        "pseudo_sdnn": float(np.std(pseudo_rr, ddof=1)) if len(pseudo_rr) > 1 else 0.0,
        "pseudo_rmssd": (
            float(np.sqrt(np.mean(np.diff(pseudo_rr) ** 2)))
            if len(pseudo_rr) > 1
            else 0.0
        ),
        "hr_range": float(np.max(hr_valid) - np.min(hr_valid)),
    }


# PUBLIC API — MAIN ENTRY POINTS
def extract_primary_features(
    ibi_ms: np.ndarray,
    clean: bool = True,
) -> dict[str, float]:
    """Extract the 6 primary time-domain HRV features from an IBI series.

    This is the main entry point for paired_data_generator.py.

    Parameters
    ----------
    ibi_ms : np.ndarray
        Inter-beat intervals in milliseconds.  If ``clean=True``
        (default), the series will be range-filtered and ectopic-
        corrected before feature computation.
    clean : bool
        Whether to apply the full cleaning pipeline.  Set to False
        if the IBI series has already been cleaned upstream.

    Returns
    -------
    dict[str, float]
        Keys match ``kd_config.PRIMARY_HRV_FEATURES`` exactly:
        [mean_rr, sdnn, rmssd, pnn50, mean_hr, std_hr].
        Returns NaN-filled dict if the window has insufficient valid
        IBIs after cleaning (< MIN_VALID_IBIS).
    """
    nan_result = dict.fromkeys(PRIMARY_HRV_FEATURES, np.nan)

    if ibi_ms is None or len(ibi_ms) == 0:
        return nan_result

    if clean:
        ibi_ms = clean_ibi_series(ibi_ms)

    if len(ibi_ms) < MIN_VALID_IBIS:
        return nan_result

    return _compute_time_domain_features(ibi_ms)


def extract_extended_features(
    ibi_ms: np.ndarray,
    clean: bool = True,
) -> dict[str, float]:
    """Extract all 12 extended HRV features from an IBI series.

    Includes the 6 primary time-domain features plus 4 frequency-domain
    and 2 nonlinear features.  Intended for the 60 s future-work ablation
    with a re-trained teacher.

    WARNING: Frequency-domain features are unreliable at 16 s windows.
    Use ``extract_primary_features()`` for the main 16 s experiments.

    Parameters
    ----------
    ibi_ms : np.ndarray
        Inter-beat intervals in milliseconds.
    clean : bool
        Whether to apply cleaning (default True).

    Returns
    -------
    dict[str, float]
        Keys match ``kd_config.EXTENDED_HRV_FEATURES`` exactly (12 keys).
        Individual features may be NaN if their specific computation
        fails, but the dict always has all 12 keys.
    """
    nan_result = dict.fromkeys(EXTENDED_HRV_FEATURES, np.nan)

    if ibi_ms is None or len(ibi_ms) == 0:
        return nan_result

    if clean:
        ibi_ms = clean_ibi_series(ibi_ms)

    if len(ibi_ms) < MIN_VALID_IBIS:
        return nan_result

    # Time-domain (6 features)
    features = _compute_time_domain_features(ibi_ms)

    # Frequency-domain (4 features)
    freq_features = _compute_frequency_domain_features(ibi_ms)
    features.update(freq_features)

    # Nonlinear (2 features)
    nl_features = _compute_nonlinear_features(ibi_ms)
    features.update(nl_features)

    return features


def extract_fitbit_features(
    hr_bpm: np.ndarray,
) -> dict[str, float]:
    """Extract 5 Fitbit-recoverable HRV features from a 1 Hz HR array.

    Convenience wrapper around ``compute_fitbit_features()``.

    Parameters
    ----------
    hr_bpm : np.ndarray
        Heart rate values in BPM (1 Hz).

    Returns
    -------
    dict[str, float]
        Keys match ``kd_config.FITBIT_HRV_FEATURES`` exactly (5 keys).
    """
    return compute_fitbit_features(hr_bpm)


# FEATURE VECTOR CONVERSION UTILITIES


def features_to_array(
    features: dict[str, float],
    feature_names: list[str],
) -> np.ndarray:
    """Convert a feature dict to a numpy array in canonical order.

    Parameters:
    features : dict[str, float]
        Feature name → value mapping.
    feature_names : list[str]
        Ordered list of feature names (e.g. PRIMARY_HRV_FEATURES).

    Returns:
    np.ndarray
        Shape (n_features,).  Missing keys are filled with NaN.
    """
    return np.array(
        [features.get(name, np.nan) for name in feature_names],
        dtype=np.float64,
    )


def is_valid_window(features: dict[str, float]) -> bool:
    """Check whether a feature dict represents a valid (non-NaN) window.

    A window is valid if ALL features are finite (not NaN, not Inf).
    Invalid windows should be excluded from training.

    Parameters:
    features : dict[str, float]
        Feature dict from any extract_* function.

    Returns:
    bool
        True if all values are finite.
    """
    return all(
        isinstance(v, (int, float)) and np.isfinite(v) for v in features.values()
    )

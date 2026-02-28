# WESAD Feature Extraction Pipeline
#
# Extracts handcrafted features from each windowed segment for traditional ML (Random Forest).
# Operates on window metadata produced by WESADWindowing to avoid
# memory explosion
#
# Feature extraction strategy per sensor:
#
#   ECG  (chest 700 Hz)  : NeuroKit2: HRV time-domain, HRV frequency-domain (LF/HF only,
#                           VLF/ULF dropped), wave morphology,
#                           ECG-derived respiration (EDR)
#   EMG  (chest 700 Hz)  : NeuroKit2 activation + manual statistical/spectral features
#   EDA  (chest 700 Hz / wrist 64 Hz) : NeuroKit2: SCR peaks, SCL, tonic/phasic decomposition
#   BVP  (wrist 64 Hz)   : NeuroKit2: PPG peaks : HRV time/frequency/nonlinear
#   RESP (chest 700 Hz)  : NeuroKit2: breathing rate, inhalation/exhalation durations
#   TEMP (chest 700 Hz / wrist 64 Hz) : Manual: mean, std, min, max, slope, range
#   ACC  (chest 700 Hz / wrist 64 Hz) : Manual on 5s sub-windows, aggregated (mean/std) per Schmidt et al. (2018)
#
# Normalization:
#   Per-subject min-max using the BASELINE condition (label 1.0) as reference.
#   This follows the literature consensus (Schmidt 2018, multiple sources in notes)
#   and prevents data leakage between subjects.
#   Normalization uses only baseline windows from that subject to compute min/max,
#   then applies those bounds to ALL windows of that subject (stress, amusement, etc.).
#   Features where min == max (zero variance) are set to 0.0.
#
# VLF/ULF HRV note:
#   NeuroKit2's hrv_frequency() will compute VLF features even on 60s windows, but the
#   values are unreliable (the Task Force of the European Society of Cardiology recommends
#   ≥300s for VLF). We explicitly drop VLF/ULF columns after extraction.
#
# References:
#   Schmidt, P. et al. (2018). Introducing WESAD. ICMI '18.
#   Makowski, D. et al. (2021). NeuroKit2. Behavior Research Methods.
#   Task Force of ESC and NASPE (1996). HRV Standards of measurement.

import json
import os
import pickle
import time
import traceback
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.integrate import simpson
from scipy.signal import find_peaks, welch

# VLF/ULF columns to drop from HRV frequency-domain output
# These require ≥300s windows for reliable estimation

# _VLF_ULF_COLS = [
#     "HRV_VLF",
#     "HRV_ULF",
#     "HRV_LFHF",  # LF/HF ratio is kept and we recompute it without VLF normalisation (there is a source in my notes that does this)
#     # NeuroKit2 VLF-related columns (varies by version, catch all known ones)
#     "HRV_VHF",
#     "HRV_LnHF",  # keep
# ]

# We keep LF, HF, LF/HF ratio, and total power. Drop VLF/ULF and any
# normalised-by-total variants that include VLF.
_VLF_ULF_DROP_PATTERNS = ["VLF", "ULF"]


def _drop_vlf_ulf(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c for c in df.columns if any(pat in c for pat in _VLF_ULF_DROP_PATTERNS)
    ]
    return df.drop(columns=drop_cols)


#  Individual sensor feature extractors


def _extract_ecg_features(signal: np.ndarray, fs: int) -> dict:
    # ECG feature extraction using NeuroKit2.

    # Extracts:
    #     - HRV time-domain (RMSSD, SDNN, pNN50, meanNN, etc.)
    #     - HRV frequency-domain (LF, HF, LF/HF — VLF/ULF dropped)
    #     - HRV nonlinear (SD1, SD2, ApEn, SampEn, DFA)
    #     - Wave morphology (P/Q/S/T peak intervals, P/R/T durations)
    #     - ECG-derived respiration rate (EDR)

    # Returns empty dict on failure (e.g. too few R-peaks detected).
    signal = signal.astype(np.float64).flatten()

    # R-peak detection
    try:
        r_peaks_signal, r_peaks_info = nk.ecg_peaks(signal, sampling_rate=fs)
    except Exception:
        # Total failure: No R-peaks means no HRV features at all so discard window
        return {}

    # Validate if enough peaks for meaningful HRV?
    # Rejects windows where peak detection produced implausible results
    # (eg noise spikes detected as heartbeats or flat signal with no peaks)
    n_peaks = r_peaks_signal["ECG_R_Peaks"].sum()
    duration_min = len(signal) / fs / 60.0
    # Physiological bounds: 40–220 bpm
    if n_peaks < (40 * duration_min) or n_peaks > (220 * duration_min):
        return {}

    features = {}

    # HRV time-domain
    # Can fail if RR interval series is too short for a specific metric (eg. pNN50 needs >1 interval >50ms)
    try:
        hrv_time = nk.hrv_time(r_peaks_signal, sampling_rate=fs)
        hrv_time = hrv_time.to_dict(orient="records")[0]
        features.update({f"ECG_{k}": v for k, v in hrv_time.items()})
    except Exception:  # nosec B110 - partial NK2 failure, other HRV blocks still run
        pass

    # HRV frequency-domain
    # Can fail if RR series is too irregular
    try:
        hrv_freq = nk.hrv_frequency(r_peaks_signal, sampling_rate=fs)
        hrv_freq = _drop_vlf_ulf(hrv_freq)
        hrv_freq = hrv_freq.to_dict(orient="records")[0]
        features.update({f"ECG_{k}": v for k, v in hrv_freq.items()})
    except (
        Exception
    ):  # nosec B110 - partial NK2 failure, imputation handles missing values
        pass

    # HRV nonlinear
    # can fail on short or irregular windows
    try:
        hrv_nl = nk.hrv_nonlinear(r_peaks_signal, sampling_rate=fs)
        hrv_nl = hrv_nl.to_dict(orient="records")[0]
        features.update({f"ECG_{k}": v for k, v in hrv_nl.items()})
    except (
        Exception
    ):  # nosec B110 - SampEn/ApEn require aprx 20 RR intervals, expected on short windows
        pass

    # Wave delineation (P, Q, S, T morphology)
    # Can fail if R-peaks are too close together
    try:
        _, waves = nk.ecg_delineate(signal, r_peaks_info, sampling_rate=fs)
        wave_feats = _ecg_wave_features(waves, fs)
        features.update(wave_feats)
    except (
        Exception
    ):  # nosec B110 - wave boundary detection fails on noisy windows, imputation handlesit
        pass

    # ECG-derived respiration (EDR)
    # Can fail if heart rate is too regular (no modulation) or if the signal is too short to estimate breathing rate
    try:
        ecg_rate = nk.signal_rate(
            r_peaks_signal, sampling_rate=fs, desired_length=len(signal)
        )
        edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs)
        edr_peaks_info = nk.signal_findpeaks(edr)
        if len(edr_peaks_info.get("Distance", [])) > 1:
            edr_dists = edr_peaks_info["Distance"]
            features["ECG_EDR_MeanDistance"] = np.nanmean(edr_dists)
            features["ECG_EDR_RMSSD"] = np.sqrt(np.nanmean(np.diff(edr_dists) ** 2))
    except (
        Exception
    ):  # nosec B110 - respiratory modulation too weak to detect, imputation handles
        pass

    return features


def _ecg_wave_features(waves: dict, fs: int) -> dict:
    features = {}

    # Peak-to-peak intervals for P, Q, S, T
    for peak_name in ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]:
        if peak_name in waves:
            peak_indices = np.array(waves[peak_name], dtype=float)
            peak_indices = peak_indices[~np.isnan(peak_indices)].astype(int)
            if len(peak_indices) > 1:
                intervals = np.diff(peak_indices) / fs * 1000  # ms
                # Physiological filter: 40–220 bpm → 273–1500 ms
                intervals = intervals[(intervals >= 273) & (intervals <= 1500)]
                if len(intervals) > 0:
                    features[f"{peak_name}_Interval_Mean"] = np.mean(intervals)
                    features[f"{peak_name}_Interval_SD"] = np.std(intervals)

    # Wave durations (onset to offset) for P, R, T
    for wave in ["P", "R", "T"]:
        onset_key = f"ECG_{wave}_Onsets"
        offset_key = f"ECG_{wave}_Offsets"
        if onset_key in waves and offset_key in waves:
            onsets = np.array(waves[onset_key], dtype=float)
            offsets = np.array(waves[offset_key], dtype=float)
            onsets = onsets[~np.isnan(onsets)].astype(int)
            offsets = offsets[~np.isnan(offsets)].astype(int)

            if len(onsets) > 0 and len(offsets) > 0:
                durations = _paired_durations(onsets, offsets, fs)
                if len(durations) > 0:
                    features[f"ECG_{wave}_Duration_Mean"] = np.mean(durations)
                    features[f"ECG_{wave}_Duration_SD"] = np.std(durations)

    return features


def _paired_durations(onsets: np.ndarray, offsets: np.ndarray, fs: int) -> np.ndarray:
    # Pair onsets with their nearest subsequent offset, return durations in ms.
    durations = []
    j = 0
    for onset in onsets:
        # Advance offset index to the first offset >= onset
        while j < len(offsets) and offsets[j] < onset:
            j += 1
        if j < len(offsets):
            dur_ms = (offsets[j] - onset) / fs * 1000
            if 0 < dur_ms < 300:  # physiological max wave duration
                durations.append(dur_ms)
            j += 1
    return np.array(durations)


def _extract_emg_features(signal: np.ndarray, fs: int) -> dict:
    # EMG feature extraction.

    # Uses NeuroKit2 for cleaning/activation, plus manual statistical and
    # spectral features following the reference repo and Schmidt 2018.
    signal = signal.astype(np.float64).flatten()
    features = {}

    # Time-domain statistical features
    features["EMG_Mean"] = np.mean(signal)
    features["EMG_STD"] = np.std(signal)
    features["EMG_Range"] = np.ptp(signal)
    features["EMG_AbsIntegral"] = np.sum(np.abs(signal))
    features["EMG_Median"] = np.median(signal)
    features["EMG_P10"] = np.percentile(signal, 10)
    features["EMG_P90"] = np.percentile(signal, 90)
    features["EMG_Variance"] = np.var(signal)
    features["EMG_SSI"] = np.sum(signal**2)  # Simple Square Integral

    # Asymmetry (skewness proxy)
    std = np.std(signal)
    if std > 0:
        features["EMG_Asymmetry"] = np.mean(signal**3) / (std**3)
    else:
        features["EMG_Asymmetry"] = 0.0

    # Spectral features via Welch's PSD estimate
    # Can fail on all-zero or all-NaN after preprocessing signals
    try:
        freqs, psd = welch(signal, fs=fs)
        psd = np.abs(psd)  # ensure real
        total_power = np.sum(psd)

        if total_power > 0:
            features["EMG_MeanFreq"] = np.sum(freqs * psd) / total_power
            cumulative = np.cumsum(psd)
            median_idx = np.searchsorted(cumulative, 0.5 * total_power)
            features["EMG_MedianFreq"] = freqs[min(median_idx, len(freqs) - 1)]
            features["EMG_PeakFreq"] = freqs[np.argmax(psd)]
            features["EMG_MeanPSD"] = np.mean(psd)

            # Energy in physiologically meaningful EMG bands (100Hz–350Hz)
            # All content below 100Hz was removed by the high-pass filter in preprocessing
            band_limits = [100, 150, 200, 250, 300, 350]
            for i in range(len(band_limits) - 1):
                mask = (freqs >= band_limits[i]) & (freqs < band_limits[i + 1])
                features[f"EMG_Band_{band_limits[i]}_{band_limits[i+1]}Hz"] = np.sum(
                    psd[mask]
                )
    except Exception:  # nosec B110 - imputation willhandle missing spectral features
        pass

    # Peak features
    try:
        peaks, _ = find_peaks(signal)
        features["EMG_NumPeaks"] = len(peaks)
        if len(peaks) > 0:
            peak_amps = signal[peaks]
            features["EMG_MeanPeakAmp"] = np.mean(peak_amps)
            features["EMG_STDPeakAmp"] = np.std(peak_amps)
            features["EMG_SumPeakAmp"] = np.sum(peak_amps)
        else:
            features["EMG_MeanPeakAmp"] = 0.0
            features["EMG_STDPeakAmp"] = 0.0
            features["EMG_SumPeakAmp"] = 0.0
    except Exception:  # nosec B110 - imputation willhandle missing peak features
        pass

    return features


def _extract_eda_features(signal: np.ndarray, fs: int) -> dict:
    # EDA feature extraction using NeuroKit2.

    # Decomposes into tonic (SCL) and phasic (SCR) components via cvxEDA,
    # then extracts features from both.
    signal = signal.astype(np.float64).flatten()
    features = {}

    # Basic statistical features (always available)
    features["EDA_Mean"] = np.mean(signal)
    features["EDA_STD"] = np.std(signal)
    features["EDA_Min"] = np.min(signal)
    features["EDA_Max"] = np.max(signal)
    features["EDA_Range"] = np.ptp(signal)
    features["EDA_Slope"] = np.mean(np.gradient(signal))

    # Correlation with time
    time_axis = np.arange(len(signal)) / fs
    if np.std(signal) > 0:
        features["EDA_CorrTime"] = np.corrcoef(signal, time_axis)[0, 1]
    else:
        features["EDA_CorrTime"] = 0.0

    # Area under curve
    features["EDA_AUC"] = np.trapz(signal, dx=1.0 / fs)

    # NeuroKit2 decomposition
    try:
        eda_processed, eda_info = nk.eda_process(signal, sampling_rate=fs)

        # Tonic (SCL) features
        scl = eda_processed["EDA_Tonic"].values
        features["EDA_SCL_Mean"] = np.mean(scl)
        features["EDA_SCL_STD"] = np.std(scl)

        # Phasic (SCR) features
        scr = eda_processed["EDA_Phasic"].values
        features["EDA_SCR_Mean"] = np.mean(scr)
        features["EDA_SCR_STD"] = np.std(scr)

        # SCR peak count and amplitude
        scr_peaks = eda_processed["SCR_Peaks"].values
        n_scr_peaks = int(np.nansum(scr_peaks))
        features["EDA_SCR_NumPeaks"] = n_scr_peaks

        if "SCR_Amplitude" in eda_processed.columns:
            scr_amps = eda_processed["SCR_Amplitude"].dropna().values
            if len(scr_amps) > 0:
                features["EDA_SCR_MeanAmp"] = np.mean(scr_amps)
                features["EDA_SCR_SumAmp"] = np.sum(scr_amps)
            else:
                features["EDA_SCR_MeanAmp"] = 0.0
                features["EDA_SCR_SumAmp"] = 0.0
        else:
            features["EDA_SCR_MeanAmp"] = 0.0
            features["EDA_SCR_SumAmp"] = 0.0

        # Interval-related features
        try:
            eda_interval = nk.eda_intervalrelated(eda_processed, sampling_rate=fs)
            for col in eda_interval.columns:
                features[f"EDA_{col}"] = eda_interval[col].iloc[0]
        except (
            Exception
        ):  # nosec B110 - NK2 eda_intervalrelated fails on short windows, imputation handlesit
            pass

    except (
        Exception
    ):  # nosec B110 - if NeuroKit2 decomposition fails, we still have the basic features
        pass

    return features


def _extract_bvp_features(signal: np.ndarray, fs: int) -> dict:
    # BVP/PPG feature extraction using NeuroKit2.

    # BVP and PPG are physiologically the same signal.
    # Extracts HRV time, frequency, and nonlinear features from detected peaks.
    signal = signal.astype(np.float64).flatten()
    features = {}

    try:
        ppg_processed, ppg_info = nk.ppg_process(signal, sampling_rate=fs)
    except Exception:
        # Total failure: no peaks detected, window unusable for HRV
        return {}

    # Validate peak count w same physiological bounds as ECG (40–220 bpm)
    n_peaks = ppg_processed["PPG_Peaks"].sum()
    duration_min = len(signal) / fs / 60.0
    if n_peaks < (40 * duration_min) or n_peaks > (220 * duration_min):
        return {}

    # HRV time-domain
    # Can fail if IBI series is too short for a specific metric
    try:
        hrv_time = nk.hrv_time(ppg_processed, sampling_rate=fs)
        hrv_time = hrv_time.to_dict(orient="records")[0]
        features.update({f"BVP_{k}": v for k, v in hrv_time.items()})
    except (
        Exception
    ):  # nosec B110 - partial NK2 failure, imputation handles missing values
        pass

    #  HRV frequency-domain (drop VLF/ULF)
    # Failure expected on irregular IBI series
    try:
        hrv_freq = nk.hrv_frequency(ppg_processed, sampling_rate=fs)
        hrv_freq = _drop_vlf_ulf(hrv_freq)
        hrv_freq = hrv_freq.to_dict(orient="records")[0]
        features.update({f"BVP_{k}": v for k, v in hrv_freq.items()})
    except (
        Exception
    ):  # nosec B110 - partial NK2 failure, imputation handles missing values
        pass

    #  HRV nonlinear
    try:
        hrv_nl = nk.hrv_nonlinear(ppg_processed, sampling_rate=fs)
        hrv_nl = hrv_nl.to_dict(orient="records")[0]
        features.update({f"BVP_{k}": v for k, v in hrv_nl.items()})
    except (
        Exception
    ):  # nosec B110 - Expected on short or irregular IBI series and imputation will handle it
        pass

    return features


def _extract_resp_features(signal: np.ndarray, fs: int) -> dict:
    # Respiration feature extraction using NeuroKit2.
    # Extracts breathing rate, inhalation/exhalation durations, and their ratio
    signal = signal.astype(np.float64).flatten()
    features = {}

    # Can fail if the signal has too few respiratory cycles in the 60s window
    try:
        rsp_processed, rsp_info = nk.rsp_process(signal, sampling_rate=fs)
    except Exception:
        # Total failure: no respiratory cycles detected, window unusable
        return {}

    # Validate breathing rate is physiologically plausible (3–60 breaths/min)
    try:
        n_cycles = rsp_processed["RSP_Phase"].diff().abs().sum() / 2
        duration_min = len(signal) / fs / 60.0
        if n_cycles < (3 * duration_min) or n_cycles > (60 * duration_min):
            return {}
    except (
        Exception
    ):  # nosec B110 RSP_Phase column missing in some NK2 versions, skip validation rather than discard
        pass

    # Interval-related features (breathing rate, phase durations)
    try:
        rsp_interval = nk.rsp_intervalrelated(rsp_processed, sampling_rate=fs)
        for col in rsp_interval.columns:
            features[f"RESP_{col}"] = rsp_interval[col].iloc[0]
    except (
        Exception
    ):  # nosec B110 - too few respiratory cycles, imputation handles missing values
        pass

    # Inhalation/exhalation ratio — computed manually for robustness
    # Normal relaxed breathing: I/E ≈ 1:2 (longer exhale)
    # Stress breathing: ratio approaches 1:1 or inverts
    insp_dur = features.get("RESP_RSP_Phase_Duration_Inspiration")
    exp_dur = features.get("RESP_RSP_Phase_Duration_Expiration")
    if insp_dur and exp_dur and insp_dur > 0 and exp_dur > 0:
        features["RESP_InhaleExhaleRatio"] = insp_dur / exp_dur

    return features


def _extract_temp_features(signal: np.ndarray) -> dict:
    # Skin temperature feature extraction (manual).

    # No NeuroKit2 module exists for temperature — it's a very slow-changing
    # signal and simple statistics are standard.
    signal = signal.astype(np.float64).flatten()

    features = {
        "TEMP_Mean": np.mean(signal),
        "TEMP_STD": np.std(signal),
        "TEMP_Min": np.min(signal),
        "TEMP_Max": np.max(signal),
        "TEMP_Range": np.ptp(signal),
    }

    # Linear slope over the window
    if len(signal) > 1:
        coeffs = np.polyfit(np.arange(len(signal)), signal, 1)
        features["TEMP_Slope"] = coeffs[0]
    else:
        features["TEMP_Slope"] = 0.0

    return features


def _extract_acc_subwindow_features(subwindow_df: pd.DataFrame, fs: int) -> dict:
    # Extract features from a single ACC sub-window (5s).

    # Features per axis + combined:
    #     - Mean, STD, absolute integral, peak frequency
    features = {}
    axis_cols = list(subwindow_df.columns)  # like ["c_acc_x", "c_acc_y", "c_acc_z"]

    for col in axis_cols:
        axis_label = col.split("_")[-1]  # "x", "y", "z"
        vals = subwindow_df[col].values.astype(np.float64)

        features[f"ACC_{axis_label}_Mean"] = np.mean(vals)
        features[f"ACC_{axis_label}_STD"] = np.std(vals)
        features[f"ACC_{axis_label}_AbsIntegral"] = simpson(np.abs(vals), dx=1.0 / fs)

        # Peak frequency
        # rfftfreq returns the corresponding frequency axis in Hz
        # argmax(|yf|) finds the bin with highest amplitude, so, the dominant motion frequency
        # Can fail on sub-windows with all-zero, or single sample
        try:
            N = len(vals)
            if N > 1:
                yf = rfft(vals)
                xf = rfftfreq(N, 1.0 / fs)
                features[f"ACC_{axis_label}_PeakFreq"] = xf[np.argmax(np.abs(yf))]
            else:
                features[f"ACC_{axis_label}_PeakFreq"] = 0.0
        except Exception:
            features[f"ACC_{axis_label}_PeakFreq"] = (
                0.0  # Fallback to 0.0 rather than NaN bc no motion is a valid state
            )

    # Combined across axes
    if len(axis_cols) == 3:
        all_vals = subwindow_df.values.astype(np.float64)
        features["ACC_all_Mean"] = np.mean(all_vals)
        features["ACC_all_STD"] = np.std(all_vals)
        features["ACC_all_AbsIntegral"] = simpson(
            np.abs(all_vals).sum(axis=1), dx=1.0 / fs
        )

    return features


def _extract_acc_features_aggregated(
    window_df: pd.DataFrame,
    acc_cols: list[str],
    acc_subwindow_samples: int,
    fs: int,
) -> dict:
    # ACC feature extraction with sub-window aggregation.

    # Per Schmidt 2018: extract features on 5s sub-windows within each 60s
    # window, then aggregate (mean and std of sub-window features) to produce
    # the final 60s-level feature vector.
    acc_present = [c for c in acc_cols if c in window_df.columns]
    if not acc_present:
        return {}

    acc_data = window_df[acc_present].reset_index(drop=True)
    n = len(acc_data)

    # Split into non-overlapping sub-windows
    subwindow_features = []
    pos = 0
    while pos + acc_subwindow_samples <= n:
        sub_df = acc_data.iloc[pos : pos + acc_subwindow_samples]
        sub_feats = _extract_acc_subwindow_features(sub_df, fs)
        subwindow_features.append(sub_feats)
        pos += acc_subwindow_samples

    if not subwindow_features:
        return {}

    # Aggregate: mean and std of each sub-window feature across all sub-windows
    sub_df = pd.DataFrame(subwindow_features)
    aggregated = {}
    for col in sub_df.columns:
        vals = sub_df[col].dropna().values
        if len(vals) > 0:
            aggregated[f"{col}_SubMean"] = np.mean(vals)
            aggregated[f"{col}_SubSTD"] = np.std(vals)
        else:
            aggregated[f"{col}_SubMean"] = np.nan
            aggregated[f"{col}_SubSTD"] = np.nan

    return aggregated


#  Main feature extraction orchestrator


class WESADFeatureExtractor:
    # Extracts features from windowed WESAD data.

    # Takes the window metadata list from WESADWindowing.segment() and the
    # loaded preprocessed DataFrame, extracts per-sensor features for each
    # window, and produces a flat feature DataFrame.

    # Parameters
    # df : pd.DataFrame
    #     The preprocessed data (already loaded by WESADWindowing.load_data()).
    # windows : list[dict]
    #     Window metadata from WESADWindowing.segment().
    # config_path : str
    #     Path to the JSON config file (for sensor list and sampling rates).
    # wrist : bool
    #     Whether this is wrist data.
    # baseline_label : float
    #     Label value for the baseline condition (default: 1.0).
    #     Used for per-subject normalization.

    def __init__(
        self,
        df: pd.DataFrame,
        windows: list[dict],
        config_path: str,
        wrist: bool = False,
        baseline_label: float = 1.0,
    ):
        self.df = df
        self.windows = windows
        self.wrist = wrist
        self.baseline_label = baseline_label

        with open(config_path) as f:
            config = json.load(f)

        self.sensors = [k for k, v in config["sensors"].items() if v]
        self._sensor_rates: dict[str, int] = config["sampling rate"]
        self.sampling_rate: int = int(max(config["sampling rate"].values()))

        # ACC configuration
        if wrist:
            self._acc_cols = ["w_acc_x", "w_acc_y", "w_acc_z"]
        else:
            self._acc_cols = ["c_acc_x", "c_acc_y", "c_acc_z"]

        # ACC sub-window: 5s per Schmidt 2018
        self._acc_subwindow_samples = int(5.0 * self.sampling_rate)

        # Pre-compute per-subject DataFrames for efficient slicing
        self._subject_dfs = {}
        for sid, group in df.groupby("sid"):
            self._subject_dfs[sid] = group.reset_index(drop=True)

        print("WESADFeatureExtractor initialized:")
        print(f"  Device: {'wrist' if wrist else 'chest'}")
        print(f"  Active sensors: {self.sensors}")
        print(f"  Windows to process: {len(windows)}")
        print(f"  Subjects: {sorted(self._subject_dfs.keys())}")

    def _get_rate(self, sensor_key: str) -> int:
        # Look up sampling rate for a sensor from config
        return self._sensor_rates.get(sensor_key, self.sampling_rate)

    def _get_window_data(self, window: dict) -> pd.DataFrame:
        # Slice raw data for a window using metadata ( no copyi, avoid memory explosion).
        subject_df = self._subject_dfs[window["sid"]]
        return subject_df.iloc[window["start_sample"] : window["end_sample"]]

    def extract_all(
        self, checkpoint_path: str | None = None, checkpoint_every: int = 1000
    ) -> pd.DataFrame:
        # Extract features from all windows with periodic checkpointing.
        #
        # KEY DESIGN: each checkpoint batch is flushed to a numbered chunk file on
        # disk rather than accumulated in RAM. This prevents memory exhaustion on large
        # runs (e.g. 121k windows at 0.25s slide). Peak RAM usage is ~checkpoint_every
        # rows at a time rather than the full dataset.
        #
        # Checkpoint file layout (co-located with checkpoint_path):
        #   <checkpoint_path>           — metadata: next_idx, failed, chunk_paths
        #   <checkpoint_path>.chunk_0   — first batch as a pickled DataFrame
        #   <checkpoint_path>.chunk_1   — second batch, etc.
        #
        # Parameters
        # checkpoint_path : str, optional
        #     Path for checkpoint metadata file. If provided and the file exists,
        #     resumes from the last saved state. If None, runs fully in-memory
        #     (original behaviour, fine for small window counts).
        # checkpoint_every : int
        #     Flush a chunk and update metadata every N windows (default: 1000).
        #
        # Returns:
        # pd.DataFrame
        #     One row per window, columns are feature names plus metadata
        #     (sid, label, window_idx, is_overlapped).

        total = len(self.windows)
        current_batch: list[dict] = []  # holds only ONE chunk at a time
        failed = 0
        start_idx = 0
        chunk_paths: list[str] = []  # paths to already-flushed chunk files on disk

        # helpers (take explicit path arg so mypy can verify str, not str | None)

        def flush_chunk(path: str, batch: list[dict], chunk_idx: int) -> str:
            chunk_file = f"{path}.chunk_{chunk_idx}"
            os.makedirs(os.path.dirname(os.path.abspath(chunk_file)), exist_ok=True)
            pd.DataFrame(batch).to_pickle(chunk_file)
            return chunk_file

        def save_meta(path: str, next_idx: int) -> None:
            meta = {"failed": failed, "next_idx": next_idx, "chunk_paths": chunk_paths}
            with open(path, "wb") as fh:
                pickle.dump(meta, fh)

        # Resume from checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as fh:
                ckpt_meta = pickle.load(fh)
            failed = ckpt_meta["failed"]
            start_idx = ckpt_meta["next_idx"]
            chunk_paths = ckpt_meta["chunk_paths"]
            already_done = sum(
                len(pd.read_pickle(p)) for p in chunk_paths if os.path.exists(p)
            )
            print(
                f"  ▶ Resuming from checkpoint: {start_idx}/{total} windows done "
                f"({already_done} rows in {len(chunk_paths)} chunks, {failed} failed)"
            )

        start_time = time.time()

        # Suppress expected NeuroKit2 and numerical warnings during batch extraction.
        # Suppression is appropriate here: warnings are expected, documented, and handled
        # at the extractor level via try/except. NK2 emits peak-detection and convergence
        # warnings on almost every window by design. Warnings are restored when the
        # with-block exits.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning
            )  # numpy/scipy numerical edge cases
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )  # NeuroKit2 peak detection warnings

            for i in range(start_idx, total):
                window = self.windows[i]

                if i > start_idx and (i - start_idx) % 100 == 0:
                    elapsed = time.time() - start_time
                    done_this_run = i - start_idx
                    rate = done_this_run / max(elapsed, 1e-9)
                    eta = (total - i) / rate
                    print(
                        f"  [{i}/{total}] {rate:.1f} win/sec  "
                        f"ETA: {eta / 60:.1f} min  failed: {failed}  "
                        f"chunks on disk: {len(chunk_paths)}"
                    )

                try:
                    feats = self._extract_window_features(window)
                    if feats:
                        feats["sid"] = window["sid"]
                        feats["label"] = window["label"]
                        feats["window_idx"] = window["window_idx"]
                        feats["is_overlapped"] = window["is_overlapped"]
                        current_batch.append(feats)
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"  Error on window {i} (S{window['sid']}): {e}")
                        traceback.print_exc()

                # Periodic flush to disk
                if (
                    checkpoint_path
                    and (i + 1) % checkpoint_every == 0
                    and current_batch
                ):
                    chunk_file = flush_chunk(
                        checkpoint_path, current_batch, len(chunk_paths)
                    )
                    chunk_paths.append(chunk_file)
                    save_meta(checkpoint_path, i + 1)
                    print(
                        f"  💾 Chunk {len(chunk_paths)} saved "
                        f"({len(current_batch)} rows → ...chunk_{len(chunk_paths) - 1})"
                    )
                    current_batch = []  # release memory

            # Flush any remaining rows in the final partial batch
            if checkpoint_path and current_batch:
                chunk_file = flush_chunk(
                    checkpoint_path, current_batch, len(chunk_paths)
                )
                chunk_paths.append(chunk_file)
                save_meta(checkpoint_path, total)
                current_batch = []

        elapsed = time.time() - start_time

        # Concat all chunks from disk (or fall back to in-memory if no checkpointing)
        if chunk_paths:
            print(f"\nConcatenating {len(chunk_paths)} chunk files from disk...")
            df_features = pd.concat(
                [pd.read_pickle(p) for p in chunk_paths if os.path.exists(p)],
                ignore_index=True,
            )
        elif current_batch:
            # checkpoint_path was None, small run, everything is still in current_batch
            df_features = pd.DataFrame(current_batch)
        else:
            raise ValueError("No features extracted from any window!")

        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("\nFeature extraction complete:")
        print(f"  Processed: {len(df_features)}/{total} windows ({failed} failed)")
        print(f"  Time: {elapsed:.1f}s ({total / max(elapsed, 1):.1f} windows/sec)")
        print(f"  Feature matrix shape: {df_features.shape}")
        print(f"  NaN count: {df_features.isna().sum().sum()}")

        return df_features

    def _extract_window_features(self, window: dict) -> dict:
        # Extract features from a single window across all active sensors.
        window_df = self._get_window_data(window)
        features = {}

        for sensor in self.sensors:
            sensor_feats = self._dispatch_sensor(sensor, window_df)
            features.update(sensor_feats)

        return features if features else {}

    def _dispatch_sensor(self, sensor: str, window_df: pd.DataFrame) -> dict:
        # Route a sensor key to its feature extractor.
        match sensor:
            # Chest signals
            case "ecg":
                return _extract_ecg_features(
                    window_df["ecg"].values, self._get_rate("ecg")
                )
            case "emg":
                return _extract_emg_features(
                    window_df["emg"].values, self._get_rate("emg")
                )
            case "c_eda":
                feats = _extract_eda_features(
                    window_df["c_eda"].values, self._get_rate("c_eda")
                )
                # Prefix to distinguish from wrist EDA
                return {f"C_{k}": v for k, v in feats.items()}
            case "c_temp":
                feats = _extract_temp_features(window_df["c_temp"].values)
                return {f"C_{k}": v for k, v in feats.items()}
            case "resp":
                return _extract_resp_features(
                    window_df["resp"].values, self._get_rate("resp")
                )
            case "c_acc":
                return _extract_acc_features_aggregated(
                    window_df,
                    self._acc_cols,
                    self._acc_subwindow_samples,
                    self._get_rate("c_acc_x"),
                )

            # Wrist signals
            case "bvp":
                return _extract_bvp_features(
                    window_df["bvp"].values, self._get_rate("bvp")
                )
            case "w_eda":
                feats = _extract_eda_features(
                    window_df["w_eda"].values, self._get_rate("w_eda")
                )
                return {f"W_{k}": v for k, v in feats.items()}
            case "w_temp":
                feats = _extract_temp_features(window_df["w_temp"].values)
                return {f"W_{k}": v for k, v in feats.items()}
            case "w_acc":
                return _extract_acc_features_aggregated(
                    window_df,
                    self._acc_cols,
                    self._acc_subwindow_samples,
                    self._get_rate("w_acc_x"),
                )

            case _:
                return {}


#  Per-subject min-max normalization using baseline


def normalize_per_subject(
    df: pd.DataFrame,
    baseline_label: float = 1.0,
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    # Per-subject min-max normalization using baseline condition.

    # For each subject, computes min and max of each feature from baseline
    # windows only (label == baseline_label), then applies [0, 1] scaling
    # to ALL windows of that subject.

    # This follows the literature consensus (Schmidt 2018, and multiple
    # WESAD studies) and prevents data leakage between subjects. In a
    # real-time deployment scenario, the baseline data would be collected
    # during a calibration phase at the start of monitoring. (literature warns that
    # in real-time applications, the entire data would not be available for normalization
    # and recommends using 5 minutes of baseline data to compute normalization parameters.)

    # df : pd.DataFrame
    #     Feature matrix from WESADFeatureExtractor.extract_all().
    # baseline_label : float
    #     Label value for baseline condition (default: 1.0).
    # metadata_cols : list[str], optional
    #     Columns to exclude from normalization (kept as-is).
    #     Defaults to ["sid", "label", "window_idx", "is_overlapped"].

    # Returns
    # pd.DataFrame
    #     Normalised feature matrix with same shape and columns.
    if metadata_cols is None:
        metadata_cols = ["sid", "label", "window_idx", "is_overlapped"]

    feature_cols = [c for c in df.columns if c not in metadata_cols]

    df_norm = df.copy()

    for sid, group in df.groupby("sid"):
        # Get baseline windows for this subject
        baseline_mask = group["label"] == baseline_label
        baseline_data = group.loc[baseline_mask, feature_cols]

        if len(baseline_data) == 0:
            print(
                f"  Warning: Subject {sid} has no baseline windows. "
                f"Falling back to overall subject min-max."
            )
            baseline_data = group[feature_cols]

        # Compute min and max from baseline
        feat_min = baseline_data.min()
        feat_max = baseline_data.max()
        feat_range = feat_max - feat_min

        # Handle zero-range features (constant in baseline → set to 0)
        zero_range = feat_range == 0
        feat_range[zero_range] = 1.0  # avoid division by zero

        # Apply normalization to al windows of this subject
        subject_idx = group.index
        normalized = (df_norm.loc[subject_idx, feature_cols] - feat_min) / feat_range

        # Features that were constant in baseline: set to 0
        normalized.loc[:, zero_range.values] = 0.0

        df_norm.loc[subject_idx, feature_cols] = normalized

    # Report
    n_nan = df_norm[feature_cols].isna().sum().sum()
    if n_nan > 0:
        print(f"  Post-normalization NaN count: {n_nan}")

    print(f"  Normalization complete. Shape: {df_norm.shape}")
    return df_norm


#  NaN imputation (run after normalization)


def impute_features(
    df: pd.DataFrame,
    strategy: str = "median",
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    # Impute NaN values in the feature matrix.

    # NeuroKit2 can produce NaN for some features on certain windowss so we fill those gaps

    # Parameters
    # df : pd.DataFrame
    #     Feature matrix (normalised or raw).
    # strategy : str
    #     "median" — fill NaN with per-column median (robust to outliers).
    #     "zero"   — fill with 0.0 (appropriate after min-max normalization).
    # metadata_cols : list[str], optional
    #     Columns to exclude from imputation.

    # Returns
    # pd.DataFrame
    #     Feature matrix with NaNs filled.
    if metadata_cols is None:
        metadata_cols = ["sid", "label", "window_idx", "is_overlapped"]

    feature_cols = [c for c in df.columns if c not in metadata_cols]
    df_imputed = df.copy()

    n_nan_before = df_imputed[feature_cols].isna().sum().sum()

    if strategy == "median":
        fill_values = df_imputed[feature_cols].median()
        # If an entire column is NaN, fill with 0
        fill_values = fill_values.fillna(0.0)
        df_imputed[feature_cols] = df_imputed[feature_cols].fillna(fill_values)
    elif strategy == "zero":
        df_imputed[feature_cols] = df_imputed[feature_cols].fillna(0.0)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    n_nan_after = df_imputed[feature_cols].isna().sum().sum()
    print(f"  Imputation ({strategy}): {n_nan_before} NaN → {n_nan_after} NaN")

    return df_imputed


# full pipeline in one call
def run_feature_pipeline(
    df: pd.DataFrame,
    windows: list[dict],
    config_path: str,
    output_path: str,
    wrist: bool = False,
    baseline_label: float = 1.0,
    normalize: bool = True,
    impute: bool = True,
    impute_strategy: str = "median",
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1000,
) -> pd.DataFrame:
    # Run the full feature extraction → normalization → imputation pipeline.

    # Parameters
    # df : pd.DataFrame
    #     Preprocessed data from WESADWindowing.load_data().
    # windows : list[dict]
    #     Window metadata from WESADWindowing.segment().
    # config_path : str
    #     Path to JSON config.
    # output_path : str
    #     Where to save the final feature matrix (.pkl).
    # wrist : bool
    #     Whether this is wrist data.
    # baseline_label : float
    #     Label for baseline condition.
    # normalize : bool
    #     Whether to apply per-subject normalization.
    # impute : bool
    #     Whether to impute NaN values.
    # impute_strategy : str
    #     Imputation method ("median" or "zero").
    # checkpoint_path : str, optional
    #     Path for extraction checkpoint file. Enables resume on crash.
    # checkpoint_every : int
    #     Save checkpoint every N windows (default: 1000).

    # Returns
    # pd.DataFrame
    #     Final feature matrix, saved to output_path.
    print("WESAD Feature Extraction Pipeline")

    # Ensure output directory exists before anything touches disk
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Auto-generate checkpoint path from output_path if not specified
    if checkpoint_path is None:
        checkpoint_path = output_path.replace(".pkl", "_checkpoint.pkl")

    # Feature extraction
    extractor = WESADFeatureExtractor(
        df=df,
        windows=windows,
        config_path=config_path,
        wrist=wrist,
        baseline_label=baseline_label,
    )
    df_features = extractor.extract_all(
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    # Normalization (per-subject, baseline-referenced)
    if normalize:
        print("\n\nApplying per-subject min-max normalization (baseline-referenced)...")
        df_features = normalize_per_subject(df_features, baseline_label=baseline_label)

    # Imputation
    if impute:
        print("\nImputing missing values...")
        df_features = impute_features(df_features, strategy=impute_strategy)

    #  Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_pickle(output_path)
    print(f"\nSaved feature matrix → {output_path}")
    print(f"  Shape: {df_features.shape}")
    print(f"  Columns: {list(df_features.columns)}")

    # Clean up checkpoint files: extraction succeeded, no longer needed.
    # Chunk deletion failures are intentionally swallowed: if a chunk file
    # is already gone (e.g. manual cleanup) that is not an error.
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as fh:
                ckpt_meta = pickle.load(fh)
            for chunk_file in ckpt_meta.get("chunk_paths", []):
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
        except Exception:  # nosec B110 - chunk files may already be deleted, non-fatal
            pass
        os.remove(checkpoint_path)
        print("  Checkpoint files removed (extraction complete)")

    # Summary statistics
    meta_cols = ["sid", "label", "window_idx", "is_overlapped"]
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    print(f"\n  Feature count: {len(feat_cols)}")
    print(f"  Window count: {len(df_features)}")
    label_dist = df_features["label"].value_counts().to_dict()
    print(f"  Label distribution: {label_dist}")

    return df_features

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

import contextlib
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
        # KEY DESIGN: each checkpoint batch is flushed to a separate file on disk
        # rather than accumulated in RAM. This prevents memory exhaustion on large
        # runs (e.g. 121k windows at 0.25s slide). On resume, only the metadata
        # file is loaded; the chunk files stay on disk until the final concat.
        #
        # Checkpoint file layout (all in same dir as checkpoint_path):
        #   <checkpoint_path>          — metadata: next_idx, failed, chunk_paths
        #   <checkpoint_path>.chunk_0  — first batch as pickled DataFrame
        #   <checkpoint_path>.chunk_1  — second batch, etc.
        #
        # Parameters
        # checkpoint_path : str, optional
        #     Path for checkpoint metadata file. Auto-generated from output_path
        #     if called via run_feature_pipeline.
        # checkpoint_every : int
        #     Flush a chunk and save metadata every N windows (default: 1000).
        #
        # Returns:
        # pd.DataFrame
        #     One row per window with metadata cols (sid, label, window_idx, is_overlapped).

        total = len(self.windows)
        current_batch: list[dict] = []  # only holds ONE chunk at a time
        failed = 0
        start_idx = 0
        chunk_paths: list[str] = []  # paths to already-flushed chunk files

        # Resume from checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as fh:
                meta = pickle.load(fh)
            failed = meta["failed"]
            start_idx = meta["next_idx"]
            chunk_paths = meta["chunk_paths"]
            already_extracted = sum(
                len(pd.read_pickle(p)) for p in chunk_paths if os.path.exists(p)
            )
            print(
                f"  ▶ Resuming from checkpoint: {start_idx}/{total} windows done "
                f"({already_extracted} rows in {len(chunk_paths)} chunks, {failed} failed)"
            )

        start_time = time.time()

        def _flush_chunk(batch: list[dict], idx: int) -> str:
            # Write a batch of feature dicts to a numbered chunk file and return its path.
            assert checkpoint_path is not None
            chunk_file = f"{checkpoint_path}.chunk_{idx}"
            pd.DataFrame(batch).to_pickle(chunk_file)
            return chunk_file

        def _save_meta(next_idx: int) -> None:
            assert checkpoint_path is not None
            meta = {"failed": failed, "next_idx": next_idx, "chunk_paths": chunk_paths}
            with open(checkpoint_path, "wb") as fh:
                pickle.dump(meta, fh)

        # Main extraction loop
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            for i in range(start_idx, total):
                window = self.windows[i]

                # Progress print every 100 windows
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

                # flush
                if (
                    checkpoint_path
                    and (i + 1) % checkpoint_every == 0
                    and current_batch
                ):
                    chunk_file = _flush_chunk(current_batch, len(chunk_paths))
                    chunk_paths.append(chunk_file)
                    _save_meta(i + 1)
                    print(
                        f"  💾 Chunk {len(chunk_paths)} saved "
                        f"({len(current_batch)} rows → {chunk_file})"
                    )
                    current_batch = []  # ← release memory

            # Flush any remaining rows
            if checkpoint_path and current_batch:
                chunk_file = _flush_chunk(current_batch, len(chunk_paths))
                chunk_paths.append(chunk_file)
                _save_meta(total)
                current_batch = []

        # Concat all chunks
        elapsed = time.time() - start_time
        print(
            f"\nExtraction loop done in {elapsed:.1f}s. Concatenating {len(chunk_paths)} chunks..."
        )

        if chunk_paths:
            df_features = pd.concat(
                [pd.read_pickle(p) for p in chunk_paths if os.path.exists(p)],
                ignore_index=True,
            )
        elif current_batch:
            # No checkpointing was used (checkpoint_path=None) : fall back to in-memory
            df_features = pd.DataFrame(current_batch)
        else:
            raise ValueError("No features extracted from any window!")

        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        total_extracted = len(df_features)
        print("\nFeature extraction complete:")
        print(f"  Processed: {total_extracted}/{total} windows ({failed} failed)")
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
    # impute_strategy: str = "median",
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
        meta = ["sid", "label", "window_idx", "is_overlapped"]
        n_nan = (
            df_features[[c for c in df_features.columns if c not in meta]]
            .isna()
            .sum()
            .sum()
        )
        print(
            f"\n  Skipping global imputation (will be done inside CV loop). NaN count: {n_nan}"
        )

    #  Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_pickle(output_path)
    print(f"\nSaved feature matrix → {output_path}")
    print(f"  Shape: {df_features.shape}")
    print(f"  Columns: {list(df_features.columns)}")

    # Clean up checkpoint files: extraction succeeded, no longer needed
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load metadata to find chunk file paths before deleting the metadata file
        with contextlib.suppress(Exception):
            with open(checkpoint_path, "rb") as fh:
                ckpt_meta = pickle.load(fh)
            for chunk_file in ckpt_meta.get("chunk_paths", []):
                with contextlib.suppress(OSError):
                    os.remove(chunk_file)
        with contextlib.suppress(OSError):
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


# Subject-streaming pipeline  (memory-safe alternative to run_feature_pipeline)


def run_feature_pipeline_streaming(
    data_path: str,
    windows: list[dict],
    config_path: str,
    output_path: str,
    wrist: bool = False,
    baseline_label: float = 1.0,
    normalize: bool = True,
    impute: bool = True,
    # impute_strategy: str = "median",
) -> pd.DataFrame:
    """Memory-safe feature extraction that loads one subject at a time.

    Instead of loading the entire 60M-row DataFrame into RAM and keeping
    15 per-subject slices alive simultaneously, this function:

      1. Iterates over subjects one at a time.
      2. Loads only that subject's rows from disk.
      3. Extracts features for all windows of that subject.
      4. Writes a per-subject chunk file and frees the data immediately.
      5. Concatenates all chunks at the end.

    Normalization and imputation are applied globally after all subjects
    are done, exactly as in run_feature_pipeline.

    Parameters
    ----------
    data_path : str
        Path to the preprocessed merged pickle (chest or wrist).
    windows : list[dict]
        Window metadata from WESADWindowing.segment().
    config_path : str
        Path to the JSON dataset config.
    output_path : str
        Destination path for the final feature matrix pickle.
    wrist, baseline_label, normalize, impute, impute_strategy
        Same semantics as run_feature_pipeline.
    """
    import gc

    print(f"\n{'='*60}")
    print(
        f"WESAD Feature Extraction  ({'wrist' if wrist else 'chest'})  — streaming mode"
    )
    print(f"{'='*60}")

    # Config
    with open(config_path) as f:
        config = json.load(f)
    sensors = [k for k, v in config["sensors"].items() if v]
    sensor_rates = config["sampling rate"]
    sampling_rate = int(max(sensor_rates.values()))

    acc_cols = (
        ["w_acc_x", "w_acc_y", "w_acc_z"]
        if wrist
        else ["c_acc_x", "c_acc_y", "c_acc_z"]
    )
    acc_subwindow_samples = int(5.0 * sampling_rate)

    meta_cols = ["sid", "label", "window_idx", "is_overlapped"]

    # Group windows by subject (avoids repeated full-list scans)
    from collections import defaultdict

    windows_by_sid: dict = defaultdict(list)
    for w in windows:
        windows_by_sid[w["sid"]].append(w)

    subject_ids = sorted(windows_by_sid.keys())
    print(f"Subjects to process : {[int(s) for s in subject_ids]}")
    print(f"Total windows       : {len(windows)}")
    print(f"Active sensors      : {sensors}\n")

    chunk_dir = os.path.dirname(output_path)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths: list[str] = []

    # Load the full DataFrame ONCE, write each subject to a temp file, then
    # free the big DataFrame.  This avoids re-reading 5 GB × 15 subjects.
    import tempfile

    subject_file_map: dict[float, str] = {}
    split_dir = tempfile.mkdtemp(prefix="wesad_fe_split_")

    print(f"Pre-splitting {data_path} by subject …")
    full_df = pd.read_pickle(data_path)
    for sid in subject_ids:
        subj_path = os.path.join(split_dir, f"subj_{int(sid)}.pkl")
        full_df[full_df["sid"] == sid].reset_index(drop=True).to_pickle(subj_path)
        subject_file_map[sid] = subj_path
    del full_df
    gc.collect()
    print(f"  Split into {len(subject_file_map)} files in {split_dir}\n")

    # Per-subject extraction
    t_start = time.time()
    total_failed = 0

    for sid in subject_ids:
        sid_windows = windows_by_sid[sid]
        print(f"\n── Subject {int(sid)}  ({len(sid_windows)} windows) ──")

        chunk_path = os.path.join(chunk_dir, f"_chunk_sid{int(sid)}.pkl")
        if os.path.exists(chunk_path):
            print(f"  Chunk EXISTS! Skipping extraction. -> {chunk_path}")
            chunk_paths.append(chunk_path)
            continue

        # Load only this subject's pre-split file (300 MB per subject)
        subject_df = pd.read_pickle(subject_file_map[sid])
        print(f"  Subject rows loaded: {len(subject_df):,}")
        rows: list[dict] = []
        failed = 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            for i, window in enumerate(sid_windows):
                if i > 0 and i % 200 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"    [{i}/{len(sid_windows)}]  total elapsed: {elapsed/60:.1f} min  failed: {failed}"
                    )

                try:
                    window_df = subject_df.iloc[
                        window["start_sample"] : window["end_sample"]
                    ]
                    feats: dict = {}

                    for sensor in sensors:
                        match sensor:
                            case "ecg":
                                feats.update(
                                    _extract_ecg_features(
                                        window_df["ecg"].values,
                                        sensor_rates.get("ecg", sampling_rate),
                                    )
                                )
                            case "emg":
                                feats.update(
                                    _extract_emg_features(
                                        window_df["emg"].values,
                                        sensor_rates.get("emg", sampling_rate),
                                    )
                                )
                            case "c_eda":
                                sensor_feats: dict = _extract_eda_features(
                                    window_df["c_eda"].values,
                                    sensor_rates.get("c_eda", sampling_rate),
                                )
                                feats.update(
                                    {f"C_{k}": v for k, v in sensor_feats.items()}
                                )
                            case "c_temp":
                                sensor_feats = _extract_temp_features(
                                    window_df["c_temp"].values
                                )
                                feats.update(
                                    {f"C_{k}": v for k, v in sensor_feats.items()}
                                )
                            case "resp":
                                feats.update(
                                    _extract_resp_features(
                                        window_df["resp"].values,
                                        sensor_rates.get("resp", sampling_rate),
                                    )
                                )
                            case "c_acc":
                                feats.update(
                                    _extract_acc_features_aggregated(
                                        window_df,
                                        acc_cols,
                                        acc_subwindow_samples,
                                        sensor_rates.get("c_acc_x", sampling_rate),
                                    )
                                )
                            case "bvp":
                                feats.update(
                                    _extract_bvp_features(
                                        window_df["bvp"].values,
                                        sensor_rates.get("bvp", sampling_rate),
                                    )
                                )
                            case "w_eda":
                                sensor_feats = _extract_eda_features(
                                    window_df["w_eda"].values,
                                    sensor_rates.get("w_eda", sampling_rate),
                                )
                                feats.update(
                                    {f"W_{k}": v for k, v in sensor_feats.items()}
                                )
                            case "w_temp":
                                sensor_feats = _extract_temp_features(
                                    window_df["w_temp"].values
                                )
                                feats.update(
                                    {f"W_{k}": v for k, v in sensor_feats.items()}
                                )
                            case "w_acc":
                                feats.update(
                                    _extract_acc_features_aggregated(
                                        window_df,
                                        acc_cols,
                                        acc_subwindow_samples,
                                        sensor_rates.get("w_acc_x", sampling_rate),
                                    )
                                )

                    if feats:
                        feats["sid"] = window["sid"]
                        feats["label"] = window["label"]
                        feats["window_idx"] = window["window_idx"]
                        feats["is_overlapped"] = window["is_overlapped"]
                        rows.append(feats)
                    else:
                        failed += 1

                except Exception as exc:
                    failed += 1
                    if failed <= 3:
                        print(f"    Error on window {i}: {exc}")

        total_failed += failed
        print(
            f"  Done: {len(rows)} windows successfully processed to extract features, {failed} failed."
        )

        # Write subject chunk and free memory
        chunk_path = os.path.join(chunk_dir, f"_chunk_sid{int(sid)}.pkl")
        pd.DataFrame(rows).to_pickle(chunk_path)
        chunk_paths.append(chunk_path)
        print(f"  Chunk saved → {chunk_path}")

        del subject_df, rows
        gc.collect()

    # Concatenate all subject chunks
    print(f"\nConcatenating {len(chunk_paths)} subject chunks …")
    df_features = pd.concat(
        [pd.read_pickle(p) for p in chunk_paths],
        ignore_index=True,
    )
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Clean up chunks
    for p in chunk_paths:
        with contextlib.suppress(OSError):
            os.remove(p)

    # Clean up pre-split temp files
    import shutil

    with contextlib.suppress(OSError):
        shutil.rmtree(split_dir)

    print(f"Raw feature matrix shape : {df_features.shape}")
    print(f"Total failed windows     : {total_failed}")

    # Normalization
    if normalize:
        print("\nApplying per-subject min-max normalization (baseline-referenced) …")
        df_features = normalize_per_subject(df_features, baseline_label=baseline_label)

    # Imputation
    if impute:
        n_nan = (
            df_features[[c for c in df_features.columns if c not in meta_cols]]
            .isna()
            .sum()
            .sum()
        )
        print(
            f"\n  Skipping global imputation (will be done inside CV loop). NaN count: {n_nan}"
        )

    # Save
    df_features.to_pickle(output_path)
    elapsed = time.time() - t_start

    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    print(f"\n{'='*60}")
    print(f"Feature extraction complete  ({elapsed/60:.1f} min total)")
    print(f"  Output           : {output_path}")
    print(f"  Shape            : {df_features.shape}")
    print(f"  Feature count    : {len(feat_cols)}")
    print(
        f"  Label dist       : {df_features['label'].value_counts().sort_index().to_dict()}"
    )
    print(f"  NaN count        : {df_features[feat_cols].isna().sum().sum()}")
    print(f"{'='*60}\n")

    return df_features

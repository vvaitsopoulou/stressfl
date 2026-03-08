# DL Data Preparation Pipeline

# Loads raw WESAD pickle files (one per subject), applies artifact correction,
# filters each signal at its NATIVE sampling rate, then downsamples to
# physiologically justified DL target rates.  Saves per-subject .npy arrays
# ready for wesad_dl_windowing.py.

# This module is the DL-branch equivalent of wesad_resample_merge.py.
# Key differences from the RF pipeline:
#   • Per-subject output (no merged dataframe): required for clean LOSO
#   • Downsamples instead of upsampling to preserve only real information
#   • Filters at native rates BEFORE downsampling (correct DSP order)
#   • Wrist EDA kept at native 4 Hz with no low-pass filter (Nyquist = 2 Hz
#     makes the 3 Hz LowPass a no-go, and in this case hardware anti-aliasing is sufficient)
#   • TEMP excluded from per-subject signal arrays, stored as a scalar
#     summary per window at the windowing stage (temperature changes over
#     minutes, not seconds so a CNN branch at 4 Hz adds no value)

# Downsampling Targets & Physiological Justification:

# Signal      Native Hz   DL Target Hz   Justification

# ECG         700         256            DL-ECG standard (Prajod & André, 2022).
#                                        QRS complex ≈ 0.6 s → 154 samples.
#                                        One heartbeat ≈ 0.8 s → 205 samples.
#                                        Hwang et al. (2018) Deep ECGNet validated
#                                        at this rate.

# BVP         64          32             Cardiac cycle transfers from ECG (same
#                                        ≈ 0.8 s period). Kernel 0.6 s → 19 samples.
#                                        Pool 0.8 s → 26 samples. Nyquist at 32 Hz
#                                        preserves all BVP content (fundamental
#                                        frequency < 5 Hz).

# EDA chest   700         8              SCR onset 1–3 s, rise 1–4 s, recovery
#                                        5–10 s.  Moser et al. (2024) downsampled
#                                        to 1 Hz. 8 Hz retains full SCR waveform
#                                        shape for CNN feature extraction.

# EDA wrist   4           4 (native)     Nyquist = 2 Hz, hardware anti-aliasing
#                                        eliminates content > 2 Hz.  3 Hz LP filter
#                                        is no use at this rate, so skipped entirely.
#                                        0.05 Hz HPF for phasic extraction is
#                                        applied (well within Nyquist).

# RESP        700         32             One breath cycle 3–5 s. Kernel 3 s → 96
#                                        samples. Stress-induced tachypnea up to
#                                        0.5 Hz is preserved. 32 Hz is well above
#                                        the 3 Hz bandpass upper bound.

# EMG         700         64             100 Hz HPF must run at native 700 Hz
#                                        (Nyquist 350 Hz).  After filtering, the
#                                        stress-correlated muscle tension energy
#                                        lies in 100–350 Hz.  Downsample to 64 Hz
#                                        intentionally sacrifices the > 32 Hz
#                                        content but retains the amplitude envelope.
#                                        NOTE: EMG is rectified (absolute value)
#                                        AFTER high-pass filtering and BEFORE
#                                        downsampling.  This converts the zero-mean
#                                        AC signal into a unipolar amplitude
#                                        envelope whose energy is concentrated in
#                                        the low-frequency range (< 20 Hz), which
#                                        survives the 64 Hz downsample without
#                                        aliasing.

# ACC chest   700         32             Motion artifacts < 10 Hz. 32 Hz is
#                                        more than sufficient.  Matches wrist ACC
#                                        native rate for architectural consistency.

# ACC wrist   32          32 (native)    Already at target. No resampling needed.

# TEMP wrist  4           —              Excluded from CNN branches.  Stored as
#                                        raw 4 Hz array for optional scalar
#                                        extraction at windowing stage.

# Labels      700         per-signal     Nearest-neighbour decimation to each
#                                        signal's target rate.  Categorical data
#                                        must never be interpolated.

# References:
# Hwang et al. (2018), Deep ECGNet: kernel/pool sizing from ECG periodicity
# Prajod & André (2022), ECG at 256 Hz DL standard, cross-dataset evaluation
# Moser et al. (2024), EDA downsampling, Z-score normalisation, augmentation
# Ordóñez & Roggen (2016), DeepConvLSTM, raw signal input, no pooling discussion
# Schmidt et al. (2018), WESAD dataset, 700 Hz chest, Empatica E4 wrist
# """

import os
import pickle
import numpy as np
from scipy.signal import resample_poly
from math import gcd

#  Import existing preprocessing modules (filter at native rate)

from src.ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from src.ml_pipeline.preprocessing.emg_preprocessing import EMGPreprocessing
from src.ml_pipeline.preprocessing.eda_preprocessing import EDAPreprocessing
from src.ml_pipeline.preprocessing.bvp_preprocessing import BVPPreprocessing
from src.ml_pipeline.preprocessing.resp_preprocessing import RespPreprocessing
from src.ml_pipeline.preprocessing.temp_preprocessing import TempPreprocessing
from src.ml_pipeline.preprocessing.acc_preprocessing import AccPreprocessing

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Native sampling rates (from WESAD README verified by metadata extractor)
NATIVE_HZ = {
    "ecg": 700,
    "emg": 700,
    "c_eda": 700,
    "resp": 700,
    "c_acc": 700,
    "c_temp": 700,
    "bvp": 64,
    "w_eda": 4,
    "w_acc": 32,
    "w_temp": 4,
    "label": 700,
}

# DL target rates: every value is physiologically justified
DL_TARGET_HZ = {
    "ecg": 256,
    "emg": 64,
    "c_eda": 8,
    "resp": 32,
    "c_acc": 32,
    "bvp": 32,
    "w_eda": 4,  # native
    "w_acc": 32,  # native
    "w_temp": 4,  # native
}


#  Utility: anti-aliased rational resampling
def _resample(signal_1d: np.ndarray, native_hz: int, target_hz: int) -> np.ndarray:
    """Resample a 1-D signal from native_hz → target_hz using
    scipy.signal.resample_poly (polyphase FIR, includes anti-alias filter).

    If native_hz == target_hz the signal is returned unchanged.
    """
    if native_hz == target_hz:
        return signal_1d

    # resample_poly needs integer up/down factors
    g = gcd(int(native_hz), int(target_hz))
    up = int(target_hz) // g
    down = int(native_hz) // g
    return resample_poly(signal_1d, up, down).astype(np.float32)


def _resample_labels(labels: np.ndarray, native_hz: int, target_hz: int) -> np.ndarray:
    """Downsample label array using nearest-neighbour decimation.
    Labels are categorical so interpolation would create invalid class values.
    """
    if native_hz == target_hz:
        return labels

    n_target = int(len(labels) * target_hz / native_hz)
    indices = np.round(np.linspace(0, len(labels) - 1, n_target)).astype(int)
    return labels[indices]


#  Artifact correction (mirrors RF pipeline's filter_chest/wrist_data)
def _fix_temp_artifacts(signal_1d: np.ndarray) -> np.ndarray:
    """Replace temperature readings ≤ 0 with linear interpolation.
    Physically impossible values from sensor dropouts."""
    mask = signal_1d <= 0
    n_bad = mask.sum()
    if n_bad == 0:
        return signal_1d

    signal_1d = signal_1d.copy().astype(np.float64)
    signal_1d[mask] = np.nan

    # Linear interpolation over NaNs
    nans = np.isnan(signal_1d)
    x = np.arange(len(signal_1d))
    signal_1d[nans] = np.interp(x[nans], x[~nans], signal_1d[~nans])

    print(f"      Fixed {n_bad} temp artifacts (≤ 0) via interpolation")
    return signal_1d


def _fix_eda_artifacts(signal_1d: np.ndarray) -> np.ndarray:
    """Replace negative EDA readings with linear interpolation.
    Skin conductance cannot be negative so these are sensor artifacts."""
    mask = signal_1d < 0
    n_bad = mask.sum()
    if n_bad == 0:
        return signal_1d

    signal_1d = signal_1d.copy().astype(np.float64)
    signal_1d[mask] = np.nan

    nans = np.isnan(signal_1d)
    x = np.arange(len(signal_1d))
    signal_1d[nans] = np.interp(x[nans], x[~nans], signal_1d[~nans])

    print(f"      Fixed {n_bad} negative EDA artifacts via interpolation")
    return signal_1d


#  Per-signal filtering (calls existing preprocessing modules)


def _filter_ecg(signal_1d: np.ndarray) -> np.ndarray:
    # ECG filtering at native 700 Hz via NeuroKit2 ecg_clean.
    # Applies: 5th-order Butterworth HPF @ 0.5 Hz + 50 Hz notch.
    # Preserves full QRS morphology for CNN feature extraction.
    # Citation: Makowski et al. (2021) NeuroKit2.
    import pandas as pd

    df = pd.DataFrame({"ecg": signal_1d})
    processor = ECGPreprocessing(df, fs=NATIVE_HZ["ecg"])
    result = processor.process()
    return result["ecg"].values


def _filter_emg(signal_1d: np.ndarray) -> np.ndarray:
    # EMG filtering at native 700 Hz via NeuroKit2 emg_clean.
    # Applies: 4th-order Butterworth HPF @ 100 Hz + detrending.
    # The 100 Hz HPF MUST run at 700 Hz (Nyquist = 350 Hz).
    # It cannot run after downsampling to 64 Hz (Nyquist = 32 Hz).
    # Citation: Makowski et al. (2021) NeuroKit2.
    import pandas as pd

    df = pd.DataFrame({"emg": signal_1d})
    processor = EMGPreprocessing(df, fs=NATIVE_HZ["emg"])
    result = processor.process()
    return result["emg"].values


def _filter_eda_chest(signal_1d: np.ndarray) -> np.ndarray:
    # Chest EDA filtering at native 700 Hz via NeuroKit2 eda_clean.
    # Applies: Butterworth LP @ 3 Hz.  Nyquist at 700 Hz = 350 Hz,
    # so the 3 Hz cutoff is well within range.
    # Citation: Makowski et al. (2021) NeuroKit2; Greco et al. (2016).
    import pandas as pd

    df = pd.DataFrame({"c_eda": signal_1d})
    processor = EDAPreprocessing(df, fs=NATIVE_HZ["c_eda"], wrist=False)
    result = processor.process()
    return result["c_eda"].values


def _filter_eda_wrist(signal_1d: np.ndarray) -> np.ndarray:
    # Wrist EDA: NO low-pass filter applied.

    # At native 4 Hz, Nyquist = 2 Hz.  The NeuroKit2 default 3 Hz LP cutoff
    # exceeds Nyquist, making the filter numerically
    # unstable.  The Empatica E4 hardware ADC already applies anti-aliasing
    # before digitisation so any content > 2 Hz is physically absent.

    # We apply ONLY the 0.05 Hz high-pass for phasic component extraction,
    # which is well within the 2 Hz Nyquist limit.
    # 2nd order butterworth is kept
    # for methodological consistency with the chest EDA branch (which uses
    # NeuroKit2's eda_clean with the same filter order at 700 Hz).

    # Citation: Moser et al. (2024) downsampled EDA to 1 Hz, at 4 Hz we
    # retain more SCR waveform shape than their approach.
    from scipy.signal import butter, filtfilt

    fs = NATIVE_HZ["w_eda"]  # 4 Hz
    nyquist = fs / 2.0  # 2 Hz
    hp_cutoff = 0.05

    b, a = butter(2, hp_cutoff / nyquist, btype="high")
    filtered = filtfilt(b, a, signal_1d.astype(np.float64))
    return filtered


def _filter_resp(signal_1d: np.ndarray) -> np.ndarray:
    # Respiration filtering at native 700 Hz via NeuroKit2 rsp_clean.
    # Applies: 2nd-order Butterworth bandpass 0.05–3 Hz (khodadad2018).
    # Preserves stress-induced tachypnea up to 30 bpm (0.5 Hz).
    # Citation: Khodadad et al. (2018); Makowski et al. (2021).
    import pandas as pd

    df = pd.DataFrame({"resp": signal_1d})
    processor = RespPreprocessing(df, fs=NATIVE_HZ["resp"])
    result = processor.process()
    return result["resp"].values


def _filter_bvp(signal_1d: np.ndarray) -> np.ndarray:
    # BVP filtering at native 64 Hz via NeuroKit2 ppg_clean.
    # Applies: Elgendi (2013) Butterworth LP for photoplethysmography.
    # Citation: Elgendi (2013); Makowski et al. (2021).
    import pandas as pd

    df = pd.DataFrame({"bvp": signal_1d})
    processor = BVPPreprocessing(df, fs=NATIVE_HZ["bvp"])
    result = processor.process()
    return result["bvp"].values


def _filter_acc_chest(signal_3d: np.ndarray) -> np.ndarray:
    # Chest ACC smoothing at native 700 Hz via Savitzky-Golay.
    # Window=31, poly_order=5 per axis.  Removes HF sensor noise
    # while preserving body motion.
    import pandas as pd

    df = pd.DataFrame(
        {
            "c_acc_x": signal_3d[:, 0],
            "c_acc_y": signal_3d[:, 1],
            "c_acc_z": signal_3d[:, 2],
        }
    )
    processor = AccPreprocessing(df, window_size=31, poly_order=5, wrist=False)
    result = processor.process()
    return np.column_stack(
        [
            result["c_acc_x"].values,
            result["c_acc_y"].values,
            result["c_acc_z"].values,
        ]
    )


def _filter_acc_wrist(signal_3d: np.ndarray) -> np.ndarray:
    # Wrist ACC filtering at native 32 Hz via FIR low-pass.
    # Cutoff 0.4 Hz normalised to Nyquist. Isolates slow orientation
    # changes from hand tremor / high-frequency vibrations.
    import pandas as pd

    df = pd.DataFrame(
        {
            "w_acc_x": signal_3d[:, 0],
            "w_acc_y": signal_3d[:, 1],
            "w_acc_z": signal_3d[:, 2],
        }
    )
    processor = AccPreprocessing(df, fs=NATIVE_HZ["w_acc"], wrist=True)
    result = processor.process()
    return np.column_stack(
        [
            result["w_acc_x"].values,
            result["w_acc_y"].values,
            result["w_acc_z"].values,
        ]
    )


def _filter_temp_chest(signal_1d: np.ndarray) -> np.ndarray:
    # Chest temperature smoothing at native 700 Hz via Savitzky-Golay.
    # Window=11, poly_order=3.  Temperature is very slow-changing so
    # SavGol is the standard approach (no NK2/BioSPPy module exists).
    import pandas as pd

    df = pd.DataFrame({"c_temp": signal_1d})
    processor = TempPreprocessing(df, window_size=11, poly_order=3, wrist=False)
    result = processor.process()
    return result["c_temp"].values


def _filter_temp_wrist(signal_1d: np.ndarray) -> np.ndarray:
    # Wrist temperature smoothing at native 4 Hz via Savitzky-Golay.
    # Window=11, poly_order=3.  At 4 Hz the native window size is appropriate
    # (no scaling needed, the RF pipeline scaled because it had upsampled
    # to 64 Hz first).
    import pandas as pd

    df = pd.DataFrame({"w_temp": signal_1d})
    processor = TempPreprocessing(df, window_size=11, poly_order=3, wrist=True)
    result = processor.process()
    return result["w_temp"].values


#  EMG rectification (DL-specific addition)


def _rectify_emg(signal_1d: np.ndarray) -> np.ndarray:
    # Rectification + envelope smoothing of the HPF EMG signal.

    # After the 100 Hz HPF, EMG is a zero-mean AC signal oscillating
    # between roughly −5 mV and +5 mV.  Downsampling this directly to
    # 64 Hz would alias the high-frequency content.

    # The standard EMG envelope extraction chain is:
    #   1. High-pass filter  (done by _filter_emg at 700 Hz)
    #   2. Rectification     (np.abs, converts to unipolar amplitude)
    #   3. Low-pass filter   (10 Hz Butterworth, smooths the envelope)

    # The 10 Hz LP removes the high-frequency ripple produced by
    # rectification while preserving the physiologically meaningful muscle activation envelope, which
    # is concentrated below 10–20 Hz.

    # For the RF pipeline, rectification was not needed because NeuroKit2
    # feature extraction handled it internally.  For the DL pipeline
    # feeding raw waveforms, we must do it explicitly.
    from scipy.signal import butter, filtfilt

    # rectification
    rectified = np.abs(signal_1d)

    # 10 Hz LP envelope smoothing at native 700 Hz
    # 4th-order Butterworth.  10 Hz is the standard cutoff for EMG
    # amplitude envelope extraction bc it retains voluntary muscle
    # activation dynamics while removing rectification ripple.
    fs = NATIVE_HZ["emg"]  # 700 Hz
    nyquist = fs / 2.0  # 350 Hz
    lp_cutoff = 10.0  # Hz

    b, a = butter(4, lp_cutoff / nyquist, btype="low")
    envelope = filtfilt(b, a, rectified)

    return envelope


#  Main: process one subject


def process_subject(subject_id: int, dataset_path: str, output_path: str):
    """Load raw pickle for one WESAD subject, filter at native rates,
    downsample to DL targets, save per-signal .npy arrays.

    Output structure:
        output_path/S{id}/
            ecg.npy         (n_samples,)  @ 256 Hz
            emg.npy         (n_samples,)  @ 64 Hz  (rectified envelope)
            c_eda.npy       (n_samples,)  @ 8 Hz
            resp.npy        (n_samples,)  @ 32 Hz
            c_acc.npy       (n_samples, 3) @ 32 Hz
            bvp.npy         (n_samples,)  @ 32 Hz
            w_eda.npy       (n_samples,)  @ 4 Hz
            w_acc.npy       (n_samples, 3) @ 32 Hz
            w_temp.npy      (n_samples,)  @ 4 Hz  (for optional scalar use)
            label_<sig>.npy per-signal label arrays at each target rate
    """
    pkl_path = os.path.join(dataset_path, f"S{subject_id}", f"S{subject_id}.pkl")
    if not os.path.exists(pkl_path):
        print(f"    File not found: {pkl_path} — skipping S{subject_id}")
        return

    print(f"\n")
    print(f"  Processing S{subject_id}")
    print(f"\n")

    # Load raw pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]
    labels_raw = data["label"].flatten()  # (n_samples,) @ 700 Hz

    # Subject output directory
    subj_dir = os.path.join(output_path, f"S{subject_id}")
    os.makedirs(subj_dir, exist_ok=True)

    #  CHEST SIGNALS

    # ECG
    print("    ECG: filter @ 700 Hz → downsample → 256 Hz")
    ecg_raw = chest["ECG"].flatten().astype(np.float64)
    ecg_filtered = _filter_ecg(ecg_raw)
    ecg_dl = _resample(ecg_filtered, NATIVE_HZ["ecg"], DL_TARGET_HZ["ecg"])
    np.save(os.path.join(subj_dir, "ecg.npy"), ecg_dl)
    labels_ecg = _resample_labels(labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["ecg"])
    np.save(os.path.join(subj_dir, "label_ecg.npy"), labels_ecg)
    print(f"      {len(ecg_raw):,} → {len(ecg_dl):,} samples")

    # EMG
    # CRITICAL: filter at 700 Hz, then rectify, THEN downsample to 64 Hz.
    print("    EMG: filter @ 700 Hz → rectify → downsample → 64 Hz")
    emg_raw = chest["EMG"].flatten().astype(np.float64)
    emg_filtered = _filter_emg(emg_raw)
    emg_rectified = _rectify_emg(emg_filtered)
    emg_dl = _resample(emg_rectified, NATIVE_HZ["emg"], DL_TARGET_HZ["emg"])
    np.save(os.path.join(subj_dir, "emg.npy"), emg_dl)
    labels_emg = _resample_labels(labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["emg"])
    np.save(os.path.join(subj_dir, "label_emg.npy"), labels_emg)
    print(f"      {len(emg_raw):,} → {len(emg_dl):,} samples (rectified envelope)")

    # Chest EDA
    print("    Chest EDA: fix artifacts → filter @ 700 Hz → downsample → 8 Hz")
    c_eda_raw = chest["EDA"].flatten().astype(np.float64)
    c_eda_raw = _fix_eda_artifacts(c_eda_raw)
    c_eda_filtered = _filter_eda_chest(c_eda_raw)
    c_eda_dl = _resample(c_eda_filtered, NATIVE_HZ["c_eda"], DL_TARGET_HZ["c_eda"])
    np.save(os.path.join(subj_dir, "c_eda.npy"), c_eda_dl)
    labels_c_eda = _resample_labels(
        labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["c_eda"]
    )
    np.save(os.path.join(subj_dir, "label_c_eda.npy"), labels_c_eda)
    print(f"      {len(c_eda_raw):,} → {len(c_eda_dl):,} samples")

    # RESP
    print("    RESP: filter @ 700 Hz → downsample → 32 Hz")
    resp_raw = chest["Resp"].flatten().astype(np.float64)
    resp_filtered = _filter_resp(resp_raw)
    resp_dl = _resample(resp_filtered, NATIVE_HZ["resp"], DL_TARGET_HZ["resp"])
    np.save(os.path.join(subj_dir, "resp.npy"), resp_dl)
    labels_resp = _resample_labels(labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["resp"])
    np.save(os.path.join(subj_dir, "label_resp.npy"), labels_resp)
    print(f"      {len(resp_raw):,} → {len(resp_dl):,} samples")

    # Chest ACC
    print("    Chest ACC: SavGol @ 700 Hz → downsample → 32 Hz")
    c_acc_raw = chest["ACC"].astype(np.float64)  # (n, 3)
    c_acc_filtered = _filter_acc_chest(c_acc_raw)
    # Resample each axis independently
    c_acc_dl = np.column_stack(
        [
            _resample(c_acc_filtered[:, i], NATIVE_HZ["c_acc"], DL_TARGET_HZ["c_acc"])
            for i in range(3)
        ]
    )
    np.save(os.path.join(subj_dir, "c_acc.npy"), c_acc_dl)
    labels_c_acc = _resample_labels(
        labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["c_acc"]
    )
    np.save(os.path.join(subj_dir, "label_c_acc.npy"), labels_c_acc)
    print(f"      {len(c_acc_raw):,} → {len(c_acc_dl):,} samples (3 axes)")

    # Chest TEMP
    # Chest temp is included for the chest DL model (upper bound baseline).
    # It will be used as a scalar feature, not a CNN branch.
    print("    Chest TEMP: fix artifacts → SavGol @ 700 Hz → downsample → 32 Hz")
    c_temp_raw = chest["Temp"].flatten().astype(np.float64)
    c_temp_raw = _fix_temp_artifacts(c_temp_raw)
    c_temp_filtered = _filter_temp_chest(c_temp_raw)
    c_temp_dl = _resample(c_temp_filtered, NATIVE_HZ["c_temp"], 32)
    np.save(os.path.join(subj_dir, "c_temp.npy"), c_temp_dl)
    labels_c_temp = _resample_labels(labels_raw, NATIVE_HZ["label"], 32)
    np.save(os.path.join(subj_dir, "label_c_temp.npy"), labels_c_temp)
    print(f"      {len(c_temp_raw):,} → {len(c_temp_dl):,} samples")

    #  WRIST SIGNALS
    #  NOTE on labels: labels_raw is the chest-rate label array at
    #  700 Hz.  ALL label resampling inclfing wrist signals
    #  starts from this 700 Hz source.  The source rate in
    #  _resample_labels refers to the label array's rate, not the
    #  signal's native rate. For all resamplinh  e.g. for w_acc (native 32 Hz), resampling labels from 700 → 32 produces
    #  the right number of label samples to match the signal length.

    # BVP
    print("    BVP: filter @ 64 Hz → downsample → 32 Hz")
    bvp_raw = wrist["BVP"].flatten().astype(np.float64)
    bvp_filtered = _filter_bvp(bvp_raw)
    bvp_dl = _resample(bvp_filtered, NATIVE_HZ["bvp"], DL_TARGET_HZ["bvp"])
    np.save(os.path.join(subj_dir, "bvp.npy"), bvp_dl)
    labels_bvp = _resample_labels(labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["bvp"])
    np.save(os.path.join(subj_dir, "label_bvp.npy"), labels_bvp)
    print(f"      {len(bvp_raw):,} → {len(bvp_dl):,} samples")

    # Wrist EDA
    # SPECIAL CASE: native 4 Hz, Nyquist = 2 Hz.
    # NO low-pass filter.  Only 0.05 Hz HPF for phasic extraction.
    print(
        "    Wrist EDA: fix artifacts → HPF 0.05 Hz @ 4 Hz (no LP — Nyquist constraint)"
    )
    w_eda_raw = wrist["EDA"].flatten().astype(np.float64)
    w_eda_raw = _fix_eda_artifacts(w_eda_raw)
    w_eda_filtered = _filter_eda_wrist(w_eda_raw)
    # already at target (4 Hz)
    w_eda_dl = w_eda_filtered.astype(np.float32)
    np.save(os.path.join(subj_dir, "w_eda.npy"), w_eda_dl)
    labels_w_eda = _resample_labels(
        labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["w_eda"]
    )
    np.save(os.path.join(subj_dir, "label_w_eda.npy"), labels_w_eda)
    print(f"      {len(w_eda_raw):,} samples (kept at native 4 Hz)")

    # Wrist ACC
    print("    Wrist ACC: FIR LP @ 32 Hz (native — no resampling)")
    w_acc_raw = wrist["ACC"].astype(np.float64)  # (n, 3)
    w_acc_filtered = _filter_acc_wrist(w_acc_raw)
    w_acc_dl = w_acc_filtered.astype(np.float32)
    np.save(os.path.join(subj_dir, "w_acc.npy"), w_acc_dl)
    labels_w_acc = _resample_labels(
        labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["w_acc"]
    )
    np.save(os.path.join(subj_dir, "label_w_acc.npy"), labels_w_acc)
    print(f"      {len(w_acc_raw):,} samples (3 axes, native 32 Hz)")

    # Wrist TEMP
    # Stored for optional scalar extraction at windowing stage.
    # NOT used as a CNN branch (changes over minutes, not seconds).
    print("    Wrist TEMP: fix artifacts → SavGol @ 4 Hz (stored for scalar use)")
    w_temp_raw = wrist["TEMP"].flatten().astype(np.float64)
    w_temp_raw = _fix_temp_artifacts(w_temp_raw)
    w_temp_filtered = _filter_temp_wrist(w_temp_raw)
    w_temp_dl = w_temp_filtered.astype(np.float32)
    np.save(os.path.join(subj_dir, "w_temp.npy"), w_temp_dl)
    labels_w_temp = _resample_labels(
        labels_raw, NATIVE_HZ["label"], DL_TARGET_HZ["w_temp"]
    )
    np.save(os.path.join(subj_dir, "label_w_temp.npy"), labels_w_temp)
    print(f"      {len(w_temp_raw):,} samples (native 4 Hz)")

    # Summary
    duration_s = len(labels_raw) / NATIVE_HZ["label"]
    print(f"\n    S{subject_id} complete → {subj_dir}")
    print(f"      Recording duration: {duration_s:.1f} s ({duration_s/60:.1f} min)")


#  Run all subjects


def prepare_all(
    dataset_path: str = "data/raw/wesad/WESAD/",
    output_path: str = "data/processed/wesad/dl_prepared/",
    subject_ids: list[int] | None = None,
):
    """Process all WESAD subjects for the DL pipeline.

    Parameters
    dataset_path : str
        Root of the WESAD dataset (contains S2/, S3/, ... directories).
    output_path : str
        Where to save per-subject .npy arrays.
    subject_ids : list[int], optional
        Subjects to process.  Defaults to all 15 WESAD subjects.
    """
    if subject_ids is None:
        subject_ids = SUBJECT_IDS

    os.makedirs(output_path, exist_ok=True)

    print("  WESAD DL Data Preparation\n")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Output:   {output_path}")
    print(f"  Subjects: {subject_ids}")
    print()
    print("  DL Target Rates:")
    for sig, hz in DL_TARGET_HZ.items():
        native = NATIVE_HZ.get(sig, "?")
        action = "native" if native == hz else f"{native} → {hz}"
        print(f"    {sig:10s}  {action} Hz")
    print()

    for sid in subject_ids:
        process_subject(sid, dataset_path, output_path)

    print("\n")
    print("  All subjects processed.")
    print(f"  Output directory: {output_path}")


if __name__ == "__main__":
    prepare_all()

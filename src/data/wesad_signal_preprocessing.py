# Signal Preprocessing Pipeline

# IMPORTANT: This operates on merged data where:
#   Chest: All signals at 700 Hz (native)
#   Wrist: All signals at 64 Hz (upsampled in wesad_resample_merge.py)
#
# Config files must reflect these post-resampling rates

# Applies signal-specific preprocessing (filtering) to each modality in the
# merged WESAD dataframe. The approach followed is a hybrid one:

# NeuroKit2 python library for every signal that is included with its own peer-reviewed
# processing module (ECG, EDA, EMG, RESP, BVP/PPG).
# Savitzky-Golay / FIR filters for the two signals that lack
# standardised complex processing modules in the NK2 Library (TEMP, ACC).

# Reference:
# D. Makowski, T. Pham, Z. J. Lau, J. C. Brammer, F. Lespinasse, H. Pham, C. Schölzel,
# and S. H. A. Chen, “NeuroKit2: A Python toolbox for neurophysiological signal processing,”
# Behavior Research Methods, vol. 53, no. 4, pp. 1689–1696, Aug. 2021, doi: 10.3758/s13428-020-01516-y.

import os
import json
import pickle
import pandas as pd
import numpy as np

# preprocessing modules
from src.ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from src.ml_pipeline.preprocessing.emg_preprocessing import EMGPreprocessing
from src.ml_pipeline.preprocessing.eda_preprocessing import EDAPreprocessing
from src.ml_pipeline.preprocessing.bvp_preprocessing import BVPPreprocessing
from src.ml_pipeline.preprocessing.resp_preprocessing import RespPreprocessing
from src.ml_pipeline.preprocessing.temp_preprocessing import TempPreprocessing
from src.ml_pipeline.preprocessing.acc_preprocessing import AccPreprocessing

# helpers
from src.utils.utils import get_max_sampling_rate, get_active_key

# pre-resampling wrist rates needed for Savitzky-Golay
# window scaling, since SavGol is sample-count-based not frequency-based
_NATIVE_WRIST_TEMP_HZ = 4


class SignalPreprocessor:
    # Signal preprocessor for the WESAD dataset.

    # Parameters:
    # data_path : str
    #     Path to the merged pickle file (``merged_chest.pkl`` or ``merged_wrist.pkl``).
    # output_path : str
    #     Destination path for the cleaned pickle file ( ``chest_preprocessed.pkl``).
    # config_path : str
    #     Path to a WESAD dataset JSON configuration file that lists which sensors are active and their sampling rates.
    # wrist : bool, optional
    #     If True , the data originates from the wrist device; otherwise from the chest device.

    def __init__(
        self,
        data_path: str,
        output_path: str,
        config_path: str,
        wrist: bool = False,
    ):
        self.data_path = data_path
        self.output_path = output_path
        self.config_path = config_path
        self.wrist = wrist

        # Load the merged df
        self.df: pd.DataFrame = pd.read_pickle(self.data_path)

        # Resolve the maximum sampling rate from the config file
        self.sampling_rate: int = get_max_sampling_rate(config_path)

        # Build a per-sensor rate lookup from the config
        with open(config_path, "r") as f:
            self._sensor_rates: dict = json.load(f)["sampling rate"]

    def _get_rate(self, sensor_key: str):
        # # Look up a sensor's sampling rate from the config fall back to self.sampling_rate (max rate) if the key isn't found
        return self._sensor_rates.get(sensor_key, self.sampling_rate)

    def preprocess_signals(self):
        sensors: list[str] = get_active_key(self.config_path, "sensors")

        # Work on a copy to preserve the original loaded data
        self.df_raw = self.df  # reference to original for integrity checks
        self.df = self.df.copy()  # working copy

        print(f"Starting signal preprocessing ({'wrist' if self.wrist else 'chest'})…")
        print(f"Sampling rate from config: {self.sampling_rate} Hz")
        print(f"Active sensors: {sensors}")

        for sensor in sensors:
            self._dispatch(sensor)

        self._save()
        print("Signal preprocessing complete.\n")
        return self.df

    # route each sensor key to its processor
    def _dispatch(self, sensor: str):
        match sensor:
            # Chest signals
            case "ecg":
                fs = self._get_rate("ecg")
                print(f"  → ECG  (NeuroKit2 ecg_clean, {fs} Hz)")
                ecg_processor = ECGPreprocessing(self.df, fs=fs)
                self.df = ecg_processor.process()

            case "emg":
                fs = self._get_rate("emg")
                print(f"  → EMG  (NeuroKit2 emg_clean, {fs} Hz)")
                emg_processor = EMGPreprocessing(self.df, fs=fs)
                self.df = emg_processor.process()

            case "c_eda":
                fs = self._get_rate("c_eda")
                print(f"  → EDA  (NeuroKit2 eda_clean, {fs} Hz)")
                ceda_processor = EDAPreprocessing(self.df, fs=fs, wrist=False)
                self.df = ceda_processor.process()

            case "resp":
                fs = self._get_rate("resp")
                print(f"  → RESP (NeuroKit2 rsp_clean, {fs} Hz)")
                resp_processor = RespPreprocessing(self.df, fs=fs)
                self.df = resp_processor.process()

            case "c_temp":
                print("  → TEMP (Savitzky-Golay, chest)")
                ctemp_processor = TempPreprocessing(
                    self.df,
                    window_size=11,
                    poly_order=3,
                    wrist=False,
                )
                self.df = ctemp_processor.process()

            case "c_acc":
                print("  → ACC  (Savitzky-Golay, chest)")
                cacc_processor = AccPreprocessing(
                    self.df,
                    window_size=31,
                    poly_order=5,
                    wrist=False,
                )
                self.df = cacc_processor.process()

            # Wrist signals
            case "bvp":
                fs = self._get_rate("bvp")
                print(f"  → BVP  (NeuroKit2 ppg_clean, {fs} Hz)")
                bvp_processor = BVPPreprocessing(self.df, fs=fs)
                self.df = bvp_processor.process()

            case "w_eda":
                fs = self._get_rate("w_eda")
                print(f"  → EDA  (NeuroKit2 eda_clean, wrist {fs} Hz)")
                weda_processor = EDAPreprocessing(self.df, fs=fs, wrist=True)
                self.df = weda_processor.process()

            case "w_temp":
                # Scale Savitzky-Golay window to match resampled rate
                # Native 4 Hz used window=11
                fs = self._get_rate("w_temp")
                native_window = 11
                scale_factor = fs // _NATIVE_WRIST_TEMP_HZ  # 64 // 4 = 16
                scaled_window = native_window * scale_factor
                if scaled_window % 2 == 0:
                    scaled_window += 1  # SavGol requires odd window because it averages all datapoints in the window around the middle one

                print(
                    f"  → TEMP (Savitzky-Golay, wrist, window={scaled_window} at {fs} Hz)"
                )
                wtemp_processor = TempPreprocessing(
                    self.df,
                    window_size=scaled_window,
                    poly_order=3,
                    wrist=True,
                )
                self.df = wtemp_processor.process()

            case "w_acc":
                fs = self._get_rate("w_acc_x")  # all axes share the same rate
                print(f"  → ACC  (FIR lowpass, wrist {fs} Hz)")
                wacc_processor = AccPreprocessing(
                    self.df,
                    fs=fs,
                    wrist=True,
                )
                self.df = wacc_processor.process()

            case _:
                print(f"  ⚠  Unknown sensor key '{sensor}' — skipping.")

    #  Save output df
    def _save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "wb") as fh:
            pickle.dump(self.df, fh)
        print(f"  Saved → {self.output_path}")

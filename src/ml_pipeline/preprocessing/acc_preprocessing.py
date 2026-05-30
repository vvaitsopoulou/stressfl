# TODO: check delay is synced
# Accelerometer Preprocessing

# Two strategies depending on the device:

# Chest (700 Hz) — Savitzky-Golay smoothing
#   Window = 31, polynomial order = 5 (per axis)
#   Removes high-frequency sensor noise while preserving body motion.

# Wrist (32 Hz - upsampled to 64Hz) — FIR low-pass filter
#   Filter length = 128 taps, cutoff = 0.4 Hz (normalised to Nyquist)
#   Designed to isolate slow orientation changes from hand tremor or
#   high-frequency vibrations.

# Neither NeuroKit2 nor BioSPPy provides a dedicated ACC preprocessing
# modyle beyond basic filtering, so manual filters are the standard.

import numpy as np
from scipy.signal import filtfilt, firwin, savgol_filter


class AccPreprocessing:

    # Column names per device
    _CHEST_COLS = ["c_acc_x", "c_acc_y", "c_acc_z"]
    _WRIST_COLS = ["w_acc_x", "w_acc_y", "w_acc_z"]

    def __init__(
        self,
        df,
        window_size: int = 31,
        poly_order: int = 5,
        fir_length: int = 128,
        fir_cutoff: float = 0.4,
        fs: int = 64,
        # Selects between the wrist FIR and chest Savitzky-Golay paths.
        wrist: bool = False,
    ):
        self.df = df
        self.window_size = window_size
        self.poly_order = poly_order
        self.fir_length = fir_length
        self.fir_cutoff = fir_cutoff
        self.fs = fs
        self.wrist = wrist

    def process(self):
        if self.wrist:
            cols = self._WRIST_COLS
            filter_fn = self._fir_lowpass
        else:
            cols = self._CHEST_COLS
            filter_fn = self._savgol_smooth

        for col in cols:
            self.df[col] = filter_fn(self.df[col].values)

        return self.df

    # Chest: Savitzky-Golay
    def _savgol_smooth(self, signal: np.ndarray):
        return savgol_filter(signal, self.window_size, self.poly_order)

    # Wrist: FIR low-pass
    def _fir_lowpass(self, signal: np.ndarray):
        nyquist = self.fs / 2.0
        normalised_cutoff = self.fir_cutoff / nyquist
        coeffs = firwin(self.fir_length, normalised_cutoff)
        return filtfilt(coeffs, 1.0, signal)

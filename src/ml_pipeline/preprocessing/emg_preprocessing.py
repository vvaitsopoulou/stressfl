# EMG Preprocessing

# electromyography cleaning using NeuroKit2's emg_clean()pipeline.

# Default method applies
#   4th-order Butterworth high-pass at 100 Hz
#   Constant detrending

# By setting the high-pass cutoff at exactly 100 Hz we are intentionally sacrificing the
# lower register of the EMG signal (20-500Hz) to completely eliminate the ECG contamination and baseline drift
# Basically we are ensuring that the remaining high-frequency signal (100–500 Hz) is purely muscle activation and not a disguised heartbeat.

# NOTE: some existing WESAD implementations apply a 0.5 Hz low-pass filter directly to the raw EMG signal. This is mathematically flawed
# as low-passing a zero-mean, non-rectified AC signal (from -5mV to +5) effectively cancels out the action potentials and destroys the data.
# If an EMG amplitude envelope is required for feature extraction, the signal must be first cleaned, then rectified (absolute value), and after that low-passed.
# Makowski et al. (2021). NeuroKit2.

import neurokit2 as nk
import pandas as pd


class EMGPreprocessing:
    # Clean a chest EMG signal using NeuroKit2.
    COLUMN = "emg"

    def __init__(self, df, fs: int = 700, method: str = "biosppy"):
        self.df = df
        self.fs = fs
        self.method = method
        self._was_series = False

    def process(self):
        # Return the dataframe with the emg column cleaned.
        if isinstance(self.df, pd.Series):
            self._was_series = True
            self.df = pd.DataFrame(self.df, columns=[self.COLUMN])

        raw = self.df[self.COLUMN].values
        cleaned = nk.emg_clean(raw, sampling_rate=self.fs)
        self.df[self.COLUMN] = cleaned

        return self.df[self.COLUMN] if self._was_series else self.df

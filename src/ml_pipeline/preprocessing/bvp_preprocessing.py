# TODO: write accompanying theory ENSURE UPSAMPLING FOR WRIST IS TAKEN CARE OF, THAT NAMES ARE CORRECT, THAT THE DF IS COPIED

# BVP Preprocessing

# Blood Volume Pulse cleaning with NeuroKit2's ppg_clean() pipeline.

# Default Elgendi (2013) method applies a Butterworth low-pass filter for photoplethysmographic signals.  BVP and PPG are
# physiologically the same signal, so ppg_clean() / ppg_process() is the correct function family.

# The cleaned signal is suitable for downstream peak detection
# (ppg_findpeaks()) and HRV / IBI feature extraction.

# Reference: Elgendi, M. (2013). Fast QRS detection with an optimized knowledge-based method. PLoS ONE, 8(4), e62578.
#            Makowski et al. (2021). NeuroKit2.

import neurokit2 as nk
import numpy as np
import pandas as pd


# Clean a wrist BVP signal using NeuroKit2's PPG pipeline
class BVPPreprocessing:

    COLUMN = "bvp"

    def __init__(self, df, fs: int = 64, method: str = "elgendi"):
        self.df = df
        self.fs = fs
        self.method = method
        self._was_series = False

    def process(self):
        # Return the dataframe with the bvp column cleaned.
        if isinstance(self.df, pd.Series):
            self._was_series = True
            self.df = pd.DataFrame(self.df, columns=[self.COLUMN])

        raw = self.df[self.COLUMN].values.astype(np.float64)
        cleaned = nk.ppg_clean(raw, sampling_rate=self.fs, method=self.method)
        self.df[self.COLUMN] = cleaned

        return self.df[self.COLUMN] if self._was_series else self.df

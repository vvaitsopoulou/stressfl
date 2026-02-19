# ECG Preprocessing
# Electrocardiogram signal cleaning using NeuroKit2's validated ecg_clean() pipeline.

# Default method ("neurokit") applies:
#   5th-order Butterworth high-pass at 0.5 Hz
#   50 Hz notch filter

# This preserves full QRS morphology  which is ssential for downstream R-peak detection, HRV analysis, and
# feature extraction.

# Makowski et al. (2021). NeuroKit2: A Python toolbox forneurophysiological signal processing.

import neurokit2 as nk
import pandas as pd


class ECGPreprocessing:
    # Clean a chest ECG signal using NeuroKit2.

    COLUMN = "ecg"

    def __init__(self, df, fs: int = 700, method: str = "neurokit"):
        self.df = df
        self.fs = fs
        self.method = method
        self._was_series = False

    def process(self):
        # Return the dataframe with the ecg column cleaned in-place.
        if isinstance(self.df, pd.Series):
            self._was_series = True
            self.df = pd.DataFrame(self.df, columns=[self.COLUMN])

        raw = self.df[self.COLUMN].values
        cleaned = nk.ecg_clean(raw, sampling_rate=self.fs, method=self.method)
        self.df[self.COLUMN] = cleaned

        return self.df[self.COLUMN] if self._was_series else self.df

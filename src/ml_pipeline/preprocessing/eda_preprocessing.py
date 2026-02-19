# TODO: write accompanying theory ENSURE UPSAMPLING FOR WRIST IS TAKEN CARE OF, THAT NAMES ARE CORRECT, THAT THE DF IS COPIED

# Electrodermal activity cleaning using NeuroKit2's eda_clean() pipeline.  Handles both chest (700 Hz) and wrist (4 Hz) - upsampled 64hz modalities.

# Chest (700 Hz):
#   Butterworth low-pass at 3 Hz
#   Well within Nyquist at 700 Hz so full filtering is effective
#   Enables proper tonic/phasic decomposition via eda_phasic() downstream

# Wrist (native 4 Hz, resampled to 64 Hz):
#   At native 4 Hz, Nyquist (2 Hz) sat below the 3 Hz cutoff, making the filter a no-op.
#   After resampling to 64 Hz (Nyquist = 32 Hz), the 3 Hz lowpass preserves all genuine EDA contentwhile removing interpolation artifacts from resampling.

# Reference
# Makowski et al. (2021). NeuroKit2.
# Greco, A. et al. (2016). cvxEDA: a convex optimization approach to EDA processing. IEEE Trans. Biomed. Eng.

import neurokit2 as nk
import numpy as np
import pandas as pd


class EDAPreprocessing:
    # Clean an EDA signal using NeuroKit2.
    def __init__(
        self,
        df,
        fs: int = 700,
        wrist: bool = False,
        method: str = "neurokit",
    ):
        self.df = df
        self.fs = fs
        self.wrist = wrist
        self.method = method
        self.column = "w_eda" if wrist else "c_eda"
        self._was_series = False

    def process(self):
        # Return the dataframe with the EDA column cleaned
        if isinstance(self.df, pd.Series):
            self._was_series = True
            self.df = pd.DataFrame(self.df, columns=[self.column])

        raw = self.df[self.column].values.astype(np.float64)
        cleaned = nk.eda_clean(raw, sampling_rate=self.fs, method=self.method)
        self.df[self.column] = cleaned

        return self.df[self.column] if self._was_series else self.df

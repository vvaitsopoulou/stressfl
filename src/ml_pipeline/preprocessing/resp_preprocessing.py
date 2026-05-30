# Respiration Preprocessing
# Respiration signal cleaning using NeuroKit2's rsp_clean() pipeline.

# Default "khodadad2018" method applies 2nd-order Butterworth band-pass at 0.05 – 3 Hz

# This preserves breathing rates from 3 to 180 breaths/min, which is preferable for stress detection over the tighter BioSPPy range
# (0.1–0.35 Hz ≈ 6–21 breaths/min) because stress-induced tachypnea can exceed 30 breaths/min (0.5 Hz).

# Reference
# Khodadad, D. et al. (2018). Optimized breath detection algorithm in
# electrical impedance tomography.
# Makowski et al. (2021). NeuroKit2.

import neurokit2 as nk
import numpy as np
import pandas as pd


class RespPreprocessing:
    # Clean a chest respiration signal using NeuroKit2.

    COLUMN = "resp"

    def __init__(self, df, fs: int = 700, method: str = "khodadad2018"):
        self.df = df
        self.fs = fs
        self.method = method
        self._was_series = False

    def process(self):
        if isinstance(self.df, pd.Series):
            self._was_series = True
            self.df = pd.DataFrame(self.df, columns=[self.COLUMN])

        raw = self.df[self.COLUMN].values.astype(np.float64)
        cleaned = nk.rsp_clean(raw, sampling_rate=self.fs, method=self.method)
        self.df[self.COLUMN] = cleaned

        return self.df[self.COLUMN] if self._was_series else self.df

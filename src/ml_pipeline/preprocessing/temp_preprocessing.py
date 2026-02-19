# Temperature Preprocessing
# Skin-temperature smoothing using Savitzky-Golay filter.

# There is no dedicated temperature module in NeuroKit2 or BioSPPy, temperature is a very slow-changing signal (variations in
# minutes) and Savitzky-Golay polynomial smoothing is the standard sufficient approach.

# Chest (700 Hz) and wrist (64 Hz upsampled) share the same filter only the key differs ( c_temp vs w_temp ).

import pandas as pd
from scipy.signal import savgol_filter


class TempPreprocessing:
    # Smooth a skin-temperature signal with Savitzky-Golay.

    def __init__(
        self,
        df,
        window_size: int = 11,
        poly_order: int = 3,
        wrist: bool = False,
    ):
        self.df = df
        self.window_size = window_size
        self.poly_order = poly_order
        self.wrist = wrist
        self.column = "w_temp" if wrist else "c_temp"
        self._was_series = False

    def process(self):
        # Return the dataframe with the temperature column smoothed.
        if isinstance(self.df, pd.Series):
            self._was_series = True
            self.df = pd.DataFrame(self.df, columns=[self.column])

        raw = self.df[self.column].values
        smoothed = savgol_filter(raw, self.window_size, self.poly_order)
        self.df[self.column] = smoothed

        return self.df[self.column] if self._was_series else self.df

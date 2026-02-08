import sys
import gc
import os
import pandas as pd
import numpy as np
import pickle
from scipy import signal
from scipy.interpolate import interp1d

# Import metadata extraction functions from our metadata extractor module
from .wesad_metadata_extractor import (
    extract_wesad_metadata,
    determine_chest_sampling_rate,
    determine_wrist_sampling_rates,
    get_column_names_from_data,
    LABEL_FREQ,
)


class WESADResampleMerge:
    def __init__(
        self,
        dataset_path="data/raw/wesad/WESAD/",
        IDs=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
    ):
        self.IDs = IDs
        self.dataset_path = dataset_path
        self.raw_path = os.path.join(dataset_path, "merged/")

        # Ensure the directories exist
        dir_name = os.path.dirname(self.dataset_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        dir_name = os.path.dirname(self.raw_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Initialize metadata as None - will be extracted on first use
        self._metadata = None
        self._sampling_rates = None
        self._columns = None
        self._chest_columns = None
        self._wrist_columns = None
        self._all_columns = None

    # use lazy loading pattern @property to save time and memory
    # get metadata from first subject file
    @property
    def metadata(self):
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    @property
    def sampling_rates(self):
        if self._sampling_rates is None:
            self._load_metadata()
        return self._sampling_rates

    # Dynamically generated chest columns
    @property
    def CHEST_COLUMNS(self):
        if self._chest_columns is None:
            self._load_metadata()
        return self._chest_columns

    @property
    def WRIST_COLUMNS(self):
        if self._wrist_columns is None:
            self._load_metadata()
        return self._wrist_columns

    # Dynamically generated all columns
    @property
    def ALL_COLUMNS(self):
        if self._all_columns is None:
            self._load_metadata()
        return self._all_columns

    # Chest sampling frequency
    @property
    def SF_CHEST(self):
        return self.sampling_rates["chest"]

    # BVP sampling frequency
    @property
    def SF_BVP(self):
        return self.sampling_rates["wrist"]["BVP"]

    # EDA sampling frequency
    @property
    def SF_EDA(self):
        return self.sampling_rates["wrist"]["EDA"]

    # Temperature sampling frequency
    @property
    def SF_TEMP(self):
        return self.sampling_rates["wrist"]["TEMP"]

    # Wrist ACC sampling frequency
    @property
    def SF_ACC(self):
        return self.sampling_rates["wrist"]["ACC"]

    # Maximum wrist sampling frequency
    @property
    def SF_WRIST(self):
        return max(self.sampling_rates["wrist"].values())

    #   Load metadata using wesad_metadata_extractor functions.

    # find first available subject file (here S2) to extract structure infro from
    def _find_first_subject_file(self):
        for subject_id in self.IDs:
            path = os.path.join(
                self.dataset_path, f"S{subject_id}", f"S{subject_id}.pkl"
            )
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"No subject files found in {self.dataset_path}. "
            "Please download the WESAD dataset first.\n"
            f"Expected structure: {self.dataset_path}/S2/S2.pkl"
        )

    def _load_metadata(self):
        print("Loading dataset metadata...")

        # Find first available subject file
        first_subject_path = self._find_first_subject_file()
        print(f"Using: {first_subject_path}")

        # Use metadata extractor functions
        self._metadata = extract_wesad_metadata(first_subject_path)

        # Determine sampling rates
        chest_rate = determine_chest_sampling_rate(first_subject_path)
        wrist_rates = determine_wrist_sampling_rates(first_subject_path, chest_rate)

        self._sampling_rates = {"chest": int(chest_rate), "wrist": wrist_rates}

        # Get column names
        self._columns = get_column_names_from_data(first_subject_path)

        self._chest_columns = self._columns["chest"]
        self._wrist_columns = self._columns["wrist"]
        self._all_columns = self._columns

        # Print summary
        print(f"Chest sampling rate: {self.SF_CHEST} Hz")
        print(f"Wrist sampling rates: {wrist_rates}")
        print(f"Chest columns ({len(self.CHEST_COLUMNS)}): {self.CHEST_COLUMNS}")
        print(f"Wrist columns ({len(self.WRIST_COLUMNS)}): {self.WRIST_COLUMNS}")
        print(f"Metadata loaded successfully!")

    # Convert chest pickle data to numpy array.
    def pkl_to_np_chest(self, filename, subject_id):
        print(f"Processing chest data for subject {subject_id}")

        with open(filename, "rb") as f:
            unpickled_df = pickle.load(f, encoding="latin1")

        # Get the  length from label
        # We need this to know how many rows of NaNs to create if a sensor is missing
        label = unpickled_df["label"].reshape(-1, 1)
        n_rows = label.shape[0]

        # pre-allocate list with Subject ID
        chest_data_parts = [np.full((n_rows, 1), subject_id)]

        #  Strict Schema Definition
        schema = [
            ("ACC", 3),  # Accelerometer has 3 columns
            ("ECG", 1),
            ("EMG", 1),
            ("EDA", 1),
            ("Temp", 1),
            ("Resp", 1),
        ]

        if "signal" in unpickled_df and "chest" in unpickled_df["signal"]:
            chest_dict = unpickled_df["signal"]["chest"]

            for sensor_name, expected_cols in schema:
                if sensor_name in chest_dict:
                    # CASE A: Sensor Exists
                    data = chest_dict[sensor_name]

                    # Fix 1D arrays to 2D
                    if len(data.shape) == 1:
                        data = data.reshape(-1, 1)

                    # Double Check
                    if data.shape[1] != expected_cols:
                        print(
                            f" Warning: {sensor_name} expected {expected_cols} cols, got {data.shape[1]}. Imputing NaNs."
                        )
                        # Fallback to NaN filling if dimensions are weird
                        data = np.full((n_rows, expected_cols), np.nan)

                    chest_data_parts.append(data)

                else:
                    #  CASE B: Sensor Missing
                    print(
                        f" Warning: Missing sensor '{sensor_name}' for S{subject_id}. Filling with NaNs."
                    )
                    # Create a column of NaNs with the correct shape
                    nan_data = np.full((n_rows, expected_cols), np.nan)
                    chest_data_parts.append(nan_data)
        else:
            raise ValueError(f"Invalid file structure for S{subject_id}")

        # Add Label
        chest_data_parts.append(label)

        # Merge
        chest_all = np.concatenate(chest_data_parts, axis=1)

        print(f"Finished S{subject_id}. Output Shape: {chest_all.shape}")
        return chest_all

    def merge_chest_data(self):
        print("Merging chest data from subjects into a single file...")

        all_data = []
        for subject_id in self.IDs:
            file_path = os.path.join(
                self.dataset_path, f"S{subject_id}", f"S{subject_id}.pkl"
            )
            if os.path.exists(file_path):
                subject_data = self.pkl_to_np_chest(file_path, subject_id)
                all_data.append(subject_data)
            else:
                print(f"Warning: File not found for subject {subject_id}")

        if not all_data:
            raise ValueError("No data files found to merge!")

        merged_data = np.vstack(all_data)
        final_df = pd.DataFrame(merged_data, columns=self.CHEST_COLUMNS)

        output_path = os.path.join(self.raw_path, "merged_chest_temp.pkl")
        final_df.to_pickle(output_path)

        print(f"Merged {len(all_data)} subjects")
        print(f"Total samples: {len(merged_data):,}")
        print(f"Saved to: {output_path}")

    # All wrist signals are resampled to BVP sampling rate (highest rate).
    def pkl_to_np_wrist(self, filename, subject_id):
        print(f"Processing wrist data for subject {subject_id}")

        with open(filename, "rb") as f:
            unpickled_df = pickle.load(f, encoding="latin1")

        # Extract Signals. If a key is missing, WESAD is corrupt for this subject.
        try:
            wrist_acc = unpickled_df["signal"]["wrist"]["ACC"]
            wrist_bvp = unpickled_df["signal"]["wrist"]["BVP"]
            wrist_eda = unpickled_df["signal"]["wrist"]["EDA"]
            wrist_temp = unpickled_df["signal"]["wrist"]["TEMP"]
            label = unpickled_df["label"].reshape(-1, 1)
        except KeyError as e:
            print(f"Missing sensor data in {filename}: {e}")
            return None

        # Setup Dimensions
        n_wrist_bvp = len(wrist_bvp)

        # Create subject ID array
        sid_bvp = np.repeat(subject_id, n_wrist_bvp).reshape((n_wrist_bvp, 1))

        # Interpolate Labels -Nearest-Neighbor Interpolation
        label = label.flatten()
        time_original = np.linspace(0, len(label) / self.SF_CHEST, len(label))
        time_target = np.linspace(0, len(label) / self.SF_CHEST, n_wrist_bvp)

        interpolator = interp1d(
            time_original, label, kind="nearest", fill_value="extrapolate"
        )
        label_bvp = interpolator(time_target).reshape(-1, 1)

        # Resample ACC (3D)
        resampled_wrist_acc = np.zeros((n_wrist_bvp, 3))
        for i in range(3):
            resampled_wrist_acc[:, i] = signal.resample(wrist_acc[:, i], n_wrist_bvp)

        # Resample EDA and Temp (1D)
        wrist_eda_resampled = signal.resample(wrist_eda, n_wrist_bvp).reshape(-1, 1)
        wrist_temp_resampled = signal.resample(wrist_temp, n_wrist_bvp).reshape(-1, 1)
        wrist_bvp = wrist_bvp.reshape(-1, 1)

        print(f"Finished processing wrist data for subject {subject_id}")

        return (
            sid_bvp,
            resampled_wrist_acc,
            wrist_bvp,
            wrist_eda_resampled,
            wrist_temp_resampled,
            label_bvp,
        )

    def merge_wrist_data(self):
        print("Merging all wrist data from subjects into a single file...")

        combined_data = []
        for subject_id in self.IDs:
            file_path = os.path.join(
                self.dataset_path, f"S{subject_id}", f"S{subject_id}.pkl"
            )

            if os.path.exists(file_path):
                result = self.pkl_to_np_wrist(file_path, subject_id)

                if result is not None:
                    sid, wrist_acc, wrist_bvp, wrist_eda, wrist_temp, label = result

                    combined_subj = np.hstack(
                        (sid, wrist_acc, wrist_bvp, wrist_eda, wrist_temp, label)
                    )
                    combined_data.append(combined_subj)
            else:
                print(f" Warning: File not found for subject {subject_id}")

        if not combined_data:
            raise ValueError("No data files found to merge!")

        # Stack all subjects
        final_data = np.vstack(combined_data)

        # Save
        output_path = os.path.join(self.raw_path, "merged_wrist.pkl")

        # USE DYNAMIC COLUMNS
        pd.DataFrame(final_data, columns=self.WRIST_COLUMNS).to_pickle(output_path)

        print(f"Merged {len(combined_data)} subjects")
        print(f"Total samples: {len(final_data):,}")
        print(f"Saved to: {output_path}")

    # Interpolate to "fix" invalid temperature readings (temp <= 0)
    def filter_chest_data(self):
        input_path = os.path.join(self.raw_path, "merged_chest_temp.pkl")
        df = pd.read_pickle(input_path)

        initial_count = len(df)
        print(f"Initial samples: {initial_count:,}")

        # LABELS: Do NOTHING yet. Initially we filtered amusement and meditation out but - see my notes - to avoid frankenstein windows we dont
        # We keep 0, 4, 5, 6, 7 so the timeline remains continuous.
        # We will drop the bad windows in Phase 2.

        # TEMPERATURE: Impute (Interpolate)
        temp_col = "c_temp" if "c_temp" in df.columns else "temp"

        if temp_col in df.columns:
            # Count artifacts (0 or less)
            zeros = (df[temp_col] <= 0).sum()
            print(f"Found {zeros:,} artifact samples (Temp <= 0) to fix.")

            if zeros > 0:
                # Replace <= 0 with NaN (Empty)
                df.loc[df[temp_col] <= 0, temp_col] = np.nan

                # Interpolate (Fill NaN with a straight line between neighbors)
                df[temp_col] = df[temp_col].interpolate(
                    method="linear", limit_direction="both"
                )

                print(f"Interpolated {zeros:,} samples. Timeline is preserved.")

                # Debug: verify immediately after interpolation
                bad = (df[temp_col] <= 0).sum()
                print("Post-interpolation bad count:", bad)
                if bad:
                    idx = df.index[df[temp_col] <= 0][:10]
                    print("Example bad values:", df.loc[idx, temp_col].tolist())
                    # show neighbors to see why interpolation ended at 0/-0.0
                    for i in idx[:3]:
                        print(
                            i,
                            "prev/cur/next:",
                            df.loc[i - 1, temp_col] if i - 1 in df.index else None,
                            df.loc[i, temp_col],
                            df.loc[i + 1, temp_col] if i + 1 in df.index else None,
                        )
        else:
            print("Warning: Temperature column not found.")

        # Save the continuous, cleaned file
        output_path = os.path.join(self.raw_path, "merged_chest.pkl")
        df.to_pickle(output_path)

        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)

        print(f"Saved cleaned data to: {output_path}")
        print(f"Total samples: {len(df):,} (No rows dropped)")

    def filter_wrist_data(self):
        input_path = os.path.join(self.raw_path, "merged_wrist.pkl")
        df = pd.read_pickle(input_path)

        print(f"Initial wrist samples: {len(df):,}")

        # TEMPERATURE: Impute (Interpolate)
        temp_col = "w_temp"

        if temp_col in df.columns:
            zeros = (df[temp_col] <= 0).sum()
            print(f"Found {zeros:,} wrist temp artifacts (Temp <= 0) to fix.")

            if zeros > 0:
                df.loc[df[temp_col] <= 0, temp_col] = np.nan
                df[temp_col] = df[temp_col].interpolate(
                    method="linear", limit_direction="both"
                )
                print(f"Interpolated {zeros:,} wrist temp samples.")

        df.to_pickle(input_path)
        print(f"Saved cleaned wrist data. Total samples: {len(df):,}")

    def resample_merge(self):
        print("Starting preprocessing...")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output path: {self.raw_path}")
        print(f"Subjects: {self.IDs}")

        # Load metadata
        _ = self.metadata

        self.merge_wrist_data()
        self.merge_chest_data()
        self.filter_chest_data()  # TEMPORAL INCONSISTENCIES. TODO move filtering to the windowing stage instead
        self.filter_wrist_data()

        print("\n" + "=" * 70)
        print("Successfully loaded merged and filtered chest and wrist data!")
        print("=" * 70)
        print("\nOutput files:")
        print(f"  - {os.path.join(self.raw_path, 'merged_wrist.pkl')}")
        print(f"  - {os.path.join(self.raw_path, 'merged_chest.pkl')}")
        print("\nSampling rates used:")
        print(f"  - Chest: {self.SF_CHEST} Hz")
        for sensor, rate in self.sampling_rates["wrist"].items():
            print(f"  - Wrist {sensor}: {rate} Hz")


if __name__ == "__main__":
    preprocessor = WESADResampleMerge()
    preprocessor.resample_merge()

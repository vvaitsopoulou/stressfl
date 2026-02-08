# Extract Metadata from WESAD Dataset Files - programmatically determine sampling rates and structure

import pickle
import numpy as np

LABEL_FREQ = 700  # from the dataset readme


# Extract metadata from a WESAD pickle file:
# Available signal types
# Signal shapes (to infer sampling rates)
# Data structure


def extract_wesad_metadata(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    metadata = {
        "chest_signals": {},
        "wrist_signals": {},
        "label_length": len(data["label"]),
    }

    # Extract chest signal information
    if "signal" in data and "chest" in data["signal"]:
        for signal_name, signal_data in data["signal"]["chest"].items():
            metadata["chest_signals"][signal_name] = {
                "shape": signal_data.shape,
                "length": len(signal_data),
            }

    # Extract wrist signal information
    if "signal" in data and "wrist" in data["signal"]:
        for signal_name, signal_data in data["signal"]["wrist"].items():
            metadata["wrist_signals"][signal_name] = {
                "shape": signal_data.shape,
                "length": len(signal_data),
            }

    return metadata


# we have to use the base frequency of 700Hz as provided in the dataset readme (label frequency)
# otherwise only relative frequency may be calculated (by using chest as the base)
def infer_sampling_rates(metadata):
    label_length = metadata["label_length"]

    sampling_rates = {"chest": {}, "wrist": {}}

    # For chest signals
    for signal_name, info in metadata["chest_signals"].items():
        signal_length = info["length"]
        # Sampling rate relative to label
        relative_rate = signal_length / label_length
        sampling_rates["chest"][signal_name] = {
            "relative_to_label": relative_rate,
            "absolute": relative_rate * LABEL_FREQ,
        }

    # For wrist signals
    for signal_name, info in metadata["wrist_signals"].items():
        signal_length = info["length"]
        relative_rate = signal_length / label_length
        sampling_rates["wrist"][signal_name] = {
            "relative_to_label": relative_rate,
            "absolute": relative_rate * LABEL_FREQ,
        }

    return sampling_rates


# Returns Chest sampling rate in Hz
def determine_chest_sampling_rate(pickle_path):
    metadata = extract_wesad_metadata(pickle_path)

    # Get any chest signal length (all the same rate)
    if metadata["chest_signals"]:
        first_signal = list(metadata["chest_signals"].values())[0]
        signal_length = first_signal["length"]
        label_length = metadata["label_length"]

        # The ratio gives us the chest sampling rate
        chest_sampling_rate = (signal_length / label_length) * LABEL_FREQ

        return chest_sampling_rate

    return None


def determine_wrist_sampling_rates(pickle_path, chest_sampling_rate):
    metadata = extract_wesad_metadata(pickle_path)
    label_length = metadata["label_length"]

    wrist_rates = {}

    for signal_name, info in metadata["wrist_signals"].items():
        signal_length = info["length"]
        # Calculate rate relative to label (which is at chest_sampling_rate)
        rate = (signal_length / label_length) * chest_sampling_rate
        wrist_rates[signal_name] = round(rate)

    return wrist_rates


#  Extract the actual signal names from the pickle file to create column names.
def get_column_names_from_data(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    columns = {
        "chest": ["sid"],  # Start with subject ID
        "wrist": ["sid"],
    }

    # Chest signals
    if "signal" in data and "chest" in data["signal"]:
        for signal_name, signal_data in data["signal"]["chest"].items():
            prefix = "c_" if signal_name.lower() in ["eda", "temp", "acc"] else ""

            if len(signal_data.shape) == 1 or (
                len(signal_data.shape) == 2 and signal_data.shape[1] == 1
            ):
                # 1D signal (ECG, EMG, EDA, TEMP, RESP)
                columns["chest"].append(f"{prefix}{signal_name.lower()}")
            elif len(signal_data.shape) == 2 and signal_data.shape[1] == 3:
                # 3D accelerometer
                columns["chest"].extend(
                    [f"{prefix}acc_{axis}" for axis in ["x", "y", "z"]]
                )

    columns["chest"].append("label")

    # Wrist signals
    if "signal" in data and "wrist" in data["signal"]:
        for signal_name, signal_data in data["signal"]["wrist"].items():
            prefix = "w_" if signal_name.lower() in ["eda", "temp", "acc"] else ""

            if len(signal_data.shape) == 1 or (
                len(signal_data.shape) == 2 and signal_data.shape[1] == 1
            ):
                # 1D signal (BVP, EDA, TEMP)
                columns["wrist"].append(f"{prefix}{signal_name.lower()}")
            elif len(signal_data.shape) == 2 and signal_data.shape[1] == 3:
                # 3D accelerometer
                columns["wrist"].extend(
                    [f"{prefix}acc_{axis}" for axis in ["x", "y", "z"]]
                )

    columns["wrist"].append("label")

    return columns


# Uncomment to debug
# if __name__ == '__main__':
#     # Example subject 2
#     example_path = 'data/raw/wesad/WESAD/S2/S2.pkl'

#     # Extract metadata
#     print("\n1. Extracting metadata...")
#     metadata = extract_wesad_metadata(example_path)
#     print(f"   Label length: {metadata['label_length']}")
#     print(f"   Chest signals: {list(metadata['chest_signals'].keys())}")
#     print(f"   Wrist signals: {list(metadata['wrist_signals'].keys())}")

#     # Determine chest sampling rate
#     print("\n2. Determining chest sampling rate...")
#     chest_rate = determine_chest_sampling_rate(example_path)
#     print(f"   Chest sampling rate: {chest_rate} Hz")

#     # Determine wrist sampling rates
#     print("\n3. Determining wrist sampling rates...")
#     wrist_rates = determine_wrist_sampling_rates(example_path, chest_rate)
#     for signal, rate in wrist_rates.items():
#         print(f"   {signal}: {rate} Hz")

#     # Get column names
#     print("\n4. Extracting column names...")
#     columns = get_column_names_from_data(example_path)
#     print(f"   Chest columns: {columns['chest']}")
#     print(f"   Wrist columns: {columns['wrist']}")

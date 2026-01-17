# Import main class
from .wesad_resample_merge import WESADResampleMerge

# Import metadata extraction utilities
from .wesad_metadata_extractor import (
    extract_wesad_metadata,
    infer_sampling_rates,
    determine_chest_sampling_rate,
    determine_wrist_sampling_rates,
    get_column_names_from_data,
    LABEL_FREQ,  # Dataset constant
)

# Define what gets imported
__all__ = [
    # Main class
    "WESADResampleMerge",
    # Metadata extraction functions
    "extract_wesad_metadata",
    "infer_sampling_rates",
    "determine_chest_sampling_rate",
    "determine_wrist_sampling_rates",
    "get_column_names_from_data",
    # Constants
    "LABEL_FREQ",
]

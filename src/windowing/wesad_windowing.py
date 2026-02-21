# WESAD Windowing & Segmentation
#
# Segments preprocessed WESAD data into fixed-length windows for feature extraction.
#
# Design decisions (justified by literature):
#
# Window size:
#   60s for all signals, Schmidt et al. (2018) WESAD benchmark.
#   This is sufficient for time-domain HRV, Poincaré, and LF/HF frequency-domain features.
#   VLF/ULF frequency-domain HRV features require ≥300s and are dropped (as in multiple
#   WESAD studies: the dataset conditions are too short for reliable VLF estimation).
#
# ACC sub-windows:
#   5s sub-windows within each 60s window, per Schmidt et al. (2018).
#   ACC is high-frequency motion data so 60s averaging washes out patterns.
#   Feature extraction will aggregate sub-window features (mean/std) into the 60s window.
#
# Slide / overlap:
#   Configurable. Schmidt benchmark uses 0.25s shift.
#   For RF, high overlap introduces correlated samples whixh according to literature is not ideal for tree-based models.
#   Therefore ablation is supported: 0.25s, 2s, 5s, 20s, 60s (no overlap).
#
# Label purity:
#   Windows spanning multiple labels are rejected. This avoids "frankenstein windows"
#   where part of the data is baseline and part is stress. We check this strictly.
#
# Subject isolation:
#   Windowing is always done per-subject. No window ever crosses subject boundaries.
#
# Reference:
#   Schmidt, P. et al. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. ICMI '18.

import json
import os
import pickle

import numpy as np
import pandas as pd


class WESADWindowing:
    # Segments preprocessed WESAD data into fixed-length windows.

    # Parameters:

    # data_path : str
    #     Path to the preprocessed pickle file
    # config_path : str
    #     Path to dataset JSON config (for sampling rate, active labels).
    # output_dir : str
    #     Directory where windowed outputs will be saved.
    # window_size_sec : float
    #     Window length in seconds (default: 60).
    # slide_sec : float
    #     Slide/shift in seconds (default: 0.25 for benchmark, but use larger for RF).
    # acc_subwindow_sec : float
    #     ACC sub-window size in seconds (default: 5, per Schmidt 2018).
    # wrist : bool
    #     Whether this is wrist data .

    # Valid WESAD labels for stress detection (from config)
    # 1 = baseline, 2 = stress, 3 = amusement

    def __init__(
        self,
        data_path: str,
        config_path: str,
        output_dir: str,
        window_size_sec: float = 60.0,
        slide_sec: float = 0.25,
        acc_subwindow_sec: float = 5.0,
        wrist: bool = False,
    ):
        self.data_path = data_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.window_size_sec = window_size_sec
        self.slide_sec = slide_sec
        self.acc_subwindow_sec = acc_subwindow_sec
        self.wrist = wrist

        with open(config_path) as f:
            self._config = json.load(f)

        # Get sampling rate and active labels
        self.sampling_rate = max(self._config["sampling rate"].values())
        self.active_labels = [
            key for key, value in self._config["labels"].items() if value
        ]

        # Convert to sample counts
        self.window_samples = int(self.window_size_sec * self.sampling_rate)
        self.slide_samples = int(self.slide_sec * self.sampling_rate)
        self.acc_subwindow_samples = int(self.acc_subwindow_sec * self.sampling_rate)

        # ACC is the only signal sub-windowed at this stage (5s chunks within each 60s window), so we need to select columns explicitly.
        # EDA and Temp are not sub-windowed so the feature extractor downstream can resolves column names itself.
        if wrist:
            self._acc_cols = ["w_acc_x", "w_acc_y", "w_acc_z"]
        else:
            self._acc_cols = ["c_acc_x", "c_acc_y", "c_acc_z"]

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        print("WESADWindowing initialized:")
        print(f"  Device: {'wrist' if wrist else 'chest'}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Window: {self.window_size_sec}s ({self.window_samples} samples)")
        print(f"  Slide: {self.slide_sec}s ({self.slide_samples} samples)")
        print(
            f"  ACC sub-window: {self.acc_subwindow_sec}s ({self.acc_subwindow_samples} samples)"
        )
        print(f"  Active labels: {self.active_labels}")

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_pickle(self.data_path)
        print(
            f"Loaded data: {len(self.df):,} samples, columns: {list(self.df.columns)}"
        )
        return self.df

    def segment(self, df: pd.DataFrame | None = None) -> list[dict]:
        # Segment the dataframe into fixed-length windows.

        # Returns a list of window dicts, each containing:
        # 'data': pd.DataFrame of shape (window_samples, n_features)
        # 'sid': subject ID
        # 'label': the (single) label for this window
        # 'window_idx': sequential index within this subject+label
        # 'start_sample': absolute start index in the subject's data
        # 'is_overlapped': True if this window overlaps with the previous non-overlapped window
        # 'acc_subwindows': list of pd.DataFrames, each of shape (acc_subwindow_samples, 3)
        #                         for the ACC channels only (for sub-window feature extraction)

        # Rejected windows:
        # Windows shorter than window_samples (tail ends)
        # Windows spanning multiple labels ("frankenstein windows")
        #  Windows whose label is not in active_labels

        if (
            df is None
        ):  # if we run ablations and this is called multiple times check if this has already been loaded in the attribute
            if not hasattr(self, "df"):
                self.load_data()
            df = self.df

        windows = []
        rejected = {"short": 0, "mixed_label": 0, "inactive_label": 0}

        grouped = df.groupby("sid")

        for sid, subject_df in grouped:
            # Reset index for clean iloc within this subject
            subject_data = subject_df.reset_index(drop=True)

            # Identify contiguous label runs
            # optimizin ghere: instead of sliding blindly and checking label continuity after, we identify contiguous segments of the same label
            # and only create windows within them
            label_col = subject_data["label"].values
            label_runs = self._get_label_runs(label_col)

            subject_window_count = 0

            for label_val, run_start, run_end in label_runs:
                # Check if this label is active
                if str(float(label_val)) not in self.active_labels:
                    rejected["inactive_label"] += run_end - run_start
                    continue

                # Slide through this contiguous label run
                pos = run_start
                local_idx = 0
                while pos + self.window_samples <= run_end:
                    window_data = subject_data.iloc[
                        pos : pos + self.window_samples
                    ].copy()

                    # Safety check: verify label purity (should always pass due to run-based approach)
                    if window_data["label"].nunique() != 1:
                        rejected["mixed_label"] += 1
                        pos += self.slide_samples
                        continue

                    # Determine if this is an overlapped window
                    # A window is "non-overlapped" if it starts at a multiple of window_samples
                    # relative to the run start
                    relative_pos = pos - run_start
                    is_overlapped = (relative_pos % self.window_samples) != 0

                    # Extract ACC sub-windows
                    acc_subwindows = self._extract_acc_subwindows(window_data)

                    window_info = {
                        "data": window_data,
                        "sid": sid,
                        "label": label_val,
                        "window_idx": local_idx,
                        "start_sample": pos,
                        "is_overlapped": is_overlapped,
                        "acc_subwindows": acc_subwindows,
                    }

                    windows.append(window_info)
                    local_idx += 1
                    subject_window_count += 1
                    pos += self.slide_samples

            # Count tail samples that didn't fill a complete window
            tail = run_end - pos
            if 0 < tail < self.window_samples:
                rejected["short"] += 1

            print(f"  Subject {int(sid)}: {subject_window_count} windows")

        # Summary
        total = len(windows)
        overlapped = sum(1 for w in windows if w["is_overlapped"])
        non_overlapped = total - overlapped

        print("\nSegmentation complete:")
        print(f"  Total windows: {total}")
        print(f"  Non-overlapped (independent): {non_overlapped}")
        print(f"  Overlapped (augmented): {overlapped}")
        print(
            f"  Rejected short: {rejected['short']}, mixed label: {rejected['mixed_label']}, "
            f"inactive label: {rejected['inactive_label']}"
        )

        # Label distribution
        label_counts: dict[float, int] = {}
        for w in windows:
            lbl = w["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  Label distribution: {label_counts}")

        return windows

    def _get_label_runs(self, labels: np.ndarray) -> list[tuple]:
        # Identify contiguous runs of the same label.

        # Returns list of (label_value, start_idx, end_idx) tuples.
        # end_idx is exclusive.
        if len(labels) == 0:
            return []

        runs = []
        current_label = labels[0]
        run_start = 0

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                runs.append((current_label, run_start, i))
                current_label = labels[i]
                run_start = i

        # last run
        runs.append((current_label, run_start, len(labels)))

        return runs

    def _extract_acc_subwindows(self, window_data: pd.DataFrame) -> list[pd.DataFrame]:
        # Split the ACC columns of a window into sub-windows.

        # Per Schmidt 2018, ACC features are extracted on 5s sub-windows
        # and then aggregated (mean/std) for the full 60s window.
        acc_cols_present = [c for c in self._acc_cols if c in window_data.columns]
        if not acc_cols_present:
            return []

        acc_data = window_data[acc_cols_present].reset_index(drop=True)
        n_samples = len(acc_data)
        subwindows = []

        pos = 0
        while pos + self.acc_subwindow_samples <= n_samples:
            sub = acc_data.iloc[pos : pos + self.acc_subwindow_samples].copy()
            subwindows.append(sub)
            pos += self.acc_subwindow_samples  # non-overlapping sub-windows

        return subwindows

    def save_windows(self, windows: list[dict], tag: str = ""):
        #     Save windowed data to disk.

        #     Saves two files, Full window list as a pickle (for feature extraction) and metadata CSV with window-level info (for quick inspection)

        #     Parameters
        #     windows : list of dict
        #         Output from segment().
        #     tag : str
        #         Optional tag for the filename (e.g. 'slide_5s', 'no_overlap').
        device = "wrist" if self.wrist else "chest"
        suffix = f"_{tag}" if tag else ""

        # Save full windows
        windows_path = os.path.join(
            self.output_dir,
            f"windows_{device}_w{int(self.window_size_sec)}s_s{self.slide_sec}s{suffix}.pkl",
        )
        with open(windows_path, "wb") as f:
            pickle.dump(windows, f)
        print(f"Saved windows → {windows_path}")

        # Save metadata CSV for quick inspection
        meta = pd.DataFrame(
            [
                {
                    "sid": w["sid"],
                    "label": w["label"],
                    "window_idx": w["window_idx"],
                    "start_sample": w["start_sample"],
                    "is_overlapped": w["is_overlapped"],
                    "n_acc_subwindows": len(w["acc_subwindows"]),
                }
                for w in windows
            ]
        )
        meta_path = os.path.join(
            self.output_dir,
            f"windows_{device}_w{int(self.window_size_sec)}s_s{self.slide_sec}s{suffix}_meta.csv",
        )
        meta.to_csv(meta_path, index=False)
        print(f"Saved metadata → {meta_path}")

        return windows_path, meta_path

    # changed to solve mutable default argument (ruff B006)
    def run_ablation(
        self,
        slide_values: list[float] | None = None,
        df: pd.DataFrame | None = None,
    ) -> dict:
        if slide_values is None:
            slide_values = [0.25, 2.0, 5.0, 20.0, 60.0]
        # Run windowing at multiple overlap settings for ablation study.
        # slide_values : list of float
        #     Slide durations in seconds to test.
        # df : pd.DataFrame, optional
        #     Pre-loaded dataframe. If None, loads from data_path.

        if df is None:
            if not hasattr(self, "df"):
                self.load_data()
            df = self.df

        results = {}

        for slide in slide_values:
            tag = "no_overlap" if slide >= self.window_size_sec else f"slide_{slide}s"
            print(f"\nAblation: slide = {slide}s ({tag})\n")

            # Update slide for this run
            self.slide_sec = slide
            self.slide_samples = int(slide * self.sampling_rate)

            windows = self.segment(df)
            self.save_windows(windows, tag=tag)
            results[slide] = windows

        return results


def summarize_ablation(results: dict, window_size_sec: float = 60.0):

    print(f"\nABLATION SUMMARY (window = {window_size_sec}s)")
    print(
        f"\n{'Slide (s)':<12} {'Overlap %':<12} {'Total':<10} {'Independent':<14} {'Overlapped':<12} {'Labels'}"
    )

    for slide, windows in sorted(results.items()):
        overlap_pct = max(0, (1 - slide / window_size_sec) * 100)
        total = len(windows)
        independent = sum(1 for w in windows if not w["is_overlapped"])
        overlapped = total - independent

        label_counts: dict[float, int] = {}
        for w in windows:
            lbl = w["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        label_str = ", ".join(
            f"{key}: {val}" for key, val in sorted(label_counts.items())
        )
        print(
            f"{slide:<12.2f} {overlap_pct:<12.1f} {total:<10} {independent:<14} {overlapped:<12} {label_str}"
        )

    print()


# to quickly run this script from lci
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WESAD Windowing Pipeline")
    parser.add_argument("--data", required=True, help="Path to preprocessed pickle")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--output", default="data/windowed/", help="Output directory")
    parser.add_argument(
        "--window", type=float, default=60.0, help="Window size in seconds"
    )
    parser.add_argument("--slide", type=float, default=0.25, help="Slide in seconds")
    parser.add_argument("--wrist", action="store_true", help="Wrist device mode")
    parser.add_argument(
        "--ablation", action="store_true", help="Run slide ablation study"
    )

    args = parser.parse_args()

    windower = WESADWindowing(
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output,
        window_size_sec=args.window,
        slide_sec=args.slide,
        wrist=args.wrist,
    )

    if args.ablation:
        results = windower.run_ablation()
        summarize_ablation(results, window_size_sec=args.window)
    else:
        windower.load_data()
        windows = windower.segment()
        windower.save_windows(windows)

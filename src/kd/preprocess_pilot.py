"""
Merge two-session Fitbit pilot data into a single preprocessing result.

Background
----------
The original pilot protocol assumed one continuous 22-minute session
(T1..T6) with a time-based cal/eval split inside it. The actual data
comes from two separate sessions on the same day:

  Morning (cal):    session_start = 11:13:00
                    T1 baseline-A    (0-4 min)   -> non-stress
                    T2 stressor-A    (4-8 min)   -> stress
                    T3..T6           (8-22 min)  -> EXCLUDED
                                                    (T5 failed manipulation
                                                    check; T4/T6 unused)

  Afternoon (eval): session_start = 13:50:00
                    T1' baseline     (0-4 min)   -> non-stress
                    T2' cold pressor (4-7 min)   -> stress
                    T3' recovery     (7-10 min)  -> EXCLUDED

Both sessions live in the same JSON file (the Fitbit API returns the
whole day's intraday data on every call), but they are processed
independently so window-timestamp numbering restarts at zero for each.

Strategy
--------
Run preprocess_fitbit_session() twice with different session_start
values and custom protocols, then concatenate the two result dicts
into one. cal_mask / eval_mask are reassigned by *which session each
window came from*, overriding the time-cutoff split inside
preprocess_fitbit_session().

The merged dict has the same shape as a single preprocess_fitbit_session
return value, so fitbit_evaluator.py works unmodified.
"""

from __future__ import annotations

import numpy as np

from src.kd.fitbit_preprocessor import preprocess_fitbit_session

# Wall-clock anchors and per-session protocols
CAL_SESSION_START = "11:13:00"
EVAL_SESSION_START = "13:50:00"

# Each session gets its own protocol. Times are minutes from session_start.
# Tuple format: (start_min, end_min, label, phase_name)
#   label = 0 -> non-stress
#   label = 1 -> stress
#   label = -1 -> excluded (recovery, failed manipulation, etc.)
CAL_PROTOCOL = [
    (0, 4, 0, "cal_baseline_A"),
    (4, 8, 1, "cal_stress_A"),
    # T3..T6 of morning session deliberately omitted; any windows after
    # 8 min of session_start fall outside the protocol and inherit
    # label = -1 ("unlabelled") from assign_labels().
]

EVAL_PROTOCOL = [
    (0, 4, 0, "eval_baseline_B"),
    (4, 7, 1, "eval_stress_B_cold_pressor"),
    # T3' recovery (7-10 min) deliberately omitted -> windows there get
    # label = -1 and are filtered out by `(labels >= 0)` masks downstream.
]


def preprocess_two_session_pilot(
    json_path: str,
    cal_session_start: str = CAL_SESSION_START,
    eval_session_start: str = EVAL_SESSION_START,
    cal_protocol: list[tuple] | None = None,
    eval_protocol: list[tuple] | None = None,
) -> dict:
    """Run preprocessing on two sessions and merge into one result dict.

    Parameters
    ----------
    json_path : str
        Path to the Fitbit intraday HR JSON. Must contain both sessions.
    cal_session_start, eval_session_start : str
        "HH:MM:SS" anchors for each session.
    cal_protocol, eval_protocol : list of tuples, optional
        Per-session protocol definitions. Defaults are module-level
        constants.

    Returns
    -------
    dict with the same keys as preprocess_fitbit_session(), but with
    arrays concatenated across both sessions and cal_mask / eval_mask
    reassigned by session of origin.
    """
    if cal_protocol is None:
        cal_protocol = CAL_PROTOCOL
    if eval_protocol is None:
        eval_protocol = EVAL_PROTOCOL

    print("=" * 70)
    print("CAL SESSION (morning)")
    print("=" * 70)
    cal = preprocess_fitbit_session(
        json_path,
        session_start=cal_session_start,
        protocol=cal_protocol,
    )

    print("\n" + "=" * 70)
    print("EVAL SESSION (afternoon)")
    print("=" * 70)
    evl = preprocess_fitbit_session(
        json_path,
        session_start=eval_session_start,
        protocol=eval_protocol,
    )

    # ----- Merge -----
    # Per-window arrays: simply concatenate. The window_timestamps are in
    # session-local seconds, so the cal session's t-axis and the eval
    # session's t-axis are independent. That's fine: nothing downstream
    # uses absolute time after this point; everything is per-window.
    merged = {
        "features": np.concatenate([cal["features"], evl["features"]], axis=0),
        "labels": np.concatenate([cal["labels"], evl["labels"]], axis=0),
        "phase_names": np.concatenate([cal["phase_names"], evl["phase_names"]], axis=0),
        "quality": np.concatenate([cal["quality"], evl["quality"]], axis=0),
        "timestamps": np.concatenate([cal["timestamps"], evl["timestamps"]], axis=0),
    }

    # cal_mask / eval_mask are reassigned by session, NOT by the
    # in-session time cutoff that preprocess_fitbit_session uses.
    n_cal = len(cal["features"])
    n_eval = len(evl["features"])
    merged["cal_mask"] = np.concatenate(
        [
            np.ones(n_cal, dtype=bool),
            np.zeros(n_eval, dtype=bool),
        ]
    )
    merged["eval_mask"] = np.concatenate(
        [
            np.zeros(n_cal, dtype=bool),
            np.ones(n_eval, dtype=bool),
        ]
    )

    # Diagnostics carried through as a list (one entry per session) so
    # nothing is lost. Downstream code reads sampling_report / gap_report
    # only for the data-quality printout, which we'll regenerate ourselves.
    merged["sampling_report"] = [cal["sampling_report"], evl["sampling_report"]]
    merged["gap_report"] = [cal["gap_report"], evl["gap_report"]]

    # Per-session 1 Hz arrays kept separately (used only for plotting in
    # the evaluator's optional viz; concatenating would create a fake
    # ~3-hour gap that isn't physically meaningful).
    merged["hr_uniform"] = {"cal": cal["hr_uniform"], "eval": evl["hr_uniform"]}
    merged["t_uniform"] = {"cal": cal["t_uniform"], "eval": evl["t_uniform"]}
    merged["gap_flags"] = {"cal": cal["gap_flags"], "eval": evl["gap_flags"]}

    return merged


def summarise(merged: dict) -> None:
    """Print a sanity-check summary of the merged result."""
    labels = merged["labels"]
    quality = merged["quality"]
    cal_mask = merged["cal_mask"]
    eval_mask = merged["eval_mask"]
    phase_names = merged["phase_names"]

    print("\n" + "=" * 70)
    print("MERGED PILOT SUMMARY")
    print("=" * 70)
    print(f"Total windows:           {len(labels)}")
    print(f"  cal:                   {cal_mask.sum()}")
    print(f"  eval:                  {eval_mask.sum()}")
    print(f"Quality-valid windows:   {quality.sum()}/{len(quality)}")

    def _stats(name: str, mask: np.ndarray) -> None:
        m = mask & quality & (labels >= 0)
        n_total = mask.sum()
        n_valid = m.sum()
        n_stress = ((labels == 1) & m).sum()
        n_nonstress = ((labels == 0) & m).sum()
        n_excluded = ((labels < 0) & mask).sum()
        print(f"\n{name}:")
        print(f"  Windows in split:        {n_total}")
        print(f"  Valid + labelled:        {n_valid}")
        print(f"    non-stress (label=0):  {n_nonstress}")
        print(f"    stress     (label=1):  {n_stress}")
        print(f"  Excluded (label=-1):     {n_excluded}")
        if n_valid > 0:
            balance = n_stress / n_valid
            print(f"  Stress fraction:         {balance:.2%}")
        # phase breakdown
        unique_phases, counts = np.unique(phase_names[mask], return_counts=True)
        print("  By phase:")
        for p, c in zip(unique_phases, counts):
            print(f"    {p}: {c}")

    _stats("CAL", cal_mask)
    _stats("EVAL", eval_mask)


if __name__ == "__main__":
    import sys

    json_path = sys.argv[1] if len(sys.argv) > 1 else "fitbit_intraday_2026-05-03.json"
    result = preprocess_two_session_pilot(json_path)
    summarise(result)

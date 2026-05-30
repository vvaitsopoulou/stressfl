"""
Drop-in fixes for the Fitbit pilot prediction-timeline figure.

Two things are replaced in src/kd/fitbit_evaluator.py:

  1. build_pilot_timeline()      -> replaces the timeline-building block at
                                    the end of evaluate_fitbit_pilot()
                                    (roughly lines 487-516).
  2. plot_prediction_timeline()  -> replaces the existing function.

Why: the Fitbit API returns the whole day's intraday HR, and the old code
interpolated/windowed all of it, so the figure spanned ~18 h with the actual
protocol squished near t=0. These helpers crop each session to its labelled
span (+ pad) and render the two sessions as separate, consistent panels.

Metrics are unaffected -- this only changes the figure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

WINDOW_SEC = 16.0  # matches kd_config.WINDOW_SEC


# ---------------------------------------------------------------------------
# 1. PER-SESSION TIMELINE BUILDER
# ---------------------------------------------------------------------------
def _session_timeline(
    model,
    feats_norm: np.ndarray,
    ts: np.ndarray,
    labels: np.ndarray,
    phases: np.ndarray,
    quality: np.ndarray,
    hr_t: np.ndarray,
    hr_v: np.ndarray,
    device: str,
    pad_s: float = 60.0,
) -> dict:
    """Build a single session's timeline, cropped to the labelled span.

    All arrays passed in are *session-local* (rebased to that session's
    session_start), so the window timestamps and the 1 Hz HR grid share
    one consistent time axis -- no cross-session shifting.
    """
    n = len(feats_norm)
    probs = np.full(n, np.nan)
    preds = np.full(n, np.nan)

    if quality.any():
        model.eval()
        with torch.no_grad():
            x = torch.tensor(feats_norm[quality], dtype=torch.float32, device=device)
            logits = model(x).cpu()
            probs[quality] = F.softmax(logits, dim=1)[:, 1].numpy()
            preds[quality] = logits.argmax(dim=1).numpy()

    # Crop to the labelled protocol span (+ pad). Falls back to full range
    # if, for some reason, there are no labelled windows.
    lab = np.where(labels >= 0)[0]
    if len(lab):
        t0 = float(ts[lab].min()) - pad_s
        t1 = float(ts[lab].max()) + WINDOW_SEC + pad_s
    else:
        t0, t1 = float(ts.min()), float(ts.max())

    win_keep = (ts >= t0) & (ts <= t1)
    hr_keep = (hr_t >= t0) & (hr_t <= t1)

    return {
        "window_times": ts[win_keep].tolist(),
        "stress_probs": probs[win_keep].tolist(),
        "preds": preds[win_keep].tolist(),
        "true_labels": labels[win_keep].tolist(),
        "phase_names": np.asarray(phases)[win_keep].tolist(),
        "hr_times": hr_t[hr_keep].tolist(),
        "hr_values": hr_v[hr_keep].tolist(),
        "t0_s": t0,
        "t1_s": t1,
    }


def build_pilot_timeline(
    model,
    preproc: dict,
    features_norm: np.ndarray,
    device: str,
    pad_s: float = 60.0,
) -> dict:
    """Build the prediction timeline, per session, cropped to the protocol.

    Replaces the timeline block in evaluate_fitbit_pilot(). Returns either
    {"cal": {...}, "eval": {...}} (two-session pilot) or {"single": {...}}.
    """
    labels = preproc["labels"]
    phases = preproc["phase_names"]
    quality = preproc["quality"]
    ts = preproc["timestamps"]

    # Two-session pilot: hr_uniform / t_uniform are dicts keyed by session,
    # and cal_mask / eval_mask select each session's windows.
    if isinstance(preproc["hr_uniform"], dict):
        out = {}
        for key, mask in [("cal", preproc["cal_mask"]), ("eval", preproc["eval_mask"])]:
            out[key] = _session_timeline(
                model,
                features_norm[mask],
                ts[mask],
                labels[mask],
                phases[mask],
                quality[mask],
                np.asarray(preproc["t_uniform"][key]),
                np.asarray(preproc["hr_uniform"][key]),
                device,
                pad_s,
            )
        return out

    # Single continuous session.
    return {
        "single": _session_timeline(
            model,
            features_norm,
            ts,
            labels,
            phases,
            quality,
            np.asarray(preproc["t_uniform"]),
            np.asarray(preproc["hr_uniform"]),
            device,
            pad_s,
        )
    }


# ---------------------------------------------------------------------------
# 2. PLOT
# ---------------------------------------------------------------------------
def plot_prediction_timeline(results: dict, output_path: str | None = None) -> None:
    """Plot the prediction timeline as one column per session, cropped.

    Each column shows three stacked panels: HR (BPM), P(stress), and the
    ground-truth protocol band. Time axes are session-local minutes, so the
    irrelevant ~3 h between sessions is broken out rather than drawn.

    Backward compatible: also accepts the old flat timeline dict, which it
    crops to the labelled span before plotting.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib required for timeline plot")
        return

    tl = results["timeline"]

    # Normalise into an ordered list of (column_title, session_dict).
    if "hr_times" in tl:  # old flat format -> crop on the fly
        tl = {"single": _crop_flat(tl)}
    titles = {
        "cal": "Calibration session (morning)",
        "eval": "Evaluation session (afternoon)",
        "single": "Pilot session",
    }
    order = [k for k in ("cal", "eval", "single") if k in tl]
    cols = [(titles[k], tl[k]) for k in order]
    ncol = len(cols)

    fig, axes = plt.subplots(
        3, ncol, figsize=(6.2 * ncol, 7.2), squeeze=False, sharey="row"
    )

    for c, (title, s) in enumerate(cols):
        hr_t = np.asarray(s["hr_times"], dtype=float)
        hr_v = np.asarray(s["hr_values"], dtype=float)
        win_t = np.asarray(s["window_times"], dtype=float)
        probs = np.asarray(s["stress_probs"], dtype=float)
        true = np.asarray(s["true_labels"], dtype=float)

        # Rebase each column to its own local zero (minutes from crop start).
        base = float(s.get("t0_s", win_t.min() if win_t.size else 0.0))
        hr_m = (hr_t - base) / 60.0
        win_m = (win_t - base) / 60.0

        ax_hr, ax_p, ax_lab = axes[0][c], axes[1][c], axes[2][c]

        # Panel 1: heart rate
        ax_hr.plot(hr_m, hr_v, color="#4477AA", linewidth=0.8)
        ax_hr.set_title(title, fontsize=11)
        if c == 0:
            ax_hr.set_ylabel("HR (BPM)")

        # Panel 2: stress probability
        valid = np.isfinite(probs)
        ax_p.plot(win_m[valid], probs[valid], color="#228833", linewidth=1.2)
        ax_p.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax_p.set_ylim(-0.05, 1.05)
        if c == 0:
            ax_p.set_ylabel("P(stress)")

        # Panel 3: ground-truth protocol bands (only labelled windows)
        for i, t in enumerate(win_m):
            if true[i] == 1:
                ax_lab.axvspan(t, t + WINDOW_SEC / 60.0, color="#EE6677", alpha=0.35)
            elif true[i] == 0:
                ax_lab.axvspan(t, t + WINDOW_SEC / 60.0, color="#4477AA", alpha=0.18)
        ax_lab.set_yticks([0, 1])
        ax_lab.set_yticklabels(["Non-stress", "Stress"])
        ax_lab.set_xlabel("Time within session (min)")
        if c == 0:
            ax_lab.set_ylabel("Protocol")

        # Keep each column tight to its cropped span.
        span = [win_m.min(), win_m.max()] if win_m.size else [hr_m.min(), hr_m.max()]
        for ax in (ax_hr, ax_p, ax_lab):
            ax.set_xlim(span[0] - 0.2, span[1] + 0.5)

    fig.suptitle("Fitbit Pilot — Prediction Timeline (N=1)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Timeline saved: {output_path}")
    plt.close(fig)


def _crop_flat(tl: dict, pad_s: float = 60.0) -> dict:
    """Crop an old flat timeline dict to its labelled span (best-effort)."""
    win_t = np.asarray(tl["window_times"], dtype=float)
    true = np.asarray(tl["true_labels"], dtype=float)
    hr_t = np.asarray(tl["hr_times"], dtype=float)
    lab = np.where(true >= 0)[0]
    if len(lab):
        t0 = float(win_t[lab].min()) - pad_s
        t1 = float(win_t[lab].max()) + WINDOW_SEC + pad_s
    else:
        t0, t1 = float(win_t.min()), float(win_t.max())
    wk = (win_t >= t0) & (win_t <= t1)
    hk = (hr_t >= t0) & (hr_t <= t1)
    return {
        "window_times": win_t[wk].tolist(),
        "stress_probs": np.asarray(tl["stress_probs"])[wk].tolist(),
        "true_labels": true[wk].tolist(),
        "hr_times": hr_t[hk].tolist(),
        "hr_values": np.asarray(tl["hr_values"])[hk].tolist(),
        "t0_s": t0,
        "t1_s": t1,
    }

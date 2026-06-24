"""Replay data source.

Streams a pre-recorded Fitbit pilot session through the *real* student
models, emitting the v1 protocol exactly like MockSource — but every
number is genuine: real HRV features, real WESAD z-scoring, real
forward passes through the three trained students.

What it does (all via existing, already-tested code):
  1. preprocess_two_session_pilot(json)        — parse → 1 Hz → window → 6 HRV feats
  2. load_paired_data + fit_scaler             — the WESAD training-set scaler
  3. load no_kd_global_best.pt, kd_global_best.pt
  4. personalise_head() on the MORNING (cal) labelled windows
  5. stream the AFTERNOON (eval) session second-by-second, emitting a
     real `prediction` per 16 s window (8 s slide).

Narrative: "calibrated on this morning's session, now running on the
afternoon session unlabelled" — which is exactly the deployment story.
Labels never cross the wire (the protocol has no label field); they are
used only internally for the one-time head personalisation, identical
to fitbit_evaluator.evaluate_fitbit_pilot(two_session_pilot=True).

IMPORTANT — session slicing:
  The Fitbit API returns the *whole day's* intraday data, so the raw
  JSON spans ~14 h with two dense ~2 s-sampled recording blocks
  (morning ≈ 11:13, afternoon ≈ 13:50) separated by long empty gaps.
  preprocess_two_session_pilot interpolates across the entire day; the
  evaluator ignores the filler via quality/label masks, but we must NOT
  stream it. We slice the eval session to t ∈ [0, eval_end_s] — the real
  ~13.7 min afternoon recording — before streaming.

Heavy work (preprocessing, scaler fit, checkpoint load, personalisation,
per-window inference) all happens ONCE in __init__. stream() then just
replays precomputed messages on protocol cadence, so nothing fragile
runs on the hot path during the viva.

time_scale: 1.0 = real-time (a ~13.7 min demo). >1 compresses for
rehearsal (e.g. 5.0 → ~2.7 min). A viva runs at 1.0.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
import copy
import datetime as dt
import logging
import os
import pathlib
import sys
import time

import numpy as np


def _ensure_repo_on_path() -> str:
    """Put the repo root (the directory containing the ``src`` package) on
    sys.path.

    The backend is usually launched from ``dashboard/`` or
    ``dashboard/backend/``, where the top-level ``src`` package is not
    importable. Walk up from this file until we find a directory that
    contains ``src/__init__.py`` and prepend it to sys.path, so
    ``from src.kd...`` works no matter where uvicorn was started.
    Returns the repo root, used to resolve relative data paths.
    """
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "__init__.py").exists():
            sp = str(parent)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return sp
    # Fallback: stressfl/ is 3 levels above sources/replay.py
    fallback = str(here.parents[3])
    if fallback not in sys.path:
        sys.path.insert(0, fallback)
    return fallback


_REPO_ROOT = _ensure_repo_on_path()


def _resolve_under_repo(path: str | None) -> str | None:
    """Resolve a possibly-relative path against the repo root.

    kd_config defines data dirs (paired npz, checkpoints) as paths relative
    to the repo root. The backend runs from dashboard/, so those relatives
    would resolve against the wrong cwd. Anchor them to the repo root; leave
    absolute paths and existing relative paths untouched.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    candidate = os.path.join(_REPO_ROOT, path)
    if os.path.exists(candidate) or not os.path.exists(path):
        return candidate
    return path


from backend.protocol import (
    SLIDE_SEC,
    WINDOW_SEC,
    CalibrationInfo,
    EmbeddingSummary,
    HelloMessage,
    HRSampleMessage,
    HRSamplePoint,
    ModelInfo,
    ModelPrediction,
    PredictionMessage,
    QualityUpdateMessage,
    ServerMessage,
    SystemEventMessage,
    TrainingNormStats,
)
from backend.sources.base import DataSource

logger = logging.getLogger("dashboard.backend.replay")

QUALITY_UPDATE_PERIOD_S = 30.0
FITBIT_SYNC_PERIOD_S = 60.0

# Model id → (label shown in the dashboard, checkpoint filename / note)
_MODEL_LABELS = {
    "no_kd": ("No KD (b\u2032)", "no_kd_global_best.pt"),
    "global_kd": ("Global KD (c\u2032)", "kd_global_best.pt"),
    "personalised": ("KD + personalised (d\u2032)", "kd_global_best.pt (head fine-tuned)"),
}


class ReplaySource(DataSource):
    """Replay a pre-recorded two-session Fitbit pilot through real models."""

    def __init__(
        self,
        json_path: str,
        checkpoint_dir: str | None = None,
        cal_session_start: str | None = None,
        eval_session_start: str | None = None,
        cal_protocol: list[tuple] | None = None,
        eval_protocol: list[tuple] | None = None,
        finetune_lr: float | None = None,
        finetune_steps: int | None = None,
        eval_end_s: float | None = None,
        time_scale: float = 1.0,
        session_id: str = "replay_session",
        device: str | None = None,
    ):
        # Imports are local so that a mock-only server never needs torch /
        # the src.kd package, and any import error surfaces clearly here
        # rather than at module import time.
        import torch  # noqa: F401

        from src.kd.kd_config import (
            ALL_SUBJECT_IDS,
            DEVICE,
            FINETUNE_LR,
            FINETUNE_STEPS_DEFAULT,
            KD_CHECKPOINT_DIR_CEN,
            KD_PAIRED_DIR,
        )

        self.json_path = json_path
        self.checkpoint_dir = _resolve_under_repo(checkpoint_dir or KD_CHECKPOINT_DIR_CEN)
        self.cal_session_start = cal_session_start
        self.eval_session_start = eval_session_start
        self.cal_protocol = cal_protocol
        self.eval_protocol = eval_protocol
        self.finetune_lr = FINETUNE_LR if finetune_lr is None else finetune_lr
        self.finetune_steps = (
            FINETUNE_STEPS_DEFAULT if finetune_steps is None else finetune_steps
        )
        self.eval_end_s = eval_end_s
        self.time_scale = float(time_scale)
        self.session_id = session_id
        self.device = device or DEVICE
        self._all_subject_ids = ALL_SUBJECT_IDS
        self._paired_dir = _resolve_under_repo(KD_PAIRED_DIR)

        self._session_start = dt.datetime.now(dt.timezone.utc)
        self._real_per_sim = 1.0 / self.time_scale

        # Filled by _prepare()
        self._hello: HelloMessage | None = None
        self._hr_samples: list[tuple[float, int, bool]] = []
        self._predictions: list[tuple[float, PredictionMessage]] = []
        self._n_cal_windows = 0
        self._eval_sampling_report: dict = {}
        self._eval_gap_pct = 0.0

        self._prepare()

    # One-time heavy preparation

    def _prepare(self) -> None:
        import torch

        from src.kd.kd_trainer import (
            fit_scaler,
            load_paired_data,
            personalise_head,
        )
        from src.kd.preprocess_pilot import preprocess_two_session_pilot
        from src.kd.student_model import build_student

        # Resolve a relative JSON path against the repo root, so the default
        # "data/fitbit/..." works regardless of the launch directory.
        if not os.path.isabs(self.json_path):
            candidate = os.path.join(_REPO_ROOT, self.json_path)
            if os.path.exists(candidate) or not os.path.exists(self.json_path):
                self.json_path = candidate

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Replay JSON not found: {self.json_path}")

        # 1. Preprocess both sessions (uses your exact pilot pipeline).
        kwargs: dict = {}
        if self.cal_session_start is not None:
            kwargs["cal_session_start"] = self.cal_session_start
        if self.eval_session_start is not None:
            kwargs["eval_session_start"] = self.eval_session_start
        if self.cal_protocol is not None:
            kwargs["cal_protocol"] = self.cal_protocol
        if self.eval_protocol is not None:
            kwargs["eval_protocol"] = self.eval_protocol

        logger.info("Replay: preprocessing two-session pilot from %s", self.json_path)
        merged = preprocess_two_session_pilot(self.json_path, **kwargs)

        # 2. WESAD scaler — fit on noisy training features (the student was
        #    trained on noisy features), identical to load_wesad_scaler().
        paired = load_paired_data(self._all_subject_ids, self._paired_dir, use_noisy=True)
        self._scaler = fit_scaler(paired["features"])

        # 3. Load the two trained students.
        no_kd_path = os.path.join(self.checkpoint_dir, "no_kd_global_best.pt")
        kd_path = os.path.join(self.checkpoint_dir, "kd_global_best.pt")
        for p in (no_kd_path, kd_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Checkpoint not found: {p}")

        model_no_kd = build_student().to(self.device)
        model_no_kd.load_state_dict(
            torch.load(no_kd_path, map_location=self.device, weights_only=True)
        )
        model_kd = build_student().to(self.device)
        model_kd.load_state_dict(
            torch.load(kd_path, map_location=self.device, weights_only=True)
        )

        # 4. Personalise a copy of the global-KD student on the MORNING
        #    (cal) labelled windows — same lr / steps as the demo.
        labels = merged["labels"]
        quality = merged["quality"]
        cal_sel = merged["cal_mask"] & quality & (labels >= 0)
        self._n_cal_windows = int(cal_sel.sum())
        if self._n_cal_windows == 0:
            raise RuntimeError(
                "No valid labelled calibration windows for personalisation. "
                "Check the cal session_start / protocol."
            )
        cal_feats = self._scaler.transform(merged["features"][cal_sel])
        cal_labels = labels[cal_sel]

        model_personal = copy.deepcopy(model_kd)
        personalise_head(
            model_personal,
            cal_feats,
            cal_labels,
            lr=self.finetune_lr,
            n_steps=self.finetune_steps,
            device=self.device,
        )

        self._models = {
            "no_kd": model_no_kd.eval(),
            "global_kd": model_kd.eval(),
            "personalised": model_personal.eval(),
        }
        self._n_params = sum(p.numel() for p in model_no_kd.parameters())

        # 5. Build the eval (afternoon) stream, sliced to the real session.
        self._build_eval_stream(merged)

        # 6. Hello.
        self._hello = self._make_hello()
        logger.info(
            "Replay ready: %d hr samples (%.1f min), %d windows, "
            "%d cal windows used for personalisation",
            len(self._hr_samples),
            len(self._hr_samples) / 60.0,
            len(self._predictions),
            self._n_cal_windows,
        )

    def _build_eval_stream(self, merged: dict) -> None:
        em = merged["eval_mask"]
        w_ts = merged["timestamps"][em]
        w_feats = merged["features"][em]
        w_quality = merged["quality"][em]

        hu = merged["hr_uniform"]["eval"]
        tu = merged["t_uniform"]["eval"]
        gf = merged["gap_flags"]["eval"]

        # The eval session's uniform grid spans the whole day (negative t is
        # earlier in the day; large positive t may be filler). Keep only the
        # real afternoon recording: t in [0, eval_end_s]. Default upper bound
        # is the last real eval sample.
        end = float(tu.max()) if self.eval_end_s is None else float(self.eval_end_s)
        hr_mask = (tu >= 0.0) & (tu <= end)
        if hr_mask.sum() == 0:
            raise RuntimeError("Eval session slice is empty — check eval_session_start.")

        tu_s = tu[hr_mask]
        hu_s = hu[hr_mask]
        gf_s = gf[hr_mask]
        t0 = float(tu_s[0])  # re-zero so the stream starts at sim_t = 0

        self._hr_samples = [
            (float(t) - t0, int(h), bool(g))
            for t, h, g in zip(tu_s, hu_s, gf_s)
        ]
        self._eval_gap_pct = float(100.0 * gf_s.mean()) if len(gf_s) else 0.0

        # eval sampling report (a list [cal, eval] in two-session mode)
        sr = merged.get("sampling_report")
        self._eval_sampling_report = sr[1] if isinstance(sr, list) and len(sr) > 1 else {}

        # Windows within the streamed span; precompute one prediction each.
        win_mask = (w_ts >= 0.0) & (w_ts + WINDOW_SEC <= end + 1e-6)
        ws_ts = w_ts[win_mask]
        ws_feats = w_feats[win_mask]
        ws_quality = w_quality[win_mask]
        ws_z = self._scaler.transform(ws_feats)

        preds: list[tuple[float, PredictionMessage]] = []
        wid = 0
        for ts, fr, fz, q in zip(ws_ts, ws_feats, ws_z, ws_quality):
            # Skip windows that failed quality or produced non-finite feats:
            # a NaN p_stress would fail protocol validation. (For a dense
            # session like this one, none are expected.)
            if not bool(q) or not np.all(np.isfinite(fz)):
                continue
            start = float(ts) - t0
            end_w = start + WINDOW_SEC
            preds.append((end_w, self._predict_window(wid, start, end_w, fr, fz)))
            wid += 1

        preds.sort(key=lambda x: x[0])
        self._predictions = preds

    def _predict_window(
        self,
        window_id: int,
        window_start: float,
        window_end: float,
        feats_raw: np.ndarray,
        feats_z: np.ndarray,
    ) -> PredictionMessage:
        import torch
        import torch.nn.functional as F

        x = torch.tensor(feats_z, dtype=torch.float32, device=self.device).unsqueeze(0)

        models: dict[str, ModelPrediction] = {}
        total_latency_ms = 0.0
        for mid, model in self._models.items():
            t_start = time.perf_counter()
            with torch.no_grad():
                logits, emb = model(x, return_features=True)
            total_latency_ms += (time.perf_counter() - t_start) * 1000.0

            probs = F.softmax(logits, dim=1)[0]
            p_stress = float(np.clip(probs[1].item(), 0.0, 1.0))
            logit_pair = [float(logits[0, 0].item()), float(logits[0, 1].item())]

            e = emb[0].detach().cpu().numpy()
            models[mid] = ModelPrediction(
                p_stress=p_stress,
                logits=logit_pair,
                embedding_summary=EmbeddingSummary(
                    mean=float(e.mean()),
                    std=float(e.std()),
                    min=float(e.min()),
                    max=float(e.max()),
                ),
            )

        return PredictionMessage(
            window_id=window_id,
            window_start_s=window_start,
            window_end_s=window_end,
            quality_pass=True,
            features_raw=[float(v) for v in feats_raw],
            features_zscore=[float(v) for v in feats_z],
            models=models,
            inference_latency_ms=total_latency_ms / max(len(self._models), 1),
        )

    def _make_hello(self) -> HelloMessage:
        cal_completed = self._session_start - dt.timedelta(hours=2, minutes=37)
        models = [
            ModelInfo(
                id=mid,
                label=_MODEL_LABELS[mid][0],
                n_params=self._n_params,
                checkpoint=_MODEL_LABELS[mid][1],
            )
            for mid in ("no_kd", "global_kd", "personalised")
        ]
        return HelloMessage(
            session_id=self.session_id,
            session_start_iso=self._session_start.isoformat().replace("+00:00", "Z"),
            data_source="replay",
            models=models,
            calibration=CalibrationInfo(
                n_windows_used=self._n_cal_windows,
                duration_min=8.0,
                fine_tune_steps=int(self.finetune_steps),
                learning_rate=float(self.finetune_lr),
                completed_at_iso=cal_completed.isoformat().replace("+00:00", "Z"),
            ),
            training_norm_stats=TrainingNormStats(
                mean=[float(v) for v in self._scaler.mean_],
                std=[float(v) for v in self._scaler.scale_],
            ),
        )

    # Stream

    async def stream(self) -> AsyncIterator[ServerMessage]:
        yield self._hello

        yield SystemEventMessage(
            event="calibration_loaded",
            severity="info",
            message=(
                f"Replaying afternoon evaluation session. Head personalised on "
                f"{self._n_cal_windows} morning calibration windows."
            ),
            payload={"n_cal_windows": self._n_cal_windows},
        )

        recent_windows: deque[bool] = deque(maxlen=300 // int(SLIDE_SEC) + 4)
        next_quality_t = QUALITY_UPDATE_PERIOD_S
        next_sync_t = FITBIT_SYNC_PERIOD_S
        pred_idx = 0
        n_pred = len(self._predictions)

        for sample_t, hr, in_gap in self._hr_samples:
            sim_t = sample_t

            # 1. one hr_sample per simulated second
            yield HRSampleMessage(
                samples=[HRSamplePoint(t=sim_t, hr=hr, in_gap=in_gap)],
            )

            # 2. predictions whose window has now closed
            while pred_idx < n_pred and self._predictions[pred_idx][0] <= sim_t + 1e-6:
                pred = self._predictions[pred_idx][1]
                recent_windows.append(pred.quality_pass)
                yield pred
                pred_idx += 1

            # 3. quality_update
            if sim_t + 1e-6 >= next_quality_t:
                yield self._make_quality_update(recent_windows)
                next_quality_t += QUALITY_UPDATE_PERIOD_S

            # 4. fitbit_sync system_event (cosmetic, keeps the event log alive)
            if sim_t + 1e-6 >= next_sync_t:
                yield SystemEventMessage(
                    event="fitbit_sync",
                    severity="info",
                    message="Fitbit synced (replay).",
                    payload={"since_last_sync_s": float(FITBIT_SYNC_PERIOD_S)},
                )
                next_sync_t += FITBIT_SYNC_PERIOD_S

            await asyncio.sleep(self._real_per_sim)

        # Drain any remaining predictions (last partial window edge cases).
        while pred_idx < n_pred:
            yield self._predictions[pred_idx][1]
            pred_idx += 1

        yield SystemEventMessage(
            event="session_end",
            severity="info",
            message="Replay session complete.",
            payload={},
        )

    def _make_quality_update(self, recent_windows: deque[bool]) -> QualityUpdateMessage:
        n_total = len(recent_windows)
        n_unreliable = sum(1 for ok in recent_windows if not ok)
        pct_unreliable = 100.0 * n_unreliable / n_total if n_total else 0.0

        sr = self._eval_sampling_report
        median_interval = float(sr.get("median_interval_s", 1.0))
        pct_at_1hz = float(sr.get("pct_1sec", 100.0))
        max_gap = float(sr.get("max_gap_s", 0.0))

        return QualityUpdateMessage(
            window_min=5,
            pct_flagged_samples=self._eval_gap_pct,
            pct_unreliable_windows=pct_unreliable,
            max_gap_s=max_gap,
            median_interval_s=median_interval,
            pct_at_1hz=pct_at_1hz,
            n_windows_total=n_total,
            n_windows_unreliable=n_unreliable,
        )

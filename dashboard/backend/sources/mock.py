"""Mock data source.

Generates a synthetic stream that satisfies the v1 protocol exactly.

Behaviour:
  * Synthetic 1 Hz HR signal: baseline 70 BPM with smooth perturbations,
    occasional 5–10 minute "stress" excursions to ~95 BPM.
  * hr_sample every 1 s with 1 sample.
  * prediction every 8 s — three models with jittered p_stress so
    Panel 2's timelines actually disagree.
  * quality_update every 30 s with mostly-good numbers, occasionally
    degraded.
  * system_event sprinkled in (fitbit_sync ~every 60 s).

Time compression toggle:
  By default, mock time runs at wall-clock speed (1 s real = 1 s sim).
  Set time_scale > 1 to compress — e.g. time_scale=10 means 10 s of
  simulated session per 1 s of wall-clock. Useful for development
  iteration. A viva-style demo uses time_scale=1.

Determinism:
  All randomness goes through one np.random.Generator seeded from the
  constructor. Two MockSources with the same seed produce identical
  streams. The frontend can rely on this for reproducible screenshots.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
import datetime as dt

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
import numpy as np

# Tunable mock parameters

BASELINE_HR = 70.0  # BPM resting
STRESS_HR = 95.0  # BPM peak during stress episode
HR_NOISE_SD = 1.5  # second-to-second BPM noise

# Episode schedule (seconds-from-session-start, label).
# Roughly: 3 min baseline, 5 min stress, 7 min baseline, 6 min stress,
# then alternate forever. Loops modulo the cycle length.
EPISODE_SCHEDULE: list[tuple[float, float, str]] = [
    (0.0, 180.0, "baseline"),
    (180.0, 480.0, "stressor"),
    (480.0, 900.0, "baseline"),
    (900.0, 1260.0, "stressor"),
    (1260.0, 1500.0, "baseline"),
]
CYCLE_LENGTH_S = EPISODE_SCHEDULE[-1][1]

HR_SAMPLE_PERIOD_S = 1.0
QUALITY_UPDATE_PERIOD_S = 30.0
FITBIT_SYNC_PERIOD_S = 60.0

P_QUALITY_FAIL = 0.04  # 4% of windows fail quality
P_GAP_SAMPLE = 0.02  # 2% of HR samples flagged in_gap

# Model parameter count (matches StudentBase + StudentHead)
# Linear(6,64)=448 + BN(64)=128 + Linear(64,32)=2080 + BN(32)=64
# + Linear(32,16)=528 + Linear(16,2)=34 = 3282. Round to plan estimate.
N_PARAMS = 3282


class MockSource(DataSource):
    """Synthetic stream satisfying the v1 protocol."""

    def __init__(
        self,
        session_id: str = "mock_session",
        seed: int = 42,
        time_scale: float = 1.0,
    ):
        self.session_id = session_id
        self.time_scale = float(time_scale)
        self.rng = np.random.default_rng(seed)
        self._session_start = dt.datetime.now(dt.timezone.utc)

        self._real_per_sim = 1.0 / self.time_scale

        # Rolling buffer of the last 5 min of windows for quality_update.
        # Each entry: (window_id, quality_pass).
        self._recent_windows: deque[tuple[int, bool]] = deque(maxlen=300 // 8 + 4)

    # Public stream method

    async def stream(self) -> AsyncIterator[ServerMessage]:
        """Yield messages forever in protocol-cadence order."""
        yield self._make_hello()

        sim_t = 0.0
        next_prediction_t = WINDOW_SEC  # first window ends at t=16
        next_quality_t = QUALITY_UPDATE_PERIOD_S
        next_sync_t = FITBIT_SYNC_PERIOD_S
        window_id = 0

        # The main loop: advance sim time in 1 s steps. At each step,
        # emit hr_sample. When sim time crosses a scheduled boundary,
        # emit the corresponding message. We schedule events by sim time
        # rather than wall time so time_scale works uniformly.
        while True:
            # 1. one sample per simulated second
            yield self._make_hr_sample(sim_t)

            # 2. prediction — when we cross the next window boundary
            while sim_t + 1e-6 >= next_prediction_t:
                window_end = next_prediction_t
                window_start = window_end - WINDOW_SEC
                yield self._make_prediction(window_id, window_start, window_end)
                window_id += 1
                next_prediction_t += SLIDE_SEC

            # 3. quality_update
            if sim_t + 1e-6 >= next_quality_t:
                yield self._make_quality_update()
                next_quality_t += QUALITY_UPDATE_PERIOD_S

            # 4. fitbit_sync system_event
            if sim_t + 1e-6 >= next_sync_t:
                yield self._make_fitbit_sync()
                next_sync_t += FITBIT_SYNC_PERIOD_S

            # Sleep one simulated second of wall-clock time.
            await asyncio.sleep(self._real_per_sim * HR_SAMPLE_PERIOD_S)
            sim_t += HR_SAMPLE_PERIOD_S

    # Message factories

    def _make_hello(self) -> HelloMessage:
        # Calibration "completed" 30 min before session start, 15 min long.
        cal_completed = self._session_start - dt.timedelta(minutes=30)
        return HelloMessage(
            session_id=self.session_id,
            session_start_iso=self._session_start.isoformat().replace("+00:00", "Z"),
            data_source="mock",
            models=[
                ModelInfo(
                    id="no_kd",
                    label="No KD (b′)",
                    n_params=N_PARAMS,
                    checkpoint="student_no_kd.pt",
                ),
                ModelInfo(
                    id="global_kd",
                    label="Global KD (c′)",
                    n_params=N_PARAMS,
                    checkpoint="student_global_kd.pt",
                ),
                ModelInfo(
                    id="personalised",
                    label="KD + personalised (d′)",
                    n_params=N_PARAMS,
                    checkpoint="student_personalised.pt",
                ),
            ],
            calibration=CalibrationInfo(
                n_windows_used=108,
                duration_min=15.0,
                fine_tune_steps=10,
                learning_rate=1e-4,
                completed_at_iso=cal_completed.isoformat().replace("+00:00", "Z"),
            ),
            training_norm_stats=TrainingNormStats(
                # Plausible values from WESAD training-set stats.
                mean=[842.1, 53.2, 31.4, 12.7, 72.5, 8.9],
                std=[98.3, 17.8, 14.2, 9.5, 9.1, 3.2],
            ),
        )

    def _make_hr_sample(self, sim_t: float) -> HRSampleMessage:
        hr = self._hr_at(sim_t)
        in_gap = bool(self.rng.random() < P_GAP_SAMPLE)
        return HRSampleMessage(
            samples=[HRSamplePoint(t=sim_t, hr=hr, in_gap=in_gap)],
        )

    def _make_prediction(
        self,
        window_id: int,
        window_start: float,
        window_end: float,
    ) -> PredictionMessage:
        # Window centre determines stress probability.
        centre = (window_start + window_end) / 2.0
        episode_label = self._episode_at(centre)
        is_stress = episode_label == "stressor"

        quality_pass = bool(self.rng.random() >= P_QUALITY_FAIL)
        self._recent_windows.append((window_id, quality_pass))

        # Synthesise features. We don't actually compute these from HR
        # because the goal is to give the frontend realistic-looking
        # numbers, not validate the feature extractor.
        features_raw = self._fake_features(is_stress)
        features_zscore = self._zscore(features_raw)

        # Three models. Personalised is closest to the true label,
        # global_kd middle, no_kd noisiest. This is what the actual
        # ablation should show, so the demo previews the result.
        true_p = 0.85 if is_stress else 0.10
        models: dict[str, ModelPrediction] = {}

        # personalised: tight around true_p (sigma 0.05)
        p_pers = self._clip01(true_p + self.rng.normal(0, 0.05))
        # global_kd: medium spread (sigma 0.10)
        p_glob = self._clip01(true_p + self.rng.normal(0, 0.10))
        # no_kd: wider (sigma 0.18)
        p_nokd = self._clip01(true_p + self.rng.normal(0, 0.18))

        # Shared base embedding for global_kd and personalised
        shared_emb = self._fake_embedding_summary()
        nokd_emb = self._fake_embedding_summary()

        models["no_kd"] = ModelPrediction(
            p_stress=p_nokd,
            logits=self._logits_from_p(p_nokd),
            embedding_summary=nokd_emb,
        )
        models["global_kd"] = ModelPrediction(
            p_stress=p_glob,
            logits=self._logits_from_p(p_glob),
            embedding_summary=shared_emb,
        )
        models["personalised"] = ModelPrediction(
            p_stress=p_pers,
            logits=self._logits_from_p(p_pers),
            embedding_summary=shared_emb,
        )

        # Plausible CPU latency in ms
        latency = float(self.rng.uniform(0.4, 1.2))

        return PredictionMessage(
            window_id=window_id,
            window_start_s=window_start,
            window_end_s=window_end,
            quality_pass=quality_pass,
            features_raw=features_raw.tolist(),
            features_zscore=features_zscore.tolist(),
            models=models,
            inference_latency_ms=latency,
        )

    def _make_quality_update(self) -> QualityUpdateMessage:
        # Compute over the rolling 5-min buffer.
        if self._recent_windows:
            n_total = len(self._recent_windows)
            n_unreliable = sum(1 for _, ok in self._recent_windows if not ok)
            pct_unreliable = 100.0 * n_unreliable / n_total
        else:
            n_total = 0
            n_unreliable = 0
            pct_unreliable = 0.0

        # Realistic numbers: median interval should be ~1.0 since we
        # interpolate to 1 Hz; pct_at_1hz close to 100; max_gap small
        # except occasionally.
        max_gap = float(self.rng.uniform(2.0, 6.0))
        if self.rng.random() < 0.1:
            max_gap = float(self.rng.uniform(8.0, 15.0))  # rare bad patch
        pct_flagged = float(self.rng.uniform(1.0, 5.0))
        pct_at_1hz = float(self.rng.uniform(94.0, 99.0))

        return QualityUpdateMessage(
            window_min=5,
            pct_flagged_samples=pct_flagged,
            pct_unreliable_windows=pct_unreliable,
            max_gap_s=max_gap,
            median_interval_s=1.0,
            pct_at_1hz=pct_at_1hz,
            n_windows_total=n_total,
            n_windows_unreliable=n_unreliable,
        )

    def _make_fitbit_sync(self) -> SystemEventMessage:
        # Mock Fitbit sync — realistic-looking payload even though
        # data_source is "mock". Lets the frontend exercise the badge
        # pulse logic.
        n_new = int(self.rng.integers(40, 80))
        return SystemEventMessage(
            event="fitbit_sync",
            severity="info",
            message=f"Fitbit synced. {n_new} new HR samples ingested.",
            payload={
                "n_new_samples": n_new,
                "since_last_sync_s": float(FITBIT_SYNC_PERIOD_S),
            },
        )

    def _hr_at(self, sim_t: float) -> int:
        """Synthetic 1 Hz HR sample at simulated time t.

        Smooth ramp between baseline and stress within each episode,
        plus low-amplitude noise. Returns integer BPM clipped to [40, 200].
        """
        # Find the current episode and compute fractional progress through it.
        for start, end, lbl in EPISODE_SCHEDULE:
            cycle_t = sim_t % CYCLE_LENGTH_S
            if start <= cycle_t < end:
                # Smooth in/out using a half-cosine ramp over 30 s at edges.
                ramp = self._ramp(cycle_t - start, end - start, ramp_s=30.0)
                base = BASELINE_HR
                target = STRESS_HR if lbl == "stressor" else BASELINE_HR
                hr = base + (target - base) * ramp
                hr += self.rng.normal(0, HR_NOISE_SD)
                return int(np.clip(round(hr), 40, 200))
        return int(BASELINE_HR)

    def _episode_at(self, sim_t: float) -> str:
        cycle_t = sim_t % CYCLE_LENGTH_S
        for start, end, lbl in EPISODE_SCHEDULE:
            if start <= cycle_t < end:
                return lbl
        return "baseline"

    @staticmethod
    def _ramp(t_in_episode: float, episode_len: float, ramp_s: float) -> float:
        """Half-cosine ramp: 0 → 1 over ramp_s, hold, then 1 → 0 over ramp_s."""
        if episode_len <= 2 * ramp_s:
            return 1.0
        if t_in_episode < ramp_s:
            return float(0.5 * (1 - np.cos(np.pi * t_in_episode / ramp_s)))
        if t_in_episode > episode_len - ramp_s:
            tail = episode_len - t_in_episode
            return float(0.5 * (1 - np.cos(np.pi * tail / ramp_s)))
        return 1.0

    def _fake_features(self, is_stress: bool) -> np.ndarray:
        """Plausible 6-feature vector. Stress → lower HRV, higher HR."""
        # Baseline: mean_rr=850, sdnn=55, rmssd=32, pnn50=15, mean_hr=70, std_hr=8
        # Stress:   mean_rr=620, sdnn=28, rmssd=14, pnn50=4,  mean_hr=95, std_hr=10
        if is_stress:
            mu = np.array([620.0, 28.0, 14.0, 4.0, 95.0, 10.0])
        else:
            mu = np.array([850.0, 55.0, 32.0, 15.0, 70.0, 8.0])
        sigma = np.array([25.0, 6.0, 4.0, 2.5, 3.0, 1.2])
        return mu + self.rng.normal(0, sigma)

    @staticmethod
    def _zscore(features_raw: np.ndarray) -> np.ndarray:
        # Use the same training_norm_stats values shipped in hello.
        train_mean = np.array([842.1, 53.2, 31.4, 12.7, 72.5, 8.9])
        train_std = np.array([98.3, 17.8, 14.2, 9.5, 9.1, 3.2])
        zscored: np.ndarray = (features_raw - train_mean) / train_std
        return zscored

    def _fake_embedding_summary(self) -> EmbeddingSummary:
        return EmbeddingSummary(
            mean=float(self.rng.normal(0, 0.05)),
            std=float(self.rng.uniform(0.85, 0.95)),
            min=float(self.rng.uniform(-2.2, -1.4)),
            max=float(self.rng.uniform(1.4, 2.2)),
        )

    @staticmethod
    def _logits_from_p(p: float) -> list[float]:
        """Inverse-softmax: logits whose softmax gives [1-p, p].

        Use logit form l = [log(1-p), log(p)] (up to additive constant).
        Frontend doesn't care about the constant, only the ratio.
        """
        eps = 1e-6
        p = max(eps, min(1 - eps, p))
        return [float(np.log(1 - p)), float(np.log(p))]

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

"""WebSocket protocol. Pydantic models for every message type.

Every outgoing message is one of these models, serialised via
`.model_dump_json()`. The discriminator is the literal `type` field on
each model, which lets the frontend dispatch on a single string.

"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION: Literal[1] = 1

WINDOW_SEC = 16.0
SLIDE_SEC = 8.0
SAMPLING_RATE_HZ = 1.0

FEATURE_NAMES: list[str] = [
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "std_hr",
]
N_FEATURES = len(FEATURE_NAMES)

MODEL_IDS: list[str] = ["no_kd", "global_kd", "personalised"]


# Shared building blocks


class _StrictBase(BaseModel):
    """Reject unknown fields. Catches typos in dev rather than at the viva."""

    model_config = ConfigDict(extra="forbid")


class _MessageBase(_StrictBase):
    """Every server→client message starts with these two fields."""

    protocol_version: Literal[1] = PROTOCOL_VERSION


# hello (sent once on connect)


class ModelInfo(_StrictBase):
    id: str
    label: str
    n_params: int
    checkpoint: str


class CalibrationInfo(_StrictBase):
    n_windows_used: int
    duration_min: float
    fine_tune_steps: int
    learning_rate: float
    completed_at_iso: str


class TrainingNormStats(_StrictBase):
    mean: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)
    std: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


class HelloMessage(_MessageBase):
    type: Literal["hello"] = "hello"
    session_id: str
    session_start_iso: str
    data_source: Literal["fitbit_live", "replay", "mock"]
    window_sec: float = WINDOW_SEC
    slide_sec: float = SLIDE_SEC
    sampling_rate_hz: float = SAMPLING_RATE_HZ
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    models: list[ModelInfo]
    calibration: CalibrationInfo
    training_norm_stats: TrainingNormStats


# hr_sample (batched 1 Hz HR samples)


class HRSamplePoint(_StrictBase):
    t: float  # seconds from session start
    hr: int  # BPM, clipped [40, 200]
    in_gap: bool


class HRSampleMessage(_MessageBase):
    type: Literal["hr_sample"] = "hr_sample"
    samples: list[HRSamplePoint]


# prediction (every 8 s, one per new window)


class EmbeddingSummary(_StrictBase):
    mean: float
    std: float
    min: float
    max: float


class ModelPrediction(_StrictBase):
    p_stress: float = Field(ge=0.0, le=1.0)
    logits: list[float] = Field(min_length=2, max_length=2)
    embedding_summary: EmbeddingSummary


class PredictionMessage(_MessageBase):
    type: Literal["prediction"] = "prediction"
    window_id: int
    window_start_s: float
    window_end_s: float
    quality_pass: bool
    features_raw: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)
    features_zscore: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)
    models: dict[str, ModelPrediction]
    inference_latency_ms: float


# quality_update (every 30 s)


class QualityUpdateMessage(_MessageBase):
    type: Literal["quality_update"] = "quality_update"
    window_min: int
    pct_flagged_samples: float
    pct_unreliable_windows: float
    max_gap_s: float
    median_interval_s: float
    pct_at_1hz: float
    n_windows_total: int
    n_windows_unreliable: int


# system_event (state changes)

SystemEventName = Literal[
    "calibration_loaded",
    "fitbit_sync",
    "fitbit_stale",
    "fitbit_auth_expired",
    "mode_switch",
    "inference_error",
    "session_end",
]

Severity = Literal["info", "warn", "error"]


class SystemEventMessage(_MessageBase):
    type: Literal["system_event"] = "system_event"
    event: str
    severity: Severity
    message: str
    payload: dict = Field(default_factory=dict)


ServerMessage = (
    HelloMessage
    | HRSampleMessage
    | PredictionMessage
    | QualityUpdateMessage
    | SystemEventMessage
)

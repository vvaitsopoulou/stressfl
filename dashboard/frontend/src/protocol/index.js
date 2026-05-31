/**
 * WebSocket protocol — frontend types and constants.
 */

// Constants  must match backend/protocol.py

export const PROTOCOL_VERSION = 1;

export const WINDOW_SEC = 16.0;
export const SLIDE_SEC = 8.0;
export const SAMPLING_RATE_HZ = 1.0;

export const FEATURE_NAMES = [
  'mean_rr',
  'sdnn',
  'rmssd',
  'pnn50',
  'mean_hr',
  'std_hr',
];

export const MODEL_IDS = ['no_kd', 'global_kd', 'personalised'];

export const HEADLINE_MODEL_ID = 'personalised';

// criminator values for type-narrowing message handlers

export const MSG = Object.freeze({
  HELLO: 'hello',
  HR_SAMPLE: 'hr_sample',
  PREDICTION: 'prediction',
  QUALITY_UPDATE: 'quality_update',
  SYSTEM_EVENT: 'system_event',
});

//JSDoc typedefs

/**
 * @typedef {Object} ModelInfo
 * @property {string} id
 * @property {string} label
 * @property {number} n_params
 * @property {string} checkpoint
 */

/**
 * @typedef {Object} CalibrationInfo
 * @property {number} n_windows_used
 * @property {number} duration_min
 * @property {number} fine_tune_steps
 * @property {number} learning_rate
 * @property {string} completed_at_iso
 */

/**
 * @typedef {Object} TrainingNormStats
 * @property {number[]} mean
 * @property {number[]} std
 */

/**
 * @typedef {Object} HelloMessage
 * @property {1} protocol_version
 * @property {'hello'} type
 * @property {string} session_id
 * @property {string} session_start_iso
 * @property {'fitbit_live'|'replay'|'mock'} data_source
 * @property {number} window_sec
 * @property {number} slide_sec
 * @property {number} sampling_rate_hz
 * @property {string[]} feature_names
 * @property {ModelInfo[]} models
 * @property {CalibrationInfo} calibration
 * @property {TrainingNormStats} training_norm_stats
 */

/**
 * @typedef {Object} HRSamplePoint
 * @property {number} t
 * @property {number} hr
 * @property {boolean} in_gap
 */

/**
 * @typedef {Object} HRSampleMessage
 * @property {1} protocol_version
 * @property {'hr_sample'} type
 * @property {HRSamplePoint[]} samples
 */

/**
 * @typedef {Object} EmbeddingSummary
 * @property {number} mean
 * @property {number} std
 * @property {number} min
 * @property {number} max
 */

/**
 * @typedef {Object} ModelPrediction
 * @property {number} p_stress
 * @property {[number, number]} logits
 * @property {EmbeddingSummary} embedding_summary
 */

/**
 * @typedef {Object} PredictionMessage
 * @property {1} protocol_version
 * @property {'prediction'} type
 * @property {number} window_id
 * @property {number} window_start_s
 * @property {number} window_end_s
 * @property {boolean} quality_pass
 * @property {number[]} features_raw
 * @property {number[]} features_zscore
 * @property {Record<string, ModelPrediction>} models
 * @property {number} inference_latency_ms
 */

/**
 * @typedef {Object} QualityUpdateMessage
 * @property {1} protocol_version
 * @property {'quality_update'} type
 * @property {number} window_min
 * @property {number} pct_flagged_samples
 * @property {number} pct_unreliable_windows
 * @property {number} max_gap_s
 * @property {number} median_interval_s
 * @property {number} pct_at_1hz
 * @property {number} n_windows_total
 * @property {number} n_windows_unreliable
 */

/**
 * @typedef {Object} SystemEventMessage
 * @property {1} protocol_version
 * @property {'system_event'} type
 * @property {string} event
 * @property {'info'|'warn'|'error'} severity
 * @property {string} message
 * @property {Object} payload
 */

/**
 * @typedef {HelloMessage|HRSampleMessage|PredictionMessage|QualityUpdateMessage|SystemEventMessage} ServerMessage
 */

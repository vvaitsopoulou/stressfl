/**
 * Selectors turn the raw `predictionBuffer`
 * (an array of full PredictionMessage objects) into the shape that
 * Recharts and metric cards actually want.
 */

import { MODEL_IDS } from 'protocol';

export function selectPerModelSeries(predictionBuffer) {
  const out = Object.fromEntries(MODEL_IDS.map((id) => [id, []]));
  for (const p of predictionBuffer) {
    for (const id of MODEL_IDS) {
      const m = p.models?.[id];
      if (!m) continue;
      out[id].push({
        t: p.window_end_s,
        p: m.p_stress,
        win: p.window_id,
        quality: p.quality_pass,
      });
    }
  }
  // Defensive: render strictly left-to-right by time. Any out-of-order
  // point (a late message, or a residual from a previous session) would
  // otherwise make the line jump backward and smear across the chart.
  for (const id of MODEL_IDS) out[id].sort((a, b) => a.t - b.t);
  return out;
}

/**
 * Running summary stats for one model over the last N windows.
 *
 * Returns:
 *   {
 *     mean: number | null,           // average P(stress)
 *     std: number | null,            // std of P(stress) — high std = unstable
 *     stableConfidence: number|null, // 1 - std, clamped to [0, 1]; for "Safety 9.3" slot
 *     n: number,                     // windows actually used
 *   }
 */
export function selectRunningStats(predictionBuffer, modelId, windowSize = 30) {
  const slice = predictionBuffer.slice(-windowSize);
  const ps = [];
  for (const p of slice) {
    const v = p.models?.[modelId]?.p_stress;
    if (typeof v === 'number') ps.push(v);
  }
  if (ps.length === 0) {
    return { mean: null, std: null, stableConfidence: null, n: 0 };
  }
  const mean = ps.reduce((a, b) => a + b, 0) / ps.length;
  const variance =
    ps.length > 1
      ? ps.reduce((a, b) => a + (b - mean) ** 2, 0) / (ps.length - 1)
      : 0;
  const std = Math.sqrt(variance);
  const stableConfidence = Math.max(0, 1 - std * 2);
  return { mean, std, stableConfidence, n: ps.length };
}

/**
 * Pairwise agreement: fraction of recent windows where two models'
 * predictions agree on the binary label (P>=0.5 vs P<0.5).
 */
export function selectAgreement(predictionBuffer, modelA, modelB, windowSize = 30) {
  const slice = predictionBuffer.slice(-windowSize);
  let agree = 0;
  let total = 0;
  for (const p of slice) {
    const a = p.models?.[modelA]?.p_stress;
    const b = p.models?.[modelB]?.p_stress;
    if (typeof a !== 'number' || typeof b !== 'number') continue;
    total += 1;
    if ((a >= 0.5) === (b >= 0.5)) agree += 1;
  }
  return total === 0 ? null : agree / total;
}

/**
 * Per-window absolute difference between two models' P(stress).
 *
 * Used by the disagreement chart to show "where models diverged most"
 * as a continuous time series rather than a thresholded binary.
 */
export function selectMeanAbsDiff(predictionBuffer, modelA, modelB) {
  const out = [];
  for (const p of predictionBuffer) {
    const a = p.models?.[modelA]?.p_stress;
    const b = p.models?.[modelB]?.p_stress;
    if (typeof a !== 'number' || typeof b !== 'number') continue;
    out.push({
      t: p.window_end_s,
      win: p.window_id,
      diff: Math.abs(a - b),
    });
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

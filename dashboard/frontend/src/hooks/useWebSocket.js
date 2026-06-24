/**
 * useWebSocket — single-connection WebSocket hook for the dashboard.
 *
 * Responsibilities:
 *   1. Manage one WebSocket connection's lifecycle (connect, reconnect
 *      with backoff, clean up on unmount).
 *   2. Dispatch incoming messages to per-type slices of state.
 *   3. Maintain ring buffers for high-volume streams (hr_sample,
 *      prediction) so the UI can render the last N rather than
 *      everything since session start.
 *
 * Reconnect strategy:
 *   Exponential backoff capped at 30 s. We do NOT replay the last hello on
 *   reconnect (the server sends a fresh one). We clear buffers on each new
 *   connection, since a reconnect is a new session boundary.
 *
 * Connection identity:
 *   The current `url` lives in a ref, and reconnects ALWAYS read that ref —
 *   never a captured value. Combined with a per-connection generation id,
 *   this guarantees that when `url` changes (e.g. the source toggle), no
 *   in-flight reconnect timer can resurrect a socket to the OLD url, and
 *   stale sockets can't stack up. (This was the mock-leak bug.)
 */

import { useEffect, useReducer, useRef } from 'react';
import { MSG } from 'protocol';

const HR_BUFFER_SEC = 300;             // 5 minutes of HR
const PREDICTION_BUFFER_WINDOWS = 600; // ~80 minutes at 8 s slide

// Reducer

const initialState = {
  connectionState: 'idle',     // 'idle' | 'connecting' | 'open' | 'closed' | 'error'
  reconnectAttempts: 0,
  hello: null,
  hrBuffer: [],
  predictionBuffer: [],
  latestPrediction: null,
  latestQuality: null,
  events: [],
};

const MAX_EVENTS = 50;

function reducer(state, action) {
  switch (action.type) {
    case 'CONNECTION_STATE':
      return { ...state, connectionState: action.value };

    case 'RECONNECT_INCREMENT':
      return { ...state, reconnectAttempts: state.reconnectAttempts + 1 };

    case 'CLEAR_SESSION':
      return {
        ...initialState,
        connectionState: state.connectionState,
        reconnectAttempts: state.reconnectAttempts,
      };

    case 'MESSAGE': {
      const msg = action.message;
      switch (msg.type) {
        case MSG.HELLO:
          // A hello marks a fresh session boundary. Reset all stream
          // buffers so data from a previous source (e.g. mock) can never
          // bleed into the new session's charts.
          return {
            ...state,
            hello: msg,
            hrBuffer: [],
            predictionBuffer: [],
            latestPrediction: null,
            latestQuality: null,
          };

        case MSG.HR_SAMPLE: {
          const next = state.hrBuffer.concat(msg.samples);
          const trimmed = next.length > HR_BUFFER_SEC
            ? next.slice(next.length - HR_BUFFER_SEC)
            : next;
          return { ...state, hrBuffer: trimmed };
        }

        case MSG.PREDICTION: {
          // Keep exactly one entry per window_id (last wins) and sorted by
          // time. Makes a backward-jumping / smeared line structurally
          // impossible for every consumer (Live, Comparison, gauge).
          const byId = new Map();
          for (const p of state.predictionBuffer) byId.set(p.window_id, p);
          byId.set(msg.window_id, msg);
          let merged = Array.from(byId.values())
            .sort((a, b) => a.window_start_s - b.window_start_s);
          if (merged.length > PREDICTION_BUFFER_WINDOWS) {
            merged = merged.slice(merged.length - PREDICTION_BUFFER_WINDOWS);
          }
          return { ...state, predictionBuffer: merged, latestPrediction: msg };
        }

        case MSG.QUALITY_UPDATE:
          return { ...state, latestQuality: msg };

        case MSG.SYSTEM_EVENT: {
          const next = state.events.concat([msg]);
          const trimmed = next.length > MAX_EVENTS
            ? next.slice(next.length - MAX_EVENTS)
            : next;
          return { ...state, events: trimmed };
        }

        default:
          console.warn('Unknown message type:', msg.type, msg);
          return state;
      }
    }

    default:
      return state;
  }
}

// Hook

/**
 * @param {string} url
 */
export default function useWebSocket(url) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const backoffRef = useRef(0);

  // Always-current url, read by (re)connect. Never close over `url` directly.
  const urlRef = useRef(url);
  urlRef.current = url;

  // Generation id: bumped every time the url changes / the effect re-runs.
  // Any socket or timer from an older generation is ignored and discarded,
  // so a stale reconnect can never open a socket to a previous url.
  const genRef = useRef(0);

  useEffect(() => {
    const myGen = ++genRef.current;

    const isStale = () => myGen !== genRef.current;

    const clearTimer = () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const closeSocket = () => {
      if (wsRef.current) {
        // Detach handlers first so the close doesn't trigger a reconnect.
        const old = wsRef.current;
        old.onopen = old.onmessage = old.onerror = old.onclose = null;
        try { old.close(); } catch { /* ignore */ }
        wsRef.current = null;
      }
    };

    const scheduleReconnect = () => {
      if (isStale()) return;
      const delays = [1000, 2000, 4000, 8000, 16000, 30000];
      const idx = Math.min(backoffRef.current, delays.length - 1);
      const delay = delays[idx];
      backoffRef.current += 1;
      dispatch({ type: 'RECONNECT_INCREMENT' });
      console.info(`Reconnecting in ${delay} ms (attempt ${backoffRef.current})`);
      clearTimer();
      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        if (!isStale()) connect();
      }, delay);
    };

    function connect() {
      if (isStale()) return;
      clearTimer();
      closeSocket();

      dispatch({ type: 'CONNECTION_STATE', value: 'connecting' });
      dispatch({ type: 'CLEAR_SESSION' });

      let ws;
      try {
        // Read the CURRENT url from the ref, never a captured value.
        ws = new WebSocket(urlRef.current);
      } catch (err) {
        console.error('WebSocket constructor failed:', err);
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        if (isStale()) { try { ws.close(); } catch { /* ignore */ } return; }
        dispatch({ type: 'CONNECTION_STATE', value: 'open' });
        backoffRef.current = 0;
      };

      ws.onmessage = (ev) => {
        if (isStale()) return;
        let parsed;
        try {
          parsed = JSON.parse(ev.data);
        } catch (err) {
          console.error('Bad JSON from server:', err, ev.data);
          return;
        }
        dispatch({ type: 'MESSAGE', message: parsed });
      };

      ws.onerror = (ev) => {
        console.warn('WebSocket error:', ev);
      };

      ws.onclose = (ev) => {
        console.info('WebSocket closed:', ev.code, ev.reason || '(no reason)');
        if (wsRef.current === ws) wsRef.current = null;
        if (!isStale()) {
          dispatch({ type: 'CONNECTION_STATE', value: 'closed' });
          scheduleReconnect();
        }
      };
    }

    connect();

    return () => {
      // Invalidate this generation: any pending timer or late event no-ops.
      genRef.current++;
      clearTimer();
      closeSocket();
    };
  }, [url]);

  return state;
}

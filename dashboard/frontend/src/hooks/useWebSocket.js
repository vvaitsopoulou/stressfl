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
 *   Exponential backoff capped at 30 s, per the spec. We do NOT
 *   replay the last hello on reconnect (the server sends a fresh one
 *   automatically), but we do clear all buffers, a reconnect means
 *   a new session boundary, so showing stale data would be misleading.
 */

import { useEffect, useReducer, useRef, useCallback } from 'react';
import { MSG } from 'protocol';

const HR_BUFFER_SEC = 300;             // 5 minutes of HR
const PREDICTION_BUFFER_WINDOWS = 600; // ~80 minutes at 8 s slide

// Reducer

const initialState = {
  // Connection
  connectionState: 'idle',     // 'idle' | 'connecting' | 'open' | 'closed' | 'error'
  reconnectAttempts: 0,

  // From hello
  hello: null,

  // High-volume streams
  hrBuffer: [],
  predictionBuffer: [],

  // Latest of each
  latestPrediction: null,
  latestQuality: null,

  // System events log
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
          return { ...state, hello: msg };

        case MSG.HR_SAMPLE: {
          const next = state.hrBuffer.concat(msg.samples);
          const trimmed = next.length > HR_BUFFER_SEC
            ? next.slice(next.length - HR_BUFFER_SEC)
            : next;
          return { ...state, hrBuffer: trimmed };
        }

        case MSG.PREDICTION: {
          const next = state.predictionBuffer.concat([msg]);
          const trimmed = next.length > PREDICTION_BUFFER_WINDOWS
            ? next.slice(next.length - PREDICTION_BUFFER_WINDOWS)
            : next;
          return {
            ...state,
            predictionBuffer: trimmed,
            latestPrediction: msg,
          };
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
  const wantConnectionRef = useRef(true);
  const backoffRef = useRef(0);

  // Connect function as a ref-stable callback.
  const connect = useCallback(() => {
    if (!wantConnectionRef.current) return;

    // Tear down any prior socket cleanly.
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }

    dispatch({ type: 'CONNECTION_STATE', value: 'connecting' });
    dispatch({ type: 'CLEAR_SESSION' });

    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      console.error('WebSocket constructor failed:', err);
      scheduleReconnect();
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      dispatch({ type: 'CONNECTION_STATE', value: 'open' });
      backoffRef.current = 0;
    };

    ws.onmessage = (ev) => {
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Bad JSON from server:', err, ev.data);
        return;
      }
      dispatch({ type: 'MESSAGE', message: parsed });
    };

    ws.onerror = (ev) => {
      // eslint-disable-next-line no-console
      console.warn('WebSocket error:', ev);
    };

    ws.onclose = (ev) => {
      // eslint-disable-next-line no-console
      console.info('WebSocket closed:', ev.code, ev.reason || '(no reason)');
      wsRef.current = null;
      if (wantConnectionRef.current) {
        dispatch({ type: 'CONNECTION_STATE', value: 'closed' });
        scheduleReconnect();
      }
    };
  }, [url]);

  const scheduleReconnect = useCallback(() => {
    if (!wantConnectionRef.current) return;

    const delays = [1000, 2000, 4000, 8000, 16000, 30000];
    const idx = Math.min(backoffRef.current, delays.length - 1);
    const delay = delays[idx];
    backoffRef.current += 1;

    dispatch({ type: 'RECONNECT_INCREMENT' });
    // eslint-disable-next-line no-console
    console.info(`Reconnecting in ${delay} ms (attempt ${backoffRef.current})`);

    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      connect();
    }, delay);
  }, [connect]);

  // Connect on mount, tear down on unmount.
  useEffect(() => {
    wantConnectionRef.current = true;
    connect();
    return () => {
      wantConnectionRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        try { wsRef.current.close(); } catch { /* ignore */ }
        wsRef.current = null;
      }
    };
  }, [connect]);

  return state;
}

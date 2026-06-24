/**
 * WebSocketProvider — single connection shared across all panels.
 *
 * Holds the current data `source` ('mock' | 'replay') and builds the
 * WebSocket URL with a ?source= query param. Changing `source` changes
 * the URL, which makes useWebSocket tear down the old socket and connect
 * a fresh one (its connect callback depends on url). The backend reads
 * ?source= per connection and constructs the matching DataSource.
 *
 * Exposes { ...wsState, source, setSource } through context, so any
 * panel can flip the source with a button.
 */

import { createContext, useContext, useMemo, useState } from 'react';
import useWebSocket from 'hooks/useWebSocket';

const WebSocketContext = createContext(null);

const WS_BASE = 'ws://127.0.0.1:8000/ws/live';

export function WebSocketProvider({ children, base = WS_BASE, initialSource = 'replay' }) {
  const [source, setSource] = useState(initialSource);

  const url = useMemo(() => `${base}?source=${encodeURIComponent(source)}`, [base, source]);

  const wsState = useWebSocket(url);

  const value = useMemo(
    () => ({ ...wsState, source, setSource }),
    [wsState, source],
  );

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocketState() {
  const ctx = useContext(WebSocketContext);
  if (ctx === null) {
    throw new Error('useWebSocketState must be used inside a WebSocketProvider');
  }
  return ctx;
}

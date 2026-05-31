/**
 * WebSocketProvider — single connection shared across all panels.
 */

import { createContext, useContext } from 'react';
import useWebSocket from 'hooks/useWebSocket';

const WebSocketContext = createContext(null);

const WS_URL = 'ws://127.0.0.1:8000/ws/live';

export function WebSocketProvider({ children, url = WS_URL }) {
  const state = useWebSocket(url);
  return (
    <WebSocketContext.Provider value={state}>
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

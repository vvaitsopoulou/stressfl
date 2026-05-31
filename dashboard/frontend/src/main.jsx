import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

import App from 'App';
import { VisionUIControllerProvider } from 'context';
import { WebSocketProvider } from 'hooks/WebSocketProvider';

const root = createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <VisionUIControllerProvider>
        <WebSocketProvider>
          <App />
        </WebSocketProvider>
      </VisionUIControllerProvider>
    </BrowserRouter>
  </React.StrictMode>,
);

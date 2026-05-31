import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Vite config tailored for absorbing Vision UI Dashboard files

export default defineConfig({
  plugins: [react()],

  // Treat .js as JSX
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.[jt]sx?$/,
    exclude: [],
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: { '.js': 'jsx' },
    },
  },

  resolve: {
    alias: {
      // Aliases match Vision UI's jsconfig.json baseUrl="src" convention.
      components: path.resolve(__dirname, './src/components'),
      layouts:    path.resolve(__dirname, './src/layouts'),
      examples:   path.resolve(__dirname, './src/examples'),
      assets:     path.resolve(__dirname, './src/assets'),
      context:    path.resolve(__dirname, './src/context'),
      hooks:      path.resolve(__dirname, './src/hooks'),
      panels:     path.resolve(__dirname, './src/panels'),
      protocol:   path.resolve(__dirname, './src/protocol'),
      App:    path.resolve(__dirname, './src/App.js'),
      routes: path.resolve(__dirname, './src/routes.js'),
    },
  },

  server: {
    port: 5173,
    host: '127.0.0.1',
  },

  base: '/',
});

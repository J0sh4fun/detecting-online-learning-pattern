import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Custom plugin to fix missing MIME types for AI assets
const mimeTypesPlugin = {
  name: 'mimetypes-plugin',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = req.url.split('?')[0];
      if (url.endsWith('.tflite') || url.endsWith('.task') || url.endsWith('.binarypb')) {
        res.setHeader('Content-Type', 'application/octet-stream');
      } else if (url.endsWith('.wasm')) {
        res.setHeader('Content-Type', 'application/wasm');
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), mimeTypesPlugin],
  server: {
    watch: {
      usePolling: true,
    },
    hmr: {
      clientPort: 5173,
    },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})

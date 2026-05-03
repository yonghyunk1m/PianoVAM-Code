import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

/**
 * Tiny dev-server API for live-saving human verdicts.
 *   GET  /api/human-verdicts  → returns JSON object (or {} if file missing)
 *   POST /api/human-verdicts  → overwrites the file with the request body
 *
 * The file is written under public/data/ so the running app can read it
 * back via a normal static fetch.
 */
function humanVerdictsApi() {
  return {
    name: 'human-verdicts-api',
    configureServer(server) {
      const dataDir = path.resolve(server.config.root, 'public/data');
      const filePath = path.join(dataDir, 'human_verdicts.json');

      server.middlewares.use('/api/human-verdicts', (req, res) => {
        if (req.method === 'GET') {
          fs.readFile(filePath, 'utf-8', (err, data) => {
            res.setHeader('Content-Type', 'application/json');
            res.end(err ? '{}' : data);
          });
          return;
        }
        if (req.method === 'POST' || req.method === 'PUT') {
          let body = '';
          req.on('data', chunk => { body += chunk; });
          req.on('end', () => {
            try {
              JSON.parse(body);  // validate
            } catch (e) {
              res.statusCode = 400;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ error: 'invalid JSON: ' + e.message }));
              return;
            }
            fs.mkdir(dataDir, { recursive: true }, (mkErr) => {
              if (mkErr) {
                res.statusCode = 500;
                res.end(JSON.stringify({ error: mkErr.message }));
                return;
              }
              fs.writeFile(filePath, body, (wrErr) => {
                if (wrErr) {
                  res.statusCode = 500;
                  res.end(JSON.stringify({ error: wrErr.message }));
                  return;
                }
                res.statusCode = 200;
                res.setHeader('Content-Type', 'application/json');
                res.end(JSON.stringify({ ok: true, bytes: body.length }));
              });
            });
          });
          return;
        }
        res.statusCode = 405;
        res.end();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), humanVerdictsApi()],
  server: {
    host: '0.0.0.0',
    port: 3333,
    strictPort: true,
    watch: {
      usePolling: true,
      interval: 1000,
      ignored: [
        '**/node_modules/**',
        '**/dist/**',
        '**/public/audio/**',
        '**/public/videos/**',
        '**/public/data/human_verdicts.json',
      ],
    },
  },
});

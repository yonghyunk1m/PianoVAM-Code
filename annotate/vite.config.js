import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';

/**
 * Tiny dev-server API for live-saving human verdicts.
 *   GET  /api/human-verdicts  → returns JSON object (or {} if file missing)
 *   POST /api/human-verdicts  → overwrites the file with the request body
 *
 * The file is written under public/data/ so the running app can read it
 * back via a normal static fetch.
 */
function makeJsonFileApi(routePath, filename, emptyValue) {
  return function configureServer(server) {
    const dataDir = path.resolve(server.config.root, 'public/data');
    const filePath = path.join(dataDir, filename);

    server.middlewares.use(routePath, (req, res) => {
      if (req.method === 'GET') {
        fs.readFile(filePath, 'utf-8', (err, data) => {
          res.setHeader('Content-Type', 'application/json');
          res.end(err ? emptyValue : data);
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
  };
}

function humanVerdictsApi() {
  return {
    name: 'human-verdicts-api',
    configureServer: makeJsonFileApi('/api/human-verdicts', 'human_verdicts.json', '{}'),
  };
}

function changeLogApi() {
  return {
    name: 'change-log-api',
    configureServer: makeJsonFileApi('/api/change-log', 'change_log.json', '[]'),
  };
}

function trialsApi() {
  return {
    name: 'trials-api',
    configureServer(server) {
      const root      = server.config.root;                       // .../annotate
      const pipelineRoot = path.resolve(root, '..');              // .../PianoVAM-Code
      const videoDir  = path.resolve(pipelineRoot, 'PianoVAM_v1.0/Video');
      const dataDir   = path.resolve(root, 'public/data');
      const JSON_EXCLUDE = new Set(['human_verdicts.json', 'change_log.json', 'notes.json']);

      // GET /api/trials → { all: [...], prepared: [...] }
      // all     = every .mp4 trial name found in VIDEO_DIR
      // prepared = trial names that already have a <trial>.json in public/data
      server.middlewares.use('/api/trials', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        const all = fs.existsSync(videoDir)
          ? fs.readdirSync(videoDir)
              .filter(f => f.endsWith('.mp4'))
              .map(f => f.slice(0, -4))
              .sort()
          : [];
        const prepared = fs.existsSync(dataDir)
          ? fs.readdirSync(dataDir)
              .filter(f => f.endsWith('.json') && !JSON_EXCLUDE.has(f)
                        && !f.startsWith('human_verdicts_'))
              .map(f => f.slice(0, -5))
          : [];
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ all, prepared }));
      });

      // POST /api/prep  { trial: "..." }
      // Runs `python pipeline.py prep <trial>` synchronously, returns when done.
      server.middlewares.use('/api/prep', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
          let trial;
          try { trial = JSON.parse(body).trial; } catch { trial = null; }
          // Validate: only date-like trial names (letters, digits, hyphens, underscores)
          if (!trial || !/^[\w\-]+$/.test(trial)) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: 'invalid trial name' }));
            return;
          }
          const result = spawnSync('python', ['pipeline.py', 'prep', trial], {
            cwd: pipelineRoot,
            timeout: 180_000,
            encoding: 'utf-8',
          });
          if (result.status === 0) {
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true }));
          } else {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: result.stderr?.slice(-500) || 'prep failed' }));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), humanVerdictsApi(), changeLogApi(), trialsApi()],
  server: {
    host: '0.0.0.0',
    port: 3333,
    strictPort: true,
    allowedHosts: true,
    fs: { allow: ['..'] },
    watch: {
      usePolling: true,
      interval: 1000,
      ignored: [
        '**/node_modules/**',
        '**/dist/**',
        '**/public/audio/**',
        '**/public/videos/**',
        '**/public/data/human_verdicts.json',
        '**/public/data/change_log.json',
      ],
    },
  },
});

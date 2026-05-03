/**
 * Lightweight IndexedDB-backed snapshot store. Every verdict change calls
 * `pushSnapshot(verdicts)`; each snapshot is keyed by an auto-incrementing
 * id and timestamped, so the user can scroll back through their session
 * even after a browser crash or accidental localStorage wipe.
 *
 * Schema:
 *   db        FingeringAnnotation
 *   store     snapshots {id (auto), ts, n_verdicts, verdicts}
 *
 * Trim policy: keep the last MAX_SNAPSHOTS rows; drop older ones to bound
 * total disk use (each snapshot is small KB-class JSON).
 */

const DB_NAME = 'FingeringAnnotation';
const STORE = 'snapshots';
const DB_VERSION = 1;
const MAX_SNAPSHOTS = 500;

let _db = null;
let _openPromise = null;

function openDB() {
  if (_db) return Promise.resolve(_db);
  if (_openPromise) return _openPromise;
  _openPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('ts', 'ts');
      }
    };
    req.onsuccess = () => { _db = req.result; resolve(_db); };
    req.onerror = () => reject(req.error);
  });
  return _openPromise;
}

export async function pushSnapshot(verdicts) {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const ts = Date.now();
    const n = Object.keys(verdicts || {}).length;
    store.add({ ts, n_verdicts: n, verdicts });
    await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
    // Trim oldest beyond MAX_SNAPSHOTS.
    await trim();
  } catch (e) {
    console.warn('[backup] pushSnapshot failed:', e);
  }
}

async function trim() {
  const db = await openDB();
  const tx = db.transaction(STORE, 'readwrite');
  const store = tx.objectStore(STORE);
  const count = await new Promise((res, rej) => {
    const r = store.count();
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  if (count <= MAX_SNAPSHOTS) return;
  const drop = count - MAX_SNAPSHOTS;
  let dropped = 0;
  await new Promise((res, rej) => {
    const cur = store.openCursor();
    cur.onsuccess = (e) => {
      const c = e.target.result;
      if (!c || dropped >= drop) { res(); return; }
      c.delete();
      dropped++;
      c.continue();
    };
    cur.onerror = () => rej(cur.error);
  });
}

export async function listSnapshots() {
  try {
    const db = await openDB();
    return await new Promise((res, rej) => {
      const tx = db.transaction(STORE, 'readonly');
      const store = tx.objectStore(STORE);
      const req = store.getAll();
      req.onsuccess = () => res(req.result.slice().reverse());
      req.onerror = () => rej(req.error);
    });
  } catch (e) {
    console.warn('[backup] listSnapshots failed:', e);
    return [];
  }
}

export async function loadSnapshot(id) {
  try {
    const db = await openDB();
    return await new Promise((res, rej) => {
      const tx = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(id);
      req.onsuccess = () => res(req.result);
      req.onerror = () => rej(req.error);
    });
  } catch (e) {
    console.warn('[backup] loadSnapshot failed:', e);
    return null;
  }
}

export async function clearAllSnapshots() {
  try {
    const db = await openDB();
    await new Promise((res, rej) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).clear();
      tx.oncomplete = res;
      tx.onerror = () => rej(tx.error);
    });
  } catch (e) {
    console.warn('[backup] clearAllSnapshots failed:', e);
  }
}

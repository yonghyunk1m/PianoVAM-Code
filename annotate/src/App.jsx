import React, { useEffect, useMemo, useRef, useState } from 'react';
import PianoRoll from './PianoRoll.jsx';
import PlaybackBar from './PlaybackBar.jsx';
import { pushSnapshot, listSnapshots, loadSnapshot, clearAllSnapshots } from './backup.js';
import { FINGER_INT_COLOR, FINGER_INT_NAME, INT_TO_LABEL } from './fingerPalette.js';
import {
  explicitAuditPriority,
  hardReasonsForNote,
  isAuditNote,
} from './auditCategories.js';

// Trial is selected via URL param ?trial=<name>.  All per-trial state is
// keyed by this name so switching trials (full page reload) never bleeds
// verdicts between sessions.
const _urlTrial = new URLSearchParams(window.location.search).get('trial') || 'default';
const VERDICT_KEY       = `fingering_verdicts_v2_${_urlTrial}`;
const HISTORY_KEY       = `fingering_history_v2_${_urlTrial}`;
const BOOKMARK_KEY      = `fingering_bookmarks_v2_${_urlTrial}`;
const BLIND_KEY         = 'fingering_blind_mode';
const SESSION_MARKER_KEY = `fingering_session_id_v1_${_urlTrial}`;
const ANNOTATOR_KEY     = 'fingering_annotator_name';
// Data file: public/data/<trial>.json  (fallback: notes.json for legacy prep)
const NOTES_URL = _urlTrial !== 'default'
  ? `/data/${_urlTrial}.json`
  : '/data/notes.json';

// Last-N cap on the undo history persisted to localStorage. The redo stack
// is in-memory only.
const HISTORY_LIMIT = 200;

// On note change, rewind the video this many seconds before the onset so the
// annotator sees the approach (helps disambiguate which finger landed).
const PRE_ONSET_LEAD_SEC = 0;

// Hand-consistency warning: two consecutive notes within this time gap and
// pitch span are likely the same hand; flip-flopping is suspicious.
const HAND_WARN_DT_SEC = 0.5;
const HAND_WARN_PITCH_SEMITONES = 7;

// While playing, advance the active note slightly ahead of the actual playhead
// so a key pressed right at the onset still selects the right note.
const PLAYBACK_LOOKAHEAD_SEC = 0.05;

function loadJSON(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
}

// Verdict provenance.
//   - source: 'algo' | 'imputed'  → bulk-accepted preload (NOT a human pick)
//   - timestamp without source    → explicit human button press
// Skipping a note never sets timestamp, so unconfirmed bulk-accepts are
// distinguishable from real human reviews in every export.
function classifyVerdict(v) {
  if (!v) return { human: null, auto: null };
  if (v.timestamp && !v.source) return { human: v, auto: null };
  return { human: null, auto: v };
}

export default function App() {
  // { all: string[], prepared: Set<string> }
  const [trialList, setTrialList] = useState({ all: [], prepared: new Set() });
  const [prepPending, setPrepPending] = useState(false);
  useEffect(() => {
    fetch('/api/trials')
      .then(r => r.ok ? r.json() : { all: [], prepared: [] })
      .then(d => setTrialList({ all: d.all || [], prepared: new Set(d.prepared || []) }))
      .catch(() => {});
  }, []);

  const [data, setData] = useState(null);
  const [idx, setIdx] = useState(0);
  const [verdicts, setVerdicts] = useState({});
  const [history, setHistory] = useState([]);
  // Server-backed verdicts: load from /api/human-verdicts on mount,
  // POST back (debounced) on every change. localStorage stays as a
  // fallback if the dev server is not reachable.
  const verdictsBootRef = useRef(false);  // skip the POST during initial load
  // Redo stack: lives in memory only (intentionally not persisted — reload
  // resets it). Cleared whenever a fresh verdict is recorded.
  const [redoStack, setRedoStack] = useState([]);
  const [bookmarks, setBookmarks] = useState(() => new Set());
  // Blind mode: hide every algorithmic hint (algo pick, imputed pick,
  // candidates, suspect markers) so the annotator labels without priming.
  // Persisted across reloads so a session is consistent.
  const [blindMode, setBlindMode] = useState(() => loadJSON(BLIND_KEY, false));
  useEffect(() => { localStorage.setItem(BLIND_KEY, JSON.stringify(blindMode)); }, [blindMode]);

  // Annotator identity — persisted in localStorage so it survives page refresh.
  const [annotatorName, setAnnotatorName] = useState(
    () => localStorage.getItem(ANNOTATOR_KEY) || '');
  useEffect(() => { localStorage.setItem(ANNOTATOR_KEY, annotatorName); }, [annotatorName]);

  // Change log — append-only audit trail of notes changed from TSV original.
  // Each entry: { annotator, trial, note_idx, pitch_name, onset_sec,
  //               from_int, from_label, to_int, to_label, timestamp }
  // Stored in public/data/change_log.json via /api/change-log.
  const [changeLog, setChangeLog] = useState([]);
  const changeLogLoadedRef = useRef(false);
  const changeLogTimerRef = useRef(null);

  // Playback state
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(NaN);
  const [paused, setPaused] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(0.5);
  // R-key loops the current note's window. Lead/tail kept tiny so adjacent
  // notes (e.g. the next onset only ~50 ms after this offset) don't sneak
  // into the loop. 30 ms each side: attack/release intact, no neighbor.
  const [replayLoop, setReplayLoop] = useState(false);
  const REPLAY_LEAD_SEC = 0.03;
  const REPLAY_TAIL_SEC = 0.03;

  // Backup panel
  const [showBackup, setShowBackup] = useState(false);
  const [snapshots, setSnapshots] = useState([]);
  // Review queue modal
  const [showReview, setShowReview] = useState(false);

  const videoRef = useRef(null);
  const audioRef = useRef(null);
  // Mirror `paused` into a ref so the snap-to-onset effect can read the
  // current playback state without re-running every time pause toggles.
  const pausedRef = useRef(true);
  useEffect(() => { pausedRef.current = paused; }, [paused]);

  // Load notes JSON. The file is gitignored — annotators populate it locally
  // from the pipeline's per-trial export. Surface a clear error if missing
  // so the user doesn't stare at a stuck "Loading…" forever.
  const [loadError, setLoadError] = useState(null);
  useEffect(() => {
    fetch(NOTES_URL)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch(err => {
        console.error('Failed to load /data/notes.json:', err);
        setLoadError(String(err.message || err));
      });
  }, []);

  // Load verdicts from the dev-server (file-backed) after the note bundle
  // arrives so we can ignore stale localStorage from a different session.
  useEffect(() => {
    if (!data) return;
    verdictsBootRef.current = false;

    const sessionId = data.session_id || 'default';
    const sameSession = localStorage.getItem(SESSION_MARKER_KEY) === sessionId;
    setHistory(sameSession ? loadJSON(HISTORY_KEY, []) : []);
    setBookmarks(new Set(sameSession ? loadJSON(BOOKMARK_KEY, []) : []));

    fetch('/api/human-verdicts')
      .then(r => r.ok ? r.json() : Promise.reject('http ' + r.status))
      .then(serverVerdicts => {
        verdictsBootRef.current = true;
        if (serverVerdicts && Object.keys(serverVerdicts).length > 0) {
          setVerdicts(serverVerdicts);
        } else if (sameSession) {
          // Server file is empty; only revive local work if it belongs to the
          // same prepared review session.
          const local = loadJSON(VERDICT_KEY, {});
          setVerdicts(local);
        } else {
          setVerdicts({});
        }
      })
      .catch(err => {
        console.warn('verdicts API unreachable, using localStorage only:', err);
        if (sameSession) setVerdicts(loadJSON(VERDICT_KEY, {}));
        else setVerdicts({});
        verdictsBootRef.current = true;
      });
  }, [data]);

  // Persist verdicts: localStorage as instant fallback + debounced POST to
  // the dev server (which writes public/data/human_verdicts.json).
  const saveTimerRef = useRef(null);
  useEffect(() => {
    if (!data) return;
    localStorage.setItem(VERDICT_KEY, JSON.stringify(verdicts));
    localStorage.setItem(SESSION_MARKER_KEY, data.session_id || 'default');
    if (Object.keys(verdicts).length > 0) pushSnapshot(verdicts);
    if (!verdictsBootRef.current) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      fetch('/api/human-verdicts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(verdicts),
      }).catch(err => console.warn('verdict POST failed:', err));
    }, 300);
  }, [verdicts]);
  useEffect(() => {
    if (!data) return;
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-HISTORY_LIMIT)));
    localStorage.setItem(SESSION_MARKER_KEY, data.session_id || 'default');
  }, [history]);
  useEffect(() => {
    if (!data) return;
    localStorage.setItem(BOOKMARK_KEY, JSON.stringify([...bookmarks]));
    localStorage.setItem(SESSION_MARKER_KEY, data.session_id || 'default');
  }, [bookmarks]);

  // Load change log from server once data is ready.
  useEffect(() => {
    if (!data) return;
    changeLogLoadedRef.current = false;
    fetch('/api/change-log')
      .then(r => r.ok ? r.json() : [])
      .then(log => {
        changeLogLoadedRef.current = true;
        setChangeLog(Array.isArray(log) ? log : []);
      })
      .catch(() => { changeLogLoadedRef.current = true; });
  }, [data]);

  // Persist change log to server (debounced 500 ms).
  useEffect(() => {
    if (!changeLogLoadedRef.current) return;
    if (changeLogTimerRef.current) clearTimeout(changeLogTimerRef.current);
    changeLogTimerRef.current = setTimeout(() => {
      fetch('/api/change-log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(changeLog),
      }).catch(err => console.warn('change-log POST failed:', err));
    }, 500);
  }, [changeLog]);

  // Auto-resume to first un-annotated note on data load.
  useEffect(() => {
    if (!data) return;
    if (Object.keys(verdicts).length === 0) return;
    const firstUnlabeled = data.notes.findIndex(
      n => !verdicts[`${n.trial}#${n.note_idx}`]);
    if (firstUnlabeled >= 0 && firstUnlabeled !== idx) setIdx(firstUnlabeled);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const note = data?.notes[idx];

  // Memoised per-trial slice for the piano roll context.
  const trialNotes = useMemo(
    () => (data && note ? data.notes.filter(n => n.trial === note.trial) : []),
    [data, note?.trial]);

  // Hand-consistency warning. Hoisted here (above the early return) so its
  // hook call order is stable across renders. Returns null if data isn't ready.
  const handWarning = useMemo(() => {
    if (!data || !note || idx === 0) return null;
    const prev = data.notes[idx - 1];
    if (!prev) return null;
    const prevV = verdicts[`${prev.trial}#${prev.note_idx}`];
    if (!prevV || prevV.finger_int < 1 || prevV.finger_int > 10) return null;
    if (prev.trial !== note.trial) return null;
    const dt = note.onset_sec - prev.onset_sec;
    if (dt > HAND_WARN_DT_SEC) return null;
    const pitchGap = Math.abs(note.pitch - prev.pitch);
    if (pitchGap > HAND_WARN_PITCH_SEMITONES) return null;  // far apart → switching hand is plausible
    const prevHand = prevV.finger_int <= 5 ? 'L' : 'R';
    const cur = verdicts[`${note.trial}#${note.note_idx}`];
    if (!cur || cur.finger_int < 1 || cur.finger_int > 10) {
      return `Previous note was ${prevV.finger_label} (hand ${prevHand}); pitch gap ${pitchGap}st within ${(dt*1000).toFixed(0)} ms suggests same hand.`;
    }
    const curHand = cur.finger_int <= 5 ? 'L' : 'R';
    if (curHand !== prevHand) {
      return `⚠ Hand switch flagged: prev ${prevV.finger_label} → cur ${cur.finger_label} (pitch gap ${pitchGap}st within ${(dt*1000).toFixed(0)} ms).`;
    }
    return null;
  }, [idx, data, verdicts, note]);

  // Set of "trial#note_idx" keys for notes that were changed from TSV original.
  const changedNoteIds = useMemo(
    () => new Set(changeLog.map(e => `${e.trial}#${e.note_idx}`)),
    [changeLog]);

  // Live stats: how many algo-labeled notes, how many human-reviewed, how many differ.
  // Denominator = all algorithm-labeled notes (not just human-reviewed ones).
  const changeStats = useMemo(() => {
    if (!data) return null;
    const n_algo_total = data.notes.filter(
      n => n.algorithm_int >= 1 && n.algorithm_int <= 10).length;
    let n_reviewed = 0, n_changed = 0;
    for (const n of data.notes) {
      const v = classifyVerdict(verdicts[`${n.trial}#${n.note_idx}`]).human;
      if (!v) continue;
      n_reviewed++;
      if (n.algorithm_int && v.finger_int >= 1 && v.finger_int <= 10
          && v.finger_int !== n.algorithm_int)
        n_changed++;
    }
    const errorPct = n_reviewed > 0 ? (100 * n_changed / n_reviewed).toFixed(1) : null;
    const reviewedPct = n_algo_total > 0 ? (100 * n_reviewed / n_algo_total).toFixed(1) : null;
    return { n_algo_total, n_reviewed, n_changed, errorPct, reviewedPct };
  }, [data, verdicts]);

  // Snap video + audio to onset whenever the active note changes — but only
  // while playback is paused. During playback the auto-sync effect drives
  // idx based on currentTime; jumping the video back would freeze it.
  // Manual nav (Prev/Next, Tab, P, click on roll, etc.) all happen while
  // paused (or right after the user pauses), so this gives them a reliable
  // jump while leaving playback alone.
  useEffect(() => {
    if (!note) return;
    if (!pausedRef.current) return;
    [videoRef.current, audioRef.current].forEach(el => {
      if (!el) return;
      const onLoaded = () => {
        el.currentTime = Math.max(0, note.onset_sec - PRE_ONSET_LEAD_SEC);
        el.playbackRate = playbackRate;
        el.pause();
      };
      if (el.readyState >= 1) onLoaded();
      else el.addEventListener('loadedmetadata', onLoaded, { once: true });
    });
  }, [note?.trial, note?.onset_sec, playbackRate]);

  // Apply playback rate change live.
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
    if (audioRef.current) audioRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  // Subscribe to <video> events for state (paused/duration/seek). Time
  // updates come from requestAnimationFrame below (60Hz instead of the
  // browser's coarse 4–30Hz timeupdate), so the piano roll scrolls smoothly.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => setCurrentTime(v.currentTime);   // fallback when paused
    const onDur = () => setDuration(v.duration);
    const onPlay = () => { setPaused(false); audioRef.current?.play().catch(()=>{}); };
    const onPause = () => { setPaused(true); audioRef.current?.pause(); };
    const onSeek = () => {
      setCurrentTime(v.currentTime);
      if (audioRef.current) audioRef.current.currentTime = v.currentTime;
    };
    v.addEventListener('timeupdate', onTime);
    v.addEventListener('loadedmetadata', onDur);
    v.addEventListener('durationchange', onDur);
    v.addEventListener('play', onPlay);
    v.addEventListener('pause', onPause);
    v.addEventListener('seeked', onSeek);
    return () => {
      v.removeEventListener('timeupdate', onTime);
      v.removeEventListener('loadedmetadata', onDur);
      v.removeEventListener('durationchange', onDur);
      v.removeEventListener('play', onPlay);
      v.removeEventListener('pause', onPause);
      v.removeEventListener('seeked', onSeek);
    };
  }, [note?.trial]);

  // 60 Hz currentTime polling while playing — feeds smooth piano-roll scroll.
  useEffect(() => {
    if (paused) return;
    let raf;
    const tick = () => {
      const v = videoRef.current;
      if (v) setCurrentTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [paused]);

  // Replay loop, isolated from the playback-driven raf so it survives
  // pause/play race conditions: this effect re-arms whenever replayLoop
  // turns on and watches currentTime via its own animation frame.
  useEffect(() => {
    if (!replayLoop || !note) return;
    let raf;
    const check = () => {
      const v = videoRef.current;
      if (v && v.currentTime > note.offset_sec + REPLAY_TAIL_SEC) {
        v.currentTime = Math.max(0, note.onset_sec - REPLAY_LEAD_SEC);
        if (v.paused) v.play().catch(() => {});
      }
      raf = requestAnimationFrame(check);
    };
    raf = requestAnimationFrame(check);
    return () => cancelAnimationFrame(raf);
  }, [replayLoop, note?.onset_sec, note?.offset_sec]);

  // When the user navigates to a different note while replay-loop is on,
  // cancel the loop AND pause the video — they wanted a snap to the new
  // note, not free playback continuing through it.
  useEffect(() => {
    if (replayLoop) {
      videoRef.current?.pause();
      audioRef.current?.pause();
      setReplayLoop(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [note?.trial, note?.note_idx]);

  // Video stays muted at all times; the separate audio file is the
  // higher-quality source and rides along synced to the video element.
  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = true;
    if (audioRef.current) audioRef.current.muted = false;
  }, [note?.trial]);

  // While the video is playing, advance the active note (idx) so the annotate
  // panel shows whatever is currently being heard. The "active" note is the
  // latest one in this trial whose onset has already started (with a 50 ms
  // lead so a key pressed slightly early still hits the right note).
  // Replay-loop suspends this so a single note can repeat in place.
  useEffect(() => {
    if (paused || !data || !note) return;
    if (replayLoop) return;
    const tLead = currentTime + PLAYBACK_LOOKAHEAD_SEC;
    let activeIdx = -1;
    for (let i = 0; i < data.notes.length; i++) {
      const n = data.notes[i];
      if (n.trial !== note.trial) continue;
      if (n.onset_sec > tLead) break;
      activeIdx = i;
    }
    if (activeIdx >= 0 && activeIdx !== idx) setIdx(activeIdx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTime, paused, data, note?.trial, replayLoop]);

  // Keyboard shortcuts (minimal set per user spec).
  //   a / d  : Prev / Next note
  //   q / e  : Prev / Next priority
  //   r      : Replay loop current note
  //   Space  : Play / pause video
  //   Ctrl+Z : Undo  ·  Ctrl+Y or Ctrl+Shift+Z : Redo
  // Finger selection itself is mouse-only (click the finger buttons).
  //
  // Listener is registered once and dispatches via a ref kept up-to-date
  // every render. Without this, the 60Hz raf re-render cycle would
  // add/remove the listener every frame and could miss a quick keystroke.
  const keyHandlerRef = useRef(null);
  keyHandlerRef.current = (e) => {
    if (e.target.tagName === 'INPUT') return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z' && !e.shiftKey) {
      e.preventDefault(); undo(); return;
    }
    if ((e.ctrlKey || e.metaKey) && (k === 'y' || (k === 'z' && e.shiftKey))) {
      e.preventDefault(); redo(); return;
    }
    if (e.ctrlKey || e.metaKey) return;
    if (k === 'a')      { e.preventDefault(); goPrev(); }
    else if (k === 'd') { e.preventDefault(); goNext(); }
    else if (k === 'q') { e.preventDefault(); jumpPrevPriority(); }
    else if (k === 'e') { e.preventDefault(); jumpNextPriority(); }
    else if (k === 'r') {
      e.preventDefault();
      toggleReplayLoop();
    }
    else if (e.key === ' ') {
      e.preventDefault();
      const v = videoRef.current;
      if (v) { if (v.paused) v.play(); else v.pause(); }
    }
  };
  useEffect(() => {
    const handler = (e) => keyHandlerRef.current?.(e);
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  if (loadError) {
    return (
      <div style={{ padding: 40, color: 'var(--text-secondary)', maxWidth: 720 }}>
        <h2 style={{ margin: '0 0 12px' }}>Cannot load notes</h2>
        <p>
          <code>/data/notes.json</code> failed to load ({loadError}).
        </p>
        <p>
          Place the per-trial export from the pipeline at{' '}
          <code>annotate/public/data/notes.json</code> and reload.
          The folder is gitignored — populate it locally for each annotation
          session.
        </p>
      </div>
    );
  }
  if (!data || !note) {
    return <div style={{ padding: 40, color: 'var(--text-secondary)' }}>Loading…</div>;
  }

  const trialLocalIdx = trialNotes.findIndex(n => n.global_idx === note.global_idx);

  function toggleReplayLoop() {
    const v = videoRef.current;
    if (!v || !note) return;
    if (replayLoop) {
      setReplayLoop(false);
      v.pause();
    } else {
      v.currentTime = Math.max(0, note.onset_sec - REPLAY_LEAD_SEC);
      v.play().catch(() => {});
      setReplayLoop(true);
    }
  }

  function setVerdictAt(notesSlice, ints, kind = 'submit') {
    const startIdx = data.notes.indexOf(notesSlice[0]);
    const endIdx = startIdx + notesSlice.length;
    const replaced = notesSlice.map(n => verdicts[`${n.trial}#${n.note_idx}`] || null);
    setHistory(h => [...h, { kind, startIdx, endIdx, replaced }]);
    setRedoStack([]);   // any new verdict invalidates the redo branch

    // Append change-log entries when human pick differs from TSV original.
    const name = annotatorName.trim();
    if (name) {
      const now = Date.now();
      const entries = [];
      for (let i = 0; i < notesSlice.length; i++) {
        const n = notesSlice[i];
        const v = ints[i];
        if (n.algorithm_int && v >= 1 && v <= 10 && v !== n.algorithm_int) {
          entries.push({
            annotator: name,
            trial: n.trial,
            note_idx: n.note_idx,
            pitch_name: n.pitch_name,
            onset_sec: n.onset_sec,
            from_int: n.algorithm_int,
            from_label: INT_TO_LABEL[n.algorithm_int] || String(n.algorithm_int),
            to_int: v,
            to_label: INT_TO_LABEL[v] || String(v),
            timestamp: now,
          });
        }
      }
      if (entries.length > 0) setChangeLog(prev => [...prev, ...entries]);
    }

    setVerdicts(prev => {
      const next = { ...prev };
      for (let i = 0; i < notesSlice.length; i++) {
        const n = notesSlice[i];
        const id = `${n.trial}#${n.note_idx}`;
        const v = ints[i];
        next[id] = {
          finger_int: v,
          finger_label: v === 0 ? 'unsure' : v === -1 ? 'no_press' : INT_TO_LABEL[v],
          algorithm_int: n.algorithm_int,
          algorithm_status: n.algorithm_status,
          timestamp: Date.now(),
        };
      }
      return next;
    });
  }

  // While playing, the playback head drives idx via the auto-sync effect, so
  // verdict actions don't self-advance (it would fight that effect). When
  // paused, advance one step like the old behavior.
  function maybeAdvance() {
    if (paused && idx < data.notes.length - 1) setIdx(idx + 1);
  }

  function confirmAlgo() {
    if (!note.algorithm_int) return;
    setVerdictAt([note], [note.algorithm_int], 'algo');
    maybeAdvance();
  }

  function applyCandidate(intVal) {
    setVerdictAt([note], [intVal], 'candidate');
    maybeAdvance();
  }

  // Generic single-finger apply (button click or keyboard shortcut).
  // intVal: 1..10 = finger; 0 = unsure; -1 = no press visible.
  function applyFinger(intVal) {
    if (!note) return;
    setVerdictAt([note], [intVal], 'finger');
    maybeAdvance();
  }

  function repeatLastVerdict() {
    if (history.length === 0) return;
    const last = history[history.length - 1];
    const lastNote = data.notes[last.endIdx - 1];
    const lastV = verdicts[`${lastNote.trial}#${lastNote.note_idx}`];
    if (!lastV) return;
    setVerdictAt([note], [lastV.finger_int], 'repeat');
    maybeAdvance();
  }

  function undo() {
    if (history.length === 0) return;
    const last = history[history.length - 1];
    // Snapshot the current values at the soon-to-be-undone positions so
    // redo can re-apply them.
    const after = [];
    for (let i = 0; i < last.endIdx - last.startIdx; i++) {
      const n = data.notes[last.startIdx + i];
      after.push(verdicts[`${n.trial}#${n.note_idx}`] || null);
    }
    setRedoStack(r => [...r, { ...last, after }]);
    setVerdicts(prev => {
      const next = { ...prev };
      for (let i = 0; i < last.endIdx - last.startIdx; i++) {
        const n = data.notes[last.startIdx + i];
        const id = `${n.trial}#${n.note_idx}`;
        if (last.replaced[i] === null) delete next[id];
        else next[id] = last.replaced[i];
      }
      return next;
    });
    setHistory(h => h.slice(0, -1));
    setIdx(last.startIdx);
  }

  function redo() {
    if (redoStack.length === 0) return;
    const next_action = redoStack[redoStack.length - 1];
    setHistory(h => [...h, { kind: next_action.kind, startIdx: next_action.startIdx,
                              endIdx: next_action.endIdx, replaced: next_action.replaced }]);
    setVerdicts(prev => {
      const out = { ...prev };
      for (let i = 0; i < next_action.endIdx - next_action.startIdx; i++) {
        const n = data.notes[next_action.startIdx + i];
        const id = `${n.trial}#${n.note_idx}`;
        if (next_action.after[i] === null) delete out[id];
        else out[id] = next_action.after[i];
      }
      return out;
    });
    setRedoStack(r => r.slice(0, -1));
    setIdx(Math.min(data.notes.length - 1, next_action.endIdx - 1));
  }

  // Skipping a note is NOT a verdict. The annotator must press an explicit
  // finger / unsure / no-press button for `timestamp` to be set. Bulk-accepted
  // entries (source: 'algo' | 'imputed') stay unconfirmed until that happens.

  function goNext() { setIdx(i => Math.min(data.notes.length - 1, i + 1)); }
  function goPrev() { setIdx(i => Math.max(0, i - 1)); }

  function jumpNextUnlabeled() {
    for (let i = idx + 1; i < data.notes.length; i++) {
      const n = data.notes[i];
      if (!verdicts[`${n.trial}#${n.note_idx}`]) { setIdx(i); return; }
    }
    // wrap-around from the start
    for (let i = 0; i < idx; i++) {
      const n = data.notes[i];
      if (!verdicts[`${n.trial}#${n.note_idx}`]) { setIdx(i); return; }
    }
  }

  function jumpNextAmbiguous() {
    for (let i = idx + 1; i < data.notes.length; i++) {
      if (data.notes[i].algorithm_status !== 'labeled') { setIdx(i); return; }
    }
  }

  function jumpNextDisagreement() {
    for (let i = idx + 1; i < data.notes.length; i++) {
      const n = data.notes[i];
      const v = verdicts[`${n.trial}#${n.note_idx}`];
      if (v && v.finger_int >= 1 && v.finger_int <= 10
          && n.algorithm_int && v.finger_int !== n.algorithm_int) {
        setIdx(i); return;
      }
    }
  }

  function jumpNextPriority() {
    // Walk forward to the next suspect note that the annotator has not yet
    // explicitly reviewed. The skip predicate keys on `timestamp` (set only
    // by an explicit button press in setVerdictAt) — bulk-accepted entries
    // never carry a timestamp.
    const start = idx + 1;
    for (let pass = 0; pass < 2; pass++) {
      const begin = pass === 0 ? start : 0;
      const end = pass === 0 ? data.notes.length : start;
      for (let i = begin; i < end; i++) {
        const n = data.notes[i];
        if (!priorityForNote(n)) continue;
        const v = verdicts[`${n.trial}#${n.note_idx}`];
        if (v && v.timestamp) continue;
        if (i === idx) continue;
        setIdx(i); return;
      }
    }
  }

  function jumpPrevPriority() {
    const start = idx - 1;
    for (let pass = 0; pass < 2; pass++) {
      const begin = pass === 0 ? start : data.notes.length - 1;
      const end = pass === 0 ? -1 : start;
      for (let i = begin; i > end; i--) {
        const n = data.notes[i];
        if (!priorityForNote(n)) continue;
        const v = verdicts[`${n.trial}#${n.note_idx}`];
        if (v && v.timestamp) continue;
        if (i === idx) continue;
        setIdx(i); return;
      }
    }
  }

  function jumpNextHard() {
    // Jump to the next explicit audit-category or legacy hard note that has
    // not been explicitly human-reviewed.
    // Reviewed = verdict has a timestamp without source (explicit button press).
    for (let pass = 0; pass < 2; pass++) {
      const begin = pass === 0 ? idx + 1 : 0;
      const end   = pass === 0 ? data.notes.length : idx;
      for (let i = begin; i < end; i++) {
        const n = data.notes[i];
        if (!isAuditNote(n)) continue;
        const v = verdicts[`${n.trial}#${n.note_idx}`];
        if (v && v.timestamp && !v.source) continue;   // already reviewed
        setIdx(i); return;
      }
    }
  }
  function jumpPrevHard() {
    for (let pass = 0; pass < 2; pass++) {
      const begin = pass === 0 ? idx - 1 : data.notes.length - 1;
      const end   = pass === 0 ? -1 : idx;
      for (let i = begin; i > end; i--) {
        const n = data.notes[i];
        if (!isAuditNote(n)) continue;
        const v = verdicts[`${n.trial}#${n.note_idx}`];
        if (v && v.timestamp && !v.source) continue;   // already reviewed
        setIdx(i); return;
      }
    }
  }

  function toggleBookmark() {
    const id = `${note.trial}#${note.note_idx}`;
    setBookmarks(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  function jumpNextBookmark() {
    const ids = [...bookmarks];
    for (let i = idx + 1; i < data.notes.length; i++) {
      const n = data.notes[i];
      if (ids.includes(`${n.trial}#${n.note_idx}`)) { setIdx(i); return; }
    }
    for (let i = 0; i < idx; i++) {
      const n = data.notes[i];
      if (ids.includes(`${n.trial}#${n.note_idx}`)) { setIdx(i); return; }
    }
  }

  function exportPayload() {
    let n_human = 0, n_auto = 0;
    for (const v of Object.values(verdicts)) {
      const c = classifyVerdict(v);
      if (c.human) n_human++;
      else if (c.auto) n_auto++;
    }
    return {
      generated_at: new Date().toISOString(),
      n_notes: data.notes.length,
      n_human_verdicts: n_human,
      n_auto_labels: n_auto,
      finger_int_encoding: data.finger_int_encoding,
      bookmarks: [...bookmarks],
      summary: computeSummary(data, verdicts),
      verdicts: data.notes.map(n => {
        const id = `${n.trial}#${n.note_idx}`;
        return {
          trial: n.trial, piece: n.piece, difficulty: n.difficulty, task: n.task,
          note_idx: n.note_idx, pitch: n.pitch, pitch_name: n.pitch_name,
          onset_sec: n.onset_sec,
          algorithm: {
            status: n.algorithm_status, int: n.algorithm_int,
            hand: n.algorithm_hand, finger: n.algorithm_finger,
            score: n.algorithm_score,
          },
          human: classifyVerdict(verdicts[id]).human,
          auto_label: classifyVerdict(verdicts[id]).auto,
          bookmarked: bookmarks.has(id),
        };
      }),
    };
  }
  function exportJSON(isAuto = false) {
    const out = exportPayload();
    const blob = new Blob([JSON.stringify(out, null, 2)],
                           { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const tag = isAuto ? '_autosave' : '';
    a.href = url;
    a.download = `fingering_verdicts_${ts}${tag}.json`;
    a.click();
  }
  function exportCSV() {
    const out = exportPayload();
    const cols = ['trial', 'piece', 'difficulty', 'note_idx', 'pitch',
                  'pitch_name', 'onset_sec', 'algo_status', 'algo_int',
                  'human_int', 'human_label',
                  'auto_label_int', 'auto_label_source',
                  'bookmarked', 'agreement'];
    const lines = [cols.join(',')];
    for (const v of out.verdicts) {
      const h = v.human;          // explicit human press only
      const a = v.auto_label;     // bulk-accepted preload (not human)
      const agree = (v.algorithm.status === 'labeled' && v.algorithm.int && h
                      && h.finger_int >= 1 && h.finger_int <= 10)
        ? (h.finger_int === v.algorithm.int ? '1' : '0') : '';
      lines.push([
        v.trial, JSON.stringify(v.piece || ''), v.difficulty,
        v.note_idx, v.pitch, v.pitch_name, v.onset_sec,
        v.algorithm.status, v.algorithm.int ?? '',
        h?.finger_int ?? '', h?.finger_label ?? '',
        a?.finger_int ?? '', a?.source ?? '',
        v.bookmarked ? '1' : '0', agree,
      ].join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = url;
    a.download = `fingering_verdicts_${ts}.csv`;
    a.click();
  }

  async function refreshSnapshots() { setSnapshots(await listSnapshots()); }
  async function restoreSnapshot(id) {
    const snap = await loadSnapshot(id);
    if (!snap || !window.confirm(
        `Restore snapshot from ${new Date(snap.ts).toLocaleString()}? This replaces your current annotation state.`))
      return;
    setVerdicts(snap.verdicts);
    setHistory([]);
    setSnapshots([]);
    setShowBackup(false);
  }
  async function clearBackups() {
    if (!window.confirm('Delete all backup snapshots? This cannot be undone.')) return;
    await clearAllSnapshots();
    setSnapshots([]);
  }

  const id = `${note.trial}#${note.note_idx}`;
  const v = verdicts[id];
  const isBookmarked = bookmarks.has(id);
  const algoLabel = note.algorithm_int ? INT_TO_LABEL[note.algorithm_int] : '—';
  const priorityStats = computePrioritySummary(data, verdicts);

  const hardStats = (() => {
    let total = 0, done = 0;
    for (const n of data.notes) {
      if (!isAuditNote(n)) continue;
      total++;
      const nv = verdicts[`${n.trial}#${n.note_idx}`];
      if (nv && nv.timestamp && !nv.source) done++;
    }
    return { total, done, remaining: total - done };
  })();

  return (
    <div className="app">
      <div className="header">
        <h1>Fingering Annotation</h1>
        <span className="progress">{idx + 1} / {data.notes.length}</span>
        {!blindMode && hardStats.total > 0 && (
          <span className="priority-progress"
                title={`Hard notes reviewed: ${hardStats.done} / ${hardStats.total} — ${hardStats.remaining} remaining`}>
            🔴 {hardStats.done}/{hardStats.total} hard
            <span className="priority-bar">
              <span className="priority-bar-fill"
                    style={{ width: `${hardStats.total ? 100 * hardStats.done / hardStats.total : 0}%` }} />
            </span>
          </span>
        )}
        <span className="meta">
          {note.difficulty && <span className="difficulty-tag">{note.difficulty}</span>}
          <b>{note.piece || note.trial}</b>
          {' · '}note {trialLocalIdx + 1}/{trialNotes.length}
          {' · '}{note.pitch_name} (MIDI {note.pitch})
          {' · '}onset {note.onset_sec.toFixed(3)} s
        </span>
        {trialList.all.length > 0 && (
          <>
            {prepPending && (
              <span className="prep-status">⏳ Preparing…</span>
            )}
            <select
              className="trial-select"
              value={_urlTrial}
              disabled={prepPending}
              title="Select trial — unprepared ones are loaded automatically"
              onChange={async e => {
                const t = e.target.value;
                if (t === _urlTrial) return;
                if (trialList.prepared.has(t)) {
                  window.location.href = `${window.location.pathname}?trial=${encodeURIComponent(t)}`;
                } else {
                  setPrepPending(true);
                  try {
                    const r = await fetch('/api/prep', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ trial: t }),
                    });
                    if (r.ok) {
                      window.location.href = `${window.location.pathname}?trial=${encodeURIComponent(t)}`;
                    } else {
                      const err = await r.json().catch(() => ({}));
                      alert('Prep failed: ' + (err.error || r.status));
                      setPrepPending(false);
                    }
                  } catch {
                    alert('Prep failed (network error)');
                    setPrepPending(false);
                  }
                }
              }}>
              {/* Show current trial even if not in video list (e.g. legacy notes.json) */}
              {_urlTrial !== 'default' && !trialList.all.includes(_urlTrial) && (
                <option value={_urlTrial}>{_urlTrial}</option>
              )}
              {trialList.all.map(t => (
                <option key={t} value={t}>
                  {trialList.prepared.has(t) ? `✓ ${t}` : t}
                </option>
              ))}
            </select>
          </>
        )}
        <input
          type="text"
          className="annotator-input"
          value={annotatorName}
          onChange={e => setAnnotatorName(e.target.value)}
          placeholder="Annotator name"
          title="Your name is recorded with each change from TSV original"
        />
        <button className={`btn-secondary ${blindMode ? 'active' : ''}`}
                onClick={() => setBlindMode(b => !b)}
                title="Blind mode: hide every algorithmic hint (algo pick, imputed pick, candidates, suspect markers).">
          {blindMode ? '👁 Blind ON' : '👁 Blind OFF'}
        </button>
        <a href={(() => { const p = new URLSearchParams(window.location.search); p.set('mode','hard'); return window.location.pathname + '?' + p.toString(); })()}
           style={{ padding: '7px 14px', background: '#D32F2F', color: '#FFF',
                    border: 'none', borderRadius: 6, fontSize: 12, fontWeight: 600,
                    textDecoration: 'none', cursor: 'pointer' }}>
          🔴 Hard Notes
        </a>
        <button className="btn-secondary" onClick={() => setShowReview(true)}>
          Review queue
        </button>
        <button className="btn-secondary" onClick={() => { setShowBackup(true); refreshSnapshots(); }}>
          Backups
        </button>
        <button className="btn-secondary" onClick={exportCSV}>CSV</button>
        <button className="btn-primary" onClick={() => exportJSON(false)}>Export JSON</button>
      </div>

      <div className="main">
        <div className="panel video">
          <div className="video-wrap">
            <div className="video-stage">
              <div className="video-frame">
                <video ref={videoRef}
                       src={note.video_path}
                       key={note.trial} />
              </div>
            </div>
            <audio ref={audioRef}
                   src={note.audio_path}
                   key={`${note.trial}_audio`}
                   preload="auto" />
          </div>
        </div>

        <div className="panel roll-panel">
          <div className="roll-wrap">
            <PianoRoll trialNotes={trialNotes} currentNoteIdx={trialLocalIdx}
                        verdicts={verdicts}
                        currentTime={currentTime}
                        duration={duration}
                        isPlaying={!paused}
                        blindMode={blindMode}
                        changedNoteIds={changedNoteIds}
                        onSeek={t => { if (videoRef.current) videoRef.current.currentTime = t; }}
                        onSelectNote={gidx => setIdx(gidx)} />
            <div className="roll-info">
              {trialLocalIdx + 1} / {trialNotes.length} in trial
            </div>
          </div>
        </div>
      </div>

      <div className="annotate-wrap">
        <div className="panel action">
            {(() => {
              const reviewed = v && v.timestamp && !v.source;
              let cls, label, detail;
              if (blindMode) {
                // Strip every algorithmic detail; show only own annotation state.
                if (reviewed) {
                  cls = 'confirmed'; label = '✓ ANNOTATED';
                  detail = `you picked ${INT_TO_LABEL[v.finger_int] ?? v.finger_label}`;
                } else {
                  cls = 'skip'; label = '○ not yet annotated'; detail = '';
                }
              } else {
                const p = priorityForNote(note);
                const isHardRuleNote = isAuditNote(note);
                const isImputed = v && v.source === 'imputed';
                const isAlgoBulk = v && v.source === 'algo' && !reviewed;
                if (p && !reviewed) {
                  cls = 'pending';
                  label = isHardRuleNote ? '🕵 HARD NOTE — verify' : '🕵 SUSPECT — verify';
                  detail = p.reason;
                } else if (p && reviewed) {
                  cls = 'done';    label = '✓ REVIEWED';
                  detail = reviewedDetail(note, v, p);
                } else if (reviewed) {
                  cls = 'confirmed'; label = '✓ ANNOTATED';
                  detail = reviewedDetail(note, v, null);
                } else if (isImputed) {
                  cls = 'skip'; label = '◇ model prediction (no human annotation yet)';
                  detail = `Sequence model picked ${INT_TO_LABEL[v.finger_int]} — looks confident`;
                } else if (isAlgoBulk) {
                  cls = 'skip'; label = '○ model prediction (no human annotation yet)';
                  detail = 'algorithm was confident — human review optional';
                } else {
                  cls = 'skip'; label = '○ not yet annotated'; detail = '';
                }
              }
              return (
                <div className={`review-status review-status-${cls}`}>
                  <span><b>{label}</b>{detail && <> — {detail}</>}
                    {handWarning && <span className="review-warn"> · ⚠ {handWarning}</span>}
                  </span>
                </div>
              );
            })()}

            {/* Change-log indicator for the current note */}
            {(() => {
              const entries = changeLog.filter(
                e => e.trial === note.trial && e.note_idx === note.note_idx);
              if (entries.length === 0) return null;
              const last = entries[entries.length - 1];
              return (
                <div className="change-indicator">
                  <span className="change-dot" />
                  Changed from TSV: <b>{last.from_label}</b> → <b>{last.to_label}</b>
                  {last.annotator && <> by <b>{last.annotator}</b></>}
                  {entries.length > 1 && <> ({entries.length}×)</>}
                </div>
              );
            })()}

            {/* Finger button grid — primary input. One click / one keystroke.
                In blindMode every algorithmic hint (algo / imputed / candidate)
                is suppressed so the annotator labels without priming. */}
            <div className="finger-grid">
              <div className="finger-half left">
                <div className="hand-label">Left  hand · pinky → thumb</div>
                <div className="finger-row">
                  {[5, 4, 3, 2, 1].map(i => (
                    <FingerBtn key={i} intVal={i}
                               isAlgo={!blindMode && note.algorithm_int === i}
                               isImputed={!blindMode && note.imputed_int === i}
                               isCurrentHuman={v?.finger_int === i}
                               isCandidate={!blindMode && note.candidates?.some(c =>
                                 ((c.hand === 'Left' ? 0 : 5) + c.finger) === i)}
                               onClick={() => applyFinger(i)} />
                  ))}
                </div>
              </div>
              <div className="finger-half right">
                <div className="hand-label">Right hand · thumb → pinky</div>
                <div className="finger-row">
                  {[6, 7, 8, 9, 10].map(i => (
                    <FingerBtn key={i} intVal={i}
                               isAlgo={!blindMode && note.algorithm_int === i}
                               isImputed={!blindMode && note.imputed_int === i}
                               isCurrentHuman={v?.finger_int === i}
                               isCandidate={!blindMode && note.candidates?.some(c =>
                                 ((c.hand === 'Left' ? 0 : 5) + c.finger) === i)}
                               onClick={() => applyFinger(i)} />
                  ))}
                </div>
              </div>
            </div>

            {/* Action row — three semantic groups: Annotation / History / Navigation */}
            <div className="actions-row">
              <div className="action-group">
                <span className="action-label">Annotation</span>
                <button className="quick-btn unsure"
                        onClick={() => applyFinger(0)}>Unsure</button>
                <button className="quick-btn nopress"
                        onClick={() => applyFinger(-1)}>No press</button>
                {!blindMode && (
                  <button className="quick-btn"
                          onClick={confirmAlgo}
                          disabled={!note.algorithm_int}>
                    ✓ Algo ({algoLabel})
                  </button>
                )}
                <button className="quick-btn"
                        onClick={repeatLastVerdict}
                        disabled={history.length === 0}>↻ Repeat</button>
                <button className="quick-btn"
                        onClick={toggleBookmark}>
                  {isBookmarked ? '★' : '☆'} Bookmark
                </button>
                <button className={`quick-btn ${replayLoop ? 'active' : ''}`}
                        onClick={toggleReplayLoop}>
                  ↻ Replay <kbd>R</kbd>
                </button>
              </div>

              <div className="action-group">
                <span className="action-label">History</span>
                <button className="quick-btn"
                        onClick={undo}
                        disabled={history.length === 0}>
                  Undo <kbd>Ctrl+Z</kbd>
                </button>
                <button className="quick-btn"
                        onClick={redo}
                        disabled={redoStack.length === 0}>
                  Redo <kbd>Ctrl+Y</kbd>
                </button>
              </div>

              <div className="action-group">
                <span className="action-label">Navigation</span>
                <button className="quick-btn" onClick={goPrev}
                        disabled={idx === 0}>← Prev <kbd>A</kbd></button>
                <button className="quick-btn" onClick={goNext}
                        disabled={idx === data.notes.length - 1}>Next → <kbd>D</kbd></button>
                {!blindMode && (
                  <>
                    <button className="quick-btn" onClick={jumpPrevPriority}>prev (priority) <kbd>Q</kbd></button>
                    <button className="quick-btn" onClick={jumpNextPriority}>next (priority) <kbd>E</kbd></button>
                    <button className="quick-btn" onClick={jumpPrevHard}>← Hard</button>
                    <button className="quick-btn" onClick={jumpNextHard}>Hard →</button>
                  </>
                )}
                <button className="quick-btn" onClick={jumpNextUnlabeled}>⇥ unlabeled</button>
                {!blindMode && (
                  <>
                    <button className="quick-btn" onClick={jumpNextAmbiguous}>⚠ ambig</button>
                    <button className="quick-btn" onClick={jumpNextDisagreement}>≠ disagree</button>
                  </>
                )}
                {bookmarks.size > 0 && (
                  <button className="quick-btn" onClick={jumpNextBookmark}>★ bookmark</button>
                )}
              </div>
            </div>

            <div className="legend">
              <span>
                <b>A</b>/<b>D</b> prev/next
                {!blindMode && <> · <b>Q</b>/<b>E</b> prev/next suspect</>}
              </span>
              <span><b>Space</b> play/pause · <b>R</b> replay-loop current note</span>
              <span><b>Ctrl+Z</b>/<b>Ctrl+Y</b> undo/redo · finger select: <b>click</b></span>
            </div>
          </div>
        </div>

      <div className="app-spacer" />

      <PlaybackBar
        videoRef={videoRef}
        currentTime={currentTime}
        duration={duration}
        paused={paused}
        onsetSec={note.onset_sec}
        playbackRate={playbackRate}
        setPlaybackRate={setPlaybackRate} />

      {showBackup && (
        <BackupPanel snapshots={snapshots}
                      onRestore={restoreSnapshot}
                      onClear={clearBackups}
                      onClose={() => setShowBackup(false)} />
      )}

      {showReview && (
        <ReviewPanel data={data}
                      verdicts={verdicts}
                      bookmarks={bookmarks}
                      blindMode={blindMode}
                      onJump={(i) => { setIdx(i); setShowReview(false); }}
                      onClose={() => setShowReview(false)} />
      )}
    </div>
  );
}

// ── Finger button ────────────────────────────────────────────────
function FingerBtn({ intVal, isAlgo, isImputed, isCurrentHuman, isCandidate, onClick }) {
  const cls = [
    'finger-btn',
    isAlgo ? 'algo' : '',
    isImputed && !isAlgo ? 'imputed' : '',
    isCurrentHuman ? 'human' : '',
    isCandidate && !isAlgo && !isImputed ? 'candidate' : '',
  ].filter(Boolean).join(' ');
  return (
    <button className={cls}
            style={{ '--finger-color': FINGER_INT_COLOR[intVal] }}
            onClick={onClick}
            title={`${INT_TO_LABEL[intVal]} (${FINGER_INT_NAME[intVal]})`}>
      <span className="finger-label">{INT_TO_LABEL[intVal]}</span>
    </button>
  );
}

// ── Helpers ──────────────────────────────────────────────────────
/**
 * Per-note review status for the suspect queue.
 * Returns one of: null | 'pending' | 'done'.
 *   null    — note is not in the suspect set (algo/imputer was confident)
 *   pending — suspect that the human has not yet touched
 *   done    — suspect that the human has explicitly verified or overridden
 *
 * "Touched" is detected by the verdict carrying a `timestamp` field — the
 * React app stamps that on every user-pressed verdict; bulk-accepted entries
 * (source: 'algo' / 'imputed') do not carry a timestamp.
 */
function reviewStatus(n, v) {
  const p = priorityForNote(n);
  if (!p) return null;
  return (v && v.timestamp) ? 'done' : 'pending';
}

function computePrioritySummary(data, verdicts) {
  let total = 0, done = 0;
  for (const n of data.notes) {
    const v = verdicts[`${n.trial}#${n.note_idx}`];
    const s = reviewStatus(n, v);
    if (s) {
      total++;
      if (s === 'done') done++;
    }
  }
  return { total, done, remaining: total - done,
           pct: total ? (100 * done / total) : 100 };
}

// Stats below are computed over EXPLICIT human verdicts only
// (classifyVerdict(v).human). Bulk-accepted preloads (source: 'algo' /
// 'imputed') are excluded so the algorithm is never compared against
// itself — that would inflate any "agreement" metric trivially.
function computeSummary(data, verdicts) {
  let n_human = 0;
  let algo_correct = 0, algo_decided = 0;
  for (const n of data.notes) {
    const v = classifyVerdict(verdicts[`${n.trial}#${n.note_idx}`]).human;
    if (!v) continue;
    n_human++;
    if (n.algorithm_status === 'labeled' && n.algorithm_int
        && v.finger_int >= 1 && v.finger_int <= 10) {
      algo_decided++;
      if (v.finger_int === n.algorithm_int) algo_correct++;
    }
  }
  return {
    n_human_verdicts: n_human, algo_correct, algo_decided,
    algo_correct_pct: algo_decided ? (100 * algo_correct / algo_decided).toFixed(1) : '—',
  };
}
// ── Review queue modal ───────────────────────────────────────────
//
// Working assumption: the algorithm's labels are mostly right, so the
// human's job is to find the small minority that needs fixing — and within
// that minority, focus on what's *most likely* wrong first.
//
// "Suspect" = a bulk-accepted note that the algorithm or imputer was unsure
// about. After the upstream bulk-accept, every note has a verdict — the
// suspect queue is the curated subset worth a human spot-check. A note stops
// being suspect once the human presses any verdict button (the React app
// stamps `timestamp` on those entries; bulk-accepted entries do not).
// Suspect-queue thresholds. Tuned together — bumping one without checking the
// others can change which notes show up first in the review panel.
const SUSPECT = {
  // algorithm_score below this is "weak pick" territory, top-priority verify
  ALGO_LOW_SCORE: 0.65,
  // close-2nd-candidate trigger: 2nd is plausible AND top-2 are within margin
  ALGO_2ND_MIN_SCORE: 0.55,
  ALGO_2ND_MAX_GAP: 0.15,
  // sequence-model imputed picks; thresholds differ by upstream ambiguity tier
  IMPUTED_MULTI_CAND_BELOW: 0.70,
  IMPUTED_NO_CAND_BELOW: 0.50,
  // Priority bucket scores. Order = sort priority in the review queue.
  // Higher = surfaced first.  no-cand imputed (75) ranks above close-2nd (70)
  // because no-candidate notes are by definition where the algorithm gave up.
  PRIO_ALGO_LOW: 80,
  PRIO_NO_CAND_IMPUTED: 75,
  PRIO_CLOSE_2ND: 70,
  PRIO_MULTI_CAND_IMPUTED: 65,
};

const HARD_RULE_PRIORITY = {
  physical_candidate_diagnostic: 100,
  impossible_fingering: 100,
  noinfo_context_k3_r2: 96,
  noinfo_cluster: 94,
  fast_jump: 92,
  hand_overlap: 88,
  rapid_alternation: 84,
  stepwise_order_violation: 78,
  noinfo: 72,
};

function hasManualPriority(data) {
  return !!data?.notes?.some(isAuditNote);
}

function reviewedLabel(v) {
  if (!v) return 'annotation';
  if (v.finger_int >= 1 && v.finger_int <= 10) return INT_TO_LABEL[v.finger_int] || v.finger_label;
  return v.finger_label || 'annotation';
}

function reviewedDetail(note, v, priority) {
  const picked = reviewedLabel(v);
  if (priority) {
    if (note.algorithm_int) {
      if (v.finger_int === note.algorithm_int) {
        return `kept algo pick (${priority.reason})`;
      }
      return `overrode algo to ${picked} (${priority.reason})`;
    }
    return `annotated ${picked} (${priority.reason})`;
  }
  if (note.algorithm_int) {
    if (v.finger_int === note.algorithm_int) return 'agrees with algorithm';
    return `differs from algorithm (algo ${INT_TO_LABEL[note.algorithm_int]})`;
  }
  return `you picked ${picked}`;
}

function priorityForNote(n) {
  if (!n) return null;
  const auditPriority = explicitAuditPriority(n);
  if (auditPriority) return auditPriority;
  const hardReasons = hardReasonsForNote(n);
  if (n.is_hard || hardReasons.length > 0) {
    const score = typeof n.review_priority === 'number'
      ? n.review_priority
      : hardReasons.reduce((best, reason) =>
          Math.max(best, HARD_RULE_PRIORITY[reason] ?? 70), 70);
    const reason = n.review_reason
      || (hardReasons.length > 0
        ? `rule-based hard note: ${hardReasons.join(', ')}`
        : 'rule-based hard note');
    return { score, reason };
  }
  // Algo-tier weak picks (algorithm gave a label but with low confidence).
  if (n.algorithm_status === 'labeled') {
    if (typeof n.algorithm_score === 'number' && n.algorithm_score < SUSPECT.ALGO_LOW_SCORE) {
      return { score: SUSPECT.PRIO_ALGO_LOW, reason: `low algo score (${n.algorithm_score.toFixed(2)})` };
    }
    if (n.candidates && n.candidates.length >= 2) {
      const top = n.candidates[0]?.score ?? 1;
      const second = n.candidates[1]?.score ?? 0;
      if (second > SUSPECT.ALGO_2ND_MIN_SCORE && top - second < SUSPECT.ALGO_2ND_MAX_GAP) {
        const sLabel = INT_TO_LABEL[(n.candidates[1].hand === 'Left' ? 0 : 5) + n.candidates[1].finger];
        return { score: SUSPECT.PRIO_CLOSE_2ND, reason: `close 2nd: ${sLabel}=${second.toFixed(2)}` };
      }
    }
    return null;
  }
  // Imputed-tier low-confidence picks.
  if (n.imputed_int && typeof n.imputed_score === 'number') {
    if (n.algorithm_status === 'multi_candidate' && n.imputed_score < SUSPECT.IMPUTED_MULTI_CAND_BELOW) {
      return { score: SUSPECT.PRIO_MULTI_CAND_IMPUTED, reason: `multi-cand imputed (${n.imputed_score.toFixed(2)})` };
    }
    if (n.algorithm_status === 'no_candidate' && n.imputed_score < SUSPECT.IMPUTED_NO_CAND_BELOW) {
      return { score: SUSPECT.PRIO_NO_CAND_IMPUTED, reason: `no-cand imputed (${n.imputed_score.toFixed(2)})` };
    }
  }
  return null;
}

function ReviewPanel({ data, verdicts, bookmarks, blindMode, onJump, onClose }) {
  const manualPriority = hasManualPriority(data);
  // In blindMode, skip the algo-derived 'priority' tab and start on 'unlabeled'
  // — every "Suspect / Disagree / Ambiguous" view inevitably surfaces the
  // algorithm's pick, defeating the purpose of blind annotation.
  const [activeTab, setActiveTab] = React.useState(blindMode ? 'unlabeled' : 'priority');
  const [hideDone, setHideDone] = React.useState(true);

  const buckets = React.useMemo(() => {
    const priority = [];
    const unlabeled = [], disagree = [], ambiguous = [], unsure = [], booked = [];
    for (let i = 0; i < data.notes.length; i++) {
      const n = data.notes[i];
      const id = `${n.trial}#${n.note_idx}`;
      const v = verdicts[id];
      // Only EXPLICIT human picks count for the review buckets below.
      // Bulk-accepted preloads (source: 'algo' / 'imputed') are routed
      // through the priority queue, never directly into disagree/ambiguous.
      const human = classifyVerdict(v).human;
      const human_touched = !!human;

      const p = priorityForNote(n);
      if (p && (!hideDone || !human_touched)) {
        priority.push({ i, n, v, priority: p.score, reason: p.reason });
      }

      if (bookmarks.has(id)) booked.push({ i, n, v });
      if (!human) { unlabeled.push({ i, n, v }); continue; }
      if (human.finger_int === 0 || human.finger_int === -1) { unsure.push({ i, n, v: human }); continue; }
      if (n.algorithm_status !== 'labeled') { ambiguous.push({ i, n, v: human }); continue; }
      if (n.algorithm_int && human.finger_int !== n.algorithm_int) disagree.push({ i, n, v: human });
    }
    priority.sort((a, b) => b.priority - a.priority || a.i - b.i);
    return { priority, unlabeled, disagree, ambiguous, unsure, booked };
  }, [data, verdicts, bookmarks, hideDone]);

  const allTabs = [
    { id: 'priority',  label: manualPriority ? '🕵 Hard notes' : '🕵 Suspect', items: buckets.priority,
      hint: manualPriority
        ? 'Rule-based hard notes from ManualCheck — review top to bottom'
        : 'Notes the algorithm/imputer was unsure about — review top to bottom',
      algo: true },
    { id: 'disagree',  label: '≠ Algo',     items: buckets.disagree,
      hint: 'Notes where your label differs from the algorithm',
      algo: true },
    { id: 'ambiguous', label: 'Ambiguous',  items: buckets.ambiguous,
      hint: 'Algorithm gave multi-/no-candidate; you labeled them',
      algo: true },
    { id: 'unsure',    label: 'Unsure',     items: buckets.unsure,
      hint: 'Notes marked 0 (unsure) or -1 (no press)' },
    { id: 'booked',    label: '★ Bookmark', items: buckets.booked,
      hint: 'Bookmarked notes' },
    { id: 'unlabeled', label: 'Unlabeled',  items: buckets.unlabeled,
      hint: 'Not yet labeled' },
  ];
  const tabs = blindMode ? allTabs.filter(t => !t.algo) : allTabs;
  const active = tabs.find(t => t.id === activeTab) || tabs[0];
  const isPriority = active.id === 'priority';

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal review-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Review queue</h3>
          <button className="btn-secondary" onClick={onClose}>Close</button>
        </div>
        <div className="review-tabs">
          {tabs.map(t => (
            <button key={t.id}
                    className={`review-tab ${activeTab === t.id ? 'active' : ''}`}
                    onClick={() => setActiveTab(t.id)}>
              {t.label} <span className="review-count">{t.items.length}</span>
            </button>
          ))}
        </div>
        <div className="modal-body">
          <div className="review-hint">
            {active.hint}
            {isPriority && (
              <label style={{ marginLeft: 12, fontSize: 11, cursor: 'pointer' }}>
                <input type="checkbox" checked={hideDone}
                       onChange={e => setHideDone(e.target.checked)}
                       style={{ marginRight: 4 }} />
                hide already-annotated
              </label>
            )}
          </div>
          {active.items.length === 0 ? (
            <p style={{ color: 'var(--text-muted)' }}>None — no notes in this category need review.</p>
          ) : (
            <table className="review-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Diff</th>
                  <th>Pitch</th>
                  <th>Onset</th>
                  {!blindMode && <th>Algo</th>}
                  <th>You</th>
                  {isPriority && !blindMode && <th>Reason</th>}
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {active.items.slice(0, 300).map(({ i, n, v, priority, reason }) => {
                  const algoLbl = n.algorithm_int
                    ? INT_TO_LABEL[n.algorithm_int]
                    : (n.algorithm_status === 'labeled' ? '—' : n.algorithm_status);
                  const humanV = classifyVerdict(v).human;
                  const youLbl = humanV?.finger_label || '—';
                  return (
                    <tr key={i}>
                      <td className="muted">{i + 1}</td>
                      <td className="diff-tag-inline">{n.difficulty?.[0] || '?'}</td>
                      <td>{n.pitch_name}</td>
                      <td className="muted">{n.onset_sec.toFixed(2)}s</td>
                      {!blindMode && (
                        <td className={n.algorithm_status === 'labeled'
                          ? 'algo-cell' : 'algo-cell ambig'}>{algoLbl}</td>
                      )}
                      <td className={!blindMode && humanV && humanV.finger_int >= 1 && humanV.finger_int <= 10
                        && humanV.finger_int !== n.algorithm_int
                        ? 'human-cell mismatch' : 'human-cell'}>{youLbl}</td>
                      {isPriority && !blindMode && (
                        <td className="reason-cell">
                          <span className={`prio-pill prio-${
                            priority >= 90 ? 'hot' : priority >= 70 ? 'warm' : 'cool'}`}>
                            {priority}
                          </span>
                          <span className="reason-text">{reason}</span>
                        </td>
                      )}
                      <td><button className="btn-secondary"
                                  onClick={() => onJump(i)}>Jump</button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {active.items.length > 300 && (
            <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>
              ... and {active.items.length - 300} more (showing first 300)
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Backup panel modal ───────────────────────────────────────────
function BackupPanel({ snapshots, onRestore, onClear, onClose }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Backup snapshots ({snapshots.length})</h3>
          <button className="btn-secondary" onClick={onClose}>Close</button>
        </div>
        <div className="modal-body">
          {snapshots.length === 0
            ? <p style={{ color: 'var(--text-muted)' }}>No snapshots yet. They are created automatically as you annotate.</p>
            : (
              <table className="backup-table">
                <thead><tr><th>Time</th><th>Entries</th><th></th></tr></thead>
                <tbody>
                  {snapshots.map(s => (
                    <tr key={s.id}>
                      <td>{new Date(s.ts).toLocaleString()}</td>
                      <td>{s.n_verdicts}</td>
                      <td><button className="btn-secondary"
                                  onClick={() => onRestore(s.id)}>Restore</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </div>
        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClear}>Delete all snapshots</button>
        </div>
      </div>
    </div>
  );
}

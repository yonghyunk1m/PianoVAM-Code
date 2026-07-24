import React, { useEffect, useMemo, useRef, useState } from 'react';
import HardPianoRoll from './HardPianoRoll.jsx';
import PlaybackBar from './PlaybackBar.jsx';
import { FINGER_INT_COLOR, FINGER_INT_NAME, INT_TO_LABEL } from './fingerPalette.js';
import {
  auditCategoryForNote,
  hardReasonsForNote,
  isAuditNote,
} from './auditCategories.js';

// Share verdict storage with the main app so labels persist across modes.
const _urlTrial = new URLSearchParams(window.location.search).get('trial') || 'default';
const VERDICT_KEY = `fingering_verdicts_v2_${_urlTrial}`;
const HISTORY_KEY = `fingering_history_hard_v1_${_urlTrial}`;
const NOTES_URL   = _urlTrial !== 'default' ? `/data/${_urlTrial}.json` : '/data/notes.json';
const HISTORY_LIMIT = 200;

// Build normal-mode URL (strips ?mode=hard, keeps other params)
const _normalModeUrl = (() => {
  const p = new URLSearchParams(window.location.search);
  p.delete('mode');
  const qs = p.toString();
  return window.location.pathname + (qs ? '?' + qs : '');
})();

function loadJSON(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
}

function classifyVerdict(v) {
  if (!v) return { human: null };
  if (v.timestamp && !v.source) return { human: v };
  return { human: null };
}

export default function HardNoteApp() {
  const [data, setData]         = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [idx, setIdx]           = useState(0);
  const [verdicts, setVerdicts] = useState({});
  const [history, setHistory]   = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const verdictsBootRef         = useRef(false);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration]       = useState(NaN);
  const [paused, setPaused]           = useState(true);
  const [playbackRate, setPlaybackRate] = useState(0.5);
  const [replayLoop, setReplayLoop]   = useState(false);
  const REPLAY_LEAD_SEC = 0.03;
  const REPLAY_TAIL_SEC = 0.03;

  const videoRef  = useRef(null);
  const audioRef  = useRef(null);
  const pausedRef = useRef(true);
  // Always-current note ref so event listeners don't capture stale state.
  const noteRef   = useRef(null);

  useEffect(() => { pausedRef.current = paused; }, [paused]);

  // ── Load notes ──
  useEffect(() => {
    fetch(NOTES_URL)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setData)
      .catch(err => setLoadError(String(err.message || err)));
  }, []);

  // ── Load verdicts from server ──
  useEffect(() => {
    if (!data) return;
    verdictsBootRef.current = false;
    fetch('/api/human-verdicts')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(sv => {
        verdictsBootRef.current = true;
        setVerdicts((sv && Object.keys(sv).length > 0) ? sv : loadJSON(VERDICT_KEY, {}));
      })
      .catch(() => {
        setVerdicts(loadJSON(VERDICT_KEY, {}));
        verdictsBootRef.current = true;
      });
  }, [data]);

  // ── Jump to first hard note once data+verdicts are ready ──
  useEffect(() => {
    if (!data) return;
    const first = data.notes.findIndex(isAuditNote);
    if (first >= 0) setIdx(first);
  }, [data]);

  // ── Persist verdicts ──
  const saveTimerRef = useRef(null);
  useEffect(() => {
    if (!data) return;
    localStorage.setItem(VERDICT_KEY, JSON.stringify(verdicts));
    if (!verdictsBootRef.current) return;
    clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      fetch('/api/human-verdicts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(verdicts),
      }).catch(() => {});
    }, 300);
  }, [verdicts]);

  useEffect(() => {
    if (!data) return;
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-HISTORY_LIMIT)));
  }, [history]);

  const note = data?.notes[idx];
  useEffect(() => { noteRef.current = note; }, [note]);

  const trialNotes = useMemo(
    () => (data && note ? data.notes.filter(n => n.trial === note.trial) : []),
    [data, note?.trial]);

  const hardNotes = useMemo(
    () => (data ? data.notes.filter(isAuditNote) : []),
    [data]);

  const hardStats = useMemo(() => {
    if (!data) return { total: 0, done: 0 };
    let total = 0, done = 0;
    for (const n of data.notes) {
      if (!isAuditNote(n)) continue;
      total++;
      const v = verdicts[`${n.trial}#${n.note_idx}`];
      if (v && v.timestamp && !v.source) done++;
    }
    return { total, done };
  }, [data, verdicts]);

  // ── Snap video + audio to onset when note changes (while paused) ──
  useEffect(() => {
    if (!note || !pausedRef.current) return;
    [videoRef.current, audioRef.current].forEach(el => {
      if (!el) return;
      const snap = () => {
        el.currentTime  = Math.max(0, note.onset_sec);
        el.playbackRate = playbackRate;
        el.pause();
      };
      if (el.readyState >= 1) snap();
      else el.addEventListener('loadedmetadata', snap, { once: true });
    });
  }, [note?.trial, note?.onset_sec, playbackRate]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
    if (audioRef.current) audioRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  // ── Video event listeners — snap back to onset on pause ──
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime  = () => setCurrentTime(v.currentTime);
    const onDur   = () => setDuration(v.duration);
    const onPlay  = () => { setPaused(false); audioRef.current?.play().catch(() => {}); };
    const onPause = () => {
      setPaused(true);
      audioRef.current?.pause();
      // Key behavior: snap back to onset when user pauses.
      const cur = noteRef.current;
      if (cur) {
        const t = Math.max(0, cur.onset_sec);
        v.currentTime = t;
        if (audioRef.current) audioRef.current.currentTime = t;
        setCurrentTime(t);
      }
    };
    const onSeek = () => {
      setCurrentTime(v.currentTime);
      if (audioRef.current) audioRef.current.currentTime = v.currentTime;
    };
    v.addEventListener('timeupdate',      onTime);
    v.addEventListener('loadedmetadata',  onDur);
    v.addEventListener('durationchange',  onDur);
    v.addEventListener('play',            onPlay);
    v.addEventListener('pause',           onPause);
    v.addEventListener('seeked',          onSeek);
    return () => {
      v.removeEventListener('timeupdate',     onTime);
      v.removeEventListener('loadedmetadata', onDur);
      v.removeEventListener('durationchange', onDur);
      v.removeEventListener('play',           onPlay);
      v.removeEventListener('pause',          onPause);
      v.removeEventListener('seeked',         onSeek);
    };
  }, [note?.trial]);

  // ── 60 Hz RAF while playing ──
  useEffect(() => {
    if (paused) return;
    let raf;
    const tick = () => {
      if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [paused]);

  // ── Replay loop ──
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

  // Cancel replay loop on note change.
  useEffect(() => {
    if (replayLoop) {
      videoRef.current?.pause();
      audioRef.current?.pause();
      setReplayLoop(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [note?.trial, note?.note_idx]);

  // Video stays muted; audio is the quality source.
  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = true;
    if (audioRef.current) audioRef.current.muted = false;
  }, [note?.trial]);

  // ── Navigation ──
  function jumpNextHard() {
    if (!data) return;
    for (let i = idx + 1; i < data.notes.length; i++) {
      if (isAuditNote(data.notes[i])) { setIdx(i); return; }
    }
    for (let i = 0; i < idx; i++) {
      if (isAuditNote(data.notes[i])) { setIdx(i); return; }
    }
  }
  function jumpPrevHard() {
    if (!data) return;
    for (let i = idx - 1; i >= 0; i--) {
      if (isAuditNote(data.notes[i])) { setIdx(i); return; }
    }
    for (let i = data.notes.length - 1; i > idx; i--) {
      if (isAuditNote(data.notes[i])) { setIdx(i); return; }
    }
  }

  // ── Verdicts ──
  function applyFinger(intVal) {
    if (!note) return;
    const startIdx = data.notes.indexOf(note);
    const replaced = [verdicts[`${note.trial}#${note.note_idx}`] || null];
    setHistory(h => [...h, { startIdx, endIdx: startIdx + 1, replaced }]);
    setRedoStack([]);
    setVerdicts(prev => ({
      ...prev,
      [`${note.trial}#${note.note_idx}`]: {
        finger_int:       intVal,
        finger_label:     intVal === 0 ? 'unsure' : intVal === -1 ? 'no_press' : INT_TO_LABEL[intVal],
        algorithm_int:    note.algorithm_int,
        algorithm_status: note.algorithm_status,
        timestamp:        Date.now(),
      },
    }));
    // Auto-advance to next hard note after each verdict.
    jumpNextHard();
  }

  function undo() {
    if (history.length === 0) return;
    const last = history[history.length - 1];
    const n    = data.notes[last.startIdx];
    const id   = `${n.trial}#${n.note_idx}`;
    const after = [verdicts[id] || null];
    setRedoStack(r => [...r, { ...last, after }]);
    setVerdicts(prev => {
      const next = { ...prev };
      if (last.replaced[0] === null) delete next[id];
      else next[id] = last.replaced[0];
      return next;
    });
    setHistory(h => h.slice(0, -1));
    setIdx(last.startIdx);
  }

  function redo() {
    if (redoStack.length === 0) return;
    const act = redoStack[redoStack.length - 1];
    const n   = data.notes[act.startIdx];
    const id  = `${n.trial}#${n.note_idx}`;
    setHistory(h => [...h, { startIdx: act.startIdx, endIdx: act.endIdx, replaced: act.replaced }]);
    setVerdicts(prev => {
      const out = { ...prev };
      if (act.after[0] === null) delete out[id];
      else out[id] = act.after[0];
      return out;
    });
    setRedoStack(r => r.slice(0, -1));
    setIdx(act.startIdx);
  }

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

  // ── Keyboard shortcuts (A/D = hard prev/next, R = replay, Space = play/pause) ──
  const keyHandlerRef = useRef(null);
  keyHandlerRef.current = (e) => {
    if (e.target.tagName === 'INPUT') return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
    if ((e.ctrlKey || e.metaKey) && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return; }
    if (e.ctrlKey || e.metaKey) return;
    if      (k === 'a')    { e.preventDefault(); jumpPrevHard(); }
    else if (k === 'd')    { e.preventDefault(); jumpNextHard(); }
    else if (k === 'r')    { e.preventDefault(); toggleReplayLoop(); }
    else if (e.key === ' ') {
      e.preventDefault();
      const v = videoRef.current;
      if (v) { if (v.paused) v.play(); else v.pause(); }
    }
  };
  useEffect(() => {
    const h = e => keyHandlerRef.current?.(e);
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);

  // ── Render guards ──
  if (loadError) return (
    <div style={{ padding: 40, maxWidth: 640 }}>
      <h2 style={{ margin: '0 0 12px' }}>Cannot load notes</h2>
      <p><code>/data/notes.json</code> failed ({loadError}).</p>
      <p>Run <code>prepare_review_data.py</code> and reload.</p>
    </div>
  );
  if (!data || !note) return <div style={{ padding: 40 }}>Loading…</div>;
  if (!isAuditNote(note)) {
    // Current idx isn't an audit note (shouldn't happen after init, but guard it).
    return <div style={{ padding: 40 }}>No audit notes found in this trial.</div>;
  }

  const v          = verdicts[`${note.trial}#${note.note_idx}`];
  const humanV     = classifyVerdict(v).human;
  const trialLocalIdx = trialNotes.findIndex(n => n.global_idx === note.global_idx);
  const hardLocalIdx  = hardNotes.findIndex(n => n.global_idx === note.global_idx);
  const donePct    = hardStats.total ? Math.round(100 * hardStats.done / hardStats.total) : 0;
  const auditCategory = auditCategoryForNote(note);
  const legacyReasons = hardReasonsForNote(note);
  const auditDetail = auditCategory
    ? `${auditCategory.label}${
        auditCategory.reasons.length > 0
          ? ` — ${auditCategory.reasons.join(', ')}`
          : ''
      }`
    : legacyReasons.join(', ');

  return (
    <div className="app">
      {/* ── Header ── */}
      <div className="header">
        <h1>Hard Note Annotator</h1>

        <span className="progress" title="Hard note position">
          {hardLocalIdx + 1} / {hardStats.total}
        </span>

        {/* Progress bar */}
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          background: 'rgba(211,47,47,0.08)', color: '#D32F2F',
          padding: '4px 10px', borderRadius: 6,
          fontFamily: 'ui-monospace, SF Mono, monospace', fontSize: 12, fontWeight: 700,
        }}>
          🔴 {hardStats.done}/{hardStats.total} reviewed
          <span style={{
            display: 'inline-block', width: 80, height: 6,
            background: 'rgba(211,47,47,0.18)', borderRadius: 3, overflow: 'hidden',
          }}>
            <span style={{
              display: 'block', height: '100%', borderRadius: 3,
              background: '#D32F2F', width: `${donePct}%`, transition: 'width 0.2s',
            }} />
          </span>
        </span>

        <span className="meta">
          <b>{note.piece || note.trial}</b>
          {' · '}{note.pitch_name} (MIDI {note.pitch})
          {' · '}onset {note.onset_sec.toFixed(3)} s
          {auditDetail && (
            <span style={{ marginLeft: 8, color: '#D32F2F', fontWeight: 700 }}>
              🔴 {auditDetail}
            </span>
          )}
        </span>

        <a href={_normalModeUrl}
           style={{
             marginLeft: 'auto', padding: '7px 14px',
             background: '#fff', color: '#0F0F70',
             border: '1px solid rgba(15,15,112,0.24)',
             borderRadius: 6, fontSize: 12, fontWeight: 600,
             textDecoration: 'none', cursor: 'pointer',
           }}>
          ← Normal mode
        </a>
      </div>

      {/* ── Main: video + piano roll ── */}
      <div className="main">
        <div className="panel video">
          <div className="video-wrap">
            <div className="video-stage">
              <div className="video-frame">
                <video ref={videoRef} src={note.video_path} key={note.trial} />
              </div>
            </div>
            <audio ref={audioRef} src={note.audio_path}
                   key={`${note.trial}_audio`} preload="auto" />
          </div>
        </div>

        <div className="panel roll-panel">
          <div className="roll-wrap">
            <HardPianoRoll
              trialNotes={trialNotes}
              currentNoteIdx={trialLocalIdx}
              verdicts={verdicts}
              currentTime={currentTime}
              isPlaying={!paused}
              onSeek={t => { if (videoRef.current) videoRef.current.currentTime = t; }}
              onSelectNote={gidx => {
                const target = data.notes.findIndex(n => n.global_idx === gidx);
                if (target >= 0 && isAuditNote(data.notes[target])) setIdx(target);
              }}
            />
            <div className="roll-info">
              {trialLocalIdx + 1} / {trialNotes.length} in trial
              {' · '}
              <span style={{ color: '#90CAF9', fontWeight: 700 }}>light blue = unlabeled</span>
              {' · '}
              <span style={{ color: '#1565C0', fontWeight: 700 }}>blue = agrees</span>
              {' · '}
              <span style={{ color: '#E65100', fontWeight: 700 }}>orange = disagrees</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Annotate panel ── */}
      <div className="annotate-wrap">
        <div className="panel action">

          {/* Status banner */}
          <div className={`review-status review-status-${humanV ? 'confirmed' : 'pending'}`}>
            <b>{humanV
              ? `✓ ${INT_TO_LABEL[humanV.finger_int] ?? humanV.finger_label}`
              : '○ not yet annotated'
            }</b>
            {note.review_reason && <> — {note.review_reason}</>}
          </div>

          {/* Finger grid */}
          <div className="finger-grid">
            <div className="finger-half left">
              <div className="hand-label">Left hand · pinky → thumb</div>
              <div className="finger-row">
                {[5, 4, 3, 2, 1].map(i => (
                  <FingerBtn key={i} intVal={i}
                    isAlgo={note.algorithm_int === i}
                    isCurrentHuman={v?.finger_int === i}
                    onClick={() => applyFinger(i)} />
                ))}
              </div>
            </div>
            <div className="finger-half right">
              <div className="hand-label">Right hand · thumb → pinky</div>
              <div className="finger-row">
                {[6, 7, 8, 9, 10].map(i => (
                  <FingerBtn key={i} intVal={i}
                    isAlgo={note.algorithm_int === i}
                    isCurrentHuman={v?.finger_int === i}
                    onClick={() => applyFinger(i)} />
                ))}
              </div>
            </div>
          </div>

          {/* Action row */}
          <div className="actions-row">
            <div className="action-group">
              <span className="action-label">Annotation</span>
              <button className="quick-btn unsure" onClick={() => applyFinger(0)}>Unsure</button>
              <button className="quick-btn nopress" onClick={() => applyFinger(-1)}>No press</button>
              <button className={`quick-btn ${replayLoop ? 'active' : ''}`}
                      onClick={toggleReplayLoop}>
                ↻ Replay <kbd>R</kbd>
              </button>
            </div>
            <div className="action-group">
              <span className="action-label">History</span>
              <button className="quick-btn" onClick={undo}
                      disabled={history.length === 0}>Undo <kbd>Ctrl+Z</kbd></button>
              <button className="quick-btn" onClick={redo}
                      disabled={redoStack.length === 0}>Redo <kbd>Ctrl+Y</kbd></button>
            </div>
            <div className="action-group">
              <span className="action-label">Navigation</span>
              <button className="quick-btn" onClick={jumpPrevHard}>← Hard <kbd>A</kbd></button>
              <button className="quick-btn" onClick={jumpNextHard}>Hard → <kbd>D</kbd></button>
            </div>
          </div>

          <div className="legend">
            <span><b>A</b>/<b>D</b> prev/next hard note (skips all others)</span>
            <span><b>Space</b> play · pause snaps back to onset · <b>R</b> replay loop</span>
            <span><b>Ctrl+Z</b>/<b>Ctrl+Y</b> undo/redo · finger: click button → auto-advance</span>
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
    </div>
  );
}

function FingerBtn({ intVal, isAlgo, isCurrentHuman, onClick }) {
  const cls = [
    'finger-btn',
    isAlgo        ? 'algo'  : '',
    isCurrentHuman ? 'human' : '',
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

import React, { useRef, useState } from 'react';

function fmt(t) {
  if (!isFinite(t)) return '0:00.000';
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(3).padStart(6, '0')}`;
}

const PLAYBACK_RATES = [0.25, 0.5, 1.0, 2.0];

/**
 * Custom video playback bar with onset marker. Replaces the default HTML5
 * controls so we can pin a "this note's onset" indicator on the timeline
 * and offer fine-grained scrub/jump that matches an annotator's workflow.
 *
 * Interaction:
 *   - click anywhere on the track to seek
 *   - press + drag the track to scrub continuously
 *   - mouse-wheel over the track for fine ±0.05 s scrub
 *   - hover shows a faint preview cursor + tooltip with that timestamp
 */
export default function PlaybackBar({ videoRef, currentTime, duration, paused,
                                       onsetSec, playbackRate, setPlaybackRate }) {
  const trackRef = useRef(null);
  const [hoverT, setHoverT] = useState(null);
  const [dragging, setDragging] = useState(false);

  const dur = isFinite(duration) && duration > 0 ? duration : 1;
  const progressPct = (currentTime / dur) * 100;
  const onsetPct = isFinite(onsetSec) ? (onsetSec / dur) * 100 : 0;
  const hoverPct = hoverT !== null ? (hoverT / dur) * 100 : null;

  const seekTo = (t) => {
    const v = videoRef.current;
    if (!v || !isFinite(duration)) return;
    v.currentTime = Math.max(0, Math.min(duration, t));
  };
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play(); else v.pause();
  };
  const jumpToOnset = () => seekTo(onsetSec);

  const tFromEvent = (e) => {
    const r = trackRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(r.width, e.clientX - r.left));
    return (x / r.width) * dur;
  };

  const onPointerDown = (e) => {
    if (!trackRef.current) return;
    setDragging(true);
    seekTo(tFromEvent(e));
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!trackRef.current) return;
    setHoverT(tFromEvent(e));
    if (dragging) seekTo(tFromEvent(e));
  };
  const onPointerUp = (e) => {
    setDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };
  const onPointerLeave = () => {
    setHoverT(null);
    if (!dragging) return;   // a drag ending outside still releases via pointerup
  };
  const onWheel = (e) => {
    if (!trackRef.current) return;
    e.preventDefault();
    // Wheel down = forward, wheel up = backward.
    // Per-tick = 0.05s by default; 0.5s when shift is held.
    const step = e.shiftKey ? 0.5 : 0.05;
    const dir = e.deltaY > 0 ? 1 : -1;
    seekTo(currentTime + dir * step);
  };

  return (
    <div className="playback">
      <div className="pb-controls">
        <button className="pb-btn play" onClick={togglePlay} title="Space">
          {paused ? '▶' : '❚❚'}
        </button>
        <button className="pb-btn jump" onClick={jumpToOnset} title="Jump to onset (J)">
          ↧ onset
        </button>
        <button className="pb-btn" onClick={() => seekTo(currentTime - 1.0)} title="−1s">−1s</button>
        <button className="pb-btn" onClick={() => seekTo(currentTime - 0.25)} title="−0.25s">−0.25s</button>
        <button className="pb-btn" onClick={() => seekTo(currentTime - 1 / 60)} title="−1 frame">−1f</button>
        <button className="pb-btn" onClick={() => seekTo(currentTime + 1 / 60)} title="+1 frame">+1f</button>
        <button className="pb-btn" onClick={() => seekTo(currentTime + 0.25)} title="+0.25s">+0.25s</button>
        <button className="pb-btn" onClick={() => seekTo(currentTime + 1.0)} title="+1s">+1s</button>

        <div className="pb-rate-group">
          {PLAYBACK_RATES.map(r => (
            <button key={r}
                    className={`pb-btn ${r === playbackRate ? 'active' : ''}`}
                    onClick={() => setPlaybackRate(r)}>
              {r}×
            </button>
          ))}
        </div>

        <span className="pb-time">
          {fmt(currentTime)} / {fmt(duration)}
          {isFinite(onsetSec) && (
            <span style={{ color: 'var(--error)', marginLeft: 8 }}>
              onset @ {fmt(onsetSec)}
            </span>
          )}
        </span>
      </div>

      <div className={`pb-track ${dragging ? 'dragging' : ''}`}
           ref={trackRef}
           onPointerDown={onPointerDown}
           onPointerMove={onPointerMove}
           onPointerUp={onPointerUp}
           onPointerLeave={onPointerLeave}
           onWheel={onWheel}>
        <div className="pb-track-fill" style={{ width: `${progressPct}%` }} />
        {isFinite(onsetSec) && onsetSec >= 0 && onsetSec <= dur && (
          <div className="pb-track-onset" style={{ left: `${onsetPct}%` }} />
        )}
        {hoverPct !== null && (
          <div className="pb-track-hover" style={{ left: `${hoverPct}%` }}>
            <span className="pb-hover-time">{fmt(hoverT)}</span>
          </div>
        )}
        <div className="pb-track-cursor" style={{ left: `${progressPct}%` }} />
      </div>
    </div>
  );
}

import React, { useRef, useEffect, useMemo } from 'react';

const BLACK_KEY_PC = new Set([1, 3, 6, 8, 10]);

const HEAD_FROM_BOTTOM = 0.18;
const LOOK_AHEAD_SEC   = 4.5;
const LOOK_BACK_SEC    = 1.0;

// Hard-mode palette
const CURRENT_FILL         = '#D32F2F';
const HARD_AGREE_FILL      = '#1565C0';   // hard + human matches algo  → solid blue
const HARD_DISAGREE_FILL   = '#E65100';   // hard + human differs algo  → orange
const HARD_PENDING_FILL    = '#90CAF9';   // hard + no human verdict    → light blue
const OTHER_FILL           = '#D8D8E0';   // non-hard notes             → grey
const PITCH_COL_W          = 56;          // left column for pitch labels

function isHard(n) {
  return n.is_hard || (Array.isArray(n.hard_reasons) && n.hard_reasons.length > 0);
}

export default function HardPianoRoll({
  trialNotes, currentNoteIdx, verdicts = {},
  currentTime, isPlaying, onSeek, onSelectNote,
}) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);

  const pitchBounds = useMemo(() => {
    if (!trialNotes?.length) return { pMin: 60, pMax: 72 };
    const ps = trialNotes.map(n => n.pitch);
    return { pMin: Math.min(...ps) - 2, pMax: Math.max(...ps) + 2 };
  }, [trialNotes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#FAFAFB';
    ctx.fillRect(0, 0, W, H);

    if (!trialNotes?.length) return;
    const cur = trialNotes[currentNoteIdx];
    if (!cur) return;

    const margin = { l: PITCH_COL_W + 6, r: 8, t: 6, b: 22 };
    const plotW  = W - margin.l - margin.r;
    const plotH  = H - margin.t - margin.b;
    const headY  = margin.t + plotH * (1 - HEAD_FROM_BOTTOM);

    const tCenter = (isPlaying && isFinite(currentTime)) ? currentTime : cur.onset_sec;
    const pps     = plotH / (LOOK_AHEAD_SEC + LOOK_BACK_SEC);
    const tTop    = tCenter + LOOK_AHEAD_SEC;
    const tBot    = tCenter - LOOK_BACK_SEC;
    const yOf     = t => headY - (t - tCenter) * pps;

    const { pMin, pMax } = pitchBounds;
    const slotW = plotW / (pMax - pMin + 1);
    const xOf   = p => margin.l + (p - pMin + 0.5) * slotW;

    layoutRef.current = { tCenter, headY, pps, margin, W, H, pMin, pMax, slotW };

    // ── Pitch lane backgrounds ──
    for (let p = pMin; p <= pMax; p++) {
      if (BLACK_KEY_PC.has(p % 12)) {
        ctx.fillStyle = '#EEEEF2';
        ctx.fillRect(xOf(p) - slotW / 2, margin.t, slotW, plotH);
      }
      if (p % 12 === 0) {
        ctx.strokeStyle = 'rgba(15,15,112,0.10)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xOf(p) - slotW / 2, margin.t);
        ctx.lineTo(xOf(p) - slotW / 2, margin.t + plotH);
        ctx.stroke();
      }
    }

    // ── Playback head ──
    ctx.strokeStyle = '#0F0F70';
    ctx.lineWidth = isPlaying ? 2 : 1.5;
    ctx.setLineDash(isPlaying ? [] : [5, 4]);
    ctx.beginPath();
    ctx.moveTo(margin.l, headY);
    ctx.lineTo(W - margin.r, headY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#0F0F70';
    ctx.font = 'bold 10px ui-monospace, SF Mono, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`▶ ${tCenter.toFixed(2)}s`, W - margin.r - 2, headY - 8);

    // ── Notes ──
    const visible = trialNotes.filter(n => n.onset_sec < tTop && n.offset_sec > tBot);
    // Track Y positions of already-drawn pitch labels to avoid collisions
    const labelYUsed = [];

    for (const n of visible) {
      const x    = xOf(n.pitch);
      const w    = slotW * 0.85;
      const yTop = yOf(n.offset_sec);
      const yBot = yOf(n.onset_sec);
      const h    = Math.max(3, yBot - yTop);
      const isCur  = n.global_idx === cur.global_idx;
      const hard   = isHard(n);
      const vv     = verdicts[`${n.trial}#${n.note_idx}`];
      const humanV = (vv && vv.timestamp && !vv.source) ? vv : null;
      const disagrees = humanV && n.algorithm_int && humanV.finger_int !== n.algorithm_int;

      let fill, stroke, strokeW = 0.5;
      if (isCur) {
        fill = CURRENT_FILL; stroke = '#FFF'; strokeW = 2;
      } else if (hard) {
        fill   = !humanV ? HARD_PENDING_FILL
               : disagrees ? HARD_DISAGREE_FILL
               : HARD_AGREE_FILL;
        stroke = 'rgba(0,0,0,0.18)';
      } else {
        fill   = OTHER_FILL;
        stroke = 'rgba(0,0,0,0.08)';
      }

      ctx.fillStyle = fill;
      ctx.fillRect(x - w / 2, yTop, w, h);
      ctx.strokeStyle = stroke;
      ctx.lineWidth   = strokeW;
      ctx.strokeRect(x - w / 2, yTop, w, h);

      // Pitch label in left column for hard notes (not current — current drawn later)
      if (hard && !isCur) {
        const cy = (yTop + yBot) / 2;
        if (cy > margin.t + 2 && cy < margin.t + plotH - 2) {
          // Only draw if no other label is within 14px
          const tooClose = labelYUsed.some(ly => Math.abs(ly - cy) < 14);
          if (!tooClose) {
            labelYUsed.push(cy);
            ctx.fillStyle   = !humanV ? '#1976D2' : disagrees ? HARD_DISAGREE_FILL : HARD_AGREE_FILL;
            ctx.font        = 'bold 13px ui-monospace, SF Mono, monospace';
            ctx.textAlign   = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(n.pitch_name, margin.l - 8, cy);
          }
        }
      }
    }

    // ── Re-draw current note on top with halo ──
    {
      const cx   = xOf(cur.pitch);
      const w    = slotW * 0.85;
      const yTop = yOf(cur.offset_sec);
      const yBot = yOf(cur.onset_sec);
      const h    = Math.max(3, yBot - yTop);
      const cy   = (yTop + yBot) / 2;

      ctx.shadowBlur  = 12;
      ctx.shadowColor = CURRENT_FILL;
      ctx.fillStyle   = CURRENT_FILL;
      ctx.fillRect(cx - w / 2, yTop, w, h);
      ctx.shadowBlur  = 0;
      ctx.strokeStyle = CURRENT_FILL;
      ctx.lineWidth   = 1.5;
      ctx.strokeRect(cx - w / 2 - 3, yTop - 3, w + 6, h + 6);
      ctx.strokeStyle = '#FFF';
      ctx.lineWidth   = 2.5;
      ctx.strokeRect(cx - w / 2, yTop, w, h);

      // Arrow pointer above
      ctx.fillStyle = CURRENT_FILL;
      ctx.beginPath();
      ctx.moveTo(cx - 5, yTop - 8);
      ctx.lineTo(cx + 5, yTop - 8);
      ctx.lineTo(cx, yTop - 2);
      ctx.closePath();
      ctx.fill();

      // Pitch label for current note (bigger, red)
      if (cy > margin.t && cy < margin.t + plotH) {
        ctx.fillStyle    = CURRENT_FILL;
        ctx.font         = 'bold 14px ui-monospace, SF Mono, monospace';
        ctx.textAlign    = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(cur.pitch_name, margin.l - 8, cy);
      }
    }

    // ── Octave labels along bottom (every C) ──
    ctx.fillStyle    = '#5A5A6E';
    ctx.font         = '10px ui-monospace, SF Mono, monospace';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    for (let p = pMin; p <= pMax; p++) {
      if (p % 12 === 0) ctx.fillText(`C${p / 12 - 1}`, xOf(p), H - margin.b + 6);
    }
  }, [trialNotes, currentNoteIdx, verdicts, currentTime, isPlaying, pitchBounds]);

  function onClick(e) {
    if (!layoutRef.current) return;
    const { headY, pps, tCenter, margin, slotW, pMin } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const x    = e.clientX - rect.left;
    const y    = e.clientY - rect.top;
    const t    = tCenter - (y - headY) / pps;
    const pitchGuess = pMin + Math.round((x - margin.l) / slotW - 0.5);
    let hit = null;
    for (const n of trialNotes) {
      if (n.pitch !== pitchGuess) continue;
      if (n.onset_sec <= t + 0.05 && n.offset_sec >= t - 0.05) { hit = n; break; }
    }
    if (hit && onSelectNote) { onSelectNote(hit.global_idx); return; }
    if (onSeek) onSeek(Math.max(0, t));
  }

  return <canvas ref={canvasRef} className="roll-canvas" onClick={onClick} />;
}

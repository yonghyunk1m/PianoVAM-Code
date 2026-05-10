import React, { useRef, useEffect, useMemo } from 'react';
import { FINGER_INT_COLOR } from './fingerPalette.js';

const BLACK_KEY_PC = new Set([1, 3, 6, 8, 10]);

// Vertical layout: notes flow top → bottom (Synthesia / falling-notes style).
// Future notes sit at the top, past notes at the bottom, the playback head is
// a horizontal line near the bottom of the canvas.
const HEAD_FROM_BOTTOM = 0.18;     // playback head sits 18% from the bottom
const LOOK_AHEAD_SEC = 4.5;        // how far into the future to render
const LOOK_BACK_SEC = 1.0;         // how far into the past to keep on screen

export default function PianoRoll({ trialNotes, currentNoteIdx, verdicts = {},
                                     currentTime, duration, isPlaying,
                                     blindMode = false,
                                     changedNoteIds = new Set(),
                                     onSeek, onSelectNote }) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);

  // Pitch range: derived from the trial's full set of notes (stable, no jitter).
  const pitchBounds = useMemo(() => {
    if (!trialNotes || trialNotes.length === 0) return { pMin: 60, pMax: 72 };
    const pitches = trialNotes.map(n => n.pitch);
    const minP = Math.min(...pitches);
    const maxP = Math.max(...pitches);
    // Pad ±2 semitones so notes never touch the edges.
    return { pMin: minP - 2, pMax: maxP + 2 };
  }, [trialNotes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#FAFAFB';
    ctx.fillRect(0, 0, W, H);

    if (!trialNotes || trialNotes.length === 0) return;
    const cur = trialNotes[currentNoteIdx];
    if (!cur) return;

    const margin = { l: 8, r: 8, t: 6, b: 22 };
    const plotW = W - margin.l - margin.r;
    const plotH = H - margin.t - margin.b;
    const headY = margin.t + plotH * (1 - HEAD_FROM_BOTTOM);

    // Time axis: head sits at headY representing tCenter.
    const tCenter = (isPlaying && isFinite(currentTime)) ? currentTime : cur.onset_sec;
    const pps = plotH / (LOOK_AHEAD_SEC + LOOK_BACK_SEC);  // pixels per second
    const tTop = tCenter + LOOK_AHEAD_SEC;     // earliest future visible
    const tBottom = tCenter - LOOK_BACK_SEC;   // latest past still visible
    const yOf = (t) => headY - (t - tCenter) * pps;

    // Pitch axis: each pitch step = constant slot width across plotW.
    const { pMin, pMax } = pitchBounds;
    const slotW = plotW / (pMax - pMin + 1);
    const xOf = (p) => margin.l + (p - pMin + 0.5) * slotW;

    layoutRef.current = { tCenter, tTop, tBottom, headY, pps,
                          margin, W, H, plotW, plotH, pMin, pMax, slotW };

    // ── Pitch lanes (background) — black-key columns slightly darker ──
    for (let p = pMin; p <= pMax; p++) {
      const isBlack = BLACK_KEY_PC.has(p % 12);
      const cx = xOf(p);
      const w = slotW;
      if (isBlack) {
        ctx.fillStyle = '#EEEEF2';
        ctx.fillRect(cx - w / 2, margin.t, w, plotH);
      }
      // octave separator at every C
      if (p % 12 === 0) {
        ctx.strokeStyle = 'rgba(15,15,112,0.10)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx - w / 2, margin.t);
        ctx.lineTo(cx - w / 2, margin.t + plotH);
        ctx.stroke();
      }
    }

    // ── Playback head (horizontal line near bottom) ──
    ctx.strokeStyle = '#0F0F70';
    ctx.lineWidth = isPlaying ? 2 : 1.5;
    ctx.setLineDash(isPlaying ? [] : [5, 4]);
    ctx.beginPath();
    ctx.moveTo(margin.l, headY);
    ctx.lineTo(W - margin.r, headY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Head label on the right
    ctx.fillStyle = '#0F0F70';
    ctx.font = 'bold 10px ui-monospace, SF Mono, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`▶ ${tCenter.toFixed(2)}s`, W - margin.r - 2, headY - 8);

    // ── Notes — vertical bars: width=keyslot, height=duration*pps ──
    const visibleNotes = trialNotes.filter(
      n => n.onset_sec < tTop && n.offset_sec > tBottom);
    for (const n of visibleNotes) {
      const x = xOf(n.pitch);
      const w = slotW * 0.85;
      const yTop = yOf(n.offset_sec);
      const yBot = yOf(n.onset_sec);
      const h = Math.max(3, yBot - yTop);
      const isCurrent = n.global_idx === cur.global_idx;
      const v = verdicts[`${n.trial}#${n.note_idx}`];
      let fill, stroke, strokeW = 0.5;

      // In blindMode: only the annotator's own picks are colored. Algo /
      // imputed colors and priority outlines are suppressed so nothing on
      // the roll betrays the model's prediction.
      const human = (v && v.timestamp && !v.source) ? v : null;
      if (isCurrent) {
        fill = '#D32F2F';
        stroke = '#FFF';
        strokeW = 2;
      } else if (human && human.finger_int >= 1 && human.finger_int <= 10) {
        fill = FINGER_INT_COLOR[human.finger_int];
        if (!blindMode && n.algorithm_int && human.finger_int !== n.algorithm_int) {
          stroke = '#ED6C02'; strokeW = 2;
        } else {
          stroke = 'rgba(0,0,0,0.2)';
        }
      } else if (human && human.finger_int === 0) {
        fill = '#ED6C02';
        stroke = 'rgba(0,0,0,0.25)';
      } else if (!blindMode && n.algorithm_int) {
        // Algorithm label — full color so the roll starts fully colored and
        // reviewers fix wrong ones rather than annotating from scratch.
        fill = FINGER_INT_COLOR[n.algorithm_int];
        stroke = 'rgba(0,0,0,0.2)';
      } else if (!blindMode && n.imputed_int) {
        fill = FINGER_INT_COLOR[n.imputed_int];
        stroke = 'rgba(0,0,0,0.2)';
      } else {
        fill = 'rgba(15,15,112,0.18)';
        stroke = 'rgba(15,15,112,0.40)';
      }

      // Hard-rule notes not yet reviewed: bold red outline so they pop.
      // Priority-not-yet-reviewed (algo uncertainty): bold orange outline.
      // Both suppressed in blindMode.
      if (!blindMode && !isCurrent) {
        const hasLabel = !!(n.algorithm_int || n.imputed_int);
        if (!hasLabel && !human) {
          stroke = '#BDBDBD';
          strokeW = 0.5;
        } else {
          const isHardPending = !human
            && (n.is_hard || (Array.isArray(n.hard_reasons) && n.hard_reasons.length > 0));
          if (isHardPending) {
            stroke = '#D32F2F';
            strokeW = 2.5;
          } else {
            const isPriorityPending = !human
              && (n.algorithm_status === 'no_candidate'
                  || n.algorithm_status === 'multi_candidate'
                  || (typeof n.algorithm_score === 'number' && n.algorithm_score < 0.65));
            if (isPriorityPending) {
              stroke = '#ED6C02';
              strokeW = 2.5;
            }
          }
        }
      }

      ctx.fillStyle = fill;
      ctx.fillRect(x - w / 2, yTop, w, h);
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = strokeW;
        ctx.strokeRect(x - w / 2, yTop, w, h);
      }
      // Changed-from-TSV indicator: small gold dot in top-right corner
      if (!isCurrent && changedNoteIds.has(`${n.trial}#${n.note_idx}`)) {
        const dotR = Math.min(3.5, w * 0.25);
        ctx.fillStyle = '#F9A825';
        ctx.beginPath();
        ctx.arc(x + w / 2 - dotR - 0.5, yTop + dotR + 1, dotR, 0, 2 * Math.PI);
        ctx.fill();
      }
      // Inscribe finger number on tall-enough notes (own picks only when blind)
      if (h >= 14) {
        const humanInt = (human && human.finger_int >= 1 && human.finger_int <= 10)
          ? human.finger_int : null;
        const autoInt = (!blindMode && v && v.finger_int >= 1 && v.finger_int <= 10)
          ? v.finger_int : (n.algorithm_int || null);
        const fingerInt = humanInt ?? (blindMode ? null : autoInt);
        if (fingerInt && !isCurrent) {
          ctx.fillStyle = '#FFF';
          ctx.font = 'bold 9px ui-monospace, SF Mono, monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(fingerInt <= 5 ? `L${fingerInt}` : `R${fingerInt - 5}`,
                       x, yTop + 2);
        }
      }
    }

    // ── Re-draw the current note on top with strong highlight ──
    // (Drawn after everything else so the focus marker is never occluded
    // by neighboring notes.)
    {
      const cx = xOf(cur.pitch);
      const w = slotW * 0.85;
      const yTop = yOf(cur.offset_sec);
      const yBot = yOf(cur.onset_sec);
      const h = Math.max(3, yBot - yTop);
      // Outer halo
      ctx.shadowBlur = 12;
      ctx.shadowColor = '#D32F2F';
      ctx.fillStyle = '#D32F2F';
      ctx.fillRect(cx - w / 2, yTop, w, h);
      ctx.shadowBlur = 0;
      // Outer ring
      ctx.strokeStyle = '#D32F2F';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cx - w / 2 - 3, yTop - 3, w + 6, h + 6);
      // Inner white outline so the red is sharp
      ctx.strokeStyle = '#FFF';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(cx - w / 2, yTop, w, h);
      // Top + bottom pointer arrows so even pixel-tiny notes are visible
      ctx.fillStyle = '#D32F2F';
      ctx.beginPath();
      ctx.moveTo(cx - 5, yTop - 8);
      ctx.lineTo(cx + 5, yTop - 8);
      ctx.lineTo(cx, yTop - 2);
      ctx.closePath();
      ctx.fill();
    }

    // ── Pitch labels along the bottom (every C) ──
    ctx.fillStyle = '#5A5A6E';
    ctx.font = '10px ui-monospace, SF Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let p = pMin; p <= pMax; p++) {
      if (p % 12 === 0) {
        ctx.fillText(`C${p / 12 - 1}`, xOf(p), H - margin.b + 6);
      }
    }
  }, [trialNotes, currentNoteIdx, verdicts, currentTime, isPlaying, pitchBounds, blindMode, changedNoteIds]);

  function onClick(e) {
    if (!layoutRef.current) return;
    const { headY, pps, tCenter, margin, slotW, pMin } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const t = tCenter - (y - headY) / pps;
    const pitchGuess = pMin + Math.round((x - margin.l) / slotW - 0.5);

    // Hit-test: any visible note whose rectangle covers (t, pitchGuess)?
    let hit = null;
    for (const n of trialNotes) {
      if (n.pitch !== pitchGuess) continue;
      if (n.onset_sec <= t + 0.05 && n.offset_sec >= t - 0.05) {
        hit = n; break;
      }
    }
    if (hit && onSelectNote) {
      onSelectNote(hit.global_idx);
      return;
    }
    if (onSeek) onSeek(Math.max(0, t));
  }

  return <canvas ref={canvasRef} className="roll-canvas" onClick={onClick} />;
}

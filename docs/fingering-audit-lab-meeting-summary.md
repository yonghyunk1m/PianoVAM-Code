# PianoVAM Fingering Audit: Lab-Meeting Summary

**Study date:** 2026-07-24
**Authoritative run:** `20260724T165823Z-authoritative-offset-audit-final-e015b28b`

## Main message

Physical and ergonomic feasibility is useful as a mandatory safety check, but
it is not sufficiently sensitive to serve as the primary fingering-error
screen.

Most assigned-finger errors need not produce an impossible fingering. A
detector can choose the wrong one of two fingers occupying the same key area
while both choices remain ergonomically plausible. The next important signal
is therefore detector uncertainty before final candidate selection, especially
top-two finger support and its temporal behavior.

## Study population

The audit used:

| Population | Count |
|---|---:|
| Complete PianoVAM corpus | 508,621 notes |
| Assigned hand/finger labels | 434,373 |
| Missing hand/finger labels | 74,248 |
| Authoritative human labels | 1,800 notes from 11 recordings |
| All exact ground-truth errors | 392 |
| Errors caused by missing prediction | 275 |
| Errors among assigned predictions | 117 |

The audit used the original native `key_offset` values from the official
PianoVAM repository. The Vite display fallback of `onset + 0.5` seconds was
not used.

For choosing assigned fingers to audit, the main metric is **assigned-error
recall**:

```text
selected errors among assigned predictions / all 117 assigned-prediction errors
```

## Main filters tested

| Filter | Method | Selected notes | Corpus % | Assigned-error recall | GT precision |
|---|---|---:|---:|---:|---:|
| Step crossing | Non-thumb crossing over at most two semitones | 3,631 | 0.71% | 2.56% | 30.00% |
| Extreme movement rate | Leave-one-recording-out 99.5th-percentile pitch/time rate | 4,745 | 0.93% | 1.71% | 25.00% |
| Non-thumb crossing | Ergonomically unusual sequential finger order | 9,707 | 1.91% | 5.98% | 25.00% |
| Practical-span violation | Outside Parncutt `MinPrac–MaxPrac` limits | 39,443 | 7.75% | 11.11% | 9.92% |
| Comfortable-span violation | Outside Parncutt comfort limits | 69,306 | 13.63% | 18.80% | 9.44% |
| Relative-span violation | Sensitive Parncutt relative-span limits | 182,738 | 35.93% | 51.28% | 9.62% |
| HMM disagreement | Published label differs from the PIG-trained HMM | 279,540 | 54.96% | 81.20% | 9.42% |
| Strict two-signal set | At least two of practical span, crossing, movement rate, and HMM disagreement | 31,066 | 6.11% | 13.68% | 17.78% |
| Two-of-three families | At least two of ergonomic, model, and temporal evidence | 49,748 | 9.78% | 20.51% | 14.20% |
| Legacy filter | Existing leap, crossing, overlap, and missingness rules | 56,271 | 11.06% | 26.50% | 21.99% |
| Nearby `Noinfo` | Assigned note beside a missing run of at least two notes | 21,651 | 4.26% | 5.13% | 8.45% |

`GT precision` is the fraction of selected authoritative-test notes that are
exact errors. It should be interpreted together with recall and workload.

## What the physical and ergonomic trials showed

### 1. Conservative physical-risk filters missed most errors

The research-supported practical-span filter selected 39,443 notes but caught
only 13 of the 117 assigned-finger errors. Its assigned-error recall was
11.11%.

The strict two-signal set was closest to the desired workload. It selected
31,066 notes but caught only 16 of the 117 assigned-finger errors, giving
13.68% assigned-error recall.

### 2. Relaxing span thresholds increased workload faster than useful recall

Moving from the practical to the sensitive relative-span boundary increased
the queue from 39,443 to 182,738 notes. Assigned-error recall increased from
11.11% to 51.28%, but almost 36% of the entire corpus would require review.

### 3. Crossing rules found concentrated outliers but covered few errors

Crossing filters had comparatively high precision, but their recall was low:

- step crossing caught 3 of 117 assigned errors;
- general non-thumb crossing caught 7 of 117 assigned errors.

They remain useful alerts, but they cannot be the main screening mechanism.

### 4. The result was nonuniform across fingers

The 31,066-note strict two-signal set caught zero authoritative errors for six
of the ten fingers: `L1`, `L2`, `L3`, `L4`, `R1`, and `R4`.

Its full-corpus selection rate also ranged from 1.70% for `R1` to 20.52% for
`L4`. A single aggregate workload number therefore hides substantial
finger-specific bias.

### 5. Model disagreement had high recall only at excessive workload

HMM disagreement caught 95 of 117 assigned errors, or 81.20%, but selected
279,540 notes. This confirms that an independent model can expose errors, but
raw disagreement is too broad to define a practical review queue.

### 6. Nearby missing labels were weak evidence by themselves

The smallest useful nearby-`Noinfo` rule selected 21,651 assigned notes and
caught 6 of 117 assigned errors. More permissive neighborhoods increased
workload substantially without providing enough enrichment to solve the
screening problem.

The missing predictions themselves are a separate publication-integrity
problem: 74,248 corpus notes are missing a label, and missing predictions
account for 275 of the 392 exact ground-truth errors. These records need repair,
not an assigned-finger audit.

## Interpretation

The results do **not** show that physical validity is irrelevant. They show
that **sequential ergonomic implausibility is a weak primary detector of wrong
assigned fingers**.

This distinction matters because:

1. a wrong finger can still be physically possible;
2. different performers can use different valid fingerings;
3. span and crossing rules inspect the plausibility of the final sequence, not
   which nearby fingertip actually depressed a key; and
4. making these rules sensitive enough to recover many errors produces an
   impractically large review queue.

The likely detector-specific failure is local candidate ambiguity: two
fingertips may occupy the played key area, but only one pushes the key. A
sequence-level physical rule may accept either candidate.

## Required scientific qualification

The strict simultaneous physical-invalidity layer was not activated in the
authoritative run because PIG v1.02 was not available locally for the required
zero-violation validation. Consequently:

- the reported physical results principally concern sequential span,
  crossing, and movement-risk rules;
- they should not be presented as a complete test of every physical
  contradiction; and
- simultaneous physical contradictions should remain mandatory alerts once
  validated against the complete authoritative PIG annotations.

The defensible presentation wording is:

> Physical and ergonomic feasibility is valuable as a mandatory safety layer,
> but it is not sufficiently sensitive to be the primary error-prioritization
> method. Our results suggest that candidate-selection uncertainty is the more
> important missing signal.

Avoid the stronger claim that physical validity is unimportant or that no
physical errors exist.

## Recommended next experiment

Recover or rerun the detector's pre-decision finger scores and record, for each
note:

- all viable finger candidates;
- top-one and top-two scores;
- top-two score margin and ratio;
- candidate entropy;
- onset-specific support;
- temporal winner changes; and
- disagreement between the chosen finger and the strongest temporally stable
  candidate.

These uncertainty signals should be calibrated using leave-one-recording-out
evaluation on the 11 authoritative recordings. Their incremental assigned-error
recall should be reported after union with mandatory physical alerts, together
with workload, precision, and per-finger results.

## One-slide conclusion

> We evaluated conservative ergonomic, temporal, model-disagreement,
> missing-context, and combined filters on 508,621 PianoVAM notes. Physically
> motivated sequence rules identified rare suspicious cases but captured only
> a small fraction of known assigned-finger errors. Increasing their
> sensitivity made the review queue impractically large. The evidence therefore
> supports retaining physical rules as a mandatory safety net while shifting
> the primary audit signal toward the detector's pre-decision candidate
> uncertainty.

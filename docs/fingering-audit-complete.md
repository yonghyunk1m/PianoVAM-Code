# PianoVAM Fingering Audit: Complete Research and Implementation Record

## Authoritative-offset rerun (2026-07-24)

The valid production run is `20260724T181031Z-crossing-ioi-cap-e015b28b`.
It uses original `key_offset` values from official `PianoVAM/PianoVAM_v1`
native TSVs at immutable revision `7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8`;
Vite `onset + 0.5`, inferred, nearest, and synthetic offsets are forbidden.
Acquisition verified 105/105 files and 508,621/508,621 exact joins, with zero
missing offsets, identity mismatches, and synthetic offsets; source fingering
TSV hashes are unchanged.

The earlier all-integrity run using the sidecar-less timing cache is invalid
and excluded from conclusions. The `20260724T165823Z` authoritative-offset
run was valid but is superseded by this run, which adds the user-specified
inclusive `1000 ms` IOI cap to crossing-based audit masks. The current run is
complete but its PIG validity gate is truthfully closed because no
checksum-verifiable PIG copy is available; therefore no recommendation or
Vite audit queue is published.

The exact generated physical, integrity, fixed/calibrated Noinfo, 189 strategy,
GT/assigned recall, precision/enrichment/incremental, methods, and all-ten-finger
tables are committed as GitHub-viewable CSV files in
[`docs/fingering-audit-results/`](fingering-audit-results/). They are a byte-for-byte
snapshot of the final run's `results/` directory at
`artifacts/fingering_audit/20260724T181031Z-crossing-ioi-cap-e015b28b/`.
The primary tables are
[`filter_sets.csv`](fingering-audit-results/filter_sets.csv),
[`individual_filters.csv`](fingering-audit-results/individual_filters.csv),
[`noinfo_sensitivity.csv`](fingering-audit-results/noinfo_sensitivity.csv),
[`queue_summary.csv`](fingering-audit-results/queue_summary.csv),
[`queue_workload_per_finger.csv`](fingering-audit-results/queue_workload_per_finger.csv),
and [`per_finger.csv`](fingering-audit-results/per_finger.csv); the directory also
contains the overlap, Pareto, per-recording, threshold, error-type, exclusion,
and general workload tables.
Headline rows:

| set | hard notes | hard % | GT recall | assigned recall | precision |
|---|---:|---:|---:|---:|---:|
| `bl_two_signal_strict` | 30,778 | 6.05% | 4.08% | 13.68% | 18.60% |
| `bl_crossing` | 8,335 | 1.64% | 1.79% | 5.98% | 35.00% |
| `bl_step_crossing` | 3,328 | 0.65% | 0.77% | 2.56% | 42.86% |
| `ni_k2_r1` | 21,651 | 4.26% | 1.53% | 5.13% | 8.45% |
| `ni_k5_r4` | 2,247 | 0.44% | 0.00% | 0.00% | — |

All ten fingers remain represented, including zero-recall fingers. Verify-only
passed; Python (125), Node audit-category (7), and Vite build checks passed.

**Study date:** 2026-07-23  
**Consolidated:** 2026-07-24  
**Historical run ID:** `20260723T122049Z-publication-audit-746e73d5` (invalid; excluded)
**Status:** authoritative-offset computation complete; PIG validity gate closed; no publication recommendation or Vite queue authorized

## 1. Executive summary

This study built and ran an unattended, evidence-governed audit of PianoVAM
fingering labels. It compared blacklist, whitelist, and hybrid strategies on
the complete local dataset while treating:

- the 1,800 human labels in 11 recordings as authoritative error ground truth;
- PIG v1.02 as the authoritative negative control for fingering validity; and
- the complete 508,621-note PianoVAM corpus as the workload population.

The approximately 30,000-note review target was deliberately treated as a soft
target. Thresholds were not adjusted merely to reach that number.

The main conclusions are:

1. No evaluated set is currently defensible as a final publication filter.
2. The closest set to 30,000 notes selects 30,778 notes but recalls only 4.08%
   of all authoritative errors (95% recording-clustered interval:
   1.68%–6.71%).
3. The corpus contains 74,248 notes without any hand/finger label. These
   account for 275 of the 392 ground-truth errors.
4. Filtering is strongly nonuniform by finger. In the 30,778-note set, the
   selected workload ranges from 1.70% of `R1` notes to 20.51% of `L4` notes.
5. PIG v1.02 is not present locally and its official page does not provide a
   checksum-verifiable unattended download. No set is therefore marked
   recommendable, and no Vite review queue was exported.

The correct next scientific step is not to tune a cutoff toward 30,000. It is
to obtain the authoritative PIG annotations, validate the applicable rules,
and address or impute the 74,248 missing labels under a separately validated
procedure.

## 2. Objective

The goal is to prioritize PianoVAM fingering labels for human review before
publication while minimizing review workload without sacrificing validity.

The study evaluates:

- **blacklisting:** selecting notes because risk conditions hold;
- **whitelisting:** removing notes from review only when sufficient reliability
  conditions hold; and
- **hybrid filtering:** combining direct risks, strict safe conditions, and
  corroboration between independent evidence families.

Scientific defensibility and error capture take priority over reaching an
arbitrary queue size.

## 3. Existing system and data

### 3.1 Repository components

| Component | Role |
|---|---|
| `PianoVAM_v1.0/Fingering/` | 105 source TSV files |
| `FingeringDetection/detection/fingergt.py` | 1,800 authoritative labels |
| `FingeringInterpolation/` | PIG-trained second-order HMM and model files |
| `ManualCheck/hard_part_selector.py` | seven legacy hard-note rules |
| `annotate/` | React/Vite fingering-correction application |
| `fingering_audit/` | automated research and reporting pipeline |

### 3.2 Audit populations

| Population | Count | Purpose |
|---|---:|---|
| Full PianoVAM notes | 508,621 | workload and hard-note percentage |
| Assigned hand/finger labels | 434,373 | assigned-fingering audit |
| Missing hand/finger labels | 74,248 | mandatory data-repair population |
| Authoritative GT labels | 1,800 | error evaluation |
| GT recordings | 11 | grouped validation and uncertainty |
| GT exact errors | 392 | target errors |
| GT errors caused by missing prediction | 275 | data-completeness failures |
| GT errors among assigned predictions | 117 | assigned-label errors |

The source TSV files are immutable inputs. The pipeline writes generated
tables under `artifacts/fingering_audit/<run-id>/`.

## 4. Terminology and validity rules

### 4.1 Invalidity rule

An invalidity rule claims that a fingering is physically or logically
impossible. Every such rule must trigger on zero PIG annotations before it can
appear in a recommended set.

### 4.2 Risk signal

A risk signal means that a detector may be wrong or that a passage may be
difficult. It does not establish that the fingering is invalid.

### 4.3 Hard-note percentage

The hard-note percentage is:

```text
selected audit notes / all 508,621 PianoVAM notes
```

It is measured before adding display-only context notes.

### 4.4 PIG policy

PIG fingerings may be rare, difficult, performer-specific, signed, or
compound. They remain authoritative examples of valid fingerings.

PIG can disprove an invalidity claim. It cannot validate a
video- or detector-specific signal because it does not contain matching
PianoVAM video evidence.

Risk rules return `not_applicable` under a PIG invalidity check rather than a
fabricated zero.

## 5. Evidence policy

Every threshold is assigned one of four grades:

1. **Physical invariant:** a schema or physical condition that is unambiguous.
2. **Research-supported:** derived from a primary source with its scope and
   assumptions retained.
3. **Empirically calibrated:** selected using training recordings only inside
   leave-one-recording-out evaluation.
4. **Exploratory:** plausible but insufficiently supported; it cannot select
   notes alone in a recommended set.

The study applies these safeguards:

- never choose a threshold merely to obtain 30,000 notes;
- keep research-derived and empirically calibrated values distinguishable;
- use pair-specific ergonomic spans instead of one global span;
- condition large leaps on available movement time;
- require corroboration for weak musical-context signals;
- retain missingness explicitly rather than treating unavailable evidence as
  confidence; and
- preserve complete PIG annotation semantics when PIG is available.

## 6. Threshold decisions

### 6.1 Finger-pair span thresholds

Parncutt et al. (1997), Table 1, gives directed semitone limits for ten finger
pairs. The source model concerns consecutive right-hand melodic fragments
under legato, moderately loud, approximately isochronous conditions. It does
not model articulation, performer hand size, two-hand interaction, or free
repositioning time.

| Pair | MinPrac | MinComf | MinRel | MaxRel | MaxComf | MaxPrac |
|---|---:|---:|---:|---:|---:|---:|
| 1–2 | -5 | -3 | 1 | 5 | 8 | 10 |
| 1–3 | -4 | -2 | 3 | 7 | 10 | 12 |
| 1–4 | -3 | -1 | 5 | 9 | 12 | 14 |
| 1–5 | -1 | 1 | 7 | 10 | 13 | 15 |
| 2–3 | 1 | 1 | 1 | 2 | 3 | 5 |
| 2–4 | 1 | 1 | 3 | 4 | 5 | 7 |
| 2–5 | 2 | 2 | 5 | 6 | 8 | 10 |
| 3–4 | 1 | 1 | 1 | 2 | 2 | 4 |
| 3–5 | 1 | 1 | 3 | 4 | 5 | 7 |
| 4–5 | 1 | 1 | 1 | 2 | 3 | 5 |

The variants are:

- **conservative:** outside `MinPrac..MaxPrac`;
- **central:** outside `MinComf..MaxComf`;
- **sensitive:** outside `MinRel..MaxRel`.

All three are risk signals, not declarations of impossibility. The left-hand
application mirrors the pitch axis and is explicitly an extrapolation.

### 6.2 Non-thumb crossings

The ergonomic literature describes most non-thumb crossings as impractical
but also recognizes legitimate historical and virtuoso exceptions. Therefore:

- crossing is a risk signal, never invalidity;
- it cannot independently support a final recommended set; and
- the raw crossing relation remains available for diagnosis;
- audit masks accept it only when the preceding same-hand IOI is at most
  `1000 ms`, inclusively; and
- the `1000 ms` cap is a user-specified conservative policy, not a published
  physical threshold.

The authoritative GT contains 17 non-thumb crossing transitions. Six have
IOI greater than `1000 ms`, and two are simultaneous chord-order relations
with zero IOI. These examples disprove treating the raw relation as physical
invalidity and motivate keeping long-gap crossings out of the audit mask.

### 6.3 Time-conditioned position changes

No reviewed primary source gives a universal piano-video error threshold that
jointly combines pitch displacement and movement time. The pipeline therefore
calibrates upper-tail position-change rates within each training fold:

| Variant | Training-fold tail |
|---|---:|
| Conservative | 99.5th percentile |
| Central | 99th percentile |
| Sensitive | 97.5th percentile |

The held-out recording never contributes to its threshold.

### 6.4 Detector and trajectory values

HaMeR and MediaPipe Hands establish useful hand reconstruction and tracking
methods, but neither publication establishes a universal piano-fingering
cutoff for:

- top candidate score;
- top-two margin;
- missing-frame duration;
- landmark velocity, acceleration, or jerk;
- key-boundary distance; or
- trajectory discontinuity.

The legacy support levels 0.50 and 0.80 are implementation-native baselines,
not published piano error thresholds. When corresponding authoritative
features become available, they must be calibrated inside the training folds.

### 6.5 HMM disagreement

The PIG-trained second-order HMM is an independent fingering estimator.
Detector/HMM disagreement is a risk signal, not proof that either label is
correct. Human fingering varies, and published sequence models have known
limitations involving phrase boundaries and interdependence between hands.

### 6.6 Legacy thresholds

| Legacy value | Final treatment | Reason |
|---|---|---|
| crossing within 500 ms | exploratory baseline | no universal timing boundary; valid exceptions exist |
| leap ≥15 semitones within 180 ms | exploratory baseline | pitch distance alone is insufficient |
| hand overlap within 200 ms | exploratory baseline | no primary piano-error cutoff |
| ≥3 consecutive `Noinfo` | exploratory baseline | missingness is relevant; cluster length is arbitrary |
| alternation IOI ≤120 ms | exploratory baseline | no universal error threshold |
| support <0.50 or <0.80 | legacy baseline | detector-native tier, not external evidence |

## 7. Automated workflow

Run:

```bash
./run_fingering_audit.sh --run-label publication-audit
```

The wrapper discovers a Python environment containing the required packages
and invokes:

```bash
python -m fingering_audit run \
  --config fingering_audit/config/research.yaml
```

Pipeline stages:

1. validate the configuration and evidence ledger;
2. discover PIG in configured roots;
3. load and canonicalize all PianoVAM notes;
4. attach all 1,800 authoritative labels;
5. extract vectorized musical and ergonomic features;
6. run the existing PIG-trained HMM independently;
7. compute fixed and fold-calibrated rule masks;
8. evaluate blacklist, whitelist, hybrid, integrity, and legacy sets;
9. generate overall, per-recording, per-error-type, and per-finger metrics;
10. compute 2,000 recording-cluster bootstrap replicates per filter set;
11. write tables, plots, manifests, and reconciliation checks; and
12. write `SUCCESS.json` only if mandatory gates pass.

When PIG is unavailable, the computational run completes but writes
`RECOMMENDATION_GATE_CLOSED.json` instead of `SUCCESS.json`.

## 8. Evaluation methodology

### 8.1 Error semantics

- A missing prediction is an exact error and a hand error.
- A wrong hand is a hand error but not a within-hand finger error.
- A wrong finger with the correct hand is a within-hand finger error.

### 8.2 Core metrics

```text
error recall       = selected errors / all errors
precision          = selected errors / selected notes
correct sieve rate = selected correct notes / all correct notes
enrichment         = precision / overall error prevalence
```

### 8.3 Grouped validation

Every empirical threshold uses leave-one-recording-out evaluation. The held-out
recording does not participate in threshold fitting.

Uncertainty intervals resample the 11 recordings with replacement rather than
resampling neighboring notes as if they were independent.

### 8.4 Per-finger reporting

Ground-truth performance is grouped by the true finger `L1`–`L5` and
`R1`–`R5`. Full-corpus workload is grouped by the predicted finger, with
missing predictions reported separately.

## 9. Ground-truth error distribution by finger

| Finger | GT notes | Exact errors | Error rate |
|---|---:|---:|---:|
| L1 | 252 | 45 | 17.86% |
| L2 | 207 | 25 | 12.08% |
| L3 | 110 | 20 | 18.18% |
| L4 | 38 | 13 | 34.21% |
| L5 | 279 | 72 | 25.81% |
| R1 | 252 | 71 | 28.17% |
| R2 | 193 | 19 | 9.84% |
| R3 | 160 | 21 | 13.12% |
| R4 | 136 | 40 | 29.41% |
| R5 | 173 | 66 | 38.15% |
| **Total** | **1,800** | **392** | **21.78%** |

The error distribution is plainly nonuniform. `R5` and `L4` have the highest
observed error rates, whereas `R2` and `L2` have the lowest.

## 10. Complete filter-set results

All percentages use the 508,621-note corpus denominator. `GT recall` includes
missing predictions; `assigned recall` conditions on notes that already have a
predicted hand and finger.

| Strategy | Filter set | Hard notes | Hard % | GT recall | Precision | Assigned recall |
|---|---|---:|---:|---:|---:|---:|
| Blacklist | `bl_step_crossing` | 3,328 | 0.65% | 0.77% | 42.86% | 2.56% |
| Blacklist | `bl_rate_q995` | 4,745 | 0.93% | 0.51% | 25.00% | 1.71% |
| Blacklist | `bl_rate_q990` | 6,351 | 1.25% | 1.02% | 23.53% | 3.42% |
| Blacklist | `bl_crossing` | 8,335 | 1.64% | 1.79% | 35.00% | 5.98% |
| Blacklist | `bl_rate_q975` | 14,655 | 2.88% | 1.53% | 14.29% | 5.13% |
| Blacklist | `bl_two_signal_strict` | 30,778 | 6.05% | 4.08% | 18.60% | 13.68% |
| Baseline | `legacy_current_default` | 35,000 | 6.88% | 6.63% | 28.89% | 22.22% |
| Blacklist | `bl_span_practical` | 39,443 | 7.75% | 3.32% | 9.92% | 11.11% |
| Blacklist | `bl_practical_or_crossing` | 39,443 | 7.75% | 3.32% | 9.92% | 11.11% |
| Hybrid | `hy_direct_plus_corroborated` | 41,928 | 8.24% | 4.08% | 11.51% | 13.68% |
| Blacklist | `bl_practical_or_rate995` | 44,172 | 8.68% | 3.83% | 10.79% | 12.82% |
| Hybrid | `hy_two_of_three_families` | 49,748 | 9.78% | 6.12% | 14.20% | 20.51% |
| Hybrid | `hy_hierarchical` | 62,225 | 12.23% | 5.87% | 10.36% | 19.66% |
| Blacklist | `bl_span_comfortable` | 69,306 | 13.63% | 5.61% | 9.44% | 18.80% |
| Integrity | `mandatory_missing` | 74,248 | 14.60% | 70.15% | 100.00% | 0.00% |
| Blacklist | `bl_span_relative` | 182,738 | 35.93% | 15.31% | 9.62% | 51.28% |
| Blacklist | `bl_hmm_disagreement` | 279,540 | 54.96% | 24.23% | 9.42% | 81.20% |
| Whitelist | `wl_model_agreement` | 279,540 | 54.96% | 24.23% | 9.42% | 81.20% |
| Whitelist | `wl_strict_obvious` | 351,748 | 69.16% | 27.81% | 8.80% | 93.16% |

No result is marked recommendable while the PIG gate is unavailable.

## 11. Detailed analysis of the nearest-to-30k set

`bl_two_signal_strict` requires at least two of four risk signals:

- practical-span violation;
- non-thumb crossing with preceding same-hand IOI at most `1000 ms`;
- central time-conditioned position-change tail; and
- HMM disagreement.

It selects 30,778 notes (6.05%) but captures only 16 of the 392 authoritative
errors:

| Metric | Result |
|---|---:|
| All-GT exact-error recall | 4.08% |
| 95% recording-clustered interval | 1.68%–6.71% |
| Assigned-label error recall | 13.68% |
| GT precision | 18.60% |

This filter is close to the requested workload only numerically. Its error
capture is too low and too uneven for publication use.

### 11.1 Per-finger GT performance

| Finger | Errors | Selected errors | Recall |
|---|---:|---:|---:|
| L1 | 45 | 0 | 0.00% |
| L2 | 25 | 0 | 0.00% |
| L3 | 20 | 0 | 0.00% |
| L4 | 13 | 0 | 0.00% |
| L5 | 72 | 7 | 9.72% |
| R1 | 71 | 0 | 0.00% |
| R2 | 19 | 1 | 5.26% |
| R3 | 21 | 1 | 4.76% |
| R4 | 40 | 0 | 0.00% |
| R5 | 66 | 7 | 10.61% |

Six of the ten fingers have zero captured authoritative errors.

### 11.2 Workload by predicted finger

| Predicted finger | Eligible notes | Hard notes | Selected % |
|---|---:|---:|---:|
| L1 | 56,961 | 1,701 | 2.99% |
| L2 | 42,151 | 3,598 | 8.54% |
| L3 | 32,568 | 4,025 | 12.36% |
| L4 | 17,383 | 3,566 | 20.51% |
| L5 | 41,112 | 4,440 | 10.80% |
| R1 | 65,498 | 1,113 | 1.70% |
| R2 | 58,395 | 2,693 | 4.61% |
| R3 | 48,125 | 3,579 | 7.44% |
| R4 | 31,856 | 3,320 | 10.42% |
| R5 | 40,324 | 2,743 | 6.80% |

The 12-fold difference between `R1` and `L4` confirms that aggregate workload
alone is misleading.

## 12. Missing labels and the workload constraint

The 74,248 missing predictions form a separate integrity problem:

- they are 14.60% of the corpus;
- they account for 275 of 392 GT errors (70.15%); and
- they already exceed the entire 30,000-note target.

Combining the missing-label queue with the nearest 30,778-note assigned-label
set would require 105,026 reviews before display context. It would capture 291
of 392 GT errors (74.23%) under the current GT sample, but the workload would
be far above the target.

An HMM or another model may propose labels for missing entries, but those
generated labels require their own held-out validation and uncertainty policy.
They cannot be treated as automatically correct merely to reduce manual work.

## 13. Available and unavailable evidence

### Available

- complete 508,621-note fingering TSV corpus;
- all 1,800 local authoritative labels;
- PIG-trained HMM model files;
- musical timing, pitch, hand, and predicted-finger context;
- vectorized ergonomic relations; and
- legacy rule outputs.

### Unavailable in the completed run

| Evidence | Status | Consequence |
|---|---|---|
| PIG v1.02 annotation files | not present locally | recommendation gate closed |
| HaMeR note-level trajectories and candidate margins | not found | trajectory-dependent sets excluded |
| MediaPipe per-frame confidence aligned to notes | not found | confidence-dependent sets excluded |

The official PIG page supplies the dataset upon request and does not expose a
direct archive with a recorded checksum. The runner refuses to substitute an
unofficial mirror.

## 14. Recommendation status

No final filter is recommended.

This is a substantive research result, not merely a software blocker:

- the nearest-to-30k candidate has inadequate recall;
- span-only filters are weak error detectors despite their stronger ergonomic
  justification;
- strict whitelisting removes too few notes to meet the workload target;
- missing predictions dominate the observed error count; and
- PIG validation is unavailable.

A candidate may only become a recommendation after:

1. a complete authoritative PIG copy is discovered;
2. all applicable invalidity rules record zero PIG violations;
3. all feature-dependent denominators are explicit;
4. out-of-fold performance is acceptable overall and for every finger;
5. count, percentage, and exported note IDs reconcile; and
6. the selected queue is exported to Vite without modifying source TSVs or
   existing human verdicts.

## 15. Implementation architecture

```text
fingering_audit/
├── __main__.py            command-line interface
├── config.py              strict configuration and path discovery
├── contracts.py           immutable shared records and enums
├── manifest.py            hashes, stage records, and terminal markers
├── acquire.py             fail-closed PIG discovery
├── canonical.py           PianoVAM, GT, and PIG canonical tables
├── evidence.py            evidence validation and PIG gate
├── study.py               full experiment construction and summaries
├── pipeline.py            unattended stage orchestration
├── report.py              tables, report, figures, and verification
├── features/
│   ├── context.py         timing, pitch, density, chord, and sequence features
│   ├── ergonomic.py       finger-pair and crossing relations
│   └── model.py           vectorized HMM and disagreement features
├── filters/
│   └── strategies.py      blacklist, whitelist, and hybrid semantics
├── evaluation/
│   ├── labels.py          exact, hand, and within-hand error labels
│   ├── metrics.py         pooled and per-finger metrics
│   └── bootstrap.py       recording-cluster uncertainty intervals
├── config/research.yaml
└── evidence/
    ├── thresholds.yaml
    └── sources.bib
```

The pipeline stores:

```text
artifacts/fingering_audit/<run-id>/
├── manifest.json
├── pig_status.json
├── RECOMMENDATION_GATE_CLOSED.json or SUCCESS.json
├── data/
│   ├── canonical_notes.parquet
│   ├── ground_truth_labels.parquet
│   ├── features.parquet
│   └── selection_masks.parquet
├── results/
│   ├── filter_sets.csv
│   ├── individual_filters.csv
│   ├── per_finger.csv
│   ├── workload_per_finger.csv
│   ├── per_recording.csv
│   ├── error_types.csv
│   ├── overlap_matrix.csv
│   ├── threshold_sensitivity.csv
│   ├── excluded_rules.csv
│   ├── pareto_tiers.csv
│   └── all_results.parquet
└── report/
    ├── research_report.md
    ├── research_report.html
    └── figures/
```

Generated artifacts are intentionally ignored by Git.

## 16. Vite correction application

The existing React/Vite application remains the review interface.

The production build originally failed because Vite followed 212 generated
audio/video symlinks whose external dataset targets were not mounted. The
build now:

- disables Vite's unconditional public-directory copier;
- copies every available public asset; and
- skips only dangling generated media links.

Development-time paths and source data are unchanged. No candidate review
queue is exported while the recommendation gate is closed.

Build verification:

```bash
npm --prefix annotate run build
```

## 17. Reproducibility and verification

Preflight:

```bash
python -m fingering_audit preflight \
  --config fingering_audit/config/research.yaml
```

Full study:

```bash
./run_fingering_audit.sh --run-label publication-audit
```

Verify a run:

```bash
python -m fingering_audit report \
  --run-dir artifacts/fingering_audit/<run-id> \
  --verify-only
```

Final verification evidence:

- 35 Python tests passed;
- all 508,621 notes were present;
- all 1,800 GT labels were present;
- every one of `L1`–`L5` and `R1`–`R5` was reported;
- all 19 filter counts matched their persisted masks;
- all required report files were present;
- no source fingering TSV was modified; and
- the Vite production build succeeded while explicitly skipping 212 dangling
  media links.

## 18. Primary sources

1. Parncutt, R., Sloboda, J. A., Clarke, E. F., Raekallio, M., & Desain, P.
   (1997). *An Ergonomic Model of Keyboard Fingering for Melodic Fragments*.
   Music Perception, 14(4), 341–382.
   DOI: [10.2307/40285730](https://doi.org/10.2307/40285730).
2. Nakamura, E., Saito, Y., & Yoshii, K. (2020). *Statistical Learning and
   Estimation of Piano Fingering*. Information Sciences, 517, 68–85.
   DOI: [10.1016/j.ins.2019.12.068](https://doi.org/10.1016/j.ins.2019.12.068).
3. Pavlakos, G., Shan, D., Radosavovic, I., Kanazawa, A., Fouhey, D., &
   Malik, J. (2024). *Reconstructing Hands in 3D with Transformers*.
   CVPR 2024, 9826–9836.
   [CVF paper](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html).
4. Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G.,
   Chang, C.-L., & Grundmann, M. (2020). *MediaPipe Hands: On-device
   Real-time Hand Tracking*. [arXiv:2006.10214](https://arxiv.org/abs/2006.10214).
5. Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M. I. (2021).
   *Distribution-Free, Risk-Controlling Prediction Sets*. Journal of the ACM,
   68(6). DOI: [10.1145/3478535](https://doi.org/10.1145/3478535).
6. Saito, Y., & Nakamura, E. *PIG: Piano Fingering Dataset v1.02*.
   [Official dataset page](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/).

## 19. Final decision

The study does not support a defensible approximately 30,000-note publication
audit queue from the currently available evidence.

The 30,778-note candidate is rejected because of low and nonuniform error
recall. The 39,443-note practical-span candidate is more directly grounded in
published ergonomic research but performs even worse as an error detector.
The 35,000-note legacy set recalls only 6.63% of all GT errors.

Publication should wait until missing labels and the PIG gate are resolved.
Any later recommendation must be selected from validity-gated, out-of-fold
results rather than by moving thresholds toward a desired queue size.

## 20. Approved physical and missing-context extension design

This section is the implementation specification approved on 2026-07-24. It
supersedes the legacy interpretation of `impossible_fingering` but does not
alter the completed results in Sections 9–12. Those tables remain the baseline
against which the extension will be evaluated.

### 20.1 Objectives and non-objectives

The extension must:

1. alert every assigned note implicated in a defensible physical
   contradiction;
2. prevent any such alert from being removed by a blacklist, whitelist, or
   hybrid strategy;
3. keep missing or malformed labels in a separate publication-blocking data
   queue;
4. alert assigned notes near substantial `Noinfo` regions;
5. report the workload, authoritative recall, precision, enrichment, and
   per-finger distribution of every threshold variant; and
6. use the same rule implementation for the research pipeline and Vite data
   preparation.

The extension must not:

- call a sequential crossing, leap, or awkward transition physically
  impossible;
- infer simultaneous notes by inventing an offset;
- choose a threshold merely to approach 30,000 notes;
- count a missing prediction as an assigned finger to audit; or
- permit a physical-invalidity rule to enter a recommended set unless it has
  zero violations on the complete authoritative PIG copy.

### 20.2 Queue model

Every PianoVAM note can carry independent boolean flags and reason tokens from
three queues:

| Queue | Population | Meaning | Publication effect |
|---|---|---|---|
| `physical_must_alert` | assigned fingers only | a PIG-gated simultaneous physical contradiction | always included in human fingering audit |
| `data_integrity_must_resolve` | incomplete or malformed records | no auditable assigned finger or insufficient valid source data | blocks publication but is not assigned-finger audit workload |
| `noinfo_context_alert` | assigned fingers only | nearby missing labels make detector output less trustworthy | always included using the selected threshold variant |

All other ergonomic, detector, model, and musical-context features remain risk
signals. For a strategy \(s\) and missing-context variant \(v\):

```text
assigned_audit(s, v)
    = physical_must_alert
      OR noinfo_context_alert(v)
      OR strategy_risk_selection(s)

publication_blockers
    = assigned_audit(s, v)
      UNION data_integrity_must_resolve
```

Reports must show `assigned_audit` and `data_integrity_must_resolve`
separately. The assigned-finger hard-note percentage uses all 508,621 notes as
its denominator for continuity with the baseline tables, but its numerator
must not contain `Noinfo` records.

### 20.3 Shared physical-rule engine

A single module will accept a canonical note table and return note-level masks
plus reason tokens. Thin adapters may rename columns for the TSV/Vite and
research callers, but may not reimplement rule logic.

Required canonical fields are:

```text
recording_id, note_id, onset_sec, offset_sec, pitch,
pred_hand, pred_finger, compound_fingering
```

The physical engine will:

- partition notes by recording and assigned hand;
- inspect every pair of simultaneously depressed keys, not only adjacent rows;
- use strict interval overlap,
  `later_onset < earlier_offset - 0.001 seconds`;
- treat the 1 ms timing epsilon as numerical tolerance, not a musical
  threshold, and keep it explicit and versioned;
- return data-integrity reasons instead of fabricating overlap when an onset,
  offset, pitch, hand, or finger is invalid; and
- flag both notes participating in a physical contradiction.

The implementation should use a sweep over active notes so its cost depends on
the number of active same-hand notes rather than comparing every pair in a
recording.

### 20.4 Physical must-alert rules

#### 20.4.1 Same finger on separated simultaneous keys

Two different pitches assigned to the same within-hand finger during a strict
key-depression overlap form an invalidity candidate. PIG validation excludes a
pair whenever either note has a compound fingering token. This conservative
treatment preserves finger-substitution semantics instead of flattening the
token to its first component.

The rule becomes `physical_must_alert` only when the complete PIG validation
returns zero violations. Otherwise the recommendation gate closes and the
rule is excluded from every recommended must-alert mask.

#### 20.4.2 Simultaneous finger-pair span

Consecutive melodic span boundaries in Parncutt et al. are ergonomic risk
limits, not universal physical limits. They therefore cannot be copied
directly into the invalidity layer.

For each unordered finger pair, the simultaneous invalidity boundary will be:

```text
physical_boundary(pair)
    = max(
        abs(published MinPrac boundary),
        abs(published MaxPrac boundary),
        maximum absolute valid simultaneous span for that pair observed in PIG
      )
```

Only spans strictly beyond this boundary are invalidity candidates. Equality
is accepted. Taking both practical directions and the observed PIG maximum
avoids asserting a smaller physical reach merely because of hand direction.
If a pair has no valid PIG coverage, no physical boundary is asserted for that
pair; its span remains an ergonomic risk signal. The frozen boundary artifact
must record:

- PIG version and dataset checksum;
- source rule and table locator;
- observed PIG maximum and observation count by pair and hand;
- chosen boundary;
- number and IDs of PIG violations; and
- implementation version and timing epsilon.

This maximum-of-two-sources construction is intentionally conservative. It
cannot establish a universal anatomical limit, so reports must describe it as
a PIG-authorized audit contradiction rather than a claim about every possible
pianist.

#### 20.4.3 Rules explicitly excluded from physical invalidity

The following remain risk signals even if the legacy Vite selector previously
placed them under `impossible_fingering`:

- non-thumb crossing;
- sequential finger-pair span;
- large sequential leap;
- fast position-change rate;
- hand-region overlap; and
- stepwise finger-order violation.

### 20.5 Data-integrity queue

`data_integrity_must_resolve` includes:

- missing hand or finger;
- hand outside `L`/`R`;
- finger outside 1–5;
- missing or invalid pitch;
- missing onset or offset; and
- offset earlier than onset.

Each record retains a specific reason token. Invalid timing prevents physical
overlap evaluation; it must never be replaced by an assumed 0.5-second note.
These records remain publication blockers but are excluded from
assigned-finger recall and predicted-finger workload tables.

### 20.6 `Noinfo`-context alert family

An assigned note is a missing-context candidate when it falls within a
recording-wide musical-sequence radius around a consecutive run of `Noinfo`
assignments. Notes are sorted stably by onset and source note index. This is
the primary definition because a missing record commonly has no reliable hand
with which to construct a hand-specific sequence.

When a missing-finger record still has a valid hand, the study also computes a
separately named same-hand variant. It is never substituted silently for the
recording-wide definition.

The fixed sensitivity grid is:

| Dimension | Values |
|---|---|
| minimum consecutive `Noinfo` run | 2, 3, 5 notes |
| assigned-note radius on each side | 1, 2, 4 assigned notes |

The existing rule—run length at least 3 with two assigned context notes on
each side—is retained as `legacy_noinfo_3_r2`. Every combination in the
3-by-3 grid is evaluated and reported; none is called research-derived.

The study will also evaluate fold-calibrated local-missingness variants.
For each assigned note it will compute in recording-wide order, and in the
available-hand subset when possible:

- adjacent `Noinfo` run length;
- number and proportion of `Noinfo` labels in centered 5-, 9-, and 17-note
  sequence windows; and
- distance in notes and seconds to the nearest `Noinfo` run.

Any empirical cutoff must be fitted on the training recordings of each
leave-one-recording-out fold. Held-out labels cannot influence its threshold.
The predeclared upper-tail variants are the 99.5th, 99th, and 97.5th
percentiles of nonzero training-fold local missingness. The threshold family
and selection rule are therefore frozen before viewing pooled test results.

The final report must contain one row per fixed and fold-calibrated variant
with:

- threshold definition and evidence grade;
- selected assigned notes and hard-note percentage;
- GT recall and assigned-finger recall;
- precision and error enrichment;
- selected notes and recall for `L1`–`L5` and `R1`–`R5`;
- incremental workload and errors beyond `physical_must_alert`; and
- overlap with each blacklist, whitelist, and hybrid risk strategy.

The final recommended missing-context variant is selected for authoritative
out-of-fold validity and recall, not proximity to a target note count. If no
variant demonstrates defensible held-out value, the report retains the table
and closes the recommendation gate rather than silently choosing the legacy
threshold.

### 20.7 Filter-set integration

The report will preserve the existing risk-only masks so their marginal
behavior remains inspectable. It will add:

1. standalone physical, integrity, and `Noinfo`-context rows;
2. every risk strategy combined with each fixed `Noinfo` variant and the
   invariant physical mask; and
3. a smaller comparison table using the chosen `Noinfo` variant, if one passes
   the recommendation criteria.

Programmatic reconciliation must assert for every combined assigned-audit
mask that:

```text
physical_must_alert <= combined_mask
noinfo_context_alert(selected_variant) <= combined_mask
combined_mask AND data_integrity_must_resolve = empty
```

This makes it impossible for whitelist logic to sieve an alerted physical
contradiction or for integrity records to leak into assigned-finger metrics.

### 20.8 Vite data contract

Prepared review JSON will add:

```text
physical_must_alert: boolean
physical_reasons: string[]
data_integrity_must_resolve: boolean
data_integrity_reasons: string[]
noinfo_context_alert: boolean
noinfo_context_reasons: string[]
```

The existing `is_hard` and `hard_reasons` fields remain available for
backward compatibility. Their value is the union of enabled assigned-audit
reasons; integrity reasons remain separately identifiable. Priority order is:

1. physical must-alert;
2. data-integrity must-resolve;
3. `Noinfo`-context alert; and
4. other strategy risks.

The Vite client displays the category and reason but does not recompute the
rules. Python preprocessing remains the authoritative implementation.

### 20.9 Failure behavior

The extension fails closed when:

- PIG is unavailable or incomplete;
- the PIG checksum differs from the frozen boundary artifact;
- an invalidity validator is missing;
- any enabled physical rule has one or more PIG violations;
- a combined strategy omits a mandatory physical or selected
  `Noinfo`-context note;
- source and exported note counts do not reconcile; or
- a required per-finger group is absent from reporting.

A failed gate may still produce diagnostic sensitivity tables, but it may not
write a recommended Vite queue or `SUCCESS.json`.

### 20.10 Test and acceptance criteria

Implementation follows test-first development. Automated tests must
demonstrate:

- same-finger overlap flags both different-pitch notes;
- repeated pitch and boundary-touching intervals are not false overlaps;
- non-adjacent active chord notes are compared;
- missing offsets produce integrity flags, not invented physical alerts;
- exact span boundaries pass and strictly larger spans flag;
- uncovered finger pairs stay risk-only;
- compound PIG tokens are handled without flattening their semantics;
- non-thumb crossings never enter `physical_must_alert`;
- all nine fixed `Noinfo` variants select the expected recording-wide context;
- recording-wide and available-hand `Noinfo` variants remain explicitly
  distinguishable;
- every combined mask contains both mandatory assigned-alert masks;
- integrity records are absent from assigned-finger denominators;
- PIG violations close the recommendation gate and expose note IDs;
- per-finger totals reconcile with overall totals; and
- Vite preparation serializes the new category fields and preserves legacy
  fields.

Completion requires a fresh full Python test run, a fresh Vite production
build, successful artifact reconciliation, and a consolidated-document update
with the resulting physical-rule and `Noinfo` sensitivity tables. No result
may be described as recommendable while the authoritative PIG gate is closed.

# Physical Must-Alert and Noinfo Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the conflated legacy “impossible fingering” rule with shared,
PIG-gated physical and integrity layers, add configurable `Noinfo`-context
alerts, and report each strategy/threshold combination overall and by finger.

**Architecture:** `fingering_audit/features/audit_flags.py` will compute
integrity, simultaneous physical-candidate, and missing-context masks for both
the research pipeline and Vite preprocessing.
`fingering_audit/physical_policy.py` will derive and validate PIG-authorized
simultaneous-span boundaries. The study will preserve marginal risk masks,
then union every strategy with the enabled physical layer and each `Noinfo`
variant.

**Tech Stack:** Python 3.10, pandas, NumPy, PyYAML, pytest, React 18, Vite 5.

## Global Constraints

- PIG v1.02 annotations are authoritative valid fingerings.
- Invalidity rules require zero complete-PIG violations before setting
  `physical_must_alert`.
- Strict overlap is `later_onset < earlier_offset - 0.001 seconds`.
- Invalid offsets create integrity flags; rule evaluation never invents a
  0.5-second offset.
- Crossings, leaps, sequential spans, and order changes remain risk signals.
- Integrity records are excluded from assigned-finger workload and recall.
- Fixed `Noinfo` runs are 2, 3, and 5; radii are 1, 2, and 4.
- Missingness windows are 5, 9, and 17 notes; fold tails are 99.5%, 99%, and
  97.5%.
- Thresholds are never selected to approach 30,000 notes.
- Source TSVs and existing human verdicts remain immutable.
- This file remains the single user-facing design, plan, and results document.

## File map

| File | Responsibility |
|---|---|
| `fingering_audit/features/audit_flags.py` | validation, overlap sweep, physical candidates, `Noinfo` contexts |
| `fingering_audit/physical_policy.py` | practical/PIG boundaries, PIG checks, policy serialization |
| `fingering_audit/evidence/thresholds.yaml` | rule evidence and variants |
| `fingering_audit/study.py` | queue masks, filter cross-products, sensitivity rows |
| `fingering_audit/report.py` | queue tables and report rendering |
| `fingering_audit/pipeline.py` | policy lifecycle, artifacts, gates |
| `ManualCheck/hard_part_selector.py` | shared-engine compatibility adapter |
| `annotate/prepare_review_data.py` | JSON category serialization |
| `annotate/src/App.jsx` | category priority and reason display |
| `tests/fingering_audit/test_audit_flags.py` | physical, integrity, and `Noinfo` tests |
| `tests/fingering_audit/test_evidence.py` | PIG policy and gate tests |
| `tests/fingering_audit/test_filters.py` | mandatory-union invariants |
| `tests/fingering_audit/test_report.py` | sensitivity and per-finger artifacts |
| `tests/test_prepare_review_data.py` | Vite JSON compatibility |

---

### Task 1: Canonical integrity and simultaneous-note engine

**Files:**

- Create: `fingering_audit/features/audit_flags.py`
- Create: `tests/fingering_audit/test_audit_flags.py`

**Interfaces:**

- Produces `compute_audit_flags(notes, span_boundaries=None,
  timing_epsilon_sec=0.001) -> AuditFlags`.
- `AuditFlags` contains `integrity`, `integrity_reasons`,
  `same_finger_candidate`, `span_candidate`, `physical_candidate`, and
  `physical_reasons`.
- Finger-pair keys are ascending strings such as `"1-5"`.

- [ ] **Step 1: Write failing integrity and overlap tests**

```python
import pandas as pd

from fingering_audit.features.audit_flags import compute_audit_flags


def fixture_notes(rows):
    records = []
    for index, values in enumerate(rows):
        records.append({
            "recording_id": "r",
            "note_id": f"r#{index}",
            "note_idx": index,
            "compound_fingering": False,
            **values,
        })
    return pd.DataFrame.from_records(records)


def test_invalid_offset_is_integrity_not_physical():
    frame = fixture_notes([{
        "onset_sec": 0.0, "offset_sec": None, "pitch": 60,
        "pred_hand": "R", "pred_finger": 1,
    }])
    flags = compute_audit_flags(frame, {"1-5": 16})
    assert flags.integrity.tolist() == [True]
    assert "missing_offset" in flags.integrity_reasons.iloc[0]
    assert flags.physical_candidate.tolist() == [False]


def test_same_finger_overlap_flags_both_notes():
    frame = fixture_notes([
        {"onset_sec": 0.0, "offset_sec": 1.0, "pitch": 60,
         "pred_hand": "R", "pred_finger": 2},
        {"onset_sec": 0.2, "offset_sec": 0.8, "pitch": 64,
         "pred_hand": "R", "pred_finger": 2},
    ])
    flags = compute_audit_flags(frame)
    assert flags.same_finger_candidate.tolist() == [True, True]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_audit_flags.py -v
```

Expected: `ModuleNotFoundError` for `audit_flags`.

- [ ] **Step 3: Implement the immutable result and active-note sweep**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class AuditFlags:
    integrity: pd.Series
    integrity_reasons: pd.Series
    same_finger_candidate: pd.Series
    span_candidate: pd.Series
    physical_candidate: pd.Series
    physical_reasons: pd.Series


def _pair_key(first, second):
    return f"{min(first, second)}-{max(first, second)}"


def compute_audit_flags(
    notes: pd.DataFrame,
    span_boundaries: Mapping[str, int] | None = None,
    *,
    timing_epsilon_sec: float = 0.001,
) -> AuditFlags:
    work = notes.reset_index(drop=True)
    hand = work["pred_hand"].astype("string")
    finger = pd.to_numeric(work["pred_finger"], errors="coerce")
    pitch = pd.to_numeric(work["pitch"], errors="coerce")
    onset = pd.to_numeric(work["onset_sec"], errors="coerce")
    offset = pd.to_numeric(work["offset_sec"], errors="coerce")
    reason_sets = [set() for _ in range(len(work))]
    for index in work.index:
        if hand.loc[index] not in {"L", "R"}:
            reason_sets[index].add("missing_or_invalid_hand")
        if pd.isna(finger.loc[index]) or not 1 <= finger.loc[index] <= 5:
            reason_sets[index].add("missing_or_invalid_finger")
        if pd.isna(pitch.loc[index]) or not 0 <= pitch.loc[index] <= 127:
            reason_sets[index].add("missing_or_invalid_pitch")
        if pd.isna(onset.loc[index]):
            reason_sets[index].add("missing_onset")
        if pd.isna(offset.loc[index]):
            reason_sets[index].add("missing_offset")
        elif not pd.isna(onset.loc[index]) and offset.loc[index] < onset.loc[index]:
            reason_sets[index].add("offset_before_onset")
    integrity = pd.Series(
        [bool(value) for value in reason_sets], index=work.index
    )
    same_finger = pd.Series(False, index=work.index)
    span = pd.Series(False, index=work.index)
    physical_reason_sets = [set() for _ in range(len(work))]
    valid = work.loc[~integrity].copy()
    valid["_hand"] = hand.loc[valid.index]
    valid["_finger"] = finger.loc[valid.index].astype(int)
    valid["_pitch"] = pitch.loc[valid.index].astype(int)
    valid["_onset"] = onset.loc[valid.index]
    valid["_offset"] = offset.loc[valid.index]
    for _, group in valid.groupby(["recording_id", "_hand"], sort=False):
        active = []
        ordered = group.sort_values(["_onset", "note_idx"], kind="stable")
        for current in ordered.index:
            current_onset = float(valid.at[current, "_onset"])
            active = [
                earlier for earlier in active
                if float(valid.at[earlier, "_offset"])
                > current_onset + timing_epsilon_sec
            ]
            for earlier in active:
                if valid.at[earlier, "_pitch"] == valid.at[current, "_pitch"]:
                    continue
                first = int(valid.at[earlier, "_finger"])
                second = int(valid.at[current, "_finger"])
                simple = not bool(work.at[earlier, "compound_fingering"])
                simple = simple and not bool(
                    work.at[current, "compound_fingering"]
                )
                if simple and first == second:
                    same_finger.loc[[earlier, current]] = True
                    physical_reason_sets[earlier].add(
                        "same_finger_simultaneous_keys"
                    )
                    physical_reason_sets[current].add(
                        "same_finger_simultaneous_keys"
                    )
                boundary = (span_boundaries or {}).get(
                    _pair_key(first, second)
                )
                distance = abs(
                    int(valid.at[earlier, "_pitch"])
                    - int(valid.at[current, "_pitch"])
                )
                if simple and boundary is not None and distance > boundary:
                    span.loc[[earlier, current]] = True
                    physical_reason_sets[earlier].add(
                        "simultaneous_span_beyond_policy"
                    )
                    physical_reason_sets[current].add(
                        "simultaneous_span_beyond_policy"
                    )
            active.append(current)
    integrity_reasons = pd.Series(
        [tuple(sorted(value)) for value in reason_sets], index=work.index
    )
    physical_reasons = pd.Series(
        [tuple(sorted(value)) for value in physical_reason_sets],
        index=work.index,
    )
    return AuditFlags(
        integrity=integrity,
        integrity_reasons=integrity_reasons,
        same_finger_candidate=same_finger,
        span_candidate=span,
        physical_candidate=same_finger | span,
        physical_reasons=physical_reasons,
    )
```

The implementation must be linear in notes plus active chord pairs and must
not compare every pair in an entire recording.

- [ ] **Step 4: Add strict-boundary and non-adjacent chord tests**

```python
def test_touching_intervals_and_repeated_pitch_do_not_flag():
    frame = fixture_notes([
        {"onset_sec": 0.0, "offset_sec": 0.5, "pitch": 60,
         "pred_hand": "R", "pred_finger": 2},
        {"onset_sec": 0.5, "offset_sec": 1.0, "pitch": 64,
         "pred_hand": "R", "pred_finger": 2},
        {"onset_sec": 0.1, "offset_sec": 0.4, "pitch": 60,
         "pred_hand": "R", "pred_finger": 2},
    ])
    assert not compute_audit_flags(frame).physical_candidate.any()


def test_span_is_strict_and_checks_non_adjacent_active_notes():
    frame = fixture_notes([
        {"onset_sec": 0.0, "offset_sec": 1.0, "pitch": 48,
         "pred_hand": "R", "pred_finger": 1},
        {"onset_sec": 0.1, "offset_sec": 0.9, "pitch": 60,
         "pred_hand": "R", "pred_finger": 3},
        {"onset_sec": 0.2, "offset_sec": 0.8, "pitch": 65,
         "pred_hand": "R", "pred_finger": 5},
    ])
    flags = compute_audit_flags(
        frame,
        {"1-5": 16, "1-3": 12, "3-5": 7},
    )
    assert flags.span_candidate.tolist() == [True, False, True]
```

- [ ] **Step 5: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_audit_flags.py -v
git add fingering_audit/features/audit_flags.py \
  tests/fingering_audit/test_audit_flags.py
git commit -m "feat: add canonical physical audit flags"
```

Expected: all Task 1 tests pass before the commit.

### Task 2: PIG-authorized physical policy

**Files:**

- Create: `fingering_audit/physical_policy.py`
- Modify: `fingering_audit/evidence.py`
- Modify: `fingering_audit/evidence/thresholds.yaml`
- Modify: `tests/fingering_audit/test_evidence.py`

**Interfaces:**

- Produces immutable `PhysicalPolicy(span_boundaries, observed_maxima,
  observation_counts, enabled_rules, validations, pig_sha256)`.
- Produces `derive_physical_policy(pig_notes, pig_root) -> PhysicalPolicy`.
- Produces `write_physical_policy(policy, path) -> Path`.

- [ ] **Step 1: Write failing policy and compound-token tests**

```python
from fingering_audit.physical_policy import (
    PRACTICAL_ABS,
    derive_physical_policy,
)


def test_policy_uses_practical_or_pig_maximum():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec"]] = [0.25, 0.75]
    policy = derive_physical_policy(pig, FIXTURES / "PIG")
    assert policy.observation_counts["1-2"] == 1
    assert policy.span_boundaries["1-2"] >= PRACTICAL_ABS["1-2"]
    assert policy.validations["simultaneous_pair_span"].violation_count == 0
    assert policy.pig_sha256


def test_compound_tokens_are_excluded_from_simple_invalidity():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[0, "finger"] = 4
    pig.loc[2, ["onset_sec", "offset_sec"]] = [0.1, 0.4]
    policy = derive_physical_policy(pig, FIXTURES / "PIG")
    assert policy.validations[
        "simultaneous_same_finger_different_pitch"
    ].violation_count == 0
```

- [ ] **Step 2: Verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_evidence.py -v
```

Expected: import fails because `physical_policy` is absent.

- [ ] **Step 3: Implement direct PIG active-pair maxima and validations**

```python
import hashlib
from dataclasses import asdict, dataclass

import pandas as pd
import yaml

from fingering_audit.contracts import PigValidation
from fingering_audit.features.audit_flags import compute_audit_flags


PRACTICAL_ABS = {
    "1-2": 10, "1-3": 12, "1-4": 14, "1-5": 15,
    "2-3": 5, "2-4": 7, "2-5": 10,
    "3-4": 4, "3-5": 7, "4-5": 5,
}


@dataclass(frozen=True)
class PhysicalPolicy:
    span_boundaries: dict[str, int]
    observed_maxima: dict[str, int]
    observation_counts: dict[str, int]
    enabled_rules: frozenset[str]
    validations: dict[str, PigValidation]
    pig_sha256: str


def sha256_dataset_tree(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def pig_to_canonical(pig):
    result = pig.rename(columns={
        "pig_note_id": "note_id",
        "hand": "pred_hand",
        "finger": "pred_finger",
    }).copy()
    result["recording_id"] = (
        result["piece_id"].astype(str)
        + "-"
        + result["performer_id"].astype(str)
    )
    result["note_idx"] = result["note_index"]
    return result.reset_index(drop=True)


def simultaneous_pair_maxima(simple):
    canonical = pig_to_canonical(simple)
    maxima = {}
    counts = {}
    for _, group in canonical.groupby(
        ["recording_id", "pred_hand"], sort=False
    ):
        active = []
        ordered = group.sort_values(
            ["onset_sec", "note_idx"], kind="stable"
        )
        for row in ordered.itertuples():
            active = [
                earlier for earlier in active
                if float(earlier.offset_sec) > float(row.onset_sec) + 0.001
            ]
            for earlier in active:
                if int(earlier.pred_finger) == int(row.pred_finger):
                    continue
                pair = "-".join(map(str, sorted([
                    int(earlier.pred_finger), int(row.pred_finger)
                ])))
                distance = abs(int(earlier.pitch) - int(row.pitch))
                maxima[pair] = max(maxima.get(pair, 0), distance)
                counts[pair] = counts.get(pair, 0) + 1
            active.append(row)
    return maxima, counts


def validations_from_flags(canonical, flags):
    same_ids = tuple(
        canonical.loc[
            flags.same_finger_candidate.to_numpy(), "note_id"
        ].astype(str)
    )
    span_ids = tuple(
        canonical.loc[
            flags.span_candidate.to_numpy(), "note_id"
        ].astype(str)
    )
    return {
        "simultaneous_same_finger_different_pitch": PigValidation(
            rule_id="simultaneous_same_finger_different_pitch",
            status="pass" if not same_ids else "fail",
            violation_count=len(same_ids),
            violating_ids=same_ids,
        ),
        "simultaneous_pair_span": PigValidation(
            rule_id="simultaneous_pair_span",
            status="pass" if not span_ids else "fail",
            violation_count=len(span_ids),
            violating_ids=span_ids,
        ),
    }


def derive_physical_policy(pig_notes, pig_root):
    simple = pig_notes.loc[~pig_notes["compound_fingering"]]
    maxima, counts = simultaneous_pair_maxima(simple)
    boundaries = {
        pair: max(PRACTICAL_ABS[pair], maxima[pair])
        for pair in PRACTICAL_ABS
        if counts.get(pair, 0) > 0
    }
    canonical = pig_to_canonical(simple)
    flags = compute_audit_flags(canonical, boundaries)
    validations = validations_from_flags(canonical, flags)
    enabled = frozenset(
        key for key, value in validations.items() if value.status == "pass"
    )
    return PhysicalPolicy(
        boundaries, maxima, counts, enabled, validations,
        sha256_dataset_tree(pig_root),
    )
```

- [ ] **Step 4: Add ledger entries for both invalidity rules**

```yaml
  - rule_id: simultaneous_same_finger_different_pitch
    kind: invalidity
    feature: simultaneous_same_finger_different_pitch
    unit: boolean
    operator: is_true
    evidence_grade: physical_invariant
    source_keys: [pig_dataset_v102]
    applicability: valid_assigned_same_hand_overlapping_keys
    may_select_alone: true
    implementation_version: 1
    sensitivity_variants:
      strict_overlap_1ms: {timing_epsilon_sec: 0.001}

  - rule_id: simultaneous_pair_span
    kind: invalidity
    feature: simultaneous_pair_span_beyond_policy
    unit: semitones
    operator: strictly_greater_than_pair_boundary
    evidence_grade: physical_invariant
    source_keys: [parncutt1997ergonomic, pig_dataset_v102]
    applicability: valid_assigned_same_hand_overlapping_keys
    may_select_alone: true
    implementation_version: 1
    sensitivity_variants:
      pig_authorized_max: {timing_epsilon_sec: 0.001}
```

Include explicit rationales matching Sections 20.4.1 and 20.4.2.

- [ ] **Step 5: Serialize the auditable policy**

```python
def write_physical_policy(policy, path):
    payload = {
        "schema_version": 1,
        "pig_sha256": policy.pig_sha256,
        "timing_epsilon_sec": 0.001,
        "span_boundaries": dict(policy.span_boundaries),
        "observed_maxima": dict(policy.observed_maxima),
        "observation_counts": dict(policy.observation_counts),
        "enabled_rules": sorted(policy.enabled_rules),
        "validations": {
            key: asdict(value) for key, value in policy.validations.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path
```

- [ ] **Step 6: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_evidence.py \
  tests/fingering_audit/test_audit_flags.py -v
git add fingering_audit/physical_policy.py fingering_audit/evidence.py \
  fingering_audit/evidence/thresholds.yaml \
  tests/fingering_audit/test_evidence.py
git commit -m "feat: gate physical rules with PIG"
```

Expected: all selected tests pass.

### Task 3: Fixed and calibrated `Noinfo` contexts

**Files:**

- Modify: `fingering_audit/features/audit_flags.py`
- Modify: `tests/fingering_audit/test_audit_flags.py`

**Interfaces:**

- Produces `noinfo_context_mask(notes, min_run, radius,
  sequence="recording") -> pd.Series`.
- Produces `local_missingness_features(notes) -> pd.DataFrame` with fractions
  for windows 5/9/17 and nearest missing-note distances.

- [ ] **Step 1: Write failing fixed-grid tests**

```python
def noinfo_fixture(run_length):
    assert run_length in {3, 5}
    total = run_length + 4
    return pd.DataFrame({
        "recording_id": ["r"] * total,
        "note_id": [f"r#{i}" for i in range(total)],
        "note_idx": range(total),
        "onset_sec": [float(i) for i in range(total)],
        "offset_sec": [i + 0.5 for i in range(total)],
        "pitch": [60] * total,
        "pred_hand": ["R"] + [None] * run_length + ["L", "R", "L"],
        "pred_finger": pd.array(
            [1] + [None] * run_length + [2, 3, 4], dtype="Int64"
        ),
        "compound_fingering": False,
    })


def test_noinfo_context_uses_recording_order_and_only_selects_assigned():
    frame = noinfo_fixture(run_length=3)
    selected = noinfo_context_mask(frame, min_run=3, radius=2)
    assert selected.tolist() == [
        True, False, False, False, True, True, False,
    ]


def test_noinfo_grid_is_monotone():
    frame = noinfo_fixture(run_length=5)
    broad = noinfo_context_mask(frame, min_run=2, radius=4)
    strict = noinfo_context_mask(frame, min_run=5, radius=1)
    assert (strict <= broad).all()
```

- [ ] **Step 2: Verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_audit_flags.py -k noinfo -v
```

Expected: import fails because `noinfo_context_mask` is absent.

- [ ] **Step 3: Implement stable run/context selection**

```python
NOINFO_RUN_LENGTHS = (2, 3, 5)
NOINFO_CONTEXT_RADII = (1, 2, 4)


def noinfo_context_mask(notes, *, min_run, radius, sequence="recording"):
    if min_run not in NOINFO_RUN_LENGTHS:
        raise ValueError(f"unsupported min_run: {min_run}")
    if radius not in NOINFO_CONTEXT_RADII:
        raise ValueError(f"unsupported radius: {radius}")
    if sequence not in {"recording", "available_hand"}:
        raise ValueError(f"unsupported sequence: {sequence}")
    work = notes.reset_index(drop=True)
    assigned = (
        work["pred_hand"].isin(["L", "R"])
        & work["pred_finger"].between(1, 5).fillna(False)
    )
    selected = pd.Series(False, index=work.index)
    group_cols = ["recording_id"]
    eligible = work
    if sequence == "available_hand":
        group_cols.append("pred_hand")
        eligible = work.loc[work["pred_hand"].isin(["L", "R"])]
    for _, group in eligible.groupby(group_cols, sort=False):
        ordered = group.sort_values(["onset_sec", "note_idx"], kind="stable")
        positions = list(ordered.index)
        missing = (~assigned.loc[positions]).to_numpy()
        start = 0
        while start < len(positions):
            if not missing[start]:
                start += 1
                continue
            end = start
            while end < len(positions) and missing[end]:
                end += 1
            if end - start >= min_run:
                context = positions[max(0, start-radius):start]
                context += positions[end:min(len(positions), end+radius)]
                selected.loc[context] = assigned.loc[context]
            start = end
    return selected
```

- [ ] **Step 4: Write failing local-feature tests**

```python
def test_local_missingness_features_have_fixed_windows_and_distances():
    result = local_missingness_features(noinfo_fixture(run_length=3))
    assert list(result) == [
        "noinfo_fraction_w5", "noinfo_fraction_w9",
        "noinfo_fraction_w17", "nearest_noinfo_note_distance",
        "nearest_noinfo_time_distance_sec",
    ]
    assert result["noinfo_fraction_w5"].between(0, 1).all()
```

- [ ] **Step 5: Implement centered windows and two-pass distances**

```python
def local_missingness_features(notes):
    work = notes.reset_index(drop=True)
    missing = ~(
        work["pred_hand"].isin(["L", "R"])
        & work["pred_finger"].between(1, 5).fillna(False)
    )
    result = pd.DataFrame(index=work.index)
    note_distance = pd.Series(np.inf, index=work.index)
    time_distance = pd.Series(np.inf, index=work.index)
    for _, group in work.groupby("recording_id", sort=False):
        ordered = group.sort_values(["onset_sec", "note_idx"], kind="stable")
        positions = np.asarray(ordered.index)
        local_missing = missing.loc[positions].to_numpy()
        for width in (5, 9, 17):
            result.loc[positions, f"noinfo_fraction_w{width}"] = (
                pd.Series(local_missing.astype(float))
                .rolling(width, center=True, min_periods=1).mean().to_numpy()
            )
        missing_positions = np.flatnonzero(local_missing)
        for local_index, global_index in enumerate(positions):
            if len(missing_positions):
                nearest = missing_positions[
                    np.abs(missing_positions-local_index).argmin()
                ]
                note_distance.loc[global_index] = abs(nearest-local_index)
                time_distance.loc[global_index] = abs(
                    float(ordered.iloc[nearest]["onset_sec"])
                    - float(work.loc[global_index, "onset_sec"])
                )
    result["nearest_noinfo_note_distance"] = note_distance
    result["nearest_noinfo_time_distance_sec"] = time_distance
    return result
```

- [ ] **Step 6: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_audit_flags.py -v
git add fingering_audit/features/audit_flags.py \
  tests/fingering_audit/test_audit_flags.py
git commit -m "feat: add noinfo context variants"
```

Expected: all physical, integrity, and `Noinfo` tests pass.

### Task 4: Mandatory unions and sensitivity tables

**Files:**

- Modify: `fingering_audit/study.py`
- Modify: `fingering_audit/filters/strategies.py`
- Modify: `tests/fingering_audit/test_filters.py`
- Create: `tests/fingering_audit/test_report.py`

**Interfaces:**

- `build_study(config, physical_policy=None) -> StudyData`.
- `StudyData` adds `queue_masks_full`, `queue_masks_gt`, and
  `noinfo_sensitivity`.
- Produces `combine_mandatory(risk, physical, noinfo, integrity) -> pd.Series`.
- Combined IDs are `<risk_set_id>__ni_k<run>_r<radius>`.

- [ ] **Step 1: Write failing mandatory-union tests**

```python
def test_mandatory_union_is_complete_and_integrity_disjoint():
    result = combine_mandatory(
        risk=pd.Series([False, True, False, False]),
        physical=pd.Series([True, False, False, False]),
        noinfo=pd.Series([False, False, True, False]),
        integrity=pd.Series([False, False, False, True]),
    )
    assert result.tolist() == [True, True, True, False]


def test_mandatory_union_rejects_overlap_with_integrity():
    with pytest.raises(ValueError, match="integrity"):
        combine_mandatory(
            risk=pd.Series([False]), physical=pd.Series([True]),
            noinfo=pd.Series([False]), integrity=pd.Series([True]),
        )
```

- [ ] **Step 2: Verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_filters.py -k mandatory -v
```

Expected: import fails because `combine_mandatory` is absent.

- [ ] **Step 3: Implement the mandatory-union invariant**

```python
def combine_mandatory(*, risk, physical, noinfo, integrity):
    masks = [
        pd.Series(value).fillna(False).astype(bool).reset_index(drop=True)
        for value in (risk, physical, noinfo, integrity)
    ]
    risk_mask, physical_mask, noinfo_mask, integrity_mask = masks
    if (physical_mask & integrity_mask).any():
        raise ValueError("physical and integrity masks overlap")
    if (noinfo_mask & integrity_mask).any():
        raise ValueError("noinfo context and integrity masks overlap")
    return (risk_mask | physical_mask | noinfo_mask) & ~integrity_mask
```

- [ ] **Step 4: Add failing cross-product and table tests**

```python
def test_each_combined_set_contains_its_mandatory_masks(study):
    physical = study.queue_masks_full["physical_must_alert"]
    integrity = study.queue_masks_full["data_integrity_must_resolve"]
    for set_id, selected in study.selections_full.items():
        if "__ni_" not in set_id:
            continue
        variant = set_id.split("__", 1)[1]
        assert (physical <= selected).all()
        assert (study.queue_masks_full[variant] <= selected).all()
        assert not (selected & integrity).any()


def test_noinfo_table_has_nine_fixed_rows_and_finger_outputs(study):
    tables = summarize_study(study, "fixture", seed=7)
    fixed = tables["noinfo_sensitivity"].query("calibration == 'fixed'")
    assert len(fixed) == 9
    assert set(fixed["min_run"]) == {2, 3, 5}
    assert set(fixed["radius"]) == {1, 2, 4}
    assert {"gt_error_recall", "assigned_gt_error_recall",
            "gt_precision", "error_enrichment",
            "incremental_count_beyond_physical"} <= set(fixed)
```

- [ ] **Step 5: Integrate the queues and strategy cross-product**

```python
NOINFO_VARIANTS = {
    f"ni_k{run}_r{radius}": (run, radius)
    for run in (2, 3, 5) for radius in (1, 2, 4)
}


def _combined_sets(risk_sets, queue_masks):
    return {
        f"{risk_id}__{variant}": combine_mandatory(
            risk=risk,
            physical=queue_masks["physical_must_alert"],
            noinfo=queue_masks[variant],
            integrity=queue_masks["data_integrity_must_resolve"],
        )
        for risk_id, risk in risk_sets.items()
        for variant in NOINFO_VARIANTS
    }


def _oof_noinfo_tail(notes, labels, feature_values, *, quantile):
    score_by_id = pd.Series(
        feature_values.to_numpy(), index=notes["note_id"]
    )
    labeled_score = labels["note_id"].map(score_by_id)
    gt_mask = pd.Series(False, index=labels.index)
    threshold_rows = []
    fold_thresholds = []
    for held_out in sorted(labels["recording_id"].unique()):
        train = labels["recording_id"].ne(held_out)
        train_scores = labeled_score.loc[train]
        train_scores = train_scores.loc[train_scores.gt(0)].dropna()
        threshold = (
            float(train_scores.quantile(quantile))
            if len(train_scores) else float("inf")
        )
        test = labels["recording_id"].eq(held_out)
        gt_mask.loc[test] = (
            labeled_score.loc[test].ge(threshold).fillna(False)
        )
        fold_thresholds.append(threshold)
        threshold_rows.append({
            "held_out_recording": held_out,
            "quantile": quantile,
            "threshold": threshold,
            "train_nonzero_notes": len(train_scores),
        })
    deployment_threshold = float(np.median(fold_thresholds))
    assigned = (
        notes["pred_hand"].isin(["L", "R"])
        & notes["pred_finger"].between(1, 5).fillna(False)
    )
    full_mask = feature_values.ge(deployment_threshold).fillna(False)
    return (
        full_mask & assigned,
        gt_mask,
        pd.DataFrame.from_records(threshold_rows),
    )
```

Without a validated policy, retain `physical_candidate_diagnostic` but set
`physical_must_alert` false. Add fixed and training-fold missingness rows to
`StudyData.noinfo_sensitivity`; held-out recordings never fit their cutoffs.

- [ ] **Step 6: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_filters.py \
  tests/fingering_audit/test_report.py -v
git add fingering_audit/study.py fingering_audit/filters/strategies.py \
  tests/fingering_audit/test_filters.py \
  tests/fingering_audit/test_report.py
git commit -m "feat: evaluate mandatory audit unions"
```

Expected: every combined-set invariant and all sensitivity assertions pass.

### Task 5: ManualCheck adapter and Vite data contract

**Files:**

- Modify: `ManualCheck/hard_part_selector.py`
- Modify: `annotate/prepare_review_data.py`
- Modify: `annotate/src/App.jsx`
- Create: `tests/test_prepare_review_data.py`

**Interfaces:**

- `select_hard_parts` adds all three queue categories while preserving
  `is_hard` and `hard_reasons`.
- Prepared JSON includes the six fields in Section 20.8.
- The browser consumes precomputed flags and never recomputes rules.

- [ ] **Step 1: Write the failing JSON contract test**

```python
def test_review_json_separates_audit_categories():
    notes = [
        {"global_idx": 0, "onset_sec": 0.0, "offset_sec": 1.0,
         "pitch": 60, "algorithm_hand": "Right", "algorithm_finger": 2,
         "algorithm_int": 7},
        {"global_idx": 1, "onset_sec": 0.1, "offset_sec": 0.9,
         "pitch": 64, "algorithm_hand": "Right", "algorithm_finger": 2,
         "algorithm_int": 7},
    ]
    result = apply_hard_rules_to_notes(
        notes, ["physical_candidate_diagnostic"]
    )
    assert result[0]["physical_reasons"] == [
        "same_finger_simultaneous_keys"
    ]
    assert result[0]["data_integrity_reasons"] == []
    assert {"is_hard", "hard_reasons"} <= set(result[0])
```

- [ ] **Step 2: Verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/test_prepare_review_data.py -v
```

Expected: category fields are absent.

- [ ] **Step 3: Replace duplicated physical logic with a canonical adapter**

```python
def _canonical_for_audit(df):
    return pd.DataFrame({
        "recording_id": "manualcheck",
        "note_id": [f"manualcheck#{i}" for i in range(len(df))],
        "note_idx": range(len(df)),
        "onset_sec": pd.to_numeric(df.get("onset"), errors="coerce"),
        "offset_sec": pd.to_numeric(df.get("key_offset"), errors="coerce"),
        "pitch": pd.to_numeric(df.get("note"), errors="coerce"),
        "pred_hand": df.get("hand"),
        "pred_finger": pd.array(df.get("finger_int"), dtype="Int64"),
        "compound_fingering": False,
    })
```

Keep `rule_impossible_fingering` as a deprecated compatibility alias for
legacy risk output; it must never populate `physical_must_alert`. Use:

```python
DEFAULT_RULES = [
    "physical_candidate_diagnostic",
    "non_thumb_crossing",
    "fast_jump",
    "noinfo_context_k3_r2",
]
```

- [ ] **Step 4: Serialize stable category fields in every input path**

```python
def audit_fields(row):
    return {
        "physical_must_alert": bool(row.physical_must_alert),
        "physical_reasons": list(row.physical_reasons),
        "data_integrity_must_resolve": bool(
            row.data_integrity_must_resolve
        ),
        "data_integrity_reasons": list(row.data_integrity_reasons),
        "noinfo_context_alert": bool(row.noinfo_context_alert),
        "noinfo_context_reasons": list(row.noinfo_context_reasons),
    }
```

Call `audit_fields` from TSV, detector, ZIP, and MIDI construction so every
note has the same JSON schema.

- [ ] **Step 5: Add category-aware Vite priority**

```jsx
function explicitAuditPriority(n) {
  if (n?.physical_must_alert) {
    return { score: 110,
      reason: `physical must-alert: ${(n.physical_reasons || []).join(', ')}` };
  }
  if (n?.data_integrity_must_resolve) {
    return { score: 105,
      reason: `data integrity: ${(n.data_integrity_reasons || []).join(', ')}` };
  }
  if (n?.noinfo_context_alert) {
    return { score: 98,
      reason: `near Noinfo region: ${(n.noinfo_context_reasons || []).join(', ')}` };
  }
  return null;
}
```

Return `explicitAuditPriority(n)` at the start of `priorityForNote` when
non-null.

- [ ] **Step 6: Verify GREEN, build, and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/test_prepare_review_data.py -v
npm --prefix annotate run build
git add ManualCheck/hard_part_selector.py annotate/prepare_review_data.py \
  annotate/src/App.jsx tests/test_prepare_review_data.py
git commit -m "feat: expose audit queues in correction app"
```

Expected: contract tests pass and Vite exits 0.

### Task 6: Pipeline artifacts and recommendation gates

**Files:**

- Modify: `fingering_audit/pipeline.py`
- Modify: `fingering_audit/report.py`
- Modify: `tests/fingering_audit/test_manifest.py`
- Modify: `tests/fingering_audit/test_report.py`

**Interfaces:**

- A PIG-present run writes `data/physical_policy.yaml`.
- Every run writes `results/noinfo_sensitivity.csv`,
  `results/queue_summary.csv`, and
  `results/queue_workload_per_finger.csv`.
- Reconciliation verifies mandatory containment and integrity disjointness.

- [ ] **Step 1: Write failing artifact and gate tests**

```python
def test_required_results_include_queue_tables():
    assert "noinfo_sensitivity.csv" in REQUIRED_RESULTS
    assert "queue_summary.csv" in REQUIRED_RESULTS
    assert "queue_workload_per_finger.csv" in REQUIRED_RESULTS


def test_failed_mandatory_reconciliation_cannot_finalize(tmp_path):
    cfg = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    manifest = RunManifest.start(cfg, run_id="mandatory-failure")
    with pytest.raises(ValueError, match="mandatory"):
        manifest.finalize({
            "counts_match": True,
            "pig_gate": True,
            "mandatory_masks_contained": False,
        })
```

- [ ] **Step 2: Verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_manifest.py \
  tests/fingering_audit/test_report.py -v
```

Expected: artifact and gate assertions fail.

- [ ] **Step 3: Derive, validate, persist, and pass the policy**

```python
physical_policy = None
if pig_notes is not None:
    physical_policy = derive_physical_policy(pig_notes, pig_root)
    enforce_recommendation_gate(
        physical_policy.validations,
        physical_policy.validations.keys(),
    )
    write_physical_policy(
        physical_policy,
        manifest.run_dir / "data/physical_policy.yaml",
    )
study = build_study(config, physical_policy=physical_policy)
```

When PIG is absent, diagnostic candidates remain reportable,
`physical_must_alert` remains false, and the recommendation gate remains
closed.

- [ ] **Step 4: Add required queue files and reconciliations**

```python
REQUIRED_RESULTS += (
    "noinfo_sensitivity.csv",
    "queue_summary.csv",
    "queue_workload_per_finger.csv",
)

reconciliations["mandatory_masks_contained"] = all(
    (study.queue_masks_full["physical_must_alert"] <= mask).all()
    for set_id, mask in study.selections_full.items()
    if "__ni_" in set_id
)
reconciliations["integrity_disjoint_from_assigned"] = all(
    not (
        study.queue_masks_full["data_integrity_must_resolve"] & mask
    ).any()
    for set_id, mask in study.selections_full.items()
    if "__ni_" in set_id
)
```

- [ ] **Step 5: Render method and queue columns**

The generated report table must include these exact columns:

```python
QUEUE_REPORT_COLUMNS = [
    "base_risk_method", "physical_policy_status",
    "noinfo_min_run", "noinfo_context_radius",
    "hard_count", "hard_percentage_all_notes",
    "gt_error_recall", "assigned_gt_error_recall",
    "gt_precision", "error_enrichment",
    "incremental_count_beyond_physical",
    "incremental_errors_beyond_physical",
]
```

Per-predicted-finger workload and per-true-finger recall remain separate
tables keyed by `set_id`.

- [ ] **Step 6: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_manifest.py \
  tests/fingering_audit/test_report.py -v
git add fingering_audit/pipeline.py fingering_audit/report.py \
  tests/fingering_audit/test_manifest.py \
  tests/fingering_audit/test_report.py
git commit -m "feat: report physical and noinfo queues"
```

Expected: all selected tests pass and fixture verification finds every queue
table.

### Task 7: Full research run and delivery

**Files:**

- Modify: `docs/fingering-audit-complete.md`
- Generate, ignored: `artifacts/fingering_audit/<run-id>/`

**Interfaces:**

- Produces exact physical/noinfo workload-recall tables in this document.
- Produces no recommended Vite queue unless PIG and reconciliation gates pass.

- [ ] **Step 1: Run the complete Python suite**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest -q
```

Expected: exit 0 with zero failures.

- [ ] **Step 2: Run the full unattended audit**

```bash
./run_fingering_audit.sh --run-label physical-noinfo-audit
```

Expected: complete run. Without authoritative local PIG it writes
`RECOMMENDATION_GATE_CLOSED.json`; with PIG it writes `SUCCESS.json` only
after validators and reconciliations pass.

- [ ] **Step 3: Verify the exact generated run**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m fingering_audit report \
  --verify-only
```

Expected: `verification_status` is `PASS` and the manifest truthfully reports
the PIG gate. With no `--run-dir`, verification selects the newest run written
by Step 2.

- [ ] **Step 4: Update consolidated results**

Update Sections 1, 10, 12, 14, 17, and 19 and append exact generated tables
for physical candidates, enabled must-alerts, integrity records, every
`Noinfo` variant, every strategy combination, GT/assigned recall, precision,
enrichment, incremental contribution, and all ten fingers.

- [ ] **Step 5: Build Vite**

```bash
npm --prefix annotate run build
```

Expected: exit 0 and only documented dangling-media skips.

- [ ] **Step 6: Verify diff and source immutability**

```bash
git diff --check
git status --short
git diff --name-only -- PianoVAM_v1.0/Fingering/
```

Expected: no whitespace errors, only intended tracked changes, and no source
fingering TSV changes.

- [ ] **Step 7: Commit the consolidated results and push**

```bash
git add docs/fingering-audit-complete.md
git commit -m "docs: record physical and noinfo audit results"
git push origin 260724-audit
```

Expected: GitHub updates `260724-audit` to the final verified commit.

## 21. Authoritative key-offset recovery design

### 21.1 Production-data finding

The first production run of Task 7 exposed a source-contract mismatch. All
105 fingering files contain exactly:

```text
onset, note, hand, finger, velocity
```

They contain 508,621 notes but no key-release timestamps. The annotation
application fills a missing display offset with `onset + 0.5` seconds. That
value is a UI playback fallback: all 508,621 archived display notes have the
same synthetic 0.5-second duration. It is forbidden as audit evidence.

Original key offsets are mandatory. The official
`PianoVAM/PianoVAM_v1` dataset contains native `TSV/*.tsv` files with:

```text
onset, key_offset, frame_offset, note, velocity
```

All 105 audited recording IDs have a native TSV at immutable dataset revision
`7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8`. Only those 105 TSVs are needed;
audio and video are not downloaded.

### 21.2 Considered approaches

1. **Pinned native TSV recovery — selected.** Download the official native
   TSV at the immutable revision and use its original `key_offset`. This
   preserves the dataset authors' timing representation and is the smallest
   authoritative input.
2. **Original MIDI reconstruction.** Parse note-on/note-off pairs from the
   official MIDI. This is authoritative but adds tempo, pedal, track-merging,
   and pairing decisions that are unnecessary while native TSV is available.
   It remains a future cross-check, not the primary source.
3. **Synthetic or inferred durations — rejected.** The Vite `+0.5` fallback,
   nearest-neighbor timing, median durations, and interpolation are forbidden.

### 21.3 Acquisition and provenance contract

The unattended workflow downloads only:

```text
https://huggingface.co/datasets/PianoVAM/PianoVAM_v1/resolve/
7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8/TSV/<recording>.tsv
```

for the exact 105 recording IDs already present in the fingering corpus.
Downloads go to an ignored, revision-named source cache. A completed run
records repository ID, immutable revision, relative source path, byte count,
SHA-256, row count, and validation status for every file. Cache reuse is
allowed only after the same validations pass.

Network or source failure is fail-closed. Partial coverage, a moving
`main` reference, an unknown extra recording, or a source whose identity
cannot be proved must prevent study/report finalization. No fallback offset is
substituted.

### 21.4 Exact identity join

For each recording, the native timing TSV and the fingering TSV must have the
same row count. Both sides are keyed by:

```text
(recording_id, round(onset, 6), note)
```

This key is unique for all 508,621 current fingering rows. Velocity is an
additional equality validator. The loader rejects:

- missing or extra records;
- duplicate identity keys on either side;
- onset, pitch, or velocity disagreement;
- nonfinite onset or key offset;
- nonintegral/out-of-range pitch;
- `key_offset < onset`; or
- anything other than exact 105-recording and 508,621-row production
  coverage.

No nearest-neighbor or index-only join is permitted. The original
`key_offset` is copied into canonical `offset_sec`; the five-column fingering
files remain immutable.

### 21.5 Eligibility and terminal gates

Acquisition and identity validation occur before physical-rule, Noinfo,
ground-truth, or report computation. Thus the production study never treats
missing offsets as ordinary fingering errors.

The existing integrity queue continues to catch genuinely malformed timing
values after enrichment. Full-corpus and GT masks must use the same integrity
eligibility for every fold-calibrated rule. In particular, GT upper-tail masks
must be intersected with the GT mapping of the full eligibility mask; a rule
cannot select a GT note that the corresponding full-corpus rule excludes.

A terminal `SUCCESS.json` or `RECOMMENDATION_GATE_CLOSED.json` additionally
requires:

```text
105 timing files
508,621 exactly joined notes
0 missing offsets
0 identity mismatches
0 synthetic offsets
full/GT eligibility parity
```

PIG remains a separate validity gate. Authoritative offsets open physical
timing evaluation; they do not authorize any invalidity rule that fails or
cannot be checked against PIG.

### 21.6 Acceptance tests

The implementation must demonstrate RED then GREEN tests for:

- pinned URL construction and exact recording-only acquisition;
- cache hashing and reuse without network;
- rejection of a moving revision or partial download;
- exact native-TSV enrichment with original offsets;
- rejection of row-count, duplicate-key, onset, pitch, velocity, and timing
  mismatches;
- proof that no code path substitutes `onset + 0.5`;
- unchanged source fingering TSV hashes;
- full/GT eligibility parity for upper-tail and queue masks;
- exact production coverage in preflight, manifest, and report verification;
  and
- a full unattended rerun whose measured tables replace the invalid
  all-integrity artifact.

# Authoritative Key-Offset Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover and validate the original PianoVAM key-release timestamps
before computing physical, Noinfo, GT, or publication-audit results.

**Architecture:** A focused acquisition module retrieves the official native
TSVs at one immutable Hugging Face revision. A timing module validates and
joins those inputs into canonical notes. The pipeline records provenance and
requires exact timing coverage and GT/full eligibility parity before terminal
markers.

**Tech Stack:** Python 3, `urllib`, pandas, NumPy, pytest, YAML, SHA-256,
React/Vite verification.

## Global constraints

- Never use `onset + 0.5`, inferred durations, or nearest-neighbor joins.
- Never edit the 105 source fingering TSV files.
- Download only official native TSVs for the exact audited recordings.
- Pin revision `7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8`.
- Fail closed unless all 105 files and all 508,621 notes validate exactly.
- Preserve the independent PIG invalidity gate and exact 189-queue gates.

### Task 8: Pinned timing acquisition and exact enrichment

**Files:**

- Create: `fingering_audit/timing.py`
- Modify: `fingering_audit/acquire.py`
- Modify: `fingering_audit/contracts.py`
- Modify: `fingering_audit/config.py`
- Modify: `fingering_audit/config/research.yaml`
- Modify: `fingering_audit/canonical.py`
- Test: `tests/fingering_audit/test_timing.py`
- Test: `tests/fingering_audit/test_canonical.py`
- Test: `tests/fingering_audit/test_config.py`

**Interfaces:**

- `ensure_authoritative_timing(config, recording_ids) -> TimingSource`
  returns the pinned cache directory and one provenance row per recording.
- `attach_authoritative_offsets(notes, timing_source) -> TimingJoin`
  returns enriched canonical notes plus validated per-recording provenance.
- `TimingJoin.complete` is true only for exact expected file and row coverage.

- [ ] **Step 1: Write acquisition RED tests**

Use a local HTTP fixture or injected downloader. Assert the URL contains the
exact repository, 40-character revision, and requested recording path.
Assert partial, wrong-revision, duplicate, and failed downloads raise before
returning `TimingSource`. Assert a valid hashed cache is reused without a
network call.

- [ ] **Step 2: Run acquisition tests and verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_timing.py -k acquisition -v
```

Expected: FAIL because the timing interfaces do not exist.

- [ ] **Step 3: Implement minimal pinned acquisition**

Add immutable repository/revision/cache fields to `AuditConfig`, reject a
non-40-hex revision, and implement exact-recording acquisition in
`fingering_audit/acquire.py`. Write atomically to the ignored revision cache;
calculate SHA-256 after download and on every reuse.

- [ ] **Step 4: Write exact-join RED tests**

Create small five-column fingering and native timing fixtures. Assert the
returned `offset_sec` equals the native `key_offset`, not `onset + 0.5`.
Individually assert rejection of row-count, duplicate-key, onset, pitch,
velocity, nonfinite timing, and offset-before-onset mismatches.

- [ ] **Step 5: Run join tests and verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_timing.py \
  tests/fingering_audit/test_canonical.py -v
```

Expected: new join tests fail for the intended missing behavior.

- [ ] **Step 6: Implement minimal exact enrichment**

Implement strict six-decimal onset/pitch identity plus velocity and count
validation in `fingering_audit/timing.py`. Update the canonical loader to
accept only a complete `TimingSource` for authoritative runs and retain
source-fingering hashes for immutability verification.

- [ ] **Step 7: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_timing.py \
  tests/fingering_audit/test_canonical.py \
  tests/fingering_audit/test_config.py -q
git diff --check
git add fingering_audit tests/fingering_audit \
  fingering_audit/config/research.yaml
git commit -m "feat: recover authoritative PianoVAM offsets"
```

Expected: all focused tests pass and no source TSV is modified.

### Task 9: Timing provenance and full/GT eligibility gates

**Files:**

- Modify: `fingering_audit/preflight.py`
- Modify: `fingering_audit/study.py`
- Modify: `fingering_audit/pipeline.py`
- Modify: `fingering_audit/report.py`
- Modify: `fingering_audit/manifest.py`
- Test: `tests/fingering_audit/test_features.py`
- Test: `tests/fingering_audit/test_manifest.py`
- Test: `tests/fingering_audit/test_report.py`

**Interfaces:**

- `build_study` consumes fully enriched canonical notes.
- `timing_provenance.csv` contains one validated row per recording.
- Terminal reconciliations include exact timing files/rows, zero missing
  offsets, source-hash immutability, and full/GT eligibility parity.

- [ ] **Step 1: Write eligibility and terminal-gate RED tests**

Add a malformed full note mapped to GT and prove an upper-tail GT mask cannot
select it when the full mask is ineligible. Add missing/extra timing-file and
row-count reconciliations and assert neither terminal marker is written.

- [ ] **Step 2: Run gate tests and verify RED**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_features.py \
  tests/fingering_audit/test_manifest.py \
  tests/fingering_audit/test_report.py -v
```

Expected: new parity and timing-gate tests fail for the intended reasons.

- [ ] **Step 3: Apply one shared eligibility mapping**

Intersect every GT rule mask, including `_oof_upper_tail`, with the GT mapping
of its full-corpus eligibility mask. Add direct parity reconciliation before
report or terminal finalization.

- [ ] **Step 4: Persist timing provenance and gates**

Write `data/timing_provenance.csv`; include repository, revision, relative
path, SHA-256, byte count, row count, and validation status. Add the six exact
timing reconciliations from Section 21.5 and make report verification require
them.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest -q
git diff --check
git add fingering_audit tests/fingering_audit
git commit -m "fix: gate audit on original timing coverage"
```

Expected: the complete Python suite passes with no terminal marker possible
under partial timing coverage.

### Task 10: Authoritative rerun, consolidated results, and delivery

**Files:**

- Modify: `docs/fingering-audit-complete.md`
- Generate, ignored: authoritative TSV cache and
  `artifacts/fingering_audit/<run-id>/`

**Interfaces:**

- Produces the exact physical/Noinfo workload-recall and all-ten-finger tables
  from original key offsets.
- Produces no recommended Vite queue unless both PIG and all reconciliations
  pass.

- [ ] **Step 1: Acquire and verify only the original TSV timing sources**

Run the unattended acquisition path. Confirm 105 files, 508,621 exact joins,
the pinned revision, and zero synthetic/missing offsets. Record all hashes in
the run artifact.

- [ ] **Step 2: Run all verification**

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest -q
./run_fingering_audit.sh --run-label authoritative-offset-audit
/home/junhyungp/autofinger/.venv/bin/python -m fingering_audit report \
  --verify-only
npm --prefix annotate run test:audit-categories
npm --prefix annotate run build
```

Expected: tests/build pass; report verification passes; with unavailable PIG,
the truthful terminal marker is `RECOMMENDATION_GATE_CLOSED.json`.

- [ ] **Step 3: Replace invalid production results**

Update Sections 1, 10, 12, 13, 14, 17, 19, and 21 with the authoritative run
ID, timing provenance, physical candidates versus enabled must-alerts,
integrity records, every Noinfo fixed/calibrated variant, all strategy
combinations, GT/assigned recall, precision, enrichment, incremental
contribution, and all ten fingers. Label the earlier all-integrity run invalid
and exclude it from conclusions.

- [ ] **Step 4: Verify immutability, commit, review, and push**

```bash
git diff --check
git diff --name-only -- PianoVAM_v1.0/Fingering/
git status --short
git add docs/fingering-audit-complete.md
git commit -m "docs: record authoritative-offset audit results"
```

Expected: no source fingering TSV changes. External push occurs only after
independent final review and fresh controller verification.


## Complete authoritative generated tables

These inline CSV tables are copied verbatim from the validated run snapshots. `NA` is the generated missing-value semantics; no PIG-derived validity is inferred.

### Physical candidates, enabled must-alerts, and integrity rows (exact generated values)

```csv
strategy,set_id,evidence_grade,threshold_summary,pig_status,recommendable,hard_count,hard_percentage_all_notes,hard_percentage_assigned_notes,gt_eligible_notes,gt_hard_count,gt_hard_percentage,gt_error_count,gt_selected_errors,gt_error_recall,gt_precision,gt_correct_sieve_rate,gt_enrichment,assigned_gt_error_recall,assigned_gt_precision,macro_finger_recall,worst_finger,cluster_count,replicates,error_recall_ci_low,error_recall_ci_high,precision_ci_low,precision_ci_high
blacklist,bl_step_crossing,exploratory,"crossing, IOI <=1000ms, and <=2 semitones","unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,3328,0.006543182448227659,0.00766161801032753,1800,7,0.0038888888888888888,392,3,0.007653061224489796,0.42857142857142855,0.002840909090909091,1.967930029154519,0.02564102564102564,0.42857142857142855,0.004545454545454545,L1,11,2000,0.0,0.015625,0.0,0.8
blacklist,bl_rate_q995,empirically_calibrated,LOPO 99.5th percentile,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,4745,0.009329146850012092,0.010923791303787298,1800,8,0.0044444444444444444,392,2,0.00510204081632653,0.25,0.004261363636363636,1.1479591836734695,0.017094017094017096,0.25,0.004015151515151515,L1,11,2000,0.0,0.01054889463362453,0.0,1.0
blacklist,bl_rate_q990,empirically_calibrated,LOPO 99th percentile,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,6351,0.0124867042454008,0.014621074514299922,1800,17,0.009444444444444445,392,4,0.01020408163265306,0.23529411764705882,0.009232954545454546,1.0804321728691477,0.03418803418803419,0.23529411764705882,0.010165945165945166,L1,11,2000,0.0,0.01815256525652565,0.0,0.4375
blacklist,bl_crossing,mixed,non-thumb crossing and IOI <=1000ms,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,8335,0.016387447627998056,0.019188577558918257,1800,20,0.011111111111111112,392,7,0.017857142857142856,0.35,0.009232954545454546,1.6071428571428572,0.05982905982905983,0.35,0.010227272727272727,L1,11,2000,0.00443337417823939,0.032573493995575256,0.125,0.5714285714285714
blacklist,bl_rate_q975,empirically_calibrated,LOPO 97.5th percentile,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,14655,0.028813202758045775,0.03373828483814602,1800,42,0.023333333333333334,392,6,0.015306122448979591,0.14285714285714285,0.02556818181818182,0.6559766763848397,0.05128205128205128,0.14285714285714285,0.020358252858252858,L1,11,2000,0.0034129692832764505,0.02616504551257145,0.047619047619047616,0.2631578947368421
blacklist,bl_span_practical,research_supported,Parncutt MaxPrac,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,39443,0.07754890183456838,0.09080444686939566,1800,131,0.07277777777777777,392,13,0.03316326530612245,0.09923664122137404,0.08380681818181818,0.4556784545879421,0.1111111111111111,0.09923664122137404,0.022687400318979263,L1,11,2000,0.014180910829548171,0.05699580369983648,0.04346657316503039,0.16842739378566896
blacklist,bl_span_comfortable,research_supported,Parncutt MaxComf,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,69306,0.1362625609245391,0.15955411593262012,1800,233,0.12944444444444445,392,22,0.05612244897959184,0.0944206008583691,0.14985795454545456,0.43356398353332753,0.18803418803418803,0.0944206008583691,0.04116142629300524,L1,11,2000,0.02958507010231148,0.08602614015572857,0.051998701298701294,0.13526641091219094
blacklist,bl_span_relative,research_supported,Parncutt MaxRel,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,182738,0.3592812723029525,0.4206937355682789,1800,624,0.3466666666666667,392,60,0.15306122448979592,0.09615384615384616,0.4005681818181818,0.4415227629513344,0.5128205128205128,0.09615384615384616,0.16215639619420197,L1,11,2000,0.12217108217332734,0.19897991312393976,0.06967713844281098,0.12666698656429942
blacklist,bl_hmm_disagreement,research_supported,PIG-trained HMM disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,279540,0.5496037324451802,0.6435482868410329,1800,1009,0.5605555555555556,392,95,0.2423469387755102,0.09415262636273539,0.6491477272727273,0.4323334884003156,0.811965811965812,0.09415262636273539,0.25879242489657617,R1,11,2000,0.15296997157071784,0.3547472992779979,0.059837298449479145,0.139002344573235
integrity,mandatory_missing,physical_invariant,schema completeness,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",False,74248,0.14597902957211756,0.0,1800,275,0.1527777777777778,392,275,0.701530612244898,1.0,0.0,4.591836734693878,0.0,,0.6812924955778922,R4,11,2000,0.6019721128059001,0.7828751260020159,1.0,1.0
```

### Every fixed and calibrated Noinfo variant (exact generated values)

```csv
calibration,variant,min_run,radius,window,quantile,held_out_recording,threshold,train_nonzero_notes,hard_count,hard_percentage_all_notes,hard_percentage_assigned_notes,gt_error_recall,assigned_gt_error_recall,gt_precision,assigned_gt_precision,error_enrichment,incremental_count_beyond_physical,incremental_errors_beyond_physical
fixed,ni_k2_r1,2.0,1.0,,,,,,21651,0.042568041822889736,0.04984425827572156,0.015306122448979591,0.05128205128205128,0.08450704225352113,0.08450704225352113,0.38804254096004603,21651,6
fixed,ni_k2_r2,2.0,2.0,,,,,,40529,0.0796840869724215,0.09330460226579461,0.03571428571428571,0.11965811965811966,0.109375,0.109375,0.5022321428571429,40529,14
fixed,ni_k2_r4,2.0,4.0,,,,,,72373,0.1422925911435037,0.1666148678670175,0.05357142857142857,0.1794871794871795,0.09375,0.09375,0.43048469387755106,72373,21
fixed,ni_k3_r1,3.0,1.0,,,,,,5397,0.010611044372922077,0.012424805409175985,0.00510204081632653,0.017094017094017096,0.11764705882352941,0.11764705882352941,0.5402160864345739,5397,2
fixed,ni_k3_r2,3.0,2.0,,,,,,10476,0.020596868788351246,0.02411752111664399,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,10476,5
fixed,ni_k3_r4,3.0,4.0,,,,,,19824,0.03897597621804841,0.04563819574421062,0.02295918367346939,0.07692307692307693,0.140625,0.140625,0.6457270408163266,19824,9
fixed,ni_k5_r1,5.0,1.0,,,,,,580,0.0011403382872512146,0.0013352579465114084,0.0,0.0,,,,580,0
fixed,ni_k5_r2,5.0,2.0,,,,,,1149,0.002259049469054561,0.002645192035416566,0.0,0.0,,,,1149,0
fixed,ni_k5_r4,5.0,4.0,,,,,,2247,0.004417827812850826,0.00517297345829506,0.0,0.0,,,,2247,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-02-15_20-07-54,0.7789999999999964,822.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-02-15_21-40-43,0.8,862.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-02-15_21-57-38,0.8,798.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-02-17_21-44-37,0.8,826.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-02-17_22-33-45,0.8,723.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-02-22_11-58-09,0.8,896.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-03-11_22-23-29,0.8,880.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-04-08_22-49-18,0.8,855.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-09-02_14-10-41,0.8,854.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-09-04_19-52-57,0.8,822.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q995,,,5.0,0.995,2024-09-05_13-25-10,0.8,882.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-02-15_20-07-54,0.6,822.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-02-15_21-40-43,0.8,862.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-02-15_21-57-38,0.8,798.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-02-17_21-44-37,0.8,826.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-02-17_22-33-45,0.7559999999999946,723.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-02-22_11-58-09,0.8,896.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-03-11_22-23-29,0.8,880.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-04-08_22-49-18,0.8,855.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-09-02_14-10-41,0.8,854.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-09-04_19-52-57,0.8,822.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q990,,,5.0,0.99,2024-09-05_13-25-10,0.8,882.0,785,0.0015433888887796611,0.0018072025655369924,0.0,0.0,0.0,0.0,0.0,785,0
training_fold,ni_w5_q975,,,5.0,0.975,2024-02-15_20-07-54,0.6,822.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-02-15_21-40-43,0.6,862.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-02-15_21-57-38,0.6,798.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-02-17_21-44-37,0.6,826.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-02-17_22-33-45,0.6,723.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-02-22_11-58-09,0.6,896.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-03-11_22-23-29,0.6,880.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-04-08_22-49-18,0.6,855.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-09-02_14-10-41,0.6,854.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-09-04_19-52-57,0.6,822.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w5_q975,,,5.0,0.975,2024-09-05_13-25-10,0.6,882.0,8627,0.016961548972614187,0.019860810869920552,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.15151515151515152,0.6957328385899815,8627,5
training_fold,ni_w9_q995,,,9.0,0.995,2024-02-15_20-07-54,0.5555555555555556,1094.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-02-15_21-40-43,0.6666666666666666,1126.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-02-15_21-57-38,0.6666666666666666,1069.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-02-17_21-44-37,0.6666666666666666,1095.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-02-17_22-33-45,0.6666666666666666,970.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-02-22_11-58-09,0.6666666666666666,1169.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-03-11_22-23-29,0.6666666666666666,1142.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-04-08_22-49-18,0.6666666666666666,1120.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-09-02_14-10-41,0.6666666666666666,1120.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-09-04_19-52-57,0.6666666666666666,1082.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q995,,,9.0,0.995,2024-09-05_13-25-10,0.6666666666666666,1153.0,1094,0.002150913941815222,0.002518572747385312,0.0,0.0,0.0,0.0,0.0,1094,0
training_fold,ni_w9_q990,,,9.0,0.99,2024-02-15_20-07-54,0.5555555555555556,1094.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-02-15_21-40-43,0.5555555555555556,1126.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-02-15_21-57-38,0.5555555555555556,1069.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-02-17_21-44-37,0.5622222222222162,1095.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-02-17_22-33-45,0.5555555555555556,970.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-02-22_11-58-09,0.5555555555555556,1169.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-03-11_22-23-29,0.5555555555555556,1142.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-04-08_22-49-18,0.5555555555555556,1120.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-09-02_14-10-41,0.5555555555555556,1120.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-09-04_19-52-57,0.5766666666666728,1082.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q990,,,9.0,0.99,2024-09-05_13-25-10,0.5555555555555556,1153.0,5060,0.009948468506019216,0.01164897449887539,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.043478260869565216,0.19964507542147295,5060,1
training_fold,ni_w9_q975,,,9.0,0.975,2024-02-15_20-07-54,0.5555555555555556,1094.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-02-15_21-40-43,0.5555555555555556,1126.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-02-15_21-57-38,0.5555555555555556,1069.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-02-17_21-44-37,0.5555555555555556,1095.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-02-17_22-33-45,0.4444444444444444,970.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-02-22_11-58-09,0.5555555555555556,1169.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-03-11_22-23-29,0.5555555555555556,1142.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-04-08_22-49-18,0.5555555555555556,1120.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-09-02_14-10-41,0.5555555555555556,1120.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-09-04_19-52-57,0.5555555555555556,1082.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w9_q975,,,9.0,0.975,2024-09-05_13-25-10,0.5555555555555556,1153.0,5060,0.009948468506019216,0.01164897449887539,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.09302325581395349,0.42714760322733747,5060,4
training_fold,ni_w17_q995,,,17.0,0.995,2024-02-15_20-07-54,0.5341176470588193,1385.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-02-15_21-40-43,0.5332352941176502,1388.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-02-15_21-57-38,0.5379411764705871,1372.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-02-17_21-44-37,0.5332352941176502,1388.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-02-17_22-33-45,0.47058823529411764,1248.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-02-22_11-58-09,0.5294117647058824,1448.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-03-11_22-23-29,0.5294117647058824,1409.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-04-08_22-49-18,0.5317647058823508,1393.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-09-02_14-10-41,0.5323529411764679,1391.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-09-04_19-52-57,0.5379411764705871,1372.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q995,,,17.0,0.995,2024-09-05_13-25-10,0.5294117647058824,1426.0,1004,0.001973964897241758,0.002311377548788714,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.058823529411764705,0.27010804321728693,1004,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-02-15_20-07-54,0.5294117647058824,1385.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-02-15_21-40-43,0.5294117647058824,1388.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-02-15_21-57-38,0.5294117647058824,1372.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-02-17_21-44-37,0.5294117647058824,1388.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-02-17_22-33-45,0.47058823529411764,1248.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-02-22_11-58-09,0.5294117647058824,1448.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-03-11_22-23-29,0.5294117647058824,1409.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-04-08_22-49-18,0.5294117647058824,1393.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-09-02_14-10-41,0.5294117647058824,1391.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-09-04_19-52-57,0.5294117647058824,1372.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q990,,,17.0,0.99,2024-09-05_13-25-10,0.5294117647058824,1426.0,2496,0.004907386836170744,0.005746213507745647,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.05263157894736842,0.24167561761546724,2496,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-02-15_20-07-54,0.4352941176470508,1385.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-02-15_21-40-43,0.47058823529411764,1388.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-02-15_21-57-38,0.47058823529411764,1372.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-02-17_21-44-37,0.47058823529411764,1388.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-02-17_22-33-45,0.4117647058823529,1248.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-02-22_11-58-09,0.47058823529411764,1448.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-03-11_22-23-29,0.47058823529411764,1409.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-04-08_22-49-18,0.47058823529411764,1393.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-09-02_14-10-41,0.47058823529411764,1391.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-09-04_19-52-57,0.47058823529411764,1372.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
training_fold,ni_w17_q975,,,17.0,0.975,2024-09-05_13-25-10,0.47058823529411764,1426.0,6006,0.011808399574535854,0.013826826253012963,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.03333333333333333,0.15306122448979592,6006,1
```

### All 189 strategy × fixed-Noinfo combinations (GT/assigned recall, precision, enrichment, incremental contribution, and method columns)

```csv
set_id,noinfo_variant,noinfo_calibration,noinfo_window,noinfo_quantile,gt_hard_count,base_risk_method,physical_policy_status,noinfo_min_run,noinfo_context_radius,hard_count,hard_percentage_all_notes,gt_error_recall,assigned_gt_error_recall,gt_precision,error_enrichment,incremental_count_beyond_physical,incremental_errors_beyond_physical
legacy_current_default__ni_k3_r1,ni_k3_r1,fixed,,,90,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,35000,0.06881351733412501,0.0663265306122449,0.2222222222222222,0.28888888888888886,1.3265306122448979,35000,26
legacy_current_default__ni_k3_r2,ni_k3_r2,fixed,,,90,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,35000,0.06881351733412501,0.0663265306122449,0.2222222222222222,0.28888888888888886,1.3265306122448979,35000,26
legacy_current_default__ni_k5_r1,ni_k5_r1,fixed,,,90,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,35000,0.06881351733412501,0.0663265306122449,0.2222222222222222,0.28888888888888886,1.3265306122448979,35000,26
legacy_current_default__ni_k5_r2,ni_k5_r2,fixed,,,90,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,35000,0.06881351733412501,0.0663265306122449,0.2222222222222222,0.28888888888888886,1.3265306122448979,35000,26
legacy_current_default__ni_k5_r4,ni_k5_r4,fixed,,,90,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,35860,0.07050436376004923,0.0663265306122449,0.2222222222222222,0.28888888888888886,1.3265306122448979,35860,26
legacy_current_default__ni_k3_r4,ni_k3_r4,fixed,,,116,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,43570,0.08566299857850934,0.07142857142857142,0.23931623931623933,0.2413793103448276,1.1083743842364533,43570,28
legacy_current_default__ni_k2_r1,ni_k2_r1,fixed,,,139,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,49431,0.09718631358123239,0.07397959183673469,0.24786324786324787,0.20863309352517986,0.9580091029217443,49431,29
legacy_current_default__ni_k2_r2,ni_k2_r2,fixed,,,180,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,62762,0.12339639928355298,0.08418367346938775,0.28205128205128205,0.18333333333333332,0.8418367346938775,62762,33
legacy_current_default__ni_k2_r4,ni_k2_r4,fixed,,,270,legacy_current_default,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,92261,0.1813943977932488,0.09438775510204081,0.3162393162393162,0.13703703703703704,0.6292517006802721,92261,37
bl_step_crossing__ni_k5_r1,ni_k5_r1,fixed,,,7,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,3899,0.007665825831021527,0.007653061224489796,0.02564102564102564,0.42857142857142855,1.967930029154519,3899,3
bl_step_crossing__ni_k5_r2,ni_k5_r2,fixed,,,7,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,4461,0.008770774309358048,0.007653061224489796,0.02564102564102564,0.42857142857142855,1.967930029154519,4461,3
bl_rate_q995__ni_k5_r1,ni_k5_r1,fixed,,,8,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,5323,0.010465552936272784,0.00510204081632653,0.017094017094017096,0.25,1.1479591836734695,5323,2
bl_step_crossing__ni_k5_r4,ni_k5_r4,fixed,,,7,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,5543,0.010898095045230143,0.007653061224489796,0.02564102564102564,0.42857142857142855,1.967930029154519,5543,3
bl_rate_q995__ni_k5_r2,ni_k5_r2,fixed,,,8,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,5887,0.011574433615599828,0.00510204081632653,0.017094017094017096,0.25,1.1479591836734695,5887,2
bl_rate_q990__ni_k5_r1,ni_k5_r1,fixed,,,17,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,6928,0.013621144231166232,0.01020408163265306,0.03418803418803419,0.23529411764705882,1.0804321728691477,6928,4
bl_rate_q995__ni_k5_r4,ni_k5_r4,fixed,,,8,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,6972,0.013707652652957703,0.00510204081632653,0.017094017094017096,0.25,1.1479591836734695,6972,2
bl_rate_q990__ni_k5_r2,ni_k5_r2,fixed,,,17,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,7492,0.014730024910493276,0.01020408163265306,0.03418803418803419,0.23529411764705882,1.0804321728691477,7492,4
bl_rate_q990__ni_k5_r4,ni_k5_r4,fixed,,,17,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,8572,0.016853413445374846,0.01020408163265306,0.03418803418803419,0.23529411764705882,1.0804321728691477,8572,4
bl_step_crossing__ni_k3_r1,ni_k3_r1,fixed,,,22,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,8661,0.01702839638945305,0.01020408163265306,0.03418803418803419,0.18181818181818182,0.8348794063079779,8661,4
bl_crossing__ni_k5_r1,ni_k5_r1,fixed,,,20,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,8877,0.017453074096429363,0.017857142857142856,0.05982905982905983,0.35,1.6071428571428572,8877,7
bl_crossing__ni_k5_r2,ni_k5_r2,fixed,,,20,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,9425,0.018530497167832238,0.017857142857142856,0.05982905982905983,0.35,1.6071428571428572,9425,7
bl_rate_q995__ni_k3_r1,ni_k3_r1,fixed,,,25,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,10101,0.01985958110262848,0.01020408163265306,0.03418803418803419,0.16,0.7346938775510204,10101,4
bl_crossing__ni_k5_r4,ni_k5_r4,fixed,,,20,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,10475,0.020594902687855986,0.017857142857142856,0.05982905982905983,0.35,1.6071428571428572,10475,7
bl_rate_q990__ni_k3_r1,ni_k3_r1,fixed,,,33,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,11690,0.022983714789597754,0.015306122448979591,0.05128205128205128,0.18181818181818182,0.8348794063079779,11690,6
bl_crossing__ni_k3_r1,ni_k3_r1,fixed,,,35,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,13482,0.026506966877104958,0.02040816326530612,0.06837606837606838,0.22857142857142856,1.0495626822157436,13482,8
bl_step_crossing__ni_k3_r2,ni_k3_r2,fixed,,,37,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,13680,0.026896254775166578,0.015306122448979591,0.05128205128205128,0.16216216216216217,0.7446221731936018,13680,6
bl_rate_q995__ni_k3_r2,ni_k3_r2,fixed,,,41,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,15115,0.029717608985865705,0.017857142857142856,0.05982905982905983,0.17073170731707318,0.7839721254355402,15115,7
bl_rate_q975__ni_k5_r1,ni_k5_r1,fixed,,,42,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,15225,0.029933880040344384,0.015306122448979591,0.05128205128205128,0.14285714285714285,0.6559766763848397,15225,6
bl_rate_q975__ni_k5_r2,ni_k5_r2,fixed,,,42,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,15774,0.031013269212242515,0.015306122448979591,0.05128205128205128,0.14285714285714285,0.6559766763848397,15774,6
bl_rate_q990__ni_k3_r2,ni_k3_r2,fixed,,,49,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,16685,0.03280438676342502,0.02295918367346939,0.07692307692307693,0.1836734693877551,0.8433985839233653,16685,9
bl_rate_q975__ni_k5_r4,ni_k5_r4,fixed,,,42,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,16831,0.03309143743573309,0.015306122448979591,0.05128205128205128,0.14285714285714285,0.6559766763848397,16831,6
bl_crossing__ni_k3_r2,ni_k3_r2,fixed,,,50,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,18383,0.03614282540437772,0.025510204081632654,0.08547008547008547,0.2,0.9183673469387756,18383,10
bl_rate_q975__ni_k3_r1,ni_k3_r1,fixed,,,58,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,19925,0.03917455236806974,0.02040816326530612,0.06837606837606838,0.13793103448275862,0.6333567909922591,19925,8
bl_step_crossing__ni_k3_r4,ni_k3_r4,fixed,,,68,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,22944,0.04511020976326184,0.025510204081632654,0.08547008547008547,0.14705882352941177,0.6752701080432173,22944,10
bl_rate_q995__ni_k3_r4,ni_k3_r4,fixed,,,72,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,24316,0.04780769964275954,0.02806122448979592,0.09401709401709402,0.1527777777777778,0.7015306122448981,24316,11
bl_step_crossing__ni_k2_r1,ni_k2_r1,fixed,,,75,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,24746,0.048653122855721646,0.02040816326530612,0.06837606837606838,0.10666666666666667,0.489795918367347,24746,8
bl_rate_q975__ni_k3_r2,ni_k3_r2,fixed,,,74,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,24806,0.04877108888543729,0.02806122448979592,0.09401709401709402,0.14864864864864866,0.6825703254274683,24806,11
bl_rate_q990__ni_k3_r4,ni_k3_r4,fixed,,,80,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,25826,0.05077651139060322,0.03316326530612245,0.1111111111111111,0.1625,0.7461734693877552,25826,13
bl_rate_q995__ni_k2_r1,ni_k2_r1,fixed,,,79,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,26248,0.05160620579960324,0.02040816326530612,0.06837606837606838,0.10126582278481013,0.46499612503229143,26248,8
bl_crossing__ni_k3_r4,ni_k3_r4,fixed,,,81,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,27474,0.054016645006792874,0.03571428571428571,0.11965811965811966,0.1728395061728395,0.7936507936507936,27474,14
bl_rate_q990__ni_k2_r1,ni_k2_r1,fixed,,,87,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,27794,0.0546457971652763,0.025510204081632654,0.08547008547008547,0.11494252873563218,0.5277973258268825,27794,10
bl_crossing__ni_k2_r1,ni_k2_r1,fixed,,,88,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,29215,0.05743962596904178,0.030612244897959183,0.10256410256410256,0.13636363636363635,0.6261595547309833,29215,12
bl_two_signal_strict__ni_k5_r1,ni_k5_r1,fixed,,,86,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,31231,0.061403284567487386,0.04081632653061224,0.13675213675213677,0.18604651162790697,0.8542952064546749,31231,16
bl_two_signal_strict__ni_k5_r2,ni_k5_r2,fixed,,,86,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,31733,0.06239026701610826,0.04081632653061224,0.13675213675213677,0.18604651162790697,0.8542952064546749,31733,16
bl_two_signal_strict__ni_k5_r4,ni_k5_r4,fixed,,,86,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,32681,0.06425413028561541,0.04081632653061224,0.13675213675213677,0.18604651162790697,0.8542952064546749,32681,16
bl_rate_q975__ni_k3_r4,ni_k3_r4,fixed,,,103,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,33699,0.06625562058979083,0.03826530612244898,0.1282051282051282,0.14563106796116504,0.6687140875767783,33699,15
bl_two_signal_strict__ni_k3_r1,ni_k3_r1,fixed,,,100,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,35380,0.06956063552232408,0.04336734693877551,0.1452991452991453,0.17,0.7806122448979593,35380,17
bl_rate_q975__ni_k2_r1,ni_k2_r1,fixed,,,112,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,35876,0.0705358213679734,0.030612244897959183,0.10256410256410256,0.10714285714285714,0.49198250728862974,35876,12
bl_practical_or_crossing__ni_k5_r1,ni_k5_r1,fixed,,,131,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,39866,0.07838056234406365,0.03316326530612245,0.1111111111111111,0.09923664122137404,0.4556784545879421,39866,13
bl_span_practical__ni_k5_r1,ni_k5_r1,fixed,,,131,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,39866,0.07838056234406365,0.03316326530612245,0.1111111111111111,0.09923664122137404,0.4556784545879421,39866,13
bl_two_signal_strict__ni_k3_r2,ni_k3_r2,fixed,,,115,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,39879,0.07840612165050205,0.04846938775510204,0.1623931623931624,0.16521739130434782,0.7586512866015972,39879,19
bl_practical_or_crossing__ni_k5_r2,ni_k5_r2,fixed,,,131,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,40336,0.07930462957683619,0.03316326530612245,0.1111111111111111,0.09923664122137404,0.4556784545879421,40336,13
bl_span_practical__ni_k5_r2,ni_k5_r2,fixed,,,131,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,40336,0.07930462957683619,0.03316326530612245,0.1111111111111111,0.09923664122137404,0.4556784545879421,40336,13
bl_practical_or_crossing__ni_k5_r4,ni_k5_r4,fixed,,,131,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,41247,0.0810957471280187,0.03316326530612245,0.1111111111111111,0.09923664122137404,0.4556784545879421,41247,13
bl_span_practical__ni_k5_r4,ni_k5_r4,fixed,,,131,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,41247,0.0810957471280187,0.03316326530612245,0.1111111111111111,0.09923664122137404,0.4556784545879421,41247,13
bl_step_crossing__ni_k2_r2,ni_k2_r2,fixed,,,131,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,43463,0.08545262582551645,0.03826530612244898,0.1282051282051282,0.11450381679389313,0.5257828322168562,43463,15
bl_practical_or_crossing__ni_k3_r1,ni_k3_r1,fixed,,,146,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,43832,0.08617811690826765,0.03571428571428571,0.11965811965811966,0.0958904109589041,0.44031311154598823,43832,14
bl_span_practical__ni_k3_r1,ni_k3_r1,fixed,,,146,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,43832,0.08617811690826765,0.03571428571428571,0.11965811965811966,0.0958904109589041,0.44031311154598823,43832,14
bl_practical_or_rate995__ni_k5_r1,ni_k5_r1,fixed,,,139,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,44593,0.08767431938516106,0.03826530612244898,0.1282051282051282,0.1079136690647482,0.4955219497871091,44593,15
bl_rate_q995__ni_k2_r2,ni_k2_r2,fixed,,,135,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,44868,0.08821499702135775,0.03826530612244898,0.1282051282051282,0.1111111111111111,0.5102040816326531,44868,15
bl_practical_or_rate995__ni_k5_r2,ni_k5_r2,fixed,,,139,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,45058,0.08858855611545728,0.03826530612244898,0.1282051282051282,0.1079136690647482,0.4955219497871091,45058,15
bl_practical_or_rate995__ni_k5_r4,ni_k5_r4,fixed,,,139,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,45956,0.09035411436020141,0.03826530612244898,0.1282051282051282,0.1079136690647482,0.4955219497871091,45956,15
bl_rate_q990__ni_k2_r2,ni_k2_r2,fixed,,,143,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,46320,0.09106977494047631,0.04336734693877551,0.1452991452991453,0.11888111888111888,0.5458826887398316,46320,17
bl_crossing__ni_k2_r2,ni_k2_r2,fixed,,,144,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,47595,0.09357655307193372,0.04846938775510204,0.1623931623931624,0.13194444444444445,0.6058673469387755,47595,19
bl_practical_or_crossing__ni_k3_r2,ni_k3_r2,fixed,,,161,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,48172,0.09471099305769914,0.04081632653061224,0.13675213675213677,0.09937888198757763,0.4563316009633667,48172,16
bl_span_practical__ni_k3_r2,ni_k3_r2,fixed,,,161,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,48172,0.09471099305769914,0.04081632653061224,0.13675213675213677,0.09937888198757763,0.4563316009633667,48172,16
bl_two_signal_strict__ni_k3_r4,ni_k3_r4,fixed,,,144,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,48248,0.09486041669533897,0.058673469387755105,0.19658119658119658,0.1597222222222222,0.7334183673469388,48248,23
bl_practical_or_rate995__ni_k3_r1,ni_k3_r1,fixed,,,154,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,48522,0.0953991282310404,0.04081632653061224,0.13675213675213677,0.1038961038961039,0.47707394646170165,48522,16
bl_two_signal_strict__ni_k2_r1,ni_k2_r1,fixed,,,149,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,49742,0.09779777083525848,0.05357142857142857,0.1794871794871795,0.14093959731543623,0.6471716203259827,49742,21
bl_practical_or_rate995__ni_k3_r2,ni_k3_r2,fixed,,,169,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,52797,0.10380420784827996,0.04591836734693878,0.15384615384615385,0.10650887573964497,0.48907136819224734,52797,18
bl_rate_q975__ni_k2_r2,ni_k2_r2,fixed,,,167,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,53954,0.1060789861212966,0.04846938775510204,0.1623931623931624,0.11377245508982035,0.5224245386777465,53954,19
bl_practical_or_crossing__ni_k3_r4,ni_k3_r4,fixed,,,188,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,56320,0.11073077989308346,0.05102040816326531,0.17094017094017094,0.10638297872340426,0.4884932696482849,56320,20
bl_span_practical__ni_k3_r4,ni_k3_r4,fixed,,,188,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,56320,0.11073077989308346,0.05102040816326531,0.17094017094017094,0.10638297872340426,0.4884932696482849,56320,20
bl_practical_or_crossing__ni_k2_r1,ni_k2_r1,fixed,,,194,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,57632,0.1133103037428655,0.04591836734693878,0.15384615384615385,0.09278350515463918,0.4260467073427309,57632,18
bl_span_practical__ni_k2_r1,ni_k2_r1,fixed,,,194,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,57632,0.1133103037428655,0.04591836734693878,0.15384615384615385,0.09278350515463918,0.4260467073427309,57632,18
bl_practical_or_rate995__ni_k3_r4,ni_k3_r4,fixed,,,196,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,60798,0.11953497791086093,0.05612244897959184,0.18803418803418803,0.11224489795918367,0.5154102457309455,60798,22
bl_practical_or_rate995__ni_k2_r1,ni_k2_r1,fixed,,,202,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,62215,0.12232094231264537,0.05102040816326531,0.17094017094017094,0.09900990099009901,0.4546373004647404,62215,20
bl_two_signal_strict__ni_k2_r2,ni_k2_r2,fixed,,,199,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,66764,0.13126473346558637,0.0663265306122449,0.2222222222222222,0.1306532663316583,0.5999384678494514,66764,26
bl_span_comfortable__ni_k5_r1,ni_k5_r1,fixed,,,233,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,69681,0.13699984861026188,0.05612244897959184,0.18803418803418803,0.0944206008583691,0.43356398353332753,69681,22
bl_span_comfortable__ni_k5_r2,ni_k5_r2,fixed,,,233,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,70113,0.1378492040242145,0.05612244897959184,0.18803418803418803,0.0944206008583691,0.43356398353332753,70113,22
bl_span_comfortable__ni_k5_r4,ni_k5_r4,fixed,,,233,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,70939,0.13947320303329985,0.05612244897959184,0.18803418803418803,0.0944206008583691,0.43356398353332753,70939,22
bl_span_comfortable__ni_k3_r1,ni_k3_r1,fixed,,,247,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,73246,0.14400899687586632,0.058673469387755105,0.19658119658119658,0.0931174089068826,0.4275799388581344,73246,23
bl_practical_or_crossing__ni_k2_r2,ni_k2_r2,fixed,,,243,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,74172,0.14582960593447775,0.061224489795918366,0.20512820512820512,0.09876543209876543,0.45351473922902497,74172,24
bl_span_practical__ni_k2_r2,ni_k2_r2,fixed,,,243,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,74172,0.14582960593447775,0.061224489795918366,0.20512820512820512,0.09876543209876543,0.45351473922902497,74172,24
bl_step_crossing__ni_k2_r4,ni_k2_r4,fixed,,,227,bl_step_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,75055,0.14756567267179294,0.05612244897959184,0.18803418803418803,0.09691629955947137,0.4450238245077767,75055,22
bl_rate_q995__ni_k2_r4,ni_k2_r4,fixed,,,231,bl_rate_q995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,76243,0.14990140006016267,0.05612244897959184,0.18803418803418803,0.09523809523809523,0.43731778425655976,76243,22
bl_span_comfortable__ni_k3_r2,ni_k3_r2,fixed,,,261,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,77152,0.15168858541035465,0.06377551020408163,0.21367521367521367,0.09578544061302682,0.43983110485573546,77152,25
bl_rate_q990__ni_k2_r4,ni_k2_r4,fixed,,,237,bl_rate_q990,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,77544,0.15245929680449685,0.061224489795918366,0.20512820512820512,0.10126582278481013,0.46499612503229143,77544,24
bl_practical_or_rate995__ni_k2_r2,ni_k2_r2,fixed,,,250,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,78497,0.15433299057648034,0.06377551020408163,0.21367521367521367,0.1,0.4591836734693878,78497,25
bl_crossing__ni_k2_r4,ni_k2_r4,fixed,,,240,bl_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,78673,0.15467902426364621,0.0663265306122449,0.2222222222222222,0.10833333333333334,0.49744897959183676,78673,26
bl_rate_q975__ni_k2_r4,ni_k2_r4,fixed,,,259,bl_rate_q975,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,84414,0.16596640720693798,0.0663265306122449,0.2222222222222222,0.10038610038610038,0.460956583405563,84414,26
bl_span_comfortable__ni_k3_r4,ni_k3_r4,fixed,,,286,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,84448,0.16603325462377683,0.07397959183673469,0.24786324786324787,0.10139860139860139,0.4656058227486799,84448,29
bl_span_comfortable__ni_k2_r1,ni_k2_r1,fixed,,,292,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,85613,0.16832376170075558,0.0663265306122449,0.2222222222222222,0.08904109589041095,0.4088621750069891,85613,26
bl_two_signal_strict__ni_k2_r4,ni_k2_r4,fixed,,,286,bl_two_signal_strict,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,95746,0.1882462580192324,0.08163265306122448,0.27350427350427353,0.11188811188811189,0.513771942343371,95746,32
bl_span_comfortable__ni_k2_r2,ni_k2_r2,fixed,,,338,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,100508,0.19760882857766393,0.08163265306122448,0.27350427350427353,0.09467455621301775,0.4347301050597754,100508,32
bl_practical_or_crossing__ni_k2_r4,ni_k2_r4,fixed,,,328,bl_practical_or_crossing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,102481,0.2014879448548133,0.07653061224489796,0.2564102564102564,0.09146341463414634,0.41998506719761075,102481,30
bl_span_practical__ni_k2_r4,ni_k2_r4,fixed,,,328,bl_span_practical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,102481,0.2014879448548133,0.07653061224489796,0.2564102564102564,0.09146341463414634,0.41998506719761075,102481,30
bl_practical_or_rate995__ni_k2_r4,ni_k2_r4,fixed,,,335,bl_practical_or_rate995,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,106337,0.20906922836453862,0.07908163265306123,0.26495726495726496,0.09253731343283582,0.42491623515077676,106337,31
bl_span_comfortable__ni_k2_r4,ni_k2_r4,fixed,,,415,bl_span_comfortable,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,125951,0.24763232347858227,0.09693877551020408,0.3247863247863248,0.09156626506024096,0.42045733956233095,125951,38
bl_span_relative__ni_k5_r1,ni_k5_r1,fixed,,,624,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,182990,0.3597767296277582,0.15306122448979592,0.5128205128205128,0.09615384615384616,0.4415227629513344,182990,60
bl_span_relative__ni_k5_r2,ni_k5_r2,fixed,,,624,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,183282,0.3603508309723743,0.15306122448979592,0.5128205128205128,0.09615384615384616,0.4415227629513344,183282,60
bl_span_relative__ni_k5_r4,ni_k5_r4,fixed,,,624,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,183817,0.3614026947373388,0.15306122448979592,0.5128205128205128,0.09615384615384616,0.4415227629513344,183817,60
bl_span_relative__ni_k3_r1,ni_k3_r1,fixed,,,631,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,185343,0.36440296409310663,0.1556122448979592,0.5213675213675214,0.09667194928684628,0.44390180794980433,185343,61
bl_span_relative__ni_k3_r2,ni_k3_r2,fixed,,,637,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,187847,0.3693260797332395,0.15816326530612246,0.5299145299145299,0.09733124018838304,0.44692916413033035,187847,62
bl_span_relative__ni_k3_r4,ni_k3_r4,fixed,,,653,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,192537,0.37854709105601225,0.16581632653061223,0.5555555555555556,0.0995405819295559,0.4570741006969404,192537,65
bl_span_relative__ni_k2_r1,ni_k2_r1,fixed,,,660,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,193540,0.38051908985275873,0.16071428571428573,0.5384615384615384,0.09545454545454546,0.4383116883116884,193540,63
bl_span_relative__ni_k2_r2,ni_k2_r2,fixed,,,688,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,203230,0.39957060365183505,0.17091836734693877,0.5726495726495726,0.09738372093023256,0.44717014712861897,203230,67
bl_span_relative__ni_k2_r4,ni_k2_r4,fixed,,,745,bl_span_relative,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,219727,0.4320053635221511,0.18112244897959184,0.6068376068376068,0.0953020134228188,0.4376112861251884,219727,71
bl_hmm_disagreement__ni_k5_r1,ni_k5_r1,fixed,,,1009,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,279738,0.5499930203432418,0.2423469387755102,0.811965811965812,0.09415262636273539,0.4323334884003156,279738,95
bl_hmm_disagreement__ni_k5_r2,ni_k5_r2,fixed,,,1009,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,279942,0.550394104844275,0.2423469387755102,0.811965811965812,0.09415262636273539,0.4323334884003156,279942,95
bl_hmm_disagreement__ni_k5_r4,ni_k5_r4,fixed,,,1009,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,280332,0.5511608840374267,0.2423469387755102,0.811965811965812,0.09415262636273539,0.4323334884003156,280332,95
bl_hmm_disagreement__ni_k3_r1,ni_k3_r1,fixed,,,1013,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,281415,0.5532901708737941,0.2423469387755102,0.811965811965812,0.09378084896347483,0.430626347281262,281415,95
bl_hmm_disagreement__ni_k3_r2,ni_k3_r2,fixed,,,1019,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,283269,0.5569353211920074,0.24744897959183673,0.8290598290598291,0.09519136408243375,0.4371032024193387,283269,97
bl_hmm_disagreement__ni_k3_r4,ni_k3_r4,fixed,,,1025,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,286666,0.563614164574408,0.24744897959183673,0.8290598290598291,0.09463414634146342,0.43454454952712795,286666,97
bl_hmm_disagreement__ni_k2_r1,ni_k2_r1,fixed,,,1033,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,287117,0.5645008758977706,0.24489795918367346,0.8205128205128205,0.09293320425943853,0.4267341011912994,287117,96
bl_hmm_disagreement__ni_k2_r2,ni_k2_r2,fixed,,,1045,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,293994,0.5780217490036785,0.25,0.8376068376068376,0.0937799043062201,0.4306220095693781,293994,98
bl_hmm_disagreement__ni_k2_r4,ni_k2_r4,fixed,,,1071,bl_hmm_disagreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,305593,0.6008265486482076,0.25,0.8376068376068376,0.0915032679738562,0.42016806722689076,305593,98
hy_direct_plus_corroborated__ni_k5_r1,ni_k5_r1,fixed,,,139,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,42349,0.08326238987379601,0.04081632653061224,0.13675213675213677,0.11510791366906475,0.5285567464395831,42349,16
hy_direct_plus_corroborated__ni_k5_r2,ni_k5_r2,fixed,,,139,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,42818,0.08418449100607328,0.04081632653061224,0.13675213675213677,0.11510791366906475,0.5285567464395831,42818,16
hy_direct_plus_corroborated__ni_k5_r4,ni_k5_r4,fixed,,,139,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,43721,0.08595987975329371,0.04081632653061224,0.13675213675213677,0.11510791366906475,0.5285567464395831,43721,16
hy_direct_plus_corroborated__ni_k3_r1,ni_k3_r1,fixed,,,153,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,46292,0.091014724126609,0.04336734693877551,0.1452991452991453,0.1111111111111111,0.5102040816326531,46292,17
hy_two_of_three_families__ni_k5_r1,ni_k5_r1,fixed,,,169,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,50181,0.09866088895267792,0.061224489795918366,0.20512820512820512,0.14201183431952663,0.6520951575896632,50181,24
hy_direct_plus_corroborated__ni_k3_r2,ni_k3_r2,fixed,,,168,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,50601,0.09948665116068743,0.04846938775510204,0.1623931623931624,0.1130952380952381,0.5193148688046647,50601,19
hy_two_of_three_families__ni_k5_r2,ni_k5_r2,fixed,,,169,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,50660,0.0996026510899078,0.061224489795918366,0.20512820512820512,0.14201183431952663,0.6520951575896632,50660,24
hy_two_of_three_families__ni_k5_r4,ni_k5_r4,fixed,,,169,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,51559,0.10137017543514719,0.061224489795918366,0.20512820512820512,0.14201183431952663,0.6520951575896632,51559,24
hy_two_of_three_families__ni_k3_r1,ni_k3_r1,fixed,,,182,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,54100,0.10636603679360467,0.06377551020408163,0.21367521367521367,0.13736263736263737,0.6307468042161921,54100,25
hy_two_of_three_families__ni_k3_r2,ni_k3_r2,fixed,,,196,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,58314,0.11465118428063333,0.06887755102040816,0.23076923076923078,0.1377551020408163,0.6325489379425239,58314,27
hy_direct_plus_corroborated__ni_k3_r4,ni_k3_r4,fixed,,,195,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,58666,0.11534325165496509,0.058673469387755105,0.19658119658119658,0.11794871794871795,0.5416012558869702,58666,23
hy_direct_plus_corroborated__ni_k2_r1,ni_k2_r1,fixed,,,201,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,60032,0.11802894493149123,0.05357142857142857,0.1794871794871795,0.1044776119402985,0.47974413646055436,60032,21
hy_hierarchical__ni_k5_r1,ni_k5_r1,fixed,,,222,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,62614,0.1231054164102544,0.058673469387755105,0.19658119658119658,0.1036036036036036,0.47573083287369,62614,23
hy_hierarchical__ni_k5_r2,ni_k5_r2,fixed,,,222,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,63058,0.12397836503015015,0.058673469387755105,0.19658119658119658,0.1036036036036036,0.47573083287369,63058,23
hy_hierarchical__ni_k5_r4,ni_k5_r4,fixed,,,222,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,63902,0.1256377538481502,0.058673469387755105,0.19658119658119658,0.1036036036036036,0.47573083287369,63902,23
hy_two_of_three_families__ni_k3_r4,ni_k3_r4,fixed,,,223,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,66137,0.1300319884550579,0.07908163265306123,0.26495726495726496,0.13901345291479822,0.6383270797108082,66137,31
hy_hierarchical__ni_k3_r1,ni_k3_r1,fixed,,,235,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,66287,0.130326903529347,0.061224489795918366,0.20512820512820512,0.10212765957446808,0.46895353886235347,66287,24
hy_two_of_three_families__ni_k2_r1,ni_k2_r1,fixed,,,229,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,67620,0.13294771548952952,0.07142857142857142,0.23931623931623933,0.1222707423580786,0.561447286338116,67620,28
hy_hierarchical__ni_k3_r2,ni_k3_r2,fixed,,,249,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,70289,0.13819523771138037,0.0663265306122449,0.2222222222222222,0.10441767068273092,0.47946889599213177,70289,26
hy_direct_plus_corroborated__ni_k2_r2,ni_k2_r2,fixed,,,249,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,76442,0.15029265405871955,0.0663265306122449,0.2222222222222222,0.10441767068273092,0.47946889599213177,76442,26
hy_hierarchical__ni_k3_r4,ni_k3_r4,fixed,,,274,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,77773,0.15290953381791156,0.07653061224489796,0.2564102564102564,0.10948905109489052,0.5027558468642932,77773,30
hy_hierarchical__ni_k2_r1,ni_k2_r1,fixed,,,281,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,79082,0.15548315936620785,0.06887755102040816,0.23076923076923078,0.09608540925266904,0.4412085118745007,79082,27
hy_two_of_three_families__ni_k2_r2,ni_k2_r2,fixed,,,276,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,83552,0.16427162858002323,0.08418367346938775,0.28205128205128205,0.11956521739130435,0.5490239574090506,83552,33
hy_hierarchical__ni_k2_r2,ni_k2_r2,fixed,,,326,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,94339,0.18547995462240058,0.08163265306122448,0.27350427350427353,0.09815950920245399,0.45073244021534997,94339,32
hy_direct_plus_corroborated__ni_k2_r4,ni_k2_r4,fixed,,,333,hy_direct_plus_corroborated,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,104504,0.20546536615672573,0.08163265306122448,0.27350427350427353,0.0960960960960961,0.441257584114727,104504,32
hy_two_of_three_families__ni_k2_r4,ni_k2_r4,fixed,,,356,hy_two_of_three_families,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,110739,0.2177240027446763,0.09948979591836735,0.3333333333333333,0.10955056179775281,0.5030382939692731,110739,39
hy_hierarchical__ni_k2_r4,ni_k2_r4,fixed,,,403,hy_hierarchical,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,120468,0.2368521944630678,0.09693877551020408,0.3247863247863248,0.09429280397022333,0.4329771610877602,120468,38
mandatory_missing__ni_k5_r1,ni_k5_r1,fixed,,,0,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,580,0.0011403382872512146,0.0,0.0,,,580,0
mandatory_missing__ni_k5_r2,ni_k5_r2,fixed,,,0,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,1149,0.002259049469054561,0.0,0.0,,,1149,0
mandatory_missing__ni_k5_r4,ni_k5_r4,fixed,,,0,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,2247,0.004417827812850826,0.0,0.0,,,2247,0
mandatory_missing__ni_k3_r1,ni_k3_r1,fixed,,,17,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,5397,0.010611044372922077,0.00510204081632653,0.017094017094017096,0.11764705882352941,0.5402160864345739,5397,2
mandatory_missing__ni_k3_r2,ni_k3_r2,fixed,,,33,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,10476,0.020596868788351246,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.6957328385899815,10476,5
mandatory_missing__ni_k3_r4,ni_k3_r4,fixed,,,64,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,19824,0.03897597621804841,0.02295918367346939,0.07692307692307693,0.140625,0.6457270408163266,19824,9
mandatory_missing__ni_k2_r1,ni_k2_r1,fixed,,,71,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,21651,0.042568041822889736,0.015306122448979591,0.05128205128205128,0.08450704225352113,0.38804254096004603,21651,6
mandatory_missing__ni_k2_r2,ni_k2_r2,fixed,,,128,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,40529,0.0796840869724215,0.03571428571428571,0.11965811965811966,0.109375,0.5022321428571429,40529,14
mandatory_missing__ni_k2_r4,ni_k2_r4,fixed,,,224,mandatory_missing,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,72373,0.1422925911435037,0.05357142857142857,0.1794871794871795,0.09375,0.43048469387755106,72373,21
ni_k5_r1,ni_k5_r1,fixed,,,0,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,580,0.0011403382872512146,0.0,0.0,,,580,0
ni_w5_q990,ni_w5_q990,training_fold,5.0,0.99,10,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,785,0.0015433888887796611,0.0,0.0,0.0,0.0,785,0
ni_w5_q995,ni_w5_q995,training_fold,5.0,0.995,3,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,785,0.0015433888887796611,0.0,0.0,0.0,0.0,785,0
ni_w17_q995,ni_w17_q995,training_fold,17.0,0.995,17,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,1004,0.001973964897241758,0.002551020408163265,0.008547008547008548,0.058823529411764705,0.27010804321728693,1004,1
ni_w9_q995,ni_w9_q995,training_fold,9.0,0.995,8,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,1094,0.002150913941815222,0.0,0.0,0.0,0.0,1094,0
ni_k5_r2,ni_k5_r2,fixed,,,0,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,1149,0.002259049469054561,0.0,0.0,,,1149,0
ni_k5_r4,ni_k5_r4,fixed,,,0,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,2247,0.004417827812850826,0.0,0.0,,,2247,0
ni_w17_q990,ni_w17_q990,training_fold,17.0,0.99,19,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,2496,0.004907386836170744,0.002551020408163265,0.008547008547008548,0.05263157894736842,0.24167561761546724,2496,1
ni_w9_q975,ni_w9_q975,training_fold,9.0,0.975,43,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,5060,0.009948468506019216,0.01020408163265306,0.03418803418803419,0.09302325581395349,0.42714760322733747,5060,4
ni_w9_q990,ni_w9_q990,training_fold,9.0,0.99,23,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,5060,0.009948468506019216,0.002551020408163265,0.008547008547008548,0.043478260869565216,0.19964507542147295,5060,1
ni_k3_r1,ni_k3_r1,fixed,,,17,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,5397,0.010611044372922077,0.00510204081632653,0.017094017094017096,0.11764705882352941,0.5402160864345739,5397,2
ni_w17_q975,ni_w17_q975,training_fold,17.0,0.975,30,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,6006,0.011808399574535854,0.002551020408163265,0.008547008547008548,0.03333333333333333,0.15306122448979592,6006,1
ni_w5_q975,ni_w5_q975,training_fold,5.0,0.975,33,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",,,8627,0.016961548972614187,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.6957328385899815,8627,5
ni_k3_r2,ni_k3_r2,fixed,,,33,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,10476,0.020596868788351246,0.012755102040816327,0.042735042735042736,0.15151515151515152,0.6957328385899815,10476,5
ni_k3_r4,ni_k3_r4,fixed,,,64,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,19824,0.03897597621804841,0.02295918367346939,0.07692307692307693,0.140625,0.6457270408163266,19824,9
ni_k2_r1,ni_k2_r1,fixed,,,71,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,21651,0.042568041822889736,0.015306122448979591,0.05128205128205128,0.08450704225352113,0.38804254096004603,21651,6
ni_k2_r2,ni_k2_r2,fixed,,,128,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,40529,0.0796840869724215,0.03571428571428571,0.11965811965811966,0.109375,0.5022321428571429,40529,14
ni_k2_r4,ni_k2_r4,fixed,,,224,,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,72373,0.1422925911435037,0.05357142857142857,0.1794871794871795,0.09375,0.43048469387755106,72373,21
wl_model_agreement__ni_k5_r1,ni_k5_r1,fixed,,,1009,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,279738,0.5499930203432418,0.2423469387755102,0.811965811965812,0.09415262636273539,0.4323334884003156,279738,95
wl_model_agreement__ni_k5_r2,ni_k5_r2,fixed,,,1009,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,279942,0.550394104844275,0.2423469387755102,0.811965811965812,0.09415262636273539,0.4323334884003156,279942,95
wl_model_agreement__ni_k5_r4,ni_k5_r4,fixed,,,1009,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,280332,0.5511608840374267,0.2423469387755102,0.811965811965812,0.09415262636273539,0.4323334884003156,280332,95
wl_model_agreement__ni_k3_r1,ni_k3_r1,fixed,,,1013,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,281415,0.5532901708737941,0.2423469387755102,0.811965811965812,0.09378084896347483,0.430626347281262,281415,95
wl_model_agreement__ni_k3_r2,ni_k3_r2,fixed,,,1019,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,283269,0.5569353211920074,0.24744897959183673,0.8290598290598291,0.09519136408243375,0.4371032024193387,283269,97
wl_model_agreement__ni_k3_r4,ni_k3_r4,fixed,,,1025,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,286666,0.563614164574408,0.24744897959183673,0.8290598290598291,0.09463414634146342,0.43454454952712795,286666,97
wl_model_agreement__ni_k2_r1,ni_k2_r1,fixed,,,1033,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,287117,0.5645008758977706,0.24489795918367346,0.8205128205128205,0.09293320425943853,0.4267341011912994,287117,96
wl_model_agreement__ni_k2_r2,ni_k2_r2,fixed,,,1045,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,293994,0.5780217490036785,0.25,0.8376068376068376,0.0937799043062201,0.4306220095693781,293994,98
wl_model_agreement__ni_k2_r4,ni_k2_r4,fixed,,,1071,wl_model_agreement,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,305593,0.6008265486482076,0.25,0.8376068376068376,0.0915032679738562,0.42016806722689076,305593,98
wl_strict_obvious__ni_k5_r1,ni_k5_r1,fixed,,,1239,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,1.0,351827,0.6917272389460914,0.2780612244897959,0.9316239316239316,0.08797417271993543,0.40396303799970357,351827,109
wl_strict_obvious__ni_k5_r2,ni_k5_r2,fixed,,,1239,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,2.0,351922,0.6919140184931413,0.2780612244897959,0.9316239316239316,0.08797417271993543,0.40396303799970357,351922,109
wl_strict_obvious__ni_k5_r4,ni_k5_r4,fixed,,,1239,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",5.0,4.0,352095,0.6922541538788214,0.2780612244897959,0.9316239316239316,0.08797417271993543,0.40396303799970357,352095,109
wl_strict_obvious__ni_k3_r1,ni_k3_r1,fixed,,,1241,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,1.0,352611,0.6932686617343758,0.2780612244897959,0.9316239316239316,0.08783239323126511,0.40331200973540104,352611,109
wl_strict_obvious__ni_k3_r2,ni_k3_r2,fixed,,,1244,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,2.0,353502,0.6950204572756532,0.28061224489795916,0.9401709401709402,0.08842443729903537,0.4060305794343461,353502,110
wl_strict_obvious__ni_k3_r4,ni_k3_r4,fixed,,,1247,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",3.0,4.0,355074,0.6981111672542031,0.28061224489795916,0.9401709401709402,0.08821170809943865,0.40505376168109586,355074,110
wl_strict_obvious__ni_k2_r1,ni_k2_r1,fixed,,,1251,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,1.0,355366,0.6986852685988192,0.2780612244897959,0.9316239316239316,0.08713029576338929,0.4000880927910733,355366,109
wl_strict_obvious__ni_k2_r2,ni_k2_r2,fixed,,,1258,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,2.0,358703,0.7052461459515041,0.28061224489795916,0.9401709401709402,0.08744038155802862,0.4015119561338049,358703,110
wl_strict_obvious__ni_k2_r4,ni_k2_r4,fixed,,,1274,wl_strict_obvious,"unavailable: PIG v1.02 annotations are distributed by the official authors upon request; no checksum-verifiable unattended download is published. Searched: /tmp/pianovam-audit-260724.djnuaa/PIG, /workspace/PIG. The strict validity gate remains closed.",2.0,4.0,364244,0.7161403087957438,0.28061224489795916,0.9401709401709402,0.08634222919937205,0.39646941979303496,364244,110
```

### True/all-GT per-finger table (all ten fingers)

```csv
set_id,finger_id,gt_notes,errors,selected_notes,selected_errors,error_recall,precision,scope
mandatory_missing,L1,252,45,35,35,0.7777777777777778,1.0,all_gt
mandatory_missing,L2,207,25,19,19,0.76,1.0,all_gt
mandatory_missing,L3,110,20,15,15,0.75,1.0,all_gt
mandatory_missing,L4,38,13,8,8,0.6153846153846154,1.0,all_gt
mandatory_missing,L5,279,72,54,54,0.75,1.0,all_gt
mandatory_missing,R1,252,71,62,62,0.8732394366197183,1.0,all_gt
mandatory_missing,R2,193,19,11,11,0.5789473684210527,1.0,all_gt
mandatory_missing,R3,160,21,14,14,0.6666666666666666,1.0,all_gt
mandatory_missing,R4,136,40,18,18,0.45,1.0,all_gt
mandatory_missing,R5,173,66,39,39,0.5909090909090909,1.0,all_gt
mandatory_missing,L1,217,10,0,0,0.0,,assigned_gt
mandatory_missing,L2,188,6,0,0,0.0,,assigned_gt
mandatory_missing,L3,95,5,0,0,0.0,,assigned_gt
mandatory_missing,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing,L5,225,18,0,0,0.0,,assigned_gt
mandatory_missing,R1,190,9,0,0,0.0,,assigned_gt
mandatory_missing,R2,182,8,0,0,0.0,,assigned_gt
mandatory_missing,R3,146,7,0,0,0.0,,assigned_gt
mandatory_missing,R4,118,22,0,0,0.0,,assigned_gt
mandatory_missing,R5,134,27,0,0,0.0,,assigned_gt
legacy_current_default,L1,252,45,5,1,0.022222222222222223,0.2,all_gt
legacy_current_default,L2,207,25,5,0,0.0,0.0,all_gt
legacy_current_default,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default,L5,279,72,21,5,0.06944444444444445,0.23809523809523808,all_gt
legacy_current_default,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
legacy_current_default,R2,193,19,2,0,0.0,0.0,all_gt
legacy_current_default,R3,160,21,8,0,0.0,0.0,all_gt
legacy_current_default,R4,136,40,15,7,0.175,0.4666666666666667,all_gt
legacy_current_default,R5,173,66,17,10,0.15151515151515152,0.5882352941176471,all_gt
legacy_current_default,L1,217,10,5,1,0.1,0.2,assigned_gt
legacy_current_default,L2,188,6,5,0,0.0,0.0,assigned_gt
legacy_current_default,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default,L5,225,18,21,5,0.2777777777777778,0.23809523809523808,assigned_gt
legacy_current_default,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
legacy_current_default,R2,182,8,2,0,0.0,0.0,assigned_gt
legacy_current_default,R3,146,7,8,0,0.0,0.0,assigned_gt
legacy_current_default,R4,118,22,15,7,0.3181818181818182,0.4666666666666667,assigned_gt
legacy_current_default,R5,134,27,17,10,0.37037037037037035,0.5882352941176471,assigned_gt
bl_span_practical,L1,252,45,13,0,0.0,0.0,all_gt
bl_span_practical,L2,207,25,10,0,0.0,0.0,all_gt
bl_span_practical,L3,110,20,7,0,0.0,0.0,all_gt
bl_span_practical,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_span_practical,R1,252,71,4,0,0.0,0.0,all_gt
bl_span_practical,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_span_practical,R3,160,21,4,0,0.0,0.0,all_gt
bl_span_practical,R4,136,40,9,0,0.0,0.0,all_gt
bl_span_practical,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_span_practical,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_span_practical,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_span_practical,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_span_practical,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_span_practical,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_span_practical,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_span_practical,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_span_practical,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_span_practical,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_span_comfortable,L1,252,45,24,0,0.0,0.0,all_gt
bl_span_comfortable,L2,207,25,22,0,0.0,0.0,all_gt
bl_span_comfortable,L3,110,20,20,0,0.0,0.0,all_gt
bl_span_comfortable,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable,L5,279,72,66,9,0.125,0.13636363636363635,all_gt
bl_span_comfortable,R1,252,71,5,0,0.0,0.0,all_gt
bl_span_comfortable,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
bl_span_comfortable,R3,160,21,17,1,0.047619047619047616,0.058823529411764705,all_gt
bl_span_comfortable,R4,136,40,21,2,0.05,0.09523809523809523,all_gt
bl_span_comfortable,R5,173,66,26,9,0.13636363636363635,0.34615384615384615,all_gt
bl_span_comfortable,L1,217,10,24,0,0.0,0.0,assigned_gt
bl_span_comfortable,L2,188,6,22,0,0.0,0.0,assigned_gt
bl_span_comfortable,L3,95,5,20,0,0.0,0.0,assigned_gt
bl_span_comfortable,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable,L5,225,18,66,9,0.5,0.13636363636363635,assigned_gt
bl_span_comfortable,R1,190,9,5,0,0.0,0.0,assigned_gt
bl_span_comfortable,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
bl_span_comfortable,R3,146,7,17,1,0.14285714285714285,0.058823529411764705,assigned_gt
bl_span_comfortable,R4,118,22,21,2,0.09090909090909091,0.09523809523809523,assigned_gt
bl_span_comfortable,R5,134,27,26,9,0.3333333333333333,0.34615384615384615,assigned_gt
bl_span_relative,L1,252,45,92,2,0.044444444444444446,0.021739130434782608,all_gt
bl_span_relative,L2,207,25,82,3,0.12,0.036585365853658534,all_gt
bl_span_relative,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative,L5,279,72,124,12,0.16666666666666666,0.0967741935483871,all_gt
bl_span_relative,R1,252,71,46,6,0.08450704225352113,0.13043478260869565,all_gt
bl_span_relative,R2,193,19,59,2,0.10526315789473684,0.03389830508474576,all_gt
bl_span_relative,R3,160,21,55,5,0.23809523809523808,0.09090909090909091,all_gt
bl_span_relative,R4,136,40,45,12,0.3,0.26666666666666666,all_gt
bl_span_relative,R5,173,66,75,12,0.18181818181818182,0.16,all_gt
bl_span_relative,L1,217,10,92,2,0.2,0.021739130434782608,assigned_gt
bl_span_relative,L2,188,6,82,3,0.5,0.036585365853658534,assigned_gt
bl_span_relative,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative,L5,225,18,124,12,0.6666666666666666,0.0967741935483871,assigned_gt
bl_span_relative,R1,190,9,46,6,0.6666666666666666,0.13043478260869565,assigned_gt
bl_span_relative,R2,182,8,59,2,0.25,0.03389830508474576,assigned_gt
bl_span_relative,R3,146,7,55,5,0.7142857142857143,0.09090909090909091,assigned_gt
bl_span_relative,R4,118,22,45,12,0.5454545454545454,0.26666666666666666,assigned_gt
bl_span_relative,R5,134,27,75,12,0.4444444444444444,0.16,assigned_gt
bl_crossing,L1,252,45,0,0,0.0,,all_gt
bl_crossing,L2,207,25,1,0,0.0,0.0,all_gt
bl_crossing,L3,110,20,1,0,0.0,0.0,all_gt
bl_crossing,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing,L5,279,72,6,3,0.041666666666666664,0.5,all_gt
bl_crossing,R1,252,71,0,0,0.0,,all_gt
bl_crossing,R2,193,19,0,0,0.0,,all_gt
bl_crossing,R3,160,21,2,0,0.0,0.0,all_gt
bl_crossing,R4,136,40,3,0,0.0,0.0,all_gt
bl_crossing,R5,173,66,6,4,0.06060606060606061,0.6666666666666666,all_gt
bl_crossing,L1,217,10,0,0,0.0,,assigned_gt
bl_crossing,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_crossing,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_crossing,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing,L5,225,18,6,3,0.16666666666666666,0.5,assigned_gt
bl_crossing,R1,190,9,0,0,0.0,,assigned_gt
bl_crossing,R2,182,8,0,0,0.0,,assigned_gt
bl_crossing,R3,146,7,2,0,0.0,0.0,assigned_gt
bl_crossing,R4,118,22,3,0,0.0,0.0,assigned_gt
bl_crossing,R5,134,27,6,4,0.14814814814814814,0.6666666666666666,assigned_gt
bl_step_crossing,L1,252,45,0,0,0.0,,all_gt
bl_step_crossing,L2,207,25,0,0,0.0,,all_gt
bl_step_crossing,L3,110,20,1,0,0.0,0.0,all_gt
bl_step_crossing,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing,L5,279,72,0,0,0.0,,all_gt
bl_step_crossing,R1,252,71,0,0,0.0,,all_gt
bl_step_crossing,R2,193,19,0,0,0.0,,all_gt
bl_step_crossing,R3,160,21,1,0,0.0,0.0,all_gt
bl_step_crossing,R4,136,40,1,0,0.0,0.0,all_gt
bl_step_crossing,R5,173,66,4,3,0.045454545454545456,0.75,all_gt
bl_step_crossing,L1,217,10,0,0,0.0,,assigned_gt
bl_step_crossing,L2,188,6,0,0,0.0,,assigned_gt
bl_step_crossing,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing,L5,225,18,0,0,0.0,,assigned_gt
bl_step_crossing,R1,190,9,0,0,0.0,,assigned_gt
bl_step_crossing,R2,182,8,0,0,0.0,,assigned_gt
bl_step_crossing,R3,146,7,1,0,0.0,0.0,assigned_gt
bl_step_crossing,R4,118,22,1,0,0.0,0.0,assigned_gt
bl_step_crossing,R5,134,27,4,3,0.1111111111111111,0.75,assigned_gt
bl_rate_q995,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q995,L2,207,25,0,0,0.0,,all_gt
bl_rate_q995,L3,110,20,0,0,0.0,,all_gt
bl_rate_q995,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995,L5,279,72,1,0,0.0,0.0,all_gt
bl_rate_q995,R1,252,71,3,0,0.0,0.0,all_gt
bl_rate_q995,R2,193,19,0,0,0.0,,all_gt
bl_rate_q995,R3,160,21,0,0,0.0,,all_gt
bl_rate_q995,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q995,R5,173,66,1,1,0.015151515151515152,1.0,all_gt
bl_rate_q995,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q995,L2,188,6,0,0,0.0,,assigned_gt
bl_rate_q995,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q995,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995,L5,225,18,1,0,0.0,0.0,assigned_gt
bl_rate_q995,R1,190,9,3,0,0.0,0.0,assigned_gt
bl_rate_q995,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q995,R3,146,7,0,0,0.0,,assigned_gt
bl_rate_q995,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q995,R5,134,27,1,1,0.037037037037037035,1.0,assigned_gt
bl_rate_q990,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q990,L2,207,25,2,0,0.0,0.0,all_gt
bl_rate_q990,L3,110,20,0,0,0.0,,all_gt
bl_rate_q990,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990,L5,279,72,4,1,0.013888888888888888,0.25,all_gt
bl_rate_q990,R1,252,71,4,0,0.0,0.0,all_gt
bl_rate_q990,R2,193,19,0,0,0.0,,all_gt
bl_rate_q990,R3,160,21,1,1,0.047619047619047616,1.0,all_gt
bl_rate_q990,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q990,R5,173,66,3,1,0.015151515151515152,0.3333333333333333,all_gt
bl_rate_q990,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q990,L2,188,6,2,0,0.0,0.0,assigned_gt
bl_rate_q990,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q990,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990,L5,225,18,4,1,0.05555555555555555,0.25,assigned_gt
bl_rate_q990,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_rate_q990,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q990,R3,146,7,1,1,0.14285714285714285,1.0,assigned_gt
bl_rate_q990,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q990,R5,134,27,3,1,0.037037037037037035,0.3333333333333333,assigned_gt
bl_rate_q975,L1,252,45,6,0,0.0,0.0,all_gt
bl_rate_q975,L2,207,25,3,0,0.0,0.0,all_gt
bl_rate_q975,L3,110,20,0,0,0.0,,all_gt
bl_rate_q975,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975,L5,279,72,8,1,0.013888888888888888,0.125,all_gt
bl_rate_q975,R1,252,71,11,0,0.0,0.0,all_gt
bl_rate_q975,R2,193,19,0,0,0.0,,all_gt
bl_rate_q975,R3,160,21,3,1,0.047619047619047616,0.3333333333333333,all_gt
bl_rate_q975,R4,136,40,3,2,0.05,0.6666666666666666,all_gt
bl_rate_q975,R5,173,66,7,1,0.015151515151515152,0.14285714285714285,all_gt
bl_rate_q975,L1,217,10,6,0,0.0,0.0,assigned_gt
bl_rate_q975,L2,188,6,3,0,0.0,0.0,assigned_gt
bl_rate_q975,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q975,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975,L5,225,18,8,1,0.05555555555555555,0.125,assigned_gt
bl_rate_q975,R1,190,9,11,0,0.0,0.0,assigned_gt
bl_rate_q975,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q975,R3,146,7,3,1,0.14285714285714285,0.3333333333333333,assigned_gt
bl_rate_q975,R4,118,22,3,2,0.09090909090909091,0.6666666666666666,assigned_gt
bl_rate_q975,R5,134,27,7,1,0.037037037037037035,0.14285714285714285,assigned_gt
bl_hmm_disagreement,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
bl_hmm_disagreement,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
bl_hmm_disagreement,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
bl_hmm_disagreement,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
bl_hmm_disagreement,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
bl_hmm_disagreement,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
bl_hmm_disagreement,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
bl_hmm_disagreement,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
bl_hmm_disagreement,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
bl_hmm_disagreement,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
bl_practical_or_rate995,L1,252,45,15,0,0.0,0.0,all_gt
bl_practical_or_rate995,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_rate995,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_rate995,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995,L5,279,72,57,6,0.08333333333333333,0.10526315789473684,all_gt
bl_practical_or_rate995,R1,252,71,7,0,0.0,0.0,all_gt
bl_practical_or_rate995,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_rate995,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_rate995,R4,136,40,10,1,0.025,0.1,all_gt
bl_practical_or_rate995,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_practical_or_rate995,L1,217,10,15,0,0.0,0.0,assigned_gt
bl_practical_or_rate995,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995,L5,225,18,57,6,0.3333333333333333,0.10526315789473684,assigned_gt
bl_practical_or_rate995,R1,190,9,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_rate995,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995,R4,118,22,10,1,0.045454545454545456,0.1,assigned_gt
bl_practical_or_rate995,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
bl_practical_or_crossing,L1,252,45,13,0,0.0,0.0,all_gt
bl_practical_or_crossing,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_crossing,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_crossing,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_practical_or_crossing,R1,252,71,4,0,0.0,0.0,all_gt
bl_practical_or_crossing,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_crossing,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_crossing,R4,136,40,9,0,0.0,0.0,all_gt
bl_practical_or_crossing,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_practical_or_crossing,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_practical_or_crossing,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_crossing,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_practical_or_crossing,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_two_signal_strict,L1,252,45,1,0,0.0,0.0,all_gt
bl_two_signal_strict,L2,207,25,9,0,0.0,0.0,all_gt
bl_two_signal_strict,L3,110,20,7,0,0.0,0.0,all_gt
bl_two_signal_strict,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict,L5,279,72,28,7,0.09722222222222222,0.25,all_gt
bl_two_signal_strict,R1,252,71,2,0,0.0,0.0,all_gt
bl_two_signal_strict,R2,193,19,7,1,0.05263157894736842,0.14285714285714285,all_gt
bl_two_signal_strict,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
bl_two_signal_strict,R4,136,40,9,0,0.0,0.0,all_gt
bl_two_signal_strict,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_two_signal_strict,L1,217,10,1,0,0.0,0.0,assigned_gt
bl_two_signal_strict,L2,188,6,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_two_signal_strict,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict,L5,225,18,28,7,0.3888888888888889,0.25,assigned_gt
bl_two_signal_strict,R1,190,9,2,0,0.0,0.0,assigned_gt
bl_two_signal_strict,R2,182,8,7,1,0.125,0.14285714285714285,assigned_gt
bl_two_signal_strict,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
bl_two_signal_strict,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
wl_model_agreement,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
wl_model_agreement,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
wl_model_agreement,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
wl_model_agreement,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
wl_model_agreement,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
wl_model_agreement,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
wl_model_agreement,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
wl_model_agreement,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
wl_model_agreement,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
wl_model_agreement,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
wl_strict_obvious,L1,252,45,150,7,0.15555555555555556,0.04666666666666667,all_gt
wl_strict_obvious,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious,L5,279,72,188,17,0.2361111111111111,0.09042553191489362,all_gt
wl_strict_obvious,R1,252,71,114,8,0.11267605633802817,0.07017543859649122,all_gt
wl_strict_obvious,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious,R4,136,40,113,20,0.5,0.17699115044247787,all_gt
wl_strict_obvious,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious,L1,217,10,150,7,0.7,0.04666666666666667,assigned_gt
wl_strict_obvious,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious,L5,225,18,188,17,0.9444444444444444,0.09042553191489362,assigned_gt
wl_strict_obvious,R1,190,9,114,8,0.8888888888888888,0.07017543859649122,assigned_gt
wl_strict_obvious,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious,R4,118,22,113,20,0.9090909090909091,0.17699115044247787,assigned_gt
wl_strict_obvious,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
hy_direct_plus_corroborated,L1,252,45,14,0,0.0,0.0,all_gt
hy_direct_plus_corroborated,L2,207,25,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated,L3,110,20,7,0,0.0,0.0,all_gt
hy_direct_plus_corroborated,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated,L5,279,72,57,7,0.09722222222222222,0.12280701754385964,all_gt
hy_direct_plus_corroborated,R1,252,71,6,0,0.0,0.0,all_gt
hy_direct_plus_corroborated,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
hy_direct_plus_corroborated,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
hy_direct_plus_corroborated,R4,136,40,9,0,0.0,0.0,all_gt
hy_direct_plus_corroborated,R5,173,66,16,7,0.10606060606060606,0.4375,all_gt
hy_direct_plus_corroborated,L1,217,10,14,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated,L2,188,6,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated,L3,95,5,7,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated,L5,225,18,57,7,0.3888888888888889,0.12280701754385964,assigned_gt
hy_direct_plus_corroborated,R1,190,9,6,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
hy_direct_plus_corroborated,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
hy_direct_plus_corroborated,R4,118,22,9,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated,R5,134,27,16,7,0.25925925925925924,0.4375,assigned_gt
hy_two_of_three_families,L1,252,45,5,0,0.0,0.0,all_gt
hy_two_of_three_families,L2,207,25,20,0,0.0,0.0,all_gt
hy_two_of_three_families,L3,110,20,20,0,0.0,0.0,all_gt
hy_two_of_three_families,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families,L5,279,72,36,10,0.1388888888888889,0.2777777777777778,all_gt
hy_two_of_three_families,R1,252,71,3,0,0.0,0.0,all_gt
hy_two_of_three_families,R2,193,19,20,1,0.05263157894736842,0.05,all_gt
hy_two_of_three_families,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_two_of_three_families,R4,136,40,19,2,0.05,0.10526315789473684,all_gt
hy_two_of_three_families,R5,173,66,21,9,0.13636363636363635,0.42857142857142855,all_gt
hy_two_of_three_families,L1,217,10,5,0,0.0,0.0,assigned_gt
hy_two_of_three_families,L2,188,6,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families,L5,225,18,36,10,0.5555555555555556,0.2777777777777778,assigned_gt
hy_two_of_three_families,R1,190,9,3,0,0.0,0.0,assigned_gt
hy_two_of_three_families,R2,182,8,20,1,0.125,0.05,assigned_gt
hy_two_of_three_families,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_two_of_three_families,R4,118,22,19,2,0.09090909090909091,0.10526315789473684,assigned_gt
hy_two_of_three_families,R5,134,27,21,9,0.3333333333333333,0.42857142857142855,assigned_gt
hy_hierarchical,L1,252,45,18,0,0.0,0.0,all_gt
hy_hierarchical,L2,207,25,21,0,0.0,0.0,all_gt
hy_hierarchical,L3,110,20,20,0,0.0,0.0,all_gt
hy_hierarchical,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical,L5,279,72,65,10,0.1388888888888889,0.15384615384615385,all_gt
hy_hierarchical,R1,252,71,7,0,0.0,0.0,all_gt
hy_hierarchical,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
hy_hierarchical,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_hierarchical,R4,136,40,18,1,0.025,0.05555555555555555,all_gt
hy_hierarchical,R5,173,66,24,9,0.13636363636363635,0.375,all_gt
hy_hierarchical,L1,217,10,18,0,0.0,0.0,assigned_gt
hy_hierarchical,L2,188,6,21,0,0.0,0.0,assigned_gt
hy_hierarchical,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_hierarchical,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical,L5,225,18,65,10,0.5555555555555556,0.15384615384615385,assigned_gt
hy_hierarchical,R1,190,9,7,0,0.0,0.0,assigned_gt
hy_hierarchical,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
hy_hierarchical,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_hierarchical,R4,118,22,18,1,0.045454545454545456,0.05555555555555555,assigned_gt
hy_hierarchical,R5,134,27,24,9,0.3333333333333333,0.375,assigned_gt
ni_k2_r1,L1,252,45,10,0,0.0,0.0,all_gt
ni_k2_r1,L2,207,25,6,1,0.04,0.16666666666666666,all_gt
ni_k2_r1,L3,110,20,2,0,0.0,0.0,all_gt
ni_k2_r1,L4,38,13,0,0,0.0,,all_gt
ni_k2_r1,L5,279,72,7,0,0.0,0.0,all_gt
ni_k2_r1,R1,252,71,14,1,0.014084507042253521,0.07142857142857142,all_gt
ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
ni_k2_r1,R3,160,21,10,0,0.0,0.0,all_gt
ni_k2_r1,R4,136,40,6,0,0.0,0.0,all_gt
ni_k2_r1,R5,173,66,10,4,0.06060606060606061,0.4,all_gt
ni_k2_r1,L1,217,10,10,0,0.0,0.0,assigned_gt
ni_k2_r1,L2,188,6,6,1,0.16666666666666666,0.16666666666666666,assigned_gt
ni_k2_r1,L3,95,5,2,0,0.0,0.0,assigned_gt
ni_k2_r1,L4,30,5,0,0,0.0,,assigned_gt
ni_k2_r1,L5,225,18,7,0,0.0,0.0,assigned_gt
ni_k2_r1,R1,190,9,14,1,0.1111111111111111,0.07142857142857142,assigned_gt
ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
ni_k2_r1,R3,146,7,10,0,0.0,0.0,assigned_gt
ni_k2_r1,R4,118,22,6,0,0.0,0.0,assigned_gt
ni_k2_r1,R5,134,27,10,4,0.14814814814814814,0.4,assigned_gt
ni_k2_r2,L1,252,45,14,0,0.0,0.0,all_gt
ni_k2_r2,L2,207,25,14,1,0.04,0.07142857142857142,all_gt
ni_k2_r2,L3,110,20,7,0,0.0,0.0,all_gt
ni_k2_r2,L4,38,13,1,0,0.0,0.0,all_gt
ni_k2_r2,L5,279,72,13,2,0.027777777777777776,0.15384615384615385,all_gt
ni_k2_r2,R1,252,71,21,3,0.04225352112676056,0.14285714285714285,all_gt
ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
ni_k2_r2,R3,160,21,16,0,0.0,0.0,all_gt
ni_k2_r2,R4,136,40,13,2,0.05,0.15384615384615385,all_gt
ni_k2_r2,R5,173,66,16,6,0.09090909090909091,0.375,all_gt
ni_k2_r2,L1,217,10,14,0,0.0,0.0,assigned_gt
ni_k2_r2,L2,188,6,14,1,0.16666666666666666,0.07142857142857142,assigned_gt
ni_k2_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
ni_k2_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
ni_k2_r2,L5,225,18,13,2,0.1111111111111111,0.15384615384615385,assigned_gt
ni_k2_r2,R1,190,9,21,3,0.3333333333333333,0.14285714285714285,assigned_gt
ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
ni_k2_r2,R3,146,7,16,0,0.0,0.0,assigned_gt
ni_k2_r2,R4,118,22,13,2,0.09090909090909091,0.15384615384615385,assigned_gt
ni_k2_r2,R5,134,27,16,6,0.2222222222222222,0.375,assigned_gt
ni_k2_r4,L1,252,45,29,0,0.0,0.0,all_gt
ni_k2_r4,L2,207,25,28,1,0.04,0.03571428571428571,all_gt
ni_k2_r4,L3,110,20,10,0,0.0,0.0,all_gt
ni_k2_r4,L4,38,13,2,0,0.0,0.0,all_gt
ni_k2_r4,L5,279,72,25,3,0.041666666666666664,0.12,all_gt
ni_k2_r4,R1,252,71,28,3,0.04225352112676056,0.10714285714285714,all_gt
ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
ni_k2_r4,R3,160,21,27,0,0.0,0.0,all_gt
ni_k2_r4,R4,136,40,28,6,0.15,0.21428571428571427,all_gt
ni_k2_r4,R5,173,66,27,8,0.12121212121212122,0.2962962962962963,all_gt
ni_k2_r4,L1,217,10,29,0,0.0,0.0,assigned_gt
ni_k2_r4,L2,188,6,28,1,0.16666666666666666,0.03571428571428571,assigned_gt
ni_k2_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
ni_k2_r4,L4,30,5,2,0,0.0,0.0,assigned_gt
ni_k2_r4,L5,225,18,25,3,0.16666666666666666,0.12,assigned_gt
ni_k2_r4,R1,190,9,28,3,0.3333333333333333,0.10714285714285714,assigned_gt
ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
ni_k2_r4,R3,146,7,27,0,0.0,0.0,assigned_gt
ni_k2_r4,R4,118,22,28,6,0.2727272727272727,0.21428571428571427,assigned_gt
ni_k2_r4,R5,134,27,27,8,0.2962962962962963,0.2962962962962963,assigned_gt
ni_k3_r1,L1,252,45,1,0,0.0,0.0,all_gt
ni_k3_r1,L2,207,25,0,0,0.0,,all_gt
ni_k3_r1,L3,110,20,0,0,0.0,,all_gt
ni_k3_r1,L4,38,13,0,0,0.0,,all_gt
ni_k3_r1,L5,279,72,2,0,0.0,0.0,all_gt
ni_k3_r1,R1,252,71,4,1,0.014084507042253521,0.25,all_gt
ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
ni_k3_r1,R3,160,21,3,0,0.0,0.0,all_gt
ni_k3_r1,R4,136,40,3,0,0.0,0.0,all_gt
ni_k3_r1,R5,173,66,3,1,0.015151515151515152,0.3333333333333333,all_gt
ni_k3_r1,L1,217,10,1,0,0.0,0.0,assigned_gt
ni_k3_r1,L2,188,6,0,0,0.0,,assigned_gt
ni_k3_r1,L3,95,5,0,0,0.0,,assigned_gt
ni_k3_r1,L4,30,5,0,0,0.0,,assigned_gt
ni_k3_r1,L5,225,18,2,0,0.0,0.0,assigned_gt
ni_k3_r1,R1,190,9,4,1,0.1111111111111111,0.25,assigned_gt
ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
ni_k3_r1,R3,146,7,3,0,0.0,0.0,assigned_gt
ni_k3_r1,R4,118,22,3,0,0.0,0.0,assigned_gt
ni_k3_r1,R5,134,27,3,1,0.037037037037037035,0.3333333333333333,assigned_gt
ni_k3_r2,L1,252,45,4,0,0.0,0.0,all_gt
ni_k3_r2,L2,207,25,1,0,0.0,0.0,all_gt
ni_k3_r2,L3,110,20,3,0,0.0,0.0,all_gt
ni_k3_r2,L4,38,13,0,0,0.0,,all_gt
ni_k3_r2,L5,279,72,2,0,0.0,0.0,all_gt
ni_k3_r2,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
ni_k3_r2,R3,160,21,3,0,0.0,0.0,all_gt
ni_k3_r2,R4,136,40,4,1,0.025,0.25,all_gt
ni_k3_r2,R5,173,66,7,2,0.030303030303030304,0.2857142857142857,all_gt
ni_k3_r2,L1,217,10,4,0,0.0,0.0,assigned_gt
ni_k3_r2,L2,188,6,1,0,0.0,0.0,assigned_gt
ni_k3_r2,L3,95,5,3,0,0.0,0.0,assigned_gt
ni_k3_r2,L4,30,5,0,0,0.0,,assigned_gt
ni_k3_r2,L5,225,18,2,0,0.0,0.0,assigned_gt
ni_k3_r2,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
ni_k3_r2,R3,146,7,3,0,0.0,0.0,assigned_gt
ni_k3_r2,R4,118,22,4,1,0.045454545454545456,0.25,assigned_gt
ni_k3_r2,R5,134,27,7,2,0.07407407407407407,0.2857142857142857,assigned_gt
ni_k3_r4,L1,252,45,9,0,0.0,0.0,all_gt
ni_k3_r4,L2,207,25,6,0,0.0,0.0,all_gt
ni_k3_r4,L3,110,20,3,0,0.0,0.0,all_gt
ni_k3_r4,L4,38,13,1,0,0.0,0.0,all_gt
ni_k3_r4,L5,279,72,6,0,0.0,0.0,all_gt
ni_k3_r4,R1,252,71,8,2,0.028169014084507043,0.25,all_gt
ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
ni_k3_r4,R3,160,21,6,0,0.0,0.0,all_gt
ni_k3_r4,R4,136,40,11,4,0.1,0.36363636363636365,all_gt
ni_k3_r4,R5,173,66,10,3,0.045454545454545456,0.3,all_gt
ni_k3_r4,L1,217,10,9,0,0.0,0.0,assigned_gt
ni_k3_r4,L2,188,6,6,0,0.0,0.0,assigned_gt
ni_k3_r4,L3,95,5,3,0,0.0,0.0,assigned_gt
ni_k3_r4,L4,30,5,1,0,0.0,0.0,assigned_gt
ni_k3_r4,L5,225,18,6,0,0.0,0.0,assigned_gt
ni_k3_r4,R1,190,9,8,2,0.2222222222222222,0.25,assigned_gt
ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
ni_k3_r4,R3,146,7,6,0,0.0,0.0,assigned_gt
ni_k3_r4,R4,118,22,11,4,0.18181818181818182,0.36363636363636365,assigned_gt
ni_k3_r4,R5,134,27,10,3,0.1111111111111111,0.3,assigned_gt
ni_k5_r1,L1,252,45,0,0,0.0,,all_gt
ni_k5_r1,L2,207,25,0,0,0.0,,all_gt
ni_k5_r1,L3,110,20,0,0,0.0,,all_gt
ni_k5_r1,L4,38,13,0,0,0.0,,all_gt
ni_k5_r1,L5,279,72,0,0,0.0,,all_gt
ni_k5_r1,R1,252,71,0,0,0.0,,all_gt
ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
ni_k5_r1,R3,160,21,0,0,0.0,,all_gt
ni_k5_r1,R4,136,40,0,0,0.0,,all_gt
ni_k5_r1,R5,173,66,0,0,0.0,,all_gt
ni_k5_r1,L1,217,10,0,0,0.0,,assigned_gt
ni_k5_r1,L2,188,6,0,0,0.0,,assigned_gt
ni_k5_r1,L3,95,5,0,0,0.0,,assigned_gt
ni_k5_r1,L4,30,5,0,0,0.0,,assigned_gt
ni_k5_r1,L5,225,18,0,0,0.0,,assigned_gt
ni_k5_r1,R1,190,9,0,0,0.0,,assigned_gt
ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
ni_k5_r1,R3,146,7,0,0,0.0,,assigned_gt
ni_k5_r1,R4,118,22,0,0,0.0,,assigned_gt
ni_k5_r1,R5,134,27,0,0,0.0,,assigned_gt
ni_k5_r2,L1,252,45,0,0,0.0,,all_gt
ni_k5_r2,L2,207,25,0,0,0.0,,all_gt
ni_k5_r2,L3,110,20,0,0,0.0,,all_gt
ni_k5_r2,L4,38,13,0,0,0.0,,all_gt
ni_k5_r2,L5,279,72,0,0,0.0,,all_gt
ni_k5_r2,R1,252,71,0,0,0.0,,all_gt
ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
ni_k5_r2,R3,160,21,0,0,0.0,,all_gt
ni_k5_r2,R4,136,40,0,0,0.0,,all_gt
ni_k5_r2,R5,173,66,0,0,0.0,,all_gt
ni_k5_r2,L1,217,10,0,0,0.0,,assigned_gt
ni_k5_r2,L2,188,6,0,0,0.0,,assigned_gt
ni_k5_r2,L3,95,5,0,0,0.0,,assigned_gt
ni_k5_r2,L4,30,5,0,0,0.0,,assigned_gt
ni_k5_r2,L5,225,18,0,0,0.0,,assigned_gt
ni_k5_r2,R1,190,9,0,0,0.0,,assigned_gt
ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
ni_k5_r2,R3,146,7,0,0,0.0,,assigned_gt
ni_k5_r2,R4,118,22,0,0,0.0,,assigned_gt
ni_k5_r2,R5,134,27,0,0,0.0,,assigned_gt
ni_k5_r4,L1,252,45,0,0,0.0,,all_gt
ni_k5_r4,L2,207,25,0,0,0.0,,all_gt
ni_k5_r4,L3,110,20,0,0,0.0,,all_gt
ni_k5_r4,L4,38,13,0,0,0.0,,all_gt
ni_k5_r4,L5,279,72,0,0,0.0,,all_gt
ni_k5_r4,R1,252,71,0,0,0.0,,all_gt
ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
ni_k5_r4,R3,160,21,0,0,0.0,,all_gt
ni_k5_r4,R4,136,40,0,0,0.0,,all_gt
ni_k5_r4,R5,173,66,0,0,0.0,,all_gt
ni_k5_r4,L1,217,10,0,0,0.0,,assigned_gt
ni_k5_r4,L2,188,6,0,0,0.0,,assigned_gt
ni_k5_r4,L3,95,5,0,0,0.0,,assigned_gt
ni_k5_r4,L4,30,5,0,0,0.0,,assigned_gt
ni_k5_r4,L5,225,18,0,0,0.0,,assigned_gt
ni_k5_r4,R1,190,9,0,0,0.0,,assigned_gt
ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
ni_k5_r4,R3,146,7,0,0,0.0,,assigned_gt
ni_k5_r4,R4,118,22,0,0,0.0,,assigned_gt
ni_k5_r4,R5,134,27,0,0,0.0,,assigned_gt
ni_w5_q995,L1,252,45,1,0,0.0,0.0,all_gt
ni_w5_q995,L2,207,25,0,0,0.0,,all_gt
ni_w5_q995,L3,110,20,0,0,0.0,,all_gt
ni_w5_q995,L4,38,13,0,0,0.0,,all_gt
ni_w5_q995,L5,279,72,0,0,0.0,,all_gt
ni_w5_q995,R1,252,71,0,0,0.0,,all_gt
ni_w5_q995,R2,193,19,0,0,0.0,,all_gt
ni_w5_q995,R3,160,21,1,0,0.0,0.0,all_gt
ni_w5_q995,R4,136,40,1,0,0.0,0.0,all_gt
ni_w5_q995,R5,173,66,0,0,0.0,,all_gt
ni_w5_q995,L1,217,10,1,0,0.0,0.0,assigned_gt
ni_w5_q995,L2,188,6,0,0,0.0,,assigned_gt
ni_w5_q995,L3,95,5,0,0,0.0,,assigned_gt
ni_w5_q995,L4,30,5,0,0,0.0,,assigned_gt
ni_w5_q995,L5,225,18,0,0,0.0,,assigned_gt
ni_w5_q995,R1,190,9,0,0,0.0,,assigned_gt
ni_w5_q995,R2,182,8,0,0,0.0,,assigned_gt
ni_w5_q995,R3,146,7,1,0,0.0,0.0,assigned_gt
ni_w5_q995,R4,118,22,1,0,0.0,0.0,assigned_gt
ni_w5_q995,R5,134,27,0,0,0.0,,assigned_gt
ni_w5_q990,L1,252,45,2,0,0.0,0.0,all_gt
ni_w5_q990,L2,207,25,0,0,0.0,,all_gt
ni_w5_q990,L3,110,20,0,0,0.0,,all_gt
ni_w5_q990,L4,38,13,0,0,0.0,,all_gt
ni_w5_q990,L5,279,72,0,0,0.0,,all_gt
ni_w5_q990,R1,252,71,1,0,0.0,0.0,all_gt
ni_w5_q990,R2,193,19,0,0,0.0,,all_gt
ni_w5_q990,R3,160,21,1,0,0.0,0.0,all_gt
ni_w5_q990,R4,136,40,4,0,0.0,0.0,all_gt
ni_w5_q990,R5,173,66,2,0,0.0,0.0,all_gt
ni_w5_q990,L1,217,10,2,0,0.0,0.0,assigned_gt
ni_w5_q990,L2,188,6,0,0,0.0,,assigned_gt
ni_w5_q990,L3,95,5,0,0,0.0,,assigned_gt
ni_w5_q990,L4,30,5,0,0,0.0,,assigned_gt
ni_w5_q990,L5,225,18,0,0,0.0,,assigned_gt
ni_w5_q990,R1,190,9,1,0,0.0,0.0,assigned_gt
ni_w5_q990,R2,182,8,0,0,0.0,,assigned_gt
ni_w5_q990,R3,146,7,1,0,0.0,0.0,assigned_gt
ni_w5_q990,R4,118,22,4,0,0.0,0.0,assigned_gt
ni_w5_q990,R5,134,27,2,0,0.0,0.0,assigned_gt
ni_w5_q975,L1,252,45,6,0,0.0,0.0,all_gt
ni_w5_q975,L2,207,25,2,1,0.04,0.5,all_gt
ni_w5_q975,L3,110,20,1,0,0.0,0.0,all_gt
ni_w5_q975,L4,38,13,0,0,0.0,,all_gt
ni_w5_q975,L5,279,72,2,0,0.0,0.0,all_gt
ni_w5_q975,R1,252,71,5,1,0.014084507042253521,0.2,all_gt
ni_w5_q975,R2,193,19,2,0,0.0,0.0,all_gt
ni_w5_q975,R3,160,21,3,0,0.0,0.0,all_gt
ni_w5_q975,R4,136,40,5,0,0.0,0.0,all_gt
ni_w5_q975,R5,173,66,7,3,0.045454545454545456,0.42857142857142855,all_gt
ni_w5_q975,L1,217,10,6,0,0.0,0.0,assigned_gt
ni_w5_q975,L2,188,6,2,1,0.16666666666666666,0.5,assigned_gt
ni_w5_q975,L3,95,5,1,0,0.0,0.0,assigned_gt
ni_w5_q975,L4,30,5,0,0,0.0,,assigned_gt
ni_w5_q975,L5,225,18,2,0,0.0,0.0,assigned_gt
ni_w5_q975,R1,190,9,5,1,0.1111111111111111,0.2,assigned_gt
ni_w5_q975,R2,182,8,2,0,0.0,0.0,assigned_gt
ni_w5_q975,R3,146,7,3,0,0.0,0.0,assigned_gt
ni_w5_q975,R4,118,22,5,0,0.0,0.0,assigned_gt
ni_w5_q975,R5,134,27,7,3,0.1111111111111111,0.42857142857142855,assigned_gt
ni_w9_q995,L1,252,45,2,0,0.0,0.0,all_gt
ni_w9_q995,L2,207,25,0,0,0.0,,all_gt
ni_w9_q995,L3,110,20,0,0,0.0,,all_gt
ni_w9_q995,L4,38,13,0,0,0.0,,all_gt
ni_w9_q995,L5,279,72,0,0,0.0,,all_gt
ni_w9_q995,R1,252,71,0,0,0.0,,all_gt
ni_w9_q995,R2,193,19,0,0,0.0,,all_gt
ni_w9_q995,R3,160,21,1,0,0.0,0.0,all_gt
ni_w9_q995,R4,136,40,3,0,0.0,0.0,all_gt
ni_w9_q995,R5,173,66,2,0,0.0,0.0,all_gt
ni_w9_q995,L1,217,10,2,0,0.0,0.0,assigned_gt
ni_w9_q995,L2,188,6,0,0,0.0,,assigned_gt
ni_w9_q995,L3,95,5,0,0,0.0,,assigned_gt
ni_w9_q995,L4,30,5,0,0,0.0,,assigned_gt
ni_w9_q995,L5,225,18,0,0,0.0,,assigned_gt
ni_w9_q995,R1,190,9,0,0,0.0,,assigned_gt
ni_w9_q995,R2,182,8,0,0,0.0,,assigned_gt
ni_w9_q995,R3,146,7,1,0,0.0,0.0,assigned_gt
ni_w9_q995,R4,118,22,3,0,0.0,0.0,assigned_gt
ni_w9_q995,R5,134,27,2,0,0.0,0.0,assigned_gt
ni_w9_q990,L1,252,45,6,0,0.0,0.0,all_gt
ni_w9_q990,L2,207,25,0,0,0.0,,all_gt
ni_w9_q990,L3,110,20,0,0,0.0,,all_gt
ni_w9_q990,L4,38,13,0,0,0.0,,all_gt
ni_w9_q990,L5,279,72,2,0,0.0,0.0,all_gt
ni_w9_q990,R1,252,71,1,0,0.0,0.0,all_gt
ni_w9_q990,R2,193,19,5,0,0.0,0.0,all_gt
ni_w9_q990,R3,160,21,2,0,0.0,0.0,all_gt
ni_w9_q990,R4,136,40,3,0,0.0,0.0,all_gt
ni_w9_q990,R5,173,66,4,1,0.015151515151515152,0.25,all_gt
ni_w9_q990,L1,217,10,6,0,0.0,0.0,assigned_gt
ni_w9_q990,L2,188,6,0,0,0.0,,assigned_gt
ni_w9_q990,L3,95,5,0,0,0.0,,assigned_gt
ni_w9_q990,L4,30,5,0,0,0.0,,assigned_gt
ni_w9_q990,L5,225,18,2,0,0.0,0.0,assigned_gt
ni_w9_q990,R1,190,9,1,0,0.0,0.0,assigned_gt
ni_w9_q990,R2,182,8,5,0,0.0,0.0,assigned_gt
ni_w9_q990,R3,146,7,2,0,0.0,0.0,assigned_gt
ni_w9_q990,R4,118,22,3,0,0.0,0.0,assigned_gt
ni_w9_q990,R5,134,27,4,1,0.037037037037037035,0.25,assigned_gt
ni_w9_q975,L1,252,45,8,0,0.0,0.0,all_gt
ni_w9_q975,L2,207,25,3,1,0.04,0.3333333333333333,all_gt
ni_w9_q975,L3,110,20,0,0,0.0,,all_gt
ni_w9_q975,L4,38,13,0,0,0.0,,all_gt
ni_w9_q975,L5,279,72,3,0,0.0,0.0,all_gt
ni_w9_q975,R1,252,71,4,0,0.0,0.0,all_gt
ni_w9_q975,R2,193,19,6,0,0.0,0.0,all_gt
ni_w9_q975,R3,160,21,4,0,0.0,0.0,all_gt
ni_w9_q975,R4,136,40,4,0,0.0,0.0,all_gt
ni_w9_q975,R5,173,66,11,3,0.045454545454545456,0.2727272727272727,all_gt
ni_w9_q975,L1,217,10,8,0,0.0,0.0,assigned_gt
ni_w9_q975,L2,188,6,3,1,0.16666666666666666,0.3333333333333333,assigned_gt
ni_w9_q975,L3,95,5,0,0,0.0,,assigned_gt
ni_w9_q975,L4,30,5,0,0,0.0,,assigned_gt
ni_w9_q975,L5,225,18,3,0,0.0,0.0,assigned_gt
ni_w9_q975,R1,190,9,4,0,0.0,0.0,assigned_gt
ni_w9_q975,R2,182,8,6,0,0.0,0.0,assigned_gt
ni_w9_q975,R3,146,7,4,0,0.0,0.0,assigned_gt
ni_w9_q975,R4,118,22,4,0,0.0,0.0,assigned_gt
ni_w9_q975,R5,134,27,11,3,0.1111111111111111,0.2727272727272727,assigned_gt
ni_w17_q995,L1,252,45,5,0,0.0,0.0,all_gt
ni_w17_q995,L2,207,25,1,0,0.0,0.0,all_gt
ni_w17_q995,L3,110,20,0,0,0.0,,all_gt
ni_w17_q995,L4,38,13,0,0,0.0,,all_gt
ni_w17_q995,L5,279,72,3,0,0.0,0.0,all_gt
ni_w17_q995,R1,252,71,0,0,0.0,,all_gt
ni_w17_q995,R2,193,19,3,0,0.0,0.0,all_gt
ni_w17_q995,R3,160,21,1,0,0.0,0.0,all_gt
ni_w17_q995,R4,136,40,0,0,0.0,,all_gt
ni_w17_q995,R5,173,66,4,1,0.015151515151515152,0.25,all_gt
ni_w17_q995,L1,217,10,5,0,0.0,0.0,assigned_gt
ni_w17_q995,L2,188,6,1,0,0.0,0.0,assigned_gt
ni_w17_q995,L3,95,5,0,0,0.0,,assigned_gt
ni_w17_q995,L4,30,5,0,0,0.0,,assigned_gt
ni_w17_q995,L5,225,18,3,0,0.0,0.0,assigned_gt
ni_w17_q995,R1,190,9,0,0,0.0,,assigned_gt
ni_w17_q995,R2,182,8,3,0,0.0,0.0,assigned_gt
ni_w17_q995,R3,146,7,1,0,0.0,0.0,assigned_gt
ni_w17_q995,R4,118,22,0,0,0.0,,assigned_gt
ni_w17_q995,R5,134,27,4,1,0.037037037037037035,0.25,assigned_gt
ni_w17_q990,L1,252,45,6,0,0.0,0.0,all_gt
ni_w17_q990,L2,207,25,1,0,0.0,0.0,all_gt
ni_w17_q990,L3,110,20,0,0,0.0,,all_gt
ni_w17_q990,L4,38,13,0,0,0.0,,all_gt
ni_w17_q990,L5,279,72,3,0,0.0,0.0,all_gt
ni_w17_q990,R1,252,71,0,0,0.0,,all_gt
ni_w17_q990,R2,193,19,3,0,0.0,0.0,all_gt
ni_w17_q990,R3,160,21,1,0,0.0,0.0,all_gt
ni_w17_q990,R4,136,40,1,0,0.0,0.0,all_gt
ni_w17_q990,R5,173,66,4,1,0.015151515151515152,0.25,all_gt
ni_w17_q990,L1,217,10,6,0,0.0,0.0,assigned_gt
ni_w17_q990,L2,188,6,1,0,0.0,0.0,assigned_gt
ni_w17_q990,L3,95,5,0,0,0.0,,assigned_gt
ni_w17_q990,L4,30,5,0,0,0.0,,assigned_gt
ni_w17_q990,L5,225,18,3,0,0.0,0.0,assigned_gt
ni_w17_q990,R1,190,9,0,0,0.0,,assigned_gt
ni_w17_q990,R2,182,8,3,0,0.0,0.0,assigned_gt
ni_w17_q990,R3,146,7,1,0,0.0,0.0,assigned_gt
ni_w17_q990,R4,118,22,1,0,0.0,0.0,assigned_gt
ni_w17_q990,R5,134,27,4,1,0.037037037037037035,0.25,assigned_gt
ni_w17_q975,L1,252,45,6,0,0.0,0.0,all_gt
ni_w17_q975,L2,207,25,1,0,0.0,0.0,all_gt
ni_w17_q975,L3,110,20,0,0,0.0,,all_gt
ni_w17_q975,L4,38,13,1,0,0.0,0.0,all_gt
ni_w17_q975,L5,279,72,3,0,0.0,0.0,all_gt
ni_w17_q975,R1,252,71,3,0,0.0,0.0,all_gt
ni_w17_q975,R2,193,19,4,0,0.0,0.0,all_gt
ni_w17_q975,R3,160,21,1,0,0.0,0.0,all_gt
ni_w17_q975,R4,136,40,4,0,0.0,0.0,all_gt
ni_w17_q975,R5,173,66,7,1,0.015151515151515152,0.14285714285714285,all_gt
ni_w17_q975,L1,217,10,6,0,0.0,0.0,assigned_gt
ni_w17_q975,L2,188,6,1,0,0.0,0.0,assigned_gt
ni_w17_q975,L3,95,5,0,0,0.0,,assigned_gt
ni_w17_q975,L4,30,5,1,0,0.0,0.0,assigned_gt
ni_w17_q975,L5,225,18,3,0,0.0,0.0,assigned_gt
ni_w17_q975,R1,190,9,3,0,0.0,0.0,assigned_gt
ni_w17_q975,R2,182,8,4,0,0.0,0.0,assigned_gt
ni_w17_q975,R3,146,7,1,0,0.0,0.0,assigned_gt
ni_w17_q975,R4,118,22,4,0,0.0,0.0,assigned_gt
ni_w17_q975,R5,134,27,7,1,0.037037037037037035,0.14285714285714285,assigned_gt
mandatory_missing__ni_k2_r1,L1,252,45,10,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r1,L2,207,25,6,1,0.04,0.16666666666666666,all_gt
mandatory_missing__ni_k2_r1,L3,110,20,2,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r1,L4,38,13,0,0,0.0,,all_gt
mandatory_missing__ni_k2_r1,L5,279,72,7,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r1,R1,252,71,14,1,0.014084507042253521,0.07142857142857142,all_gt
mandatory_missing__ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r1,R3,160,21,10,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r1,R4,136,40,6,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r1,R5,173,66,10,4,0.06060606060606061,0.4,all_gt
mandatory_missing__ni_k2_r1,L1,217,10,10,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r1,L2,188,6,6,1,0.16666666666666666,0.16666666666666666,assigned_gt
mandatory_missing__ni_k2_r1,L3,95,5,2,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r1,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k2_r1,L5,225,18,7,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r1,R1,190,9,14,1,0.1111111111111111,0.07142857142857142,assigned_gt
mandatory_missing__ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r1,R3,146,7,10,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r1,R4,118,22,6,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r1,R5,134,27,10,4,0.14814814814814814,0.4,assigned_gt
mandatory_missing__ni_k2_r2,L1,252,45,14,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r2,L2,207,25,14,1,0.04,0.07142857142857142,all_gt
mandatory_missing__ni_k2_r2,L3,110,20,7,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r2,L4,38,13,1,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r2,L5,279,72,13,2,0.027777777777777776,0.15384615384615385,all_gt
mandatory_missing__ni_k2_r2,R1,252,71,21,3,0.04225352112676056,0.14285714285714285,all_gt
mandatory_missing__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r2,R3,160,21,16,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r2,R4,136,40,13,2,0.05,0.15384615384615385,all_gt
mandatory_missing__ni_k2_r2,R5,173,66,16,6,0.09090909090909091,0.375,all_gt
mandatory_missing__ni_k2_r2,L1,217,10,14,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r2,L2,188,6,14,1,0.16666666666666666,0.07142857142857142,assigned_gt
mandatory_missing__ni_k2_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r2,L5,225,18,13,2,0.1111111111111111,0.15384615384615385,assigned_gt
mandatory_missing__ni_k2_r2,R1,190,9,21,3,0.3333333333333333,0.14285714285714285,assigned_gt
mandatory_missing__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r2,R3,146,7,16,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r2,R4,118,22,13,2,0.09090909090909091,0.15384615384615385,assigned_gt
mandatory_missing__ni_k2_r2,R5,134,27,16,6,0.2222222222222222,0.375,assigned_gt
mandatory_missing__ni_k2_r4,L1,252,45,29,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r4,L2,207,25,28,1,0.04,0.03571428571428571,all_gt
mandatory_missing__ni_k2_r4,L3,110,20,10,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r4,L4,38,13,2,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r4,L5,279,72,25,3,0.041666666666666664,0.12,all_gt
mandatory_missing__ni_k2_r4,R1,252,71,28,3,0.04225352112676056,0.10714285714285714,all_gt
mandatory_missing__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r4,R3,160,21,27,0,0.0,0.0,all_gt
mandatory_missing__ni_k2_r4,R4,136,40,28,6,0.15,0.21428571428571427,all_gt
mandatory_missing__ni_k2_r4,R5,173,66,27,8,0.12121212121212122,0.2962962962962963,all_gt
mandatory_missing__ni_k2_r4,L1,217,10,29,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r4,L2,188,6,28,1,0.16666666666666666,0.03571428571428571,assigned_gt
mandatory_missing__ni_k2_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r4,L4,30,5,2,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r4,L5,225,18,25,3,0.16666666666666666,0.12,assigned_gt
mandatory_missing__ni_k2_r4,R1,190,9,28,3,0.3333333333333333,0.10714285714285714,assigned_gt
mandatory_missing__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r4,R3,146,7,27,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k2_r4,R4,118,22,28,6,0.2727272727272727,0.21428571428571427,assigned_gt
mandatory_missing__ni_k2_r4,R5,134,27,27,8,0.2962962962962963,0.2962962962962963,assigned_gt
mandatory_missing__ni_k3_r1,L1,252,45,1,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r1,L2,207,25,0,0,0.0,,all_gt
mandatory_missing__ni_k3_r1,L3,110,20,0,0,0.0,,all_gt
mandatory_missing__ni_k3_r1,L4,38,13,0,0,0.0,,all_gt
mandatory_missing__ni_k3_r1,L5,279,72,2,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r1,R1,252,71,4,1,0.014084507042253521,0.25,all_gt
mandatory_missing__ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r1,R3,160,21,3,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r1,R4,136,40,3,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r1,R5,173,66,3,1,0.015151515151515152,0.3333333333333333,all_gt
mandatory_missing__ni_k3_r1,L1,217,10,1,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r1,L2,188,6,0,0,0.0,,assigned_gt
mandatory_missing__ni_k3_r1,L3,95,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k3_r1,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k3_r1,L5,225,18,2,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r1,R1,190,9,4,1,0.1111111111111111,0.25,assigned_gt
mandatory_missing__ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r1,R3,146,7,3,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r1,R4,118,22,3,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r1,R5,134,27,3,1,0.037037037037037035,0.3333333333333333,assigned_gt
mandatory_missing__ni_k3_r2,L1,252,45,4,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r2,L2,207,25,1,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r2,L3,110,20,3,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r2,L4,38,13,0,0,0.0,,all_gt
mandatory_missing__ni_k3_r2,L5,279,72,2,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r2,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
mandatory_missing__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r2,R3,160,21,3,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r2,R4,136,40,4,1,0.025,0.25,all_gt
mandatory_missing__ni_k3_r2,R5,173,66,7,2,0.030303030303030304,0.2857142857142857,all_gt
mandatory_missing__ni_k3_r2,L1,217,10,4,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r2,L2,188,6,1,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r2,L3,95,5,3,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r2,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k3_r2,L5,225,18,2,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r2,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
mandatory_missing__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r2,R3,146,7,3,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r2,R4,118,22,4,1,0.045454545454545456,0.25,assigned_gt
mandatory_missing__ni_k3_r2,R5,134,27,7,2,0.07407407407407407,0.2857142857142857,assigned_gt
mandatory_missing__ni_k3_r4,L1,252,45,9,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,L2,207,25,6,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,L3,110,20,3,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,L4,38,13,1,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,L5,279,72,6,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,R1,252,71,8,2,0.028169014084507043,0.25,all_gt
mandatory_missing__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,R3,160,21,6,0,0.0,0.0,all_gt
mandatory_missing__ni_k3_r4,R4,136,40,11,4,0.1,0.36363636363636365,all_gt
mandatory_missing__ni_k3_r4,R5,173,66,10,3,0.045454545454545456,0.3,all_gt
mandatory_missing__ni_k3_r4,L1,217,10,9,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,L2,188,6,6,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,L3,95,5,3,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,L4,30,5,1,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,L5,225,18,6,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,R1,190,9,8,2,0.2222222222222222,0.25,assigned_gt
mandatory_missing__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,R3,146,7,6,0,0.0,0.0,assigned_gt
mandatory_missing__ni_k3_r4,R4,118,22,11,4,0.18181818181818182,0.36363636363636365,assigned_gt
mandatory_missing__ni_k3_r4,R5,134,27,10,3,0.1111111111111111,0.3,assigned_gt
mandatory_missing__ni_k5_r1,L1,252,45,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,L2,207,25,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,L3,110,20,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,L4,38,13,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,L5,279,72,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,R1,252,71,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,R3,160,21,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,R4,136,40,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,R5,173,66,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r1,L1,217,10,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,L2,188,6,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,L3,95,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,L5,225,18,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,R1,190,9,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,R3,146,7,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,R4,118,22,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r1,R5,134,27,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,L1,252,45,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,L2,207,25,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,L3,110,20,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,L4,38,13,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,L5,279,72,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,R1,252,71,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,R3,160,21,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,R4,136,40,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,R5,173,66,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r2,L1,217,10,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,L2,188,6,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,L3,95,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,L5,225,18,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,R1,190,9,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,R3,146,7,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,R4,118,22,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r2,R5,134,27,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,L1,252,45,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,L2,207,25,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,L3,110,20,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,L4,38,13,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,L5,279,72,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,R1,252,71,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,R3,160,21,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,R4,136,40,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,R5,173,66,0,0,0.0,,all_gt
mandatory_missing__ni_k5_r4,L1,217,10,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,L2,188,6,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,L3,95,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,L4,30,5,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,L5,225,18,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,R1,190,9,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,R3,146,7,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,R4,118,22,0,0,0.0,,assigned_gt
mandatory_missing__ni_k5_r4,R5,134,27,0,0,0.0,,assigned_gt
legacy_current_default__ni_k2_r1,L1,252,45,12,1,0.022222222222222223,0.08333333333333333,all_gt
legacy_current_default__ni_k2_r1,L2,207,25,11,1,0.04,0.09090909090909091,all_gt
legacy_current_default__ni_k2_r1,L3,110,20,7,1,0.05,0.14285714285714285,all_gt
legacy_current_default__ni_k2_r1,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r1,L5,279,72,26,5,0.06944444444444445,0.19230769230769232,all_gt
legacy_current_default__ni_k2_r1,R1,252,71,17,2,0.028169014084507043,0.11764705882352941,all_gt
legacy_current_default__ni_k2_r1,R2,193,19,7,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r1,R3,160,21,14,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r1,R4,136,40,18,7,0.175,0.3888888888888889,all_gt
legacy_current_default__ni_k2_r1,R5,173,66,22,12,0.18181818181818182,0.5454545454545454,all_gt
legacy_current_default__ni_k2_r1,L1,217,10,12,1,0.1,0.08333333333333333,assigned_gt
legacy_current_default__ni_k2_r1,L2,188,6,11,1,0.16666666666666666,0.09090909090909091,assigned_gt
legacy_current_default__ni_k2_r1,L3,95,5,7,1,0.2,0.14285714285714285,assigned_gt
legacy_current_default__ni_k2_r1,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r1,L5,225,18,26,5,0.2777777777777778,0.19230769230769232,assigned_gt
legacy_current_default__ni_k2_r1,R1,190,9,17,2,0.2222222222222222,0.11764705882352941,assigned_gt
legacy_current_default__ni_k2_r1,R2,182,8,7,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r1,R3,146,7,14,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r1,R4,118,22,18,7,0.3181818181818182,0.3888888888888889,assigned_gt
legacy_current_default__ni_k2_r1,R5,134,27,22,12,0.4444444444444444,0.5454545454545454,assigned_gt
legacy_current_default__ni_k2_r2,L1,252,45,15,1,0.022222222222222223,0.06666666666666667,all_gt
legacy_current_default__ni_k2_r2,L2,207,25,18,1,0.04,0.05555555555555555,all_gt
legacy_current_default__ni_k2_r2,L3,110,20,9,1,0.05,0.1111111111111111,all_gt
legacy_current_default__ni_k2_r2,L4,38,13,6,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r2,L5,279,72,31,7,0.09722222222222222,0.22580645161290322,all_gt
legacy_current_default__ni_k2_r2,R1,252,71,21,3,0.04225352112676056,0.14285714285714285,all_gt
legacy_current_default__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r2,R3,160,21,20,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r2,R4,136,40,22,7,0.175,0.3181818181818182,all_gt
legacy_current_default__ni_k2_r2,R5,173,66,25,13,0.19696969696969696,0.52,all_gt
legacy_current_default__ni_k2_r2,L1,217,10,15,1,0.1,0.06666666666666667,assigned_gt
legacy_current_default__ni_k2_r2,L2,188,6,18,1,0.16666666666666666,0.05555555555555555,assigned_gt
legacy_current_default__ni_k2_r2,L3,95,5,9,1,0.2,0.1111111111111111,assigned_gt
legacy_current_default__ni_k2_r2,L4,30,5,6,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r2,L5,225,18,31,7,0.3888888888888889,0.22580645161290322,assigned_gt
legacy_current_default__ni_k2_r2,R1,190,9,21,3,0.3333333333333333,0.14285714285714285,assigned_gt
legacy_current_default__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r2,R3,146,7,20,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r2,R4,118,22,22,7,0.3181818181818182,0.3181818181818182,assigned_gt
legacy_current_default__ni_k2_r2,R5,134,27,25,13,0.48148148148148145,0.52,assigned_gt
legacy_current_default__ni_k2_r4,L1,252,45,30,1,0.022222222222222223,0.03333333333333333,all_gt
legacy_current_default__ni_k2_r4,L2,207,25,32,1,0.04,0.03125,all_gt
legacy_current_default__ni_k2_r4,L3,110,20,12,1,0.05,0.08333333333333333,all_gt
legacy_current_default__ni_k2_r4,L4,38,13,7,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r4,L5,279,72,43,8,0.1111111111111111,0.18604651162790697,all_gt
legacy_current_default__ni_k2_r4,R1,252,71,28,3,0.04225352112676056,0.10714285714285714,all_gt
legacy_current_default__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r4,R3,160,21,30,0,0.0,0.0,all_gt
legacy_current_default__ni_k2_r4,R4,136,40,33,9,0.225,0.2727272727272727,all_gt
legacy_current_default__ni_k2_r4,R5,173,66,35,14,0.21212121212121213,0.4,all_gt
legacy_current_default__ni_k2_r4,L1,217,10,30,1,0.1,0.03333333333333333,assigned_gt
legacy_current_default__ni_k2_r4,L2,188,6,32,1,0.16666666666666666,0.03125,assigned_gt
legacy_current_default__ni_k2_r4,L3,95,5,12,1,0.2,0.08333333333333333,assigned_gt
legacy_current_default__ni_k2_r4,L4,30,5,7,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r4,L5,225,18,43,8,0.4444444444444444,0.18604651162790697,assigned_gt
legacy_current_default__ni_k2_r4,R1,190,9,28,3,0.3333333333333333,0.10714285714285714,assigned_gt
legacy_current_default__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r4,R3,146,7,30,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k2_r4,R4,118,22,33,9,0.4090909090909091,0.2727272727272727,assigned_gt
legacy_current_default__ni_k2_r4,R5,134,27,35,14,0.5185185185185185,0.4,assigned_gt
legacy_current_default__ni_k3_r1,L1,252,45,5,1,0.022222222222222223,0.2,all_gt
legacy_current_default__ni_k3_r1,L2,207,25,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r1,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default__ni_k3_r1,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r1,L5,279,72,21,5,0.06944444444444445,0.23809523809523808,all_gt
legacy_current_default__ni_k3_r1,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
legacy_current_default__ni_k3_r1,R2,193,19,2,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r1,R3,160,21,8,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r1,R4,136,40,15,7,0.175,0.4666666666666667,all_gt
legacy_current_default__ni_k3_r1,R5,173,66,17,10,0.15151515151515152,0.5882352941176471,all_gt
legacy_current_default__ni_k3_r1,L1,217,10,5,1,0.1,0.2,assigned_gt
legacy_current_default__ni_k3_r1,L2,188,6,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r1,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default__ni_k3_r1,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r1,L5,225,18,21,5,0.2777777777777778,0.23809523809523808,assigned_gt
legacy_current_default__ni_k3_r1,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
legacy_current_default__ni_k3_r1,R2,182,8,2,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r1,R3,146,7,8,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r1,R4,118,22,15,7,0.3181818181818182,0.4666666666666667,assigned_gt
legacy_current_default__ni_k3_r1,R5,134,27,17,10,0.37037037037037035,0.5882352941176471,assigned_gt
legacy_current_default__ni_k3_r2,L1,252,45,5,1,0.022222222222222223,0.2,all_gt
legacy_current_default__ni_k3_r2,L2,207,25,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r2,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default__ni_k3_r2,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r2,L5,279,72,21,5,0.06944444444444445,0.23809523809523808,all_gt
legacy_current_default__ni_k3_r2,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
legacy_current_default__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r2,R3,160,21,8,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r2,R4,136,40,15,7,0.175,0.4666666666666667,all_gt
legacy_current_default__ni_k3_r2,R5,173,66,17,10,0.15151515151515152,0.5882352941176471,all_gt
legacy_current_default__ni_k3_r2,L1,217,10,5,1,0.1,0.2,assigned_gt
legacy_current_default__ni_k3_r2,L2,188,6,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r2,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default__ni_k3_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r2,L5,225,18,21,5,0.2777777777777778,0.23809523809523808,assigned_gt
legacy_current_default__ni_k3_r2,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
legacy_current_default__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r2,R3,146,7,8,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r2,R4,118,22,15,7,0.3181818181818182,0.4666666666666667,assigned_gt
legacy_current_default__ni_k3_r2,R5,134,27,17,10,0.37037037037037035,0.5882352941176471,assigned_gt
legacy_current_default__ni_k3_r4,L1,252,45,10,1,0.022222222222222223,0.1,all_gt
legacy_current_default__ni_k3_r4,L2,207,25,10,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r4,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default__ni_k3_r4,L4,38,13,6,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r4,L5,279,72,25,5,0.06944444444444445,0.2,all_gt
legacy_current_default__ni_k3_r4,R1,252,71,8,2,0.028169014084507043,0.25,all_gt
legacy_current_default__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r4,R3,160,21,10,0,0.0,0.0,all_gt
legacy_current_default__ni_k3_r4,R4,136,40,18,8,0.2,0.4444444444444444,all_gt
legacy_current_default__ni_k3_r4,R5,173,66,20,11,0.16666666666666666,0.55,all_gt
legacy_current_default__ni_k3_r4,L1,217,10,10,1,0.1,0.1,assigned_gt
legacy_current_default__ni_k3_r4,L2,188,6,10,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r4,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default__ni_k3_r4,L4,30,5,6,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r4,L5,225,18,25,5,0.2777777777777778,0.2,assigned_gt
legacy_current_default__ni_k3_r4,R1,190,9,8,2,0.2222222222222222,0.25,assigned_gt
legacy_current_default__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r4,R3,146,7,10,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k3_r4,R4,118,22,18,8,0.36363636363636365,0.4444444444444444,assigned_gt
legacy_current_default__ni_k3_r4,R5,134,27,20,11,0.4074074074074074,0.55,assigned_gt
legacy_current_default__ni_k5_r1,L1,252,45,5,1,0.022222222222222223,0.2,all_gt
legacy_current_default__ni_k5_r1,L2,207,25,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r1,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default__ni_k5_r1,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r1,L5,279,72,21,5,0.06944444444444445,0.23809523809523808,all_gt
legacy_current_default__ni_k5_r1,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
legacy_current_default__ni_k5_r1,R2,193,19,2,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r1,R3,160,21,8,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r1,R4,136,40,15,7,0.175,0.4666666666666667,all_gt
legacy_current_default__ni_k5_r1,R5,173,66,17,10,0.15151515151515152,0.5882352941176471,all_gt
legacy_current_default__ni_k5_r1,L1,217,10,5,1,0.1,0.2,assigned_gt
legacy_current_default__ni_k5_r1,L2,188,6,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r1,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default__ni_k5_r1,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r1,L5,225,18,21,5,0.2777777777777778,0.23809523809523808,assigned_gt
legacy_current_default__ni_k5_r1,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
legacy_current_default__ni_k5_r1,R2,182,8,2,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r1,R3,146,7,8,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r1,R4,118,22,15,7,0.3181818181818182,0.4666666666666667,assigned_gt
legacy_current_default__ni_k5_r1,R5,134,27,17,10,0.37037037037037035,0.5882352941176471,assigned_gt
legacy_current_default__ni_k5_r2,L1,252,45,5,1,0.022222222222222223,0.2,all_gt
legacy_current_default__ni_k5_r2,L2,207,25,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r2,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default__ni_k5_r2,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r2,L5,279,72,21,5,0.06944444444444445,0.23809523809523808,all_gt
legacy_current_default__ni_k5_r2,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
legacy_current_default__ni_k5_r2,R2,193,19,2,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r2,R3,160,21,8,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r2,R4,136,40,15,7,0.175,0.4666666666666667,all_gt
legacy_current_default__ni_k5_r2,R5,173,66,17,10,0.15151515151515152,0.5882352941176471,all_gt
legacy_current_default__ni_k5_r2,L1,217,10,5,1,0.1,0.2,assigned_gt
legacy_current_default__ni_k5_r2,L2,188,6,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r2,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default__ni_k5_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r2,L5,225,18,21,5,0.2777777777777778,0.23809523809523808,assigned_gt
legacy_current_default__ni_k5_r2,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
legacy_current_default__ni_k5_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r2,R3,146,7,8,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r2,R4,118,22,15,7,0.3181818181818182,0.4666666666666667,assigned_gt
legacy_current_default__ni_k5_r2,R5,134,27,17,10,0.37037037037037035,0.5882352941176471,assigned_gt
legacy_current_default__ni_k5_r4,L1,252,45,5,1,0.022222222222222223,0.2,all_gt
legacy_current_default__ni_k5_r4,L2,207,25,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r4,L3,110,20,5,1,0.05,0.2,all_gt
legacy_current_default__ni_k5_r4,L4,38,13,5,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r4,L5,279,72,21,5,0.06944444444444445,0.23809523809523808,all_gt
legacy_current_default__ni_k5_r4,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
legacy_current_default__ni_k5_r4,R2,193,19,2,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r4,R3,160,21,8,0,0.0,0.0,all_gt
legacy_current_default__ni_k5_r4,R4,136,40,15,7,0.175,0.4666666666666667,all_gt
legacy_current_default__ni_k5_r4,R5,173,66,17,10,0.15151515151515152,0.5882352941176471,all_gt
legacy_current_default__ni_k5_r4,L1,217,10,5,1,0.1,0.2,assigned_gt
legacy_current_default__ni_k5_r4,L2,188,6,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r4,L3,95,5,5,1,0.2,0.2,assigned_gt
legacy_current_default__ni_k5_r4,L4,30,5,5,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r4,L5,225,18,21,5,0.2777777777777778,0.23809523809523808,assigned_gt
legacy_current_default__ni_k5_r4,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
legacy_current_default__ni_k5_r4,R2,182,8,2,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r4,R3,146,7,8,0,0.0,0.0,assigned_gt
legacy_current_default__ni_k5_r4,R4,118,22,15,7,0.3181818181818182,0.4666666666666667,assigned_gt
legacy_current_default__ni_k5_r4,R5,134,27,17,10,0.37037037037037035,0.5882352941176471,assigned_gt
bl_span_practical__ni_k2_r1,L1,252,45,23,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r1,L2,207,25,15,1,0.04,0.06666666666666667,all_gt
bl_span_practical__ni_k2_r1,L3,110,20,8,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r1,L5,279,72,63,6,0.08333333333333333,0.09523809523809523,all_gt
bl_span_practical__ni_k2_r1,R1,252,71,18,1,0.014084507042253521,0.05555555555555555,all_gt
bl_span_practical__ni_k2_r1,R2,193,19,16,1,0.05263157894736842,0.0625,all_gt
bl_span_practical__ni_k2_r1,R3,160,21,13,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r1,R4,136,40,13,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r1,R5,173,66,21,9,0.13636363636363635,0.42857142857142855,all_gt
bl_span_practical__ni_k2_r1,L1,217,10,23,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r1,L2,188,6,15,1,0.16666666666666666,0.06666666666666667,assigned_gt
bl_span_practical__ni_k2_r1,L3,95,5,8,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r1,L5,225,18,63,6,0.3333333333333333,0.09523809523809523,assigned_gt
bl_span_practical__ni_k2_r1,R1,190,9,18,1,0.1111111111111111,0.05555555555555555,assigned_gt
bl_span_practical__ni_k2_r1,R2,182,8,16,1,0.125,0.0625,assigned_gt
bl_span_practical__ni_k2_r1,R3,146,7,13,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r1,R4,118,22,13,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r1,R5,134,27,21,9,0.3333333333333333,0.42857142857142855,assigned_gt
bl_span_practical__ni_k2_r2,L1,252,45,27,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r2,L2,207,25,23,1,0.04,0.043478260869565216,all_gt
bl_span_practical__ni_k2_r2,L3,110,20,13,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r2,L4,38,13,5,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r2,L5,279,72,66,7,0.09722222222222222,0.10606060606060606,all_gt
bl_span_practical__ni_k2_r2,R1,252,71,25,3,0.04225352112676056,0.12,all_gt
bl_span_practical__ni_k2_r2,R2,193,19,22,1,0.05263157894736842,0.045454545454545456,all_gt
bl_span_practical__ni_k2_r2,R3,160,21,18,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r2,R4,136,40,18,2,0.05,0.1111111111111111,all_gt
bl_span_practical__ni_k2_r2,R5,173,66,26,10,0.15151515151515152,0.38461538461538464,all_gt
bl_span_practical__ni_k2_r2,L1,217,10,27,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r2,L2,188,6,23,1,0.16666666666666666,0.043478260869565216,assigned_gt
bl_span_practical__ni_k2_r2,L3,95,5,13,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r2,L5,225,18,66,7,0.3888888888888889,0.10606060606060606,assigned_gt
bl_span_practical__ni_k2_r2,R1,190,9,25,3,0.3333333333333333,0.12,assigned_gt
bl_span_practical__ni_k2_r2,R2,182,8,22,1,0.125,0.045454545454545456,assigned_gt
bl_span_practical__ni_k2_r2,R3,146,7,18,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r2,R4,118,22,18,2,0.09090909090909091,0.1111111111111111,assigned_gt
bl_span_practical__ni_k2_r2,R5,134,27,26,10,0.37037037037037035,0.38461538461538464,assigned_gt
bl_span_practical__ni_k2_r4,L1,252,45,42,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r4,L2,207,25,34,1,0.04,0.029411764705882353,all_gt
bl_span_practical__ni_k2_r4,L3,110,20,15,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r4,L4,38,13,6,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r4,L5,279,72,76,8,0.1111111111111111,0.10526315789473684,all_gt
bl_span_practical__ni_k2_r4,R1,252,71,31,3,0.04225352112676056,0.0967741935483871,all_gt
bl_span_practical__ni_k2_r4,R2,193,19,28,1,0.05263157894736842,0.03571428571428571,all_gt
bl_span_practical__ni_k2_r4,R3,160,21,28,0,0.0,0.0,all_gt
bl_span_practical__ni_k2_r4,R4,136,40,32,6,0.15,0.1875,all_gt
bl_span_practical__ni_k2_r4,R5,173,66,36,11,0.16666666666666666,0.3055555555555556,all_gt
bl_span_practical__ni_k2_r4,L1,217,10,42,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r4,L2,188,6,34,1,0.16666666666666666,0.029411764705882353,assigned_gt
bl_span_practical__ni_k2_r4,L3,95,5,15,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r4,L4,30,5,6,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r4,L5,225,18,76,8,0.4444444444444444,0.10526315789473684,assigned_gt
bl_span_practical__ni_k2_r4,R1,190,9,31,3,0.3333333333333333,0.0967741935483871,assigned_gt
bl_span_practical__ni_k2_r4,R2,182,8,28,1,0.125,0.03571428571428571,assigned_gt
bl_span_practical__ni_k2_r4,R3,146,7,28,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k2_r4,R4,118,22,32,6,0.2727272727272727,0.1875,assigned_gt
bl_span_practical__ni_k2_r4,R5,134,27,36,11,0.4074074074074074,0.3055555555555556,assigned_gt
bl_span_practical__ni_k3_r1,L1,252,45,14,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r1,L2,207,25,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r1,L5,279,72,58,6,0.08333333333333333,0.10344827586206896,all_gt
bl_span_practical__ni_k3_r1,R1,252,71,8,1,0.014084507042253521,0.125,all_gt
bl_span_practical__ni_k3_r1,R2,193,19,12,1,0.05263157894736842,0.08333333333333333,all_gt
bl_span_practical__ni_k3_r1,R3,160,21,7,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r1,R4,136,40,11,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r1,R5,173,66,15,6,0.09090909090909091,0.4,all_gt
bl_span_practical__ni_k3_r1,L1,217,10,14,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r1,L5,225,18,58,6,0.3333333333333333,0.10344827586206896,assigned_gt
bl_span_practical__ni_k3_r1,R1,190,9,8,1,0.1111111111111111,0.125,assigned_gt
bl_span_practical__ni_k3_r1,R2,182,8,12,1,0.125,0.08333333333333333,assigned_gt
bl_span_practical__ni_k3_r1,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r1,R4,118,22,11,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r1,R5,134,27,15,6,0.2222222222222222,0.4,assigned_gt
bl_span_practical__ni_k3_r2,L1,252,45,17,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r2,L2,207,25,11,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r2,L3,110,20,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r2,L5,279,72,58,6,0.08333333333333333,0.10344827586206896,all_gt
bl_span_practical__ni_k3_r2,R1,252,71,11,2,0.028169014084507043,0.18181818181818182,all_gt
bl_span_practical__ni_k3_r2,R2,193,19,13,1,0.05263157894736842,0.07692307692307693,all_gt
bl_span_practical__ni_k3_r2,R3,160,21,7,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r2,R4,136,40,12,1,0.025,0.08333333333333333,all_gt
bl_span_practical__ni_k3_r2,R5,173,66,18,6,0.09090909090909091,0.3333333333333333,all_gt
bl_span_practical__ni_k3_r2,L1,217,10,17,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r2,L2,188,6,11,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r2,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r2,L5,225,18,58,6,0.3333333333333333,0.10344827586206896,assigned_gt
bl_span_practical__ni_k3_r2,R1,190,9,11,2,0.2222222222222222,0.18181818181818182,assigned_gt
bl_span_practical__ni_k3_r2,R2,182,8,13,1,0.125,0.07692307692307693,assigned_gt
bl_span_practical__ni_k3_r2,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r2,R4,118,22,12,1,0.045454545454545456,0.08333333333333333,assigned_gt
bl_span_practical__ni_k3_r2,R5,134,27,18,6,0.2222222222222222,0.3333333333333333,assigned_gt
bl_span_practical__ni_k3_r4,L1,252,45,22,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r4,L2,207,25,15,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r4,L4,38,13,5,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r4,L5,279,72,60,6,0.08333333333333333,0.1,all_gt
bl_span_practical__ni_k3_r4,R1,252,71,12,2,0.028169014084507043,0.16666666666666666,all_gt
bl_span_practical__ni_k3_r4,R2,193,19,14,1,0.05263157894736842,0.07142857142857142,all_gt
bl_span_practical__ni_k3_r4,R3,160,21,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k3_r4,R4,136,40,19,4,0.1,0.21052631578947367,all_gt
bl_span_practical__ni_k3_r4,R5,173,66,21,7,0.10606060606060606,0.3333333333333333,all_gt
bl_span_practical__ni_k3_r4,L1,217,10,22,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r4,L2,188,6,15,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r4,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r4,L5,225,18,60,6,0.3333333333333333,0.1,assigned_gt
bl_span_practical__ni_k3_r4,R1,190,9,12,2,0.2222222222222222,0.16666666666666666,assigned_gt
bl_span_practical__ni_k3_r4,R2,182,8,14,1,0.125,0.07142857142857142,assigned_gt
bl_span_practical__ni_k3_r4,R3,146,7,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k3_r4,R4,118,22,19,4,0.18181818181818182,0.21052631578947367,assigned_gt
bl_span_practical__ni_k3_r4,R5,134,27,21,7,0.25925925925925924,0.3333333333333333,assigned_gt
bl_span_practical__ni_k5_r1,L1,252,45,13,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,L2,207,25,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_span_practical__ni_k5_r1,R1,252,71,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_span_practical__ni_k5_r1,R3,160,21,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,R4,136,40,9,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r1,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_span_practical__ni_k5_r1,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_span_practical__ni_k5_r1,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_span_practical__ni_k5_r1,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r1,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_span_practical__ni_k5_r2,L1,252,45,13,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,L2,207,25,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_span_practical__ni_k5_r2,R1,252,71,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_span_practical__ni_k5_r2,R3,160,21,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,R4,136,40,9,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r2,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_span_practical__ni_k5_r2,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_span_practical__ni_k5_r2,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_span_practical__ni_k5_r2,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r2,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_span_practical__ni_k5_r4,L1,252,45,13,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,L2,207,25,10,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,L3,110,20,7,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,L4,38,13,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_span_practical__ni_k5_r4,R1,252,71,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_span_practical__ni_k5_r4,R3,160,21,4,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,R4,136,40,9,0,0.0,0.0,all_gt
bl_span_practical__ni_k5_r4,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_span_practical__ni_k5_r4,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_span_practical__ni_k5_r4,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_span_practical__ni_k5_r4,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_span_practical__ni_k5_r4,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_span_comfortable__ni_k2_r1,L1,252,45,34,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r1,L2,207,25,26,1,0.04,0.038461538461538464,all_gt
bl_span_comfortable__ni_k2_r1,L3,110,20,21,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r1,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r1,L5,279,72,73,9,0.125,0.1232876712328767,all_gt
bl_span_comfortable__ni_k2_r1,R1,252,71,19,1,0.014084507042253521,0.05263157894736842,all_gt
bl_span_comfortable__ni_k2_r1,R2,193,19,29,1,0.05263157894736842,0.034482758620689655,all_gt
bl_span_comfortable__ni_k2_r1,R3,160,21,25,1,0.047619047619047616,0.04,all_gt
bl_span_comfortable__ni_k2_r1,R4,136,40,25,2,0.05,0.08,all_gt
bl_span_comfortable__ni_k2_r1,R5,173,66,32,11,0.16666666666666666,0.34375,all_gt
bl_span_comfortable__ni_k2_r1,L1,217,10,34,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r1,L2,188,6,26,1,0.16666666666666666,0.038461538461538464,assigned_gt
bl_span_comfortable__ni_k2_r1,L3,95,5,21,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r1,L5,225,18,73,9,0.5,0.1232876712328767,assigned_gt
bl_span_comfortable__ni_k2_r1,R1,190,9,19,1,0.1111111111111111,0.05263157894736842,assigned_gt
bl_span_comfortable__ni_k2_r1,R2,182,8,29,1,0.125,0.034482758620689655,assigned_gt
bl_span_comfortable__ni_k2_r1,R3,146,7,25,1,0.14285714285714285,0.04,assigned_gt
bl_span_comfortable__ni_k2_r1,R4,118,22,25,2,0.09090909090909091,0.08,assigned_gt
bl_span_comfortable__ni_k2_r1,R5,134,27,32,11,0.4074074074074074,0.34375,assigned_gt
bl_span_comfortable__ni_k2_r2,L1,252,45,38,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r2,L2,207,25,32,1,0.04,0.03125,all_gt
bl_span_comfortable__ni_k2_r2,L3,110,20,25,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r2,L4,38,13,9,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r2,L5,279,72,76,10,0.1388888888888889,0.13157894736842105,all_gt
bl_span_comfortable__ni_k2_r2,R1,252,71,26,3,0.04225352112676056,0.11538461538461539,all_gt
bl_span_comfortable__ni_k2_r2,R2,193,19,35,1,0.05263157894736842,0.02857142857142857,all_gt
bl_span_comfortable__ni_k2_r2,R3,160,21,30,1,0.047619047619047616,0.03333333333333333,all_gt
bl_span_comfortable__ni_k2_r2,R4,136,40,30,4,0.1,0.13333333333333333,all_gt
bl_span_comfortable__ni_k2_r2,R5,173,66,37,12,0.18181818181818182,0.32432432432432434,all_gt
bl_span_comfortable__ni_k2_r2,L1,217,10,38,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r2,L2,188,6,32,1,0.16666666666666666,0.03125,assigned_gt
bl_span_comfortable__ni_k2_r2,L3,95,5,25,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r2,L4,30,5,9,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r2,L5,225,18,76,10,0.5555555555555556,0.13157894736842105,assigned_gt
bl_span_comfortable__ni_k2_r2,R1,190,9,26,3,0.3333333333333333,0.11538461538461539,assigned_gt
bl_span_comfortable__ni_k2_r2,R2,182,8,35,1,0.125,0.02857142857142857,assigned_gt
bl_span_comfortable__ni_k2_r2,R3,146,7,30,1,0.14285714285714285,0.03333333333333333,assigned_gt
bl_span_comfortable__ni_k2_r2,R4,118,22,30,4,0.18181818181818182,0.13333333333333333,assigned_gt
bl_span_comfortable__ni_k2_r2,R5,134,27,37,12,0.4444444444444444,0.32432432432432434,assigned_gt
bl_span_comfortable__ni_k2_r4,L1,252,45,52,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r4,L2,207,25,40,1,0.04,0.025,all_gt
bl_span_comfortable__ni_k2_r4,L3,110,20,26,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r4,L4,38,13,10,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k2_r4,L5,279,72,84,11,0.1527777777777778,0.13095238095238096,all_gt
bl_span_comfortable__ni_k2_r4,R1,252,71,32,3,0.04225352112676056,0.09375,all_gt
bl_span_comfortable__ni_k2_r4,R2,193,19,41,1,0.05263157894736842,0.024390243902439025,all_gt
bl_span_comfortable__ni_k2_r4,R3,160,21,40,1,0.047619047619047616,0.025,all_gt
bl_span_comfortable__ni_k2_r4,R4,136,40,43,8,0.2,0.18604651162790697,all_gt
bl_span_comfortable__ni_k2_r4,R5,173,66,47,13,0.19696969696969696,0.2765957446808511,all_gt
bl_span_comfortable__ni_k2_r4,L1,217,10,52,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r4,L2,188,6,40,1,0.16666666666666666,0.025,assigned_gt
bl_span_comfortable__ni_k2_r4,L3,95,5,26,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r4,L4,30,5,10,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k2_r4,L5,225,18,84,11,0.6111111111111112,0.13095238095238096,assigned_gt
bl_span_comfortable__ni_k2_r4,R1,190,9,32,3,0.3333333333333333,0.09375,assigned_gt
bl_span_comfortable__ni_k2_r4,R2,182,8,41,1,0.125,0.024390243902439025,assigned_gt
bl_span_comfortable__ni_k2_r4,R3,146,7,40,1,0.14285714285714285,0.025,assigned_gt
bl_span_comfortable__ni_k2_r4,R4,118,22,43,8,0.36363636363636365,0.18604651162790697,assigned_gt
bl_span_comfortable__ni_k2_r4,R5,134,27,47,13,0.48148148148148145,0.2765957446808511,assigned_gt
bl_span_comfortable__ni_k3_r1,L1,252,45,25,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r1,L2,207,25,22,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r1,L3,110,20,20,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r1,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r1,L5,279,72,68,9,0.125,0.1323529411764706,all_gt
bl_span_comfortable__ni_k3_r1,R1,252,71,9,1,0.014084507042253521,0.1111111111111111,all_gt
bl_span_comfortable__ni_k3_r1,R2,193,19,25,1,0.05263157894736842,0.04,all_gt
bl_span_comfortable__ni_k3_r1,R3,160,21,19,1,0.047619047619047616,0.05263157894736842,all_gt
bl_span_comfortable__ni_k3_r1,R4,136,40,23,2,0.05,0.08695652173913043,all_gt
bl_span_comfortable__ni_k3_r1,R5,173,66,28,9,0.13636363636363635,0.32142857142857145,all_gt
bl_span_comfortable__ni_k3_r1,L1,217,10,25,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r1,L2,188,6,22,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r1,L3,95,5,20,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r1,L5,225,18,68,9,0.5,0.1323529411764706,assigned_gt
bl_span_comfortable__ni_k3_r1,R1,190,9,9,1,0.1111111111111111,0.1111111111111111,assigned_gt
bl_span_comfortable__ni_k3_r1,R2,182,8,25,1,0.125,0.04,assigned_gt
bl_span_comfortable__ni_k3_r1,R3,146,7,19,1,0.14285714285714285,0.05263157894736842,assigned_gt
bl_span_comfortable__ni_k3_r1,R4,118,22,23,2,0.09090909090909091,0.08695652173913043,assigned_gt
bl_span_comfortable__ni_k3_r1,R5,134,27,28,9,0.3333333333333333,0.32142857142857145,assigned_gt
bl_span_comfortable__ni_k3_r2,L1,252,45,28,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r2,L2,207,25,23,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r2,L3,110,20,22,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r2,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r2,L5,279,72,68,9,0.125,0.1323529411764706,all_gt
bl_span_comfortable__ni_k3_r2,R1,252,71,12,2,0.028169014084507043,0.16666666666666666,all_gt
bl_span_comfortable__ni_k3_r2,R2,193,19,26,1,0.05263157894736842,0.038461538461538464,all_gt
bl_span_comfortable__ni_k3_r2,R3,160,21,19,1,0.047619047619047616,0.05263157894736842,all_gt
bl_span_comfortable__ni_k3_r2,R4,136,40,24,3,0.075,0.125,all_gt
bl_span_comfortable__ni_k3_r2,R5,173,66,31,9,0.13636363636363635,0.2903225806451613,all_gt
bl_span_comfortable__ni_k3_r2,L1,217,10,28,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r2,L2,188,6,23,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r2,L3,95,5,22,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r2,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r2,L5,225,18,68,9,0.5,0.1323529411764706,assigned_gt
bl_span_comfortable__ni_k3_r2,R1,190,9,12,2,0.2222222222222222,0.16666666666666666,assigned_gt
bl_span_comfortable__ni_k3_r2,R2,182,8,26,1,0.125,0.038461538461538464,assigned_gt
bl_span_comfortable__ni_k3_r2,R3,146,7,19,1,0.14285714285714285,0.05263157894736842,assigned_gt
bl_span_comfortable__ni_k3_r2,R4,118,22,24,3,0.13636363636363635,0.125,assigned_gt
bl_span_comfortable__ni_k3_r2,R5,134,27,31,9,0.3333333333333333,0.2903225806451613,assigned_gt
bl_span_comfortable__ni_k3_r4,L1,252,45,33,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r4,L2,207,25,25,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r4,L3,110,20,22,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r4,L4,38,13,9,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k3_r4,L5,279,72,70,9,0.125,0.12857142857142856,all_gt
bl_span_comfortable__ni_k3_r4,R1,252,71,13,2,0.028169014084507043,0.15384615384615385,all_gt
bl_span_comfortable__ni_k3_r4,R2,193,19,27,1,0.05263157894736842,0.037037037037037035,all_gt
bl_span_comfortable__ni_k3_r4,R3,160,21,22,1,0.047619047619047616,0.045454545454545456,all_gt
bl_span_comfortable__ni_k3_r4,R4,136,40,31,6,0.15,0.1935483870967742,all_gt
bl_span_comfortable__ni_k3_r4,R5,173,66,34,10,0.15151515151515152,0.29411764705882354,all_gt
bl_span_comfortable__ni_k3_r4,L1,217,10,33,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r4,L2,188,6,25,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r4,L3,95,5,22,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r4,L4,30,5,9,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k3_r4,L5,225,18,70,9,0.5,0.12857142857142856,assigned_gt
bl_span_comfortable__ni_k3_r4,R1,190,9,13,2,0.2222222222222222,0.15384615384615385,assigned_gt
bl_span_comfortable__ni_k3_r4,R2,182,8,27,1,0.125,0.037037037037037035,assigned_gt
bl_span_comfortable__ni_k3_r4,R3,146,7,22,1,0.14285714285714285,0.045454545454545456,assigned_gt
bl_span_comfortable__ni_k3_r4,R4,118,22,31,6,0.2727272727272727,0.1935483870967742,assigned_gt
bl_span_comfortable__ni_k3_r4,R5,134,27,34,10,0.37037037037037035,0.29411764705882354,assigned_gt
bl_span_comfortable__ni_k5_r1,L1,252,45,24,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r1,L2,207,25,22,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r1,L3,110,20,20,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r1,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r1,L5,279,72,66,9,0.125,0.13636363636363635,all_gt
bl_span_comfortable__ni_k5_r1,R1,252,71,5,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r1,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
bl_span_comfortable__ni_k5_r1,R3,160,21,17,1,0.047619047619047616,0.058823529411764705,all_gt
bl_span_comfortable__ni_k5_r1,R4,136,40,21,2,0.05,0.09523809523809523,all_gt
bl_span_comfortable__ni_k5_r1,R5,173,66,26,9,0.13636363636363635,0.34615384615384615,all_gt
bl_span_comfortable__ni_k5_r1,L1,217,10,24,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r1,L2,188,6,22,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r1,L3,95,5,20,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r1,L5,225,18,66,9,0.5,0.13636363636363635,assigned_gt
bl_span_comfortable__ni_k5_r1,R1,190,9,5,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r1,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
bl_span_comfortable__ni_k5_r1,R3,146,7,17,1,0.14285714285714285,0.058823529411764705,assigned_gt
bl_span_comfortable__ni_k5_r1,R4,118,22,21,2,0.09090909090909091,0.09523809523809523,assigned_gt
bl_span_comfortable__ni_k5_r1,R5,134,27,26,9,0.3333333333333333,0.34615384615384615,assigned_gt
bl_span_comfortable__ni_k5_r2,L1,252,45,24,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r2,L2,207,25,22,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r2,L3,110,20,20,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r2,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r2,L5,279,72,66,9,0.125,0.13636363636363635,all_gt
bl_span_comfortable__ni_k5_r2,R1,252,71,5,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r2,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
bl_span_comfortable__ni_k5_r2,R3,160,21,17,1,0.047619047619047616,0.058823529411764705,all_gt
bl_span_comfortable__ni_k5_r2,R4,136,40,21,2,0.05,0.09523809523809523,all_gt
bl_span_comfortable__ni_k5_r2,R5,173,66,26,9,0.13636363636363635,0.34615384615384615,all_gt
bl_span_comfortable__ni_k5_r2,L1,217,10,24,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r2,L2,188,6,22,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r2,L3,95,5,20,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r2,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r2,L5,225,18,66,9,0.5,0.13636363636363635,assigned_gt
bl_span_comfortable__ni_k5_r2,R1,190,9,5,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r2,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
bl_span_comfortable__ni_k5_r2,R3,146,7,17,1,0.14285714285714285,0.058823529411764705,assigned_gt
bl_span_comfortable__ni_k5_r2,R4,118,22,21,2,0.09090909090909091,0.09523809523809523,assigned_gt
bl_span_comfortable__ni_k5_r2,R5,134,27,26,9,0.3333333333333333,0.34615384615384615,assigned_gt
bl_span_comfortable__ni_k5_r4,L1,252,45,24,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r4,L2,207,25,22,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r4,L3,110,20,20,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r4,L4,38,13,8,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r4,L5,279,72,66,9,0.125,0.13636363636363635,all_gt
bl_span_comfortable__ni_k5_r4,R1,252,71,5,0,0.0,0.0,all_gt
bl_span_comfortable__ni_k5_r4,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
bl_span_comfortable__ni_k5_r4,R3,160,21,17,1,0.047619047619047616,0.058823529411764705,all_gt
bl_span_comfortable__ni_k5_r4,R4,136,40,21,2,0.05,0.09523809523809523,all_gt
bl_span_comfortable__ni_k5_r4,R5,173,66,26,9,0.13636363636363635,0.34615384615384615,all_gt
bl_span_comfortable__ni_k5_r4,L1,217,10,24,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r4,L2,188,6,22,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r4,L3,95,5,20,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r4,L4,30,5,8,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r4,L5,225,18,66,9,0.5,0.13636363636363635,assigned_gt
bl_span_comfortable__ni_k5_r4,R1,190,9,5,0,0.0,0.0,assigned_gt
bl_span_comfortable__ni_k5_r4,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
bl_span_comfortable__ni_k5_r4,R3,146,7,17,1,0.14285714285714285,0.058823529411764705,assigned_gt
bl_span_comfortable__ni_k5_r4,R4,118,22,21,2,0.09090909090909091,0.09523809523809523,assigned_gt
bl_span_comfortable__ni_k5_r4,R5,134,27,26,9,0.3333333333333333,0.34615384615384615,assigned_gt
bl_span_relative__ni_k2_r1,L1,252,45,99,2,0.044444444444444446,0.020202020202020204,all_gt
bl_span_relative__ni_k2_r1,L2,207,25,83,3,0.12,0.03614457831325301,all_gt
bl_span_relative__ni_k2_r1,L3,110,20,34,3,0.15,0.08823529411764706,all_gt
bl_span_relative__ni_k2_r1,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k2_r1,L5,279,72,127,12,0.16666666666666666,0.09448818897637795,all_gt
bl_span_relative__ni_k2_r1,R1,252,71,55,7,0.09859154929577464,0.12727272727272726,all_gt
bl_span_relative__ni_k2_r1,R2,193,19,62,2,0.10526315789473684,0.03225806451612903,all_gt
bl_span_relative__ni_k2_r1,R3,160,21,61,5,0.23809523809523808,0.08196721311475409,all_gt
bl_span_relative__ni_k2_r1,R4,136,40,48,12,0.3,0.25,all_gt
bl_span_relative__ni_k2_r1,R5,173,66,78,14,0.21212121212121213,0.1794871794871795,all_gt
bl_span_relative__ni_k2_r1,L1,217,10,99,2,0.2,0.020202020202020204,assigned_gt
bl_span_relative__ni_k2_r1,L2,188,6,83,3,0.5,0.03614457831325301,assigned_gt
bl_span_relative__ni_k2_r1,L3,95,5,34,3,0.6,0.08823529411764706,assigned_gt
bl_span_relative__ni_k2_r1,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k2_r1,L5,225,18,127,12,0.6666666666666666,0.09448818897637795,assigned_gt
bl_span_relative__ni_k2_r1,R1,190,9,55,7,0.7777777777777778,0.12727272727272726,assigned_gt
bl_span_relative__ni_k2_r1,R2,182,8,62,2,0.25,0.03225806451612903,assigned_gt
bl_span_relative__ni_k2_r1,R3,146,7,61,5,0.7142857142857143,0.08196721311475409,assigned_gt
bl_span_relative__ni_k2_r1,R4,118,22,48,12,0.5454545454545454,0.25,assigned_gt
bl_span_relative__ni_k2_r1,R5,134,27,78,14,0.5185185185185185,0.1794871794871795,assigned_gt
bl_span_relative__ni_k2_r2,L1,252,45,101,2,0.044444444444444446,0.019801980198019802,all_gt
bl_span_relative__ni_k2_r2,L2,207,25,86,3,0.12,0.03488372093023256,all_gt
bl_span_relative__ni_k2_r2,L3,110,20,35,3,0.15,0.08571428571428572,all_gt
bl_span_relative__ni_k2_r2,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k2_r2,L5,279,72,129,13,0.18055555555555555,0.10077519379844961,all_gt
bl_span_relative__ni_k2_r2,R1,252,71,60,7,0.09859154929577464,0.11666666666666667,all_gt
bl_span_relative__ni_k2_r2,R2,193,19,65,2,0.10526315789473684,0.03076923076923077,all_gt
bl_span_relative__ni_k2_r2,R3,160,21,65,5,0.23809523809523808,0.07692307692307693,all_gt
bl_span_relative__ni_k2_r2,R4,136,40,52,14,0.35,0.2692307692307692,all_gt
bl_span_relative__ni_k2_r2,R5,173,66,82,15,0.22727272727272727,0.18292682926829268,all_gt
bl_span_relative__ni_k2_r2,L1,217,10,101,2,0.2,0.019801980198019802,assigned_gt
bl_span_relative__ni_k2_r2,L2,188,6,86,3,0.5,0.03488372093023256,assigned_gt
bl_span_relative__ni_k2_r2,L3,95,5,35,3,0.6,0.08571428571428572,assigned_gt
bl_span_relative__ni_k2_r2,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k2_r2,L5,225,18,129,13,0.7222222222222222,0.10077519379844961,assigned_gt
bl_span_relative__ni_k2_r2,R1,190,9,60,7,0.7777777777777778,0.11666666666666667,assigned_gt
bl_span_relative__ni_k2_r2,R2,182,8,65,2,0.25,0.03076923076923077,assigned_gt
bl_span_relative__ni_k2_r2,R3,146,7,65,5,0.7142857142857143,0.07692307692307693,assigned_gt
bl_span_relative__ni_k2_r2,R4,118,22,52,14,0.6363636363636364,0.2692307692307692,assigned_gt
bl_span_relative__ni_k2_r2,R5,134,27,82,15,0.5555555555555556,0.18292682926829268,assigned_gt
bl_span_relative__ni_k2_r4,L1,252,45,111,2,0.044444444444444446,0.018018018018018018,all_gt
bl_span_relative__ni_k2_r4,L2,207,25,92,3,0.12,0.03260869565217391,all_gt
bl_span_relative__ni_k2_r4,L3,110,20,36,3,0.15,0.08333333333333333,all_gt
bl_span_relative__ni_k2_r4,L4,38,13,14,3,0.23076923076923078,0.21428571428571427,all_gt
bl_span_relative__ni_k2_r4,L5,279,72,133,14,0.19444444444444445,0.10526315789473684,all_gt
bl_span_relative__ni_k2_r4,R1,252,71,66,7,0.09859154929577464,0.10606060606060606,all_gt
bl_span_relative__ni_k2_r4,R2,193,19,70,2,0.10526315789473684,0.02857142857142857,all_gt
bl_span_relative__ni_k2_r4,R3,160,21,73,5,0.23809523809523808,0.0684931506849315,all_gt
bl_span_relative__ni_k2_r4,R4,136,40,62,16,0.4,0.25806451612903225,all_gt
bl_span_relative__ni_k2_r4,R5,173,66,88,16,0.24242424242424243,0.18181818181818182,all_gt
bl_span_relative__ni_k2_r4,L1,217,10,111,2,0.2,0.018018018018018018,assigned_gt
bl_span_relative__ni_k2_r4,L2,188,6,92,3,0.5,0.03260869565217391,assigned_gt
bl_span_relative__ni_k2_r4,L3,95,5,36,3,0.6,0.08333333333333333,assigned_gt
bl_span_relative__ni_k2_r4,L4,30,5,14,3,0.6,0.21428571428571427,assigned_gt
bl_span_relative__ni_k2_r4,L5,225,18,133,14,0.7777777777777778,0.10526315789473684,assigned_gt
bl_span_relative__ni_k2_r4,R1,190,9,66,7,0.7777777777777778,0.10606060606060606,assigned_gt
bl_span_relative__ni_k2_r4,R2,182,8,70,2,0.25,0.02857142857142857,assigned_gt
bl_span_relative__ni_k2_r4,R3,146,7,73,5,0.7142857142857143,0.0684931506849315,assigned_gt
bl_span_relative__ni_k2_r4,R4,118,22,62,16,0.7272727272727273,0.25806451612903225,assigned_gt
bl_span_relative__ni_k2_r4,R5,134,27,88,16,0.5925925925925926,0.18181818181818182,assigned_gt
bl_span_relative__ni_k3_r1,L1,252,45,93,2,0.044444444444444446,0.021505376344086023,all_gt
bl_span_relative__ni_k3_r1,L2,207,25,82,3,0.12,0.036585365853658534,all_gt
bl_span_relative__ni_k3_r1,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative__ni_k3_r1,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k3_r1,L5,279,72,125,12,0.16666666666666666,0.096,all_gt
bl_span_relative__ni_k3_r1,R1,252,71,48,7,0.09859154929577464,0.14583333333333334,all_gt
bl_span_relative__ni_k3_r1,R2,193,19,60,2,0.10526315789473684,0.03333333333333333,all_gt
bl_span_relative__ni_k3_r1,R3,160,21,56,5,0.23809523809523808,0.08928571428571429,all_gt
bl_span_relative__ni_k3_r1,R4,136,40,46,12,0.3,0.2608695652173913,all_gt
bl_span_relative__ni_k3_r1,R5,173,66,75,12,0.18181818181818182,0.16,all_gt
bl_span_relative__ni_k3_r1,L1,217,10,93,2,0.2,0.021505376344086023,assigned_gt
bl_span_relative__ni_k3_r1,L2,188,6,82,3,0.5,0.036585365853658534,assigned_gt
bl_span_relative__ni_k3_r1,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative__ni_k3_r1,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k3_r1,L5,225,18,125,12,0.6666666666666666,0.096,assigned_gt
bl_span_relative__ni_k3_r1,R1,190,9,48,7,0.7777777777777778,0.14583333333333334,assigned_gt
bl_span_relative__ni_k3_r1,R2,182,8,60,2,0.25,0.03333333333333333,assigned_gt
bl_span_relative__ni_k3_r1,R3,146,7,56,5,0.7142857142857143,0.08928571428571429,assigned_gt
bl_span_relative__ni_k3_r1,R4,118,22,46,12,0.5454545454545454,0.2608695652173913,assigned_gt
bl_span_relative__ni_k3_r1,R5,134,27,75,12,0.4444444444444444,0.16,assigned_gt
bl_span_relative__ni_k3_r2,L1,252,45,94,2,0.044444444444444446,0.02127659574468085,all_gt
bl_span_relative__ni_k3_r2,L2,207,25,82,3,0.12,0.036585365853658534,all_gt
bl_span_relative__ni_k3_r2,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative__ni_k3_r2,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k3_r2,L5,279,72,125,12,0.16666666666666666,0.096,all_gt
bl_span_relative__ni_k3_r2,R1,252,71,50,7,0.09859154929577464,0.14,all_gt
bl_span_relative__ni_k3_r2,R2,193,19,60,2,0.10526315789473684,0.03333333333333333,all_gt
bl_span_relative__ni_k3_r2,R3,160,21,56,5,0.23809523809523808,0.08928571428571429,all_gt
bl_span_relative__ni_k3_r2,R4,136,40,47,13,0.325,0.2765957446808511,all_gt
bl_span_relative__ni_k3_r2,R5,173,66,77,12,0.18181818181818182,0.15584415584415584,all_gt
bl_span_relative__ni_k3_r2,L1,217,10,94,2,0.2,0.02127659574468085,assigned_gt
bl_span_relative__ni_k3_r2,L2,188,6,82,3,0.5,0.036585365853658534,assigned_gt
bl_span_relative__ni_k3_r2,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative__ni_k3_r2,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k3_r2,L5,225,18,125,12,0.6666666666666666,0.096,assigned_gt
bl_span_relative__ni_k3_r2,R1,190,9,50,7,0.7777777777777778,0.14,assigned_gt
bl_span_relative__ni_k3_r2,R2,182,8,60,2,0.25,0.03333333333333333,assigned_gt
bl_span_relative__ni_k3_r2,R3,146,7,56,5,0.7142857142857143,0.08928571428571429,assigned_gt
bl_span_relative__ni_k3_r2,R4,118,22,47,13,0.5909090909090909,0.2765957446808511,assigned_gt
bl_span_relative__ni_k3_r2,R5,134,27,77,12,0.4444444444444444,0.15584415584415584,assigned_gt
bl_span_relative__ni_k3_r4,L1,252,45,97,2,0.044444444444444446,0.020618556701030927,all_gt
bl_span_relative__ni_k3_r4,L2,207,25,83,3,0.12,0.03614457831325301,all_gt
bl_span_relative__ni_k3_r4,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative__ni_k3_r4,L4,38,13,14,3,0.23076923076923078,0.21428571428571427,all_gt
bl_span_relative__ni_k3_r4,L5,279,72,125,12,0.16666666666666666,0.096,all_gt
bl_span_relative__ni_k3_r4,R1,252,71,51,7,0.09859154929577464,0.13725490196078433,all_gt
bl_span_relative__ni_k3_r4,R2,193,19,61,2,0.10526315789473684,0.03278688524590164,all_gt
bl_span_relative__ni_k3_r4,R3,160,21,59,5,0.23809523809523808,0.0847457627118644,all_gt
bl_span_relative__ni_k3_r4,R4,136,40,52,15,0.375,0.28846153846153844,all_gt
bl_span_relative__ni_k3_r4,R5,173,66,78,13,0.19696969696969696,0.16666666666666666,all_gt
bl_span_relative__ni_k3_r4,L1,217,10,97,2,0.2,0.020618556701030927,assigned_gt
bl_span_relative__ni_k3_r4,L2,188,6,83,3,0.5,0.03614457831325301,assigned_gt
bl_span_relative__ni_k3_r4,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative__ni_k3_r4,L4,30,5,14,3,0.6,0.21428571428571427,assigned_gt
bl_span_relative__ni_k3_r4,L5,225,18,125,12,0.6666666666666666,0.096,assigned_gt
bl_span_relative__ni_k3_r4,R1,190,9,51,7,0.7777777777777778,0.13725490196078433,assigned_gt
bl_span_relative__ni_k3_r4,R2,182,8,61,2,0.25,0.03278688524590164,assigned_gt
bl_span_relative__ni_k3_r4,R3,146,7,59,5,0.7142857142857143,0.0847457627118644,assigned_gt
bl_span_relative__ni_k3_r4,R4,118,22,52,15,0.6818181818181818,0.28846153846153844,assigned_gt
bl_span_relative__ni_k3_r4,R5,134,27,78,13,0.48148148148148145,0.16666666666666666,assigned_gt
bl_span_relative__ni_k5_r1,L1,252,45,92,2,0.044444444444444446,0.021739130434782608,all_gt
bl_span_relative__ni_k5_r1,L2,207,25,82,3,0.12,0.036585365853658534,all_gt
bl_span_relative__ni_k5_r1,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative__ni_k5_r1,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k5_r1,L5,279,72,124,12,0.16666666666666666,0.0967741935483871,all_gt
bl_span_relative__ni_k5_r1,R1,252,71,46,6,0.08450704225352113,0.13043478260869565,all_gt
bl_span_relative__ni_k5_r1,R2,193,19,59,2,0.10526315789473684,0.03389830508474576,all_gt
bl_span_relative__ni_k5_r1,R3,160,21,55,5,0.23809523809523808,0.09090909090909091,all_gt
bl_span_relative__ni_k5_r1,R4,136,40,45,12,0.3,0.26666666666666666,all_gt
bl_span_relative__ni_k5_r1,R5,173,66,75,12,0.18181818181818182,0.16,all_gt
bl_span_relative__ni_k5_r1,L1,217,10,92,2,0.2,0.021739130434782608,assigned_gt
bl_span_relative__ni_k5_r1,L2,188,6,82,3,0.5,0.036585365853658534,assigned_gt
bl_span_relative__ni_k5_r1,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative__ni_k5_r1,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k5_r1,L5,225,18,124,12,0.6666666666666666,0.0967741935483871,assigned_gt
bl_span_relative__ni_k5_r1,R1,190,9,46,6,0.6666666666666666,0.13043478260869565,assigned_gt
bl_span_relative__ni_k5_r1,R2,182,8,59,2,0.25,0.03389830508474576,assigned_gt
bl_span_relative__ni_k5_r1,R3,146,7,55,5,0.7142857142857143,0.09090909090909091,assigned_gt
bl_span_relative__ni_k5_r1,R4,118,22,45,12,0.5454545454545454,0.26666666666666666,assigned_gt
bl_span_relative__ni_k5_r1,R5,134,27,75,12,0.4444444444444444,0.16,assigned_gt
bl_span_relative__ni_k5_r2,L1,252,45,92,2,0.044444444444444446,0.021739130434782608,all_gt
bl_span_relative__ni_k5_r2,L2,207,25,82,3,0.12,0.036585365853658534,all_gt
bl_span_relative__ni_k5_r2,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative__ni_k5_r2,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k5_r2,L5,279,72,124,12,0.16666666666666666,0.0967741935483871,all_gt
bl_span_relative__ni_k5_r2,R1,252,71,46,6,0.08450704225352113,0.13043478260869565,all_gt
bl_span_relative__ni_k5_r2,R2,193,19,59,2,0.10526315789473684,0.03389830508474576,all_gt
bl_span_relative__ni_k5_r2,R3,160,21,55,5,0.23809523809523808,0.09090909090909091,all_gt
bl_span_relative__ni_k5_r2,R4,136,40,45,12,0.3,0.26666666666666666,all_gt
bl_span_relative__ni_k5_r2,R5,173,66,75,12,0.18181818181818182,0.16,all_gt
bl_span_relative__ni_k5_r2,L1,217,10,92,2,0.2,0.021739130434782608,assigned_gt
bl_span_relative__ni_k5_r2,L2,188,6,82,3,0.5,0.036585365853658534,assigned_gt
bl_span_relative__ni_k5_r2,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative__ni_k5_r2,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k5_r2,L5,225,18,124,12,0.6666666666666666,0.0967741935483871,assigned_gt
bl_span_relative__ni_k5_r2,R1,190,9,46,6,0.6666666666666666,0.13043478260869565,assigned_gt
bl_span_relative__ni_k5_r2,R2,182,8,59,2,0.25,0.03389830508474576,assigned_gt
bl_span_relative__ni_k5_r2,R3,146,7,55,5,0.7142857142857143,0.09090909090909091,assigned_gt
bl_span_relative__ni_k5_r2,R4,118,22,45,12,0.5454545454545454,0.26666666666666666,assigned_gt
bl_span_relative__ni_k5_r2,R5,134,27,75,12,0.4444444444444444,0.16,assigned_gt
bl_span_relative__ni_k5_r4,L1,252,45,92,2,0.044444444444444446,0.021739130434782608,all_gt
bl_span_relative__ni_k5_r4,L2,207,25,82,3,0.12,0.036585365853658534,all_gt
bl_span_relative__ni_k5_r4,L3,110,20,33,3,0.15,0.09090909090909091,all_gt
bl_span_relative__ni_k5_r4,L4,38,13,13,3,0.23076923076923078,0.23076923076923078,all_gt
bl_span_relative__ni_k5_r4,L5,279,72,124,12,0.16666666666666666,0.0967741935483871,all_gt
bl_span_relative__ni_k5_r4,R1,252,71,46,6,0.08450704225352113,0.13043478260869565,all_gt
bl_span_relative__ni_k5_r4,R2,193,19,59,2,0.10526315789473684,0.03389830508474576,all_gt
bl_span_relative__ni_k5_r4,R3,160,21,55,5,0.23809523809523808,0.09090909090909091,all_gt
bl_span_relative__ni_k5_r4,R4,136,40,45,12,0.3,0.26666666666666666,all_gt
bl_span_relative__ni_k5_r4,R5,173,66,75,12,0.18181818181818182,0.16,all_gt
bl_span_relative__ni_k5_r4,L1,217,10,92,2,0.2,0.021739130434782608,assigned_gt
bl_span_relative__ni_k5_r4,L2,188,6,82,3,0.5,0.036585365853658534,assigned_gt
bl_span_relative__ni_k5_r4,L3,95,5,33,3,0.6,0.09090909090909091,assigned_gt
bl_span_relative__ni_k5_r4,L4,30,5,13,3,0.6,0.23076923076923078,assigned_gt
bl_span_relative__ni_k5_r4,L5,225,18,124,12,0.6666666666666666,0.0967741935483871,assigned_gt
bl_span_relative__ni_k5_r4,R1,190,9,46,6,0.6666666666666666,0.13043478260869565,assigned_gt
bl_span_relative__ni_k5_r4,R2,182,8,59,2,0.25,0.03389830508474576,assigned_gt
bl_span_relative__ni_k5_r4,R3,146,7,55,5,0.7142857142857143,0.09090909090909091,assigned_gt
bl_span_relative__ni_k5_r4,R4,118,22,45,12,0.5454545454545454,0.26666666666666666,assigned_gt
bl_span_relative__ni_k5_r4,R5,134,27,75,12,0.4444444444444444,0.16,assigned_gt
bl_crossing__ni_k2_r1,L1,252,45,10,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r1,L2,207,25,7,1,0.04,0.14285714285714285,all_gt
bl_crossing__ni_k2_r1,L3,110,20,3,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r1,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r1,L5,279,72,13,3,0.041666666666666664,0.23076923076923078,all_gt
bl_crossing__ni_k2_r1,R1,252,71,14,1,0.014084507042253521,0.07142857142857142,all_gt
bl_crossing__ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r1,R3,160,21,11,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r1,R4,136,40,8,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r1,R5,173,66,15,7,0.10606060606060606,0.4666666666666667,all_gt
bl_crossing__ni_k2_r1,L1,217,10,10,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r1,L2,188,6,7,1,0.16666666666666666,0.14285714285714285,assigned_gt
bl_crossing__ni_k2_r1,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r1,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r1,L5,225,18,13,3,0.16666666666666666,0.23076923076923078,assigned_gt
bl_crossing__ni_k2_r1,R1,190,9,14,1,0.1111111111111111,0.07142857142857142,assigned_gt
bl_crossing__ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r1,R3,146,7,11,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r1,R4,118,22,8,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r1,R5,134,27,15,7,0.25925925925925924,0.4666666666666667,assigned_gt
bl_crossing__ni_k2_r2,L1,252,45,14,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r2,L2,207,25,15,1,0.04,0.06666666666666667,all_gt
bl_crossing__ni_k2_r2,L3,110,20,8,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r2,L4,38,13,2,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r2,L5,279,72,19,5,0.06944444444444445,0.2631578947368421,all_gt
bl_crossing__ni_k2_r2,R1,252,71,21,3,0.04225352112676056,0.14285714285714285,all_gt
bl_crossing__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r2,R3,160,21,17,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r2,R4,136,40,15,2,0.05,0.13333333333333333,all_gt
bl_crossing__ni_k2_r2,R5,173,66,20,8,0.12121212121212122,0.4,all_gt
bl_crossing__ni_k2_r2,L1,217,10,14,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r2,L2,188,6,15,1,0.16666666666666666,0.06666666666666667,assigned_gt
bl_crossing__ni_k2_r2,L3,95,5,8,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r2,L4,30,5,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r2,L5,225,18,19,5,0.2777777777777778,0.2631578947368421,assigned_gt
bl_crossing__ni_k2_r2,R1,190,9,21,3,0.3333333333333333,0.14285714285714285,assigned_gt
bl_crossing__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r2,R3,146,7,17,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r2,R4,118,22,15,2,0.09090909090909091,0.13333333333333333,assigned_gt
bl_crossing__ni_k2_r2,R5,134,27,20,8,0.2962962962962963,0.4,assigned_gt
bl_crossing__ni_k2_r4,L1,252,45,29,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r4,L2,207,25,29,1,0.04,0.034482758620689655,all_gt
bl_crossing__ni_k2_r4,L3,110,20,11,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r4,L4,38,13,3,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r4,L5,279,72,31,6,0.08333333333333333,0.1935483870967742,all_gt
bl_crossing__ni_k2_r4,R1,252,71,28,3,0.04225352112676056,0.10714285714285714,all_gt
bl_crossing__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r4,R3,160,21,28,0,0.0,0.0,all_gt
bl_crossing__ni_k2_r4,R4,136,40,30,6,0.15,0.2,all_gt
bl_crossing__ni_k2_r4,R5,173,66,31,10,0.15151515151515152,0.3225806451612903,all_gt
bl_crossing__ni_k2_r4,L1,217,10,29,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r4,L2,188,6,29,1,0.16666666666666666,0.034482758620689655,assigned_gt
bl_crossing__ni_k2_r4,L3,95,5,11,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r4,L4,30,5,3,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r4,L5,225,18,31,6,0.3333333333333333,0.1935483870967742,assigned_gt
bl_crossing__ni_k2_r4,R1,190,9,28,3,0.3333333333333333,0.10714285714285714,assigned_gt
bl_crossing__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r4,R3,146,7,28,0,0.0,0.0,assigned_gt
bl_crossing__ni_k2_r4,R4,118,22,30,6,0.2727272727272727,0.2,assigned_gt
bl_crossing__ni_k2_r4,R5,134,27,31,10,0.37037037037037035,0.3225806451612903,assigned_gt
bl_crossing__ni_k3_r1,L1,252,45,1,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,L2,207,25,1,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,L3,110,20,1,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,L5,279,72,8,3,0.041666666666666664,0.375,all_gt
bl_crossing__ni_k3_r1,R1,252,71,4,1,0.014084507042253521,0.25,all_gt
bl_crossing__ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,R3,160,21,5,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,R4,136,40,5,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r1,R5,173,66,8,4,0.06060606060606061,0.5,all_gt
bl_crossing__ni_k3_r1,L1,217,10,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,L5,225,18,8,3,0.16666666666666666,0.375,assigned_gt
bl_crossing__ni_k3_r1,R1,190,9,4,1,0.1111111111111111,0.25,assigned_gt
bl_crossing__ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,R3,146,7,5,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,R4,118,22,5,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r1,R5,134,27,8,4,0.14814814814814814,0.5,assigned_gt
bl_crossing__ni_k3_r2,L1,252,45,4,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r2,L2,207,25,2,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r2,L3,110,20,4,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r2,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r2,L5,279,72,8,3,0.041666666666666664,0.375,all_gt
bl_crossing__ni_k3_r2,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
bl_crossing__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r2,R3,160,21,5,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r2,R4,136,40,6,1,0.025,0.16666666666666666,all_gt
bl_crossing__ni_k3_r2,R5,173,66,11,4,0.06060606060606061,0.36363636363636365,all_gt
bl_crossing__ni_k3_r2,L1,217,10,4,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r2,L2,188,6,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r2,L3,95,5,4,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r2,L5,225,18,8,3,0.16666666666666666,0.375,assigned_gt
bl_crossing__ni_k3_r2,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
bl_crossing__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r2,R3,146,7,5,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r2,R4,118,22,6,1,0.045454545454545456,0.16666666666666666,assigned_gt
bl_crossing__ni_k3_r2,R5,134,27,11,4,0.14814814814814814,0.36363636363636365,assigned_gt
bl_crossing__ni_k3_r4,L1,252,45,9,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r4,L2,207,25,7,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r4,L3,110,20,4,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r4,L4,38,13,2,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r4,L5,279,72,12,3,0.041666666666666664,0.25,all_gt
bl_crossing__ni_k3_r4,R1,252,71,8,2,0.028169014084507043,0.25,all_gt
bl_crossing__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r4,R3,160,21,8,0,0.0,0.0,all_gt
bl_crossing__ni_k3_r4,R4,136,40,13,4,0.1,0.3076923076923077,all_gt
bl_crossing__ni_k3_r4,R5,173,66,14,5,0.07575757575757576,0.35714285714285715,all_gt
bl_crossing__ni_k3_r4,L1,217,10,9,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r4,L2,188,6,7,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r4,L3,95,5,4,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r4,L4,30,5,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r4,L5,225,18,12,3,0.16666666666666666,0.25,assigned_gt
bl_crossing__ni_k3_r4,R1,190,9,8,2,0.2222222222222222,0.25,assigned_gt
bl_crossing__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r4,R3,146,7,8,0,0.0,0.0,assigned_gt
bl_crossing__ni_k3_r4,R4,118,22,13,4,0.18181818181818182,0.3076923076923077,assigned_gt
bl_crossing__ni_k3_r4,R5,134,27,14,5,0.18518518518518517,0.35714285714285715,assigned_gt
bl_crossing__ni_k5_r1,L1,252,45,0,0,0.0,,all_gt
bl_crossing__ni_k5_r1,L2,207,25,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r1,L3,110,20,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r1,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r1,L5,279,72,6,3,0.041666666666666664,0.5,all_gt
bl_crossing__ni_k5_r1,R1,252,71,0,0,0.0,,all_gt
bl_crossing__ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
bl_crossing__ni_k5_r1,R3,160,21,2,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r1,R4,136,40,3,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r1,R5,173,66,6,4,0.06060606060606061,0.6666666666666666,all_gt
bl_crossing__ni_k5_r1,L1,217,10,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r1,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r1,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r1,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r1,L5,225,18,6,3,0.16666666666666666,0.5,assigned_gt
bl_crossing__ni_k5_r1,R1,190,9,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r1,R3,146,7,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r1,R4,118,22,3,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r1,R5,134,27,6,4,0.14814814814814814,0.6666666666666666,assigned_gt
bl_crossing__ni_k5_r2,L1,252,45,0,0,0.0,,all_gt
bl_crossing__ni_k5_r2,L2,207,25,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r2,L3,110,20,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r2,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r2,L5,279,72,6,3,0.041666666666666664,0.5,all_gt
bl_crossing__ni_k5_r2,R1,252,71,0,0,0.0,,all_gt
bl_crossing__ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
bl_crossing__ni_k5_r2,R3,160,21,2,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r2,R4,136,40,3,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r2,R5,173,66,6,4,0.06060606060606061,0.6666666666666666,all_gt
bl_crossing__ni_k5_r2,L1,217,10,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r2,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r2,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r2,L5,225,18,6,3,0.16666666666666666,0.5,assigned_gt
bl_crossing__ni_k5_r2,R1,190,9,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r2,R3,146,7,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r2,R4,118,22,3,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r2,R5,134,27,6,4,0.14814814814814814,0.6666666666666666,assigned_gt
bl_crossing__ni_k5_r4,L1,252,45,0,0,0.0,,all_gt
bl_crossing__ni_k5_r4,L2,207,25,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r4,L3,110,20,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r4,L4,38,13,1,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r4,L5,279,72,6,3,0.041666666666666664,0.5,all_gt
bl_crossing__ni_k5_r4,R1,252,71,0,0,0.0,,all_gt
bl_crossing__ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
bl_crossing__ni_k5_r4,R3,160,21,2,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r4,R4,136,40,3,0,0.0,0.0,all_gt
bl_crossing__ni_k5_r4,R5,173,66,6,4,0.06060606060606061,0.6666666666666666,all_gt
bl_crossing__ni_k5_r4,L1,217,10,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r4,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r4,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r4,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r4,L5,225,18,6,3,0.16666666666666666,0.5,assigned_gt
bl_crossing__ni_k5_r4,R1,190,9,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
bl_crossing__ni_k5_r4,R3,146,7,2,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r4,R4,118,22,3,0,0.0,0.0,assigned_gt
bl_crossing__ni_k5_r4,R5,134,27,6,4,0.14814814814814814,0.6666666666666666,assigned_gt
bl_step_crossing__ni_k2_r1,L1,252,45,10,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r1,L2,207,25,6,1,0.04,0.16666666666666666,all_gt
bl_step_crossing__ni_k2_r1,L3,110,20,3,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r1,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing__ni_k2_r1,L5,279,72,7,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r1,R1,252,71,14,1,0.014084507042253521,0.07142857142857142,all_gt
bl_step_crossing__ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r1,R3,160,21,10,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r1,R4,136,40,6,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r1,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_step_crossing__ni_k2_r1,L1,217,10,10,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r1,L2,188,6,6,1,0.16666666666666666,0.16666666666666666,assigned_gt
bl_step_crossing__ni_k2_r1,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k2_r1,L5,225,18,7,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r1,R1,190,9,14,1,0.1111111111111111,0.07142857142857142,assigned_gt
bl_step_crossing__ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r1,R3,146,7,10,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r1,R4,118,22,6,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r1,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_step_crossing__ni_k2_r2,L1,252,45,14,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r2,L2,207,25,14,1,0.04,0.07142857142857142,all_gt
bl_step_crossing__ni_k2_r2,L3,110,20,8,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r2,L4,38,13,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r2,L5,279,72,13,2,0.027777777777777776,0.15384615384615385,all_gt
bl_step_crossing__ni_k2_r2,R1,252,71,21,3,0.04225352112676056,0.14285714285714285,all_gt
bl_step_crossing__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r2,R3,160,21,16,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r2,R4,136,40,13,2,0.05,0.15384615384615385,all_gt
bl_step_crossing__ni_k2_r2,R5,173,66,18,7,0.10606060606060606,0.3888888888888889,all_gt
bl_step_crossing__ni_k2_r2,L1,217,10,14,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r2,L2,188,6,14,1,0.16666666666666666,0.07142857142857142,assigned_gt
bl_step_crossing__ni_k2_r2,L3,95,5,8,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r2,L5,225,18,13,2,0.1111111111111111,0.15384615384615385,assigned_gt
bl_step_crossing__ni_k2_r2,R1,190,9,21,3,0.3333333333333333,0.14285714285714285,assigned_gt
bl_step_crossing__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r2,R3,146,7,16,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r2,R4,118,22,13,2,0.09090909090909091,0.15384615384615385,assigned_gt
bl_step_crossing__ni_k2_r2,R5,134,27,18,7,0.25925925925925924,0.3888888888888889,assigned_gt
bl_step_crossing__ni_k2_r4,L1,252,45,29,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r4,L2,207,25,28,1,0.04,0.03571428571428571,all_gt
bl_step_crossing__ni_k2_r4,L3,110,20,11,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r4,L4,38,13,2,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r4,L5,279,72,25,3,0.041666666666666664,0.12,all_gt
bl_step_crossing__ni_k2_r4,R1,252,71,28,3,0.04225352112676056,0.10714285714285714,all_gt
bl_step_crossing__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r4,R3,160,21,27,0,0.0,0.0,all_gt
bl_step_crossing__ni_k2_r4,R4,136,40,28,6,0.15,0.21428571428571427,all_gt
bl_step_crossing__ni_k2_r4,R5,173,66,29,9,0.13636363636363635,0.3103448275862069,all_gt
bl_step_crossing__ni_k2_r4,L1,217,10,29,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r4,L2,188,6,28,1,0.16666666666666666,0.03571428571428571,assigned_gt
bl_step_crossing__ni_k2_r4,L3,95,5,11,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r4,L4,30,5,2,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r4,L5,225,18,25,3,0.16666666666666666,0.12,assigned_gt
bl_step_crossing__ni_k2_r4,R1,190,9,28,3,0.3333333333333333,0.10714285714285714,assigned_gt
bl_step_crossing__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r4,R3,146,7,27,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k2_r4,R4,118,22,28,6,0.2727272727272727,0.21428571428571427,assigned_gt
bl_step_crossing__ni_k2_r4,R5,134,27,29,9,0.3333333333333333,0.3103448275862069,assigned_gt
bl_step_crossing__ni_k3_r1,L1,252,45,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r1,L2,207,25,0,0,0.0,,all_gt
bl_step_crossing__ni_k3_r1,L3,110,20,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r1,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing__ni_k3_r1,L5,279,72,2,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r1,R1,252,71,4,1,0.014084507042253521,0.25,all_gt
bl_step_crossing__ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r1,R3,160,21,4,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r1,R4,136,40,3,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r1,R5,173,66,6,3,0.045454545454545456,0.5,all_gt
bl_step_crossing__ni_k3_r1,L1,217,10,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r1,L2,188,6,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k3_r1,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k3_r1,L5,225,18,2,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r1,R1,190,9,4,1,0.1111111111111111,0.25,assigned_gt
bl_step_crossing__ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r1,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r1,R4,118,22,3,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r1,R5,134,27,6,3,0.1111111111111111,0.5,assigned_gt
bl_step_crossing__ni_k3_r2,L1,252,45,4,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r2,L2,207,25,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r2,L3,110,20,4,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r2,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing__ni_k3_r2,L5,279,72,2,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r2,R1,252,71,7,2,0.028169014084507043,0.2857142857142857,all_gt
bl_step_crossing__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r2,R3,160,21,4,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r2,R4,136,40,4,1,0.025,0.25,all_gt
bl_step_crossing__ni_k3_r2,R5,173,66,9,3,0.045454545454545456,0.3333333333333333,all_gt
bl_step_crossing__ni_k3_r2,L1,217,10,4,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r2,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r2,L3,95,5,4,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r2,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k3_r2,L5,225,18,2,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r2,R1,190,9,7,2,0.2222222222222222,0.2857142857142857,assigned_gt
bl_step_crossing__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r2,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r2,R4,118,22,4,1,0.045454545454545456,0.25,assigned_gt
bl_step_crossing__ni_k3_r2,R5,134,27,9,3,0.1111111111111111,0.3333333333333333,assigned_gt
bl_step_crossing__ni_k3_r4,L1,252,45,9,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,L2,207,25,6,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,L3,110,20,4,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,L4,38,13,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,L5,279,72,6,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,R1,252,71,8,2,0.028169014084507043,0.25,all_gt
bl_step_crossing__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,R3,160,21,7,0,0.0,0.0,all_gt
bl_step_crossing__ni_k3_r4,R4,136,40,11,4,0.1,0.36363636363636365,all_gt
bl_step_crossing__ni_k3_r4,R5,173,66,12,4,0.06060606060606061,0.3333333333333333,all_gt
bl_step_crossing__ni_k3_r4,L1,217,10,9,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,L2,188,6,6,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,L3,95,5,4,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,L5,225,18,6,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,R1,190,9,8,2,0.2222222222222222,0.25,assigned_gt
bl_step_crossing__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k3_r4,R4,118,22,11,4,0.18181818181818182,0.36363636363636365,assigned_gt
bl_step_crossing__ni_k3_r4,R5,134,27,12,4,0.14814814814814814,0.3333333333333333,assigned_gt
bl_step_crossing__ni_k5_r1,L1,252,45,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r1,L2,207,25,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r1,L3,110,20,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r1,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r1,L5,279,72,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r1,R1,252,71,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r1,R3,160,21,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r1,R4,136,40,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r1,R5,173,66,4,3,0.045454545454545456,0.75,all_gt
bl_step_crossing__ni_k5_r1,L1,217,10,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r1,L2,188,6,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r1,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r1,L5,225,18,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r1,R1,190,9,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r1,R3,146,7,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r1,R4,118,22,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r1,R5,134,27,4,3,0.1111111111111111,0.75,assigned_gt
bl_step_crossing__ni_k5_r2,L1,252,45,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r2,L2,207,25,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r2,L3,110,20,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r2,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r2,L5,279,72,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r2,R1,252,71,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r2,R3,160,21,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r2,R4,136,40,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r2,R5,173,66,4,3,0.045454545454545456,0.75,all_gt
bl_step_crossing__ni_k5_r2,L1,217,10,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r2,L2,188,6,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r2,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r2,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r2,L5,225,18,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r2,R1,190,9,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r2,R3,146,7,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r2,R4,118,22,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r2,R5,134,27,4,3,0.1111111111111111,0.75,assigned_gt
bl_step_crossing__ni_k5_r4,L1,252,45,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r4,L2,207,25,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r4,L3,110,20,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r4,L4,38,13,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r4,L5,279,72,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r4,R1,252,71,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
bl_step_crossing__ni_k5_r4,R3,160,21,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r4,R4,136,40,1,0,0.0,0.0,all_gt
bl_step_crossing__ni_k5_r4,R5,173,66,4,3,0.045454545454545456,0.75,all_gt
bl_step_crossing__ni_k5_r4,L1,217,10,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r4,L2,188,6,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r4,L3,95,5,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r4,L4,30,5,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r4,L5,225,18,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r4,R1,190,9,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
bl_step_crossing__ni_k5_r4,R3,146,7,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r4,R4,118,22,1,0,0.0,0.0,assigned_gt
bl_step_crossing__ni_k5_r4,R5,134,27,4,3,0.1111111111111111,0.75,assigned_gt
bl_rate_q995__ni_k2_r1,L1,252,45,12,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r1,L2,207,25,6,1,0.04,0.16666666666666666,all_gt
bl_rate_q995__ni_k2_r1,L3,110,20,2,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r1,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995__ni_k2_r1,L5,279,72,8,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r1,R1,252,71,17,1,0.014084507042253521,0.058823529411764705,all_gt
bl_rate_q995__ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r1,R3,160,21,10,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r1,R4,136,40,7,1,0.025,0.14285714285714285,all_gt
bl_rate_q995__ni_k2_r1,R5,173,66,11,5,0.07575757575757576,0.45454545454545453,all_gt
bl_rate_q995__ni_k2_r1,L1,217,10,12,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r1,L2,188,6,6,1,0.16666666666666666,0.16666666666666666,assigned_gt
bl_rate_q995__ni_k2_r1,L3,95,5,2,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k2_r1,L5,225,18,8,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r1,R1,190,9,17,1,0.1111111111111111,0.058823529411764705,assigned_gt
bl_rate_q995__ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r1,R3,146,7,10,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r1,R4,118,22,7,1,0.045454545454545456,0.14285714285714285,assigned_gt
bl_rate_q995__ni_k2_r1,R5,134,27,11,5,0.18518518518518517,0.45454545454545453,assigned_gt
bl_rate_q995__ni_k2_r2,L1,252,45,16,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r2,L2,207,25,14,1,0.04,0.07142857142857142,all_gt
bl_rate_q995__ni_k2_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r2,L4,38,13,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r2,L5,279,72,14,2,0.027777777777777776,0.14285714285714285,all_gt
bl_rate_q995__ni_k2_r2,R1,252,71,24,3,0.04225352112676056,0.125,all_gt
bl_rate_q995__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r2,R3,160,21,16,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r2,R4,136,40,14,3,0.075,0.21428571428571427,all_gt
bl_rate_q995__ni_k2_r2,R5,173,66,16,6,0.09090909090909091,0.375,all_gt
bl_rate_q995__ni_k2_r2,L1,217,10,16,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r2,L2,188,6,14,1,0.16666666666666666,0.07142857142857142,assigned_gt
bl_rate_q995__ni_k2_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r2,L5,225,18,14,2,0.1111111111111111,0.14285714285714285,assigned_gt
bl_rate_q995__ni_k2_r2,R1,190,9,24,3,0.3333333333333333,0.125,assigned_gt
bl_rate_q995__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r2,R3,146,7,16,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r2,R4,118,22,14,3,0.13636363636363635,0.21428571428571427,assigned_gt
bl_rate_q995__ni_k2_r2,R5,134,27,16,6,0.2222222222222222,0.375,assigned_gt
bl_rate_q995__ni_k2_r4,L1,252,45,31,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r4,L2,207,25,28,1,0.04,0.03571428571428571,all_gt
bl_rate_q995__ni_k2_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r4,L4,38,13,2,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r4,L5,279,72,26,3,0.041666666666666664,0.11538461538461539,all_gt
bl_rate_q995__ni_k2_r4,R1,252,71,31,3,0.04225352112676056,0.0967741935483871,all_gt
bl_rate_q995__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r4,R3,160,21,27,0,0.0,0.0,all_gt
bl_rate_q995__ni_k2_r4,R4,136,40,29,7,0.175,0.2413793103448276,all_gt
bl_rate_q995__ni_k2_r4,R5,173,66,27,8,0.12121212121212122,0.2962962962962963,all_gt
bl_rate_q995__ni_k2_r4,L1,217,10,31,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r4,L2,188,6,28,1,0.16666666666666666,0.03571428571428571,assigned_gt
bl_rate_q995__ni_k2_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r4,L4,30,5,2,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r4,L5,225,18,26,3,0.16666666666666666,0.11538461538461539,assigned_gt
bl_rate_q995__ni_k2_r4,R1,190,9,31,3,0.3333333333333333,0.0967741935483871,assigned_gt
bl_rate_q995__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r4,R3,146,7,27,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k2_r4,R4,118,22,29,7,0.3181818181818182,0.2413793103448276,assigned_gt
bl_rate_q995__ni_k2_r4,R5,134,27,27,8,0.2962962962962963,0.2962962962962963,assigned_gt
bl_rate_q995__ni_k3_r1,L1,252,45,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r1,L2,207,25,0,0,0.0,,all_gt
bl_rate_q995__ni_k3_r1,L3,110,20,0,0,0.0,,all_gt
bl_rate_q995__ni_k3_r1,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995__ni_k3_r1,L5,279,72,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r1,R1,252,71,7,1,0.014084507042253521,0.14285714285714285,all_gt
bl_rate_q995__ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r1,R3,160,21,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r1,R4,136,40,4,1,0.025,0.25,all_gt
bl_rate_q995__ni_k3_r1,R5,173,66,4,2,0.030303030303030304,0.5,all_gt
bl_rate_q995__ni_k3_r1,L1,217,10,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r1,L2,188,6,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k3_r1,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k3_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k3_r1,L5,225,18,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r1,R1,190,9,7,1,0.1111111111111111,0.14285714285714285,assigned_gt
bl_rate_q995__ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r1,R3,146,7,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r1,R4,118,22,4,1,0.045454545454545456,0.25,assigned_gt
bl_rate_q995__ni_k3_r1,R5,134,27,4,2,0.07407407407407407,0.5,assigned_gt
bl_rate_q995__ni_k3_r2,L1,252,45,6,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r2,L2,207,25,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r2,L3,110,20,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r2,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995__ni_k3_r2,L5,279,72,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r2,R1,252,71,10,2,0.028169014084507043,0.2,all_gt
bl_rate_q995__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r2,R3,160,21,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r2,R4,136,40,5,2,0.05,0.4,all_gt
bl_rate_q995__ni_k3_r2,R5,173,66,8,3,0.045454545454545456,0.375,all_gt
bl_rate_q995__ni_k3_r2,L1,217,10,6,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r2,L2,188,6,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r2,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r2,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k3_r2,L5,225,18,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r2,R1,190,9,10,2,0.2222222222222222,0.2,assigned_gt
bl_rate_q995__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r2,R3,146,7,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r2,R4,118,22,5,2,0.09090909090909091,0.4,assigned_gt
bl_rate_q995__ni_k3_r2,R5,134,27,8,3,0.1111111111111111,0.375,assigned_gt
bl_rate_q995__ni_k3_r4,L1,252,45,11,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,L2,207,25,6,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,L3,110,20,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,L4,38,13,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,L5,279,72,7,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,R1,252,71,11,2,0.028169014084507043,0.18181818181818182,all_gt
bl_rate_q995__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,R3,160,21,6,0,0.0,0.0,all_gt
bl_rate_q995__ni_k3_r4,R4,136,40,12,5,0.125,0.4166666666666667,all_gt
bl_rate_q995__ni_k3_r4,R5,173,66,11,4,0.06060606060606061,0.36363636363636365,all_gt
bl_rate_q995__ni_k3_r4,L1,217,10,11,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,L2,188,6,6,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,L5,225,18,7,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,R1,190,9,11,2,0.2222222222222222,0.18181818181818182,assigned_gt
bl_rate_q995__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,R3,146,7,6,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k3_r4,R4,118,22,12,5,0.22727272727272727,0.4166666666666667,assigned_gt
bl_rate_q995__ni_k3_r4,R5,134,27,11,4,0.14814814814814814,0.36363636363636365,assigned_gt
bl_rate_q995__ni_k5_r1,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r1,L2,207,25,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r1,L3,110,20,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r1,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r1,L5,279,72,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r1,R1,252,71,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r1,R3,160,21,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r1,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q995__ni_k5_r1,R5,173,66,1,1,0.015151515151515152,1.0,all_gt
bl_rate_q995__ni_k5_r1,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r1,L2,188,6,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r1,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r1,L5,225,18,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r1,R1,190,9,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r1,R3,146,7,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r1,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q995__ni_k5_r1,R5,134,27,1,1,0.037037037037037035,1.0,assigned_gt
bl_rate_q995__ni_k5_r2,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r2,L2,207,25,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r2,L3,110,20,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r2,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r2,L5,279,72,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r2,R1,252,71,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r2,R3,160,21,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r2,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q995__ni_k5_r2,R5,173,66,1,1,0.015151515151515152,1.0,all_gt
bl_rate_q995__ni_k5_r2,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r2,L2,188,6,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r2,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r2,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r2,L5,225,18,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r2,R1,190,9,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r2,R3,146,7,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r2,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q995__ni_k5_r2,R5,134,27,1,1,0.037037037037037035,1.0,assigned_gt
bl_rate_q995__ni_k5_r4,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r4,L2,207,25,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r4,L3,110,20,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r4,L4,38,13,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r4,L5,279,72,1,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r4,R1,252,71,3,0,0.0,0.0,all_gt
bl_rate_q995__ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r4,R3,160,21,0,0,0.0,,all_gt
bl_rate_q995__ni_k5_r4,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q995__ni_k5_r4,R5,173,66,1,1,0.015151515151515152,1.0,all_gt
bl_rate_q995__ni_k5_r4,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r4,L2,188,6,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r4,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r4,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r4,L5,225,18,1,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r4,R1,190,9,3,0,0.0,0.0,assigned_gt
bl_rate_q995__ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r4,R3,146,7,0,0,0.0,,assigned_gt
bl_rate_q995__ni_k5_r4,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q995__ni_k5_r4,R5,134,27,1,1,0.037037037037037035,1.0,assigned_gt
bl_rate_q990__ni_k2_r1,L1,252,45,12,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r1,L2,207,25,8,1,0.04,0.125,all_gt
bl_rate_q990__ni_k2_r1,L3,110,20,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r1,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990__ni_k2_r1,L5,279,72,11,1,0.013888888888888888,0.09090909090909091,all_gt
bl_rate_q990__ni_k2_r1,R1,252,71,17,1,0.014084507042253521,0.058823529411764705,all_gt
bl_rate_q990__ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r1,R3,160,21,11,1,0.047619047619047616,0.09090909090909091,all_gt
bl_rate_q990__ni_k2_r1,R4,136,40,7,1,0.025,0.14285714285714285,all_gt
bl_rate_q990__ni_k2_r1,R5,173,66,13,5,0.07575757575757576,0.38461538461538464,all_gt
bl_rate_q990__ni_k2_r1,L1,217,10,12,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r1,L2,188,6,8,1,0.16666666666666666,0.125,assigned_gt
bl_rate_q990__ni_k2_r1,L3,95,5,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k2_r1,L5,225,18,11,1,0.05555555555555555,0.09090909090909091,assigned_gt
bl_rate_q990__ni_k2_r1,R1,190,9,17,1,0.1111111111111111,0.058823529411764705,assigned_gt
bl_rate_q990__ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r1,R3,146,7,11,1,0.14285714285714285,0.09090909090909091,assigned_gt
bl_rate_q990__ni_k2_r1,R4,118,22,7,1,0.045454545454545456,0.14285714285714285,assigned_gt
bl_rate_q990__ni_k2_r1,R5,134,27,13,5,0.18518518518518517,0.38461538461538464,assigned_gt
bl_rate_q990__ni_k2_r2,L1,252,45,16,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r2,L2,207,25,16,1,0.04,0.0625,all_gt
bl_rate_q990__ni_k2_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r2,L4,38,13,1,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r2,L5,279,72,17,3,0.041666666666666664,0.17647058823529413,all_gt
bl_rate_q990__ni_k2_r2,R1,252,71,24,3,0.04225352112676056,0.125,all_gt
bl_rate_q990__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r2,R3,160,21,17,1,0.047619047619047616,0.058823529411764705,all_gt
bl_rate_q990__ni_k2_r2,R4,136,40,14,3,0.075,0.21428571428571427,all_gt
bl_rate_q990__ni_k2_r2,R5,173,66,18,6,0.09090909090909091,0.3333333333333333,all_gt
bl_rate_q990__ni_k2_r2,L1,217,10,16,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r2,L2,188,6,16,1,0.16666666666666666,0.0625,assigned_gt
bl_rate_q990__ni_k2_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r2,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r2,L5,225,18,17,3,0.16666666666666666,0.17647058823529413,assigned_gt
bl_rate_q990__ni_k2_r2,R1,190,9,24,3,0.3333333333333333,0.125,assigned_gt
bl_rate_q990__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r2,R3,146,7,17,1,0.14285714285714285,0.058823529411764705,assigned_gt
bl_rate_q990__ni_k2_r2,R4,118,22,14,3,0.13636363636363635,0.21428571428571427,assigned_gt
bl_rate_q990__ni_k2_r2,R5,134,27,18,6,0.2222222222222222,0.3333333333333333,assigned_gt
bl_rate_q990__ni_k2_r4,L1,252,45,31,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r4,L2,207,25,30,1,0.04,0.03333333333333333,all_gt
bl_rate_q990__ni_k2_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r4,L4,38,13,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r4,L5,279,72,28,4,0.05555555555555555,0.14285714285714285,all_gt
bl_rate_q990__ni_k2_r4,R1,252,71,31,3,0.04225352112676056,0.0967741935483871,all_gt
bl_rate_q990__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
bl_rate_q990__ni_k2_r4,R3,160,21,28,1,0.047619047619047616,0.03571428571428571,all_gt
bl_rate_q990__ni_k2_r4,R4,136,40,29,7,0.175,0.2413793103448276,all_gt
bl_rate_q990__ni_k2_r4,R5,173,66,28,8,0.12121212121212122,0.2857142857142857,all_gt
bl_rate_q990__ni_k2_r4,L1,217,10,31,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r4,L2,188,6,30,1,0.16666666666666666,0.03333333333333333,assigned_gt
bl_rate_q990__ni_k2_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r4,L4,30,5,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r4,L5,225,18,28,4,0.2222222222222222,0.14285714285714285,assigned_gt
bl_rate_q990__ni_k2_r4,R1,190,9,31,3,0.3333333333333333,0.0967741935483871,assigned_gt
bl_rate_q990__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k2_r4,R3,146,7,28,1,0.14285714285714285,0.03571428571428571,assigned_gt
bl_rate_q990__ni_k2_r4,R4,118,22,29,7,0.3181818181818182,0.2413793103448276,assigned_gt
bl_rate_q990__ni_k2_r4,R5,134,27,28,8,0.2962962962962963,0.2857142857142857,assigned_gt
bl_rate_q990__ni_k3_r1,L1,252,45,3,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r1,L2,207,25,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r1,L3,110,20,0,0,0.0,,all_gt
bl_rate_q990__ni_k3_r1,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990__ni_k3_r1,L5,279,72,6,1,0.013888888888888888,0.16666666666666666,all_gt
bl_rate_q990__ni_k3_r1,R1,252,71,7,1,0.014084507042253521,0.14285714285714285,all_gt
bl_rate_q990__ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r1,R3,160,21,4,1,0.047619047619047616,0.25,all_gt
bl_rate_q990__ni_k3_r1,R4,136,40,4,1,0.025,0.25,all_gt
bl_rate_q990__ni_k3_r1,R5,173,66,6,2,0.030303030303030304,0.3333333333333333,all_gt
bl_rate_q990__ni_k3_r1,L1,217,10,3,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r1,L2,188,6,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r1,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k3_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k3_r1,L5,225,18,6,1,0.05555555555555555,0.16666666666666666,assigned_gt
bl_rate_q990__ni_k3_r1,R1,190,9,7,1,0.1111111111111111,0.14285714285714285,assigned_gt
bl_rate_q990__ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r1,R3,146,7,4,1,0.14285714285714285,0.25,assigned_gt
bl_rate_q990__ni_k3_r1,R4,118,22,4,1,0.045454545454545456,0.25,assigned_gt
bl_rate_q990__ni_k3_r1,R5,134,27,6,2,0.07407407407407407,0.3333333333333333,assigned_gt
bl_rate_q990__ni_k3_r2,L1,252,45,6,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r2,L2,207,25,3,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r2,L3,110,20,3,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r2,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990__ni_k3_r2,L5,279,72,6,1,0.013888888888888888,0.16666666666666666,all_gt
bl_rate_q990__ni_k3_r2,R1,252,71,10,2,0.028169014084507043,0.2,all_gt
bl_rate_q990__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r2,R3,160,21,4,1,0.047619047619047616,0.25,all_gt
bl_rate_q990__ni_k3_r2,R4,136,40,5,2,0.05,0.4,all_gt
bl_rate_q990__ni_k3_r2,R5,173,66,10,3,0.045454545454545456,0.3,all_gt
bl_rate_q990__ni_k3_r2,L1,217,10,6,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r2,L2,188,6,3,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r2,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r2,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k3_r2,L5,225,18,6,1,0.05555555555555555,0.16666666666666666,assigned_gt
bl_rate_q990__ni_k3_r2,R1,190,9,10,2,0.2222222222222222,0.2,assigned_gt
bl_rate_q990__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r2,R3,146,7,4,1,0.14285714285714285,0.25,assigned_gt
bl_rate_q990__ni_k3_r2,R4,118,22,5,2,0.09090909090909091,0.4,assigned_gt
bl_rate_q990__ni_k3_r2,R5,134,27,10,3,0.1111111111111111,0.3,assigned_gt
bl_rate_q990__ni_k3_r4,L1,252,45,11,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r4,L2,207,25,8,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r4,L3,110,20,3,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r4,L4,38,13,1,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r4,L5,279,72,10,1,0.013888888888888888,0.1,all_gt
bl_rate_q990__ni_k3_r4,R1,252,71,11,2,0.028169014084507043,0.18181818181818182,all_gt
bl_rate_q990__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
bl_rate_q990__ni_k3_r4,R3,160,21,7,1,0.047619047619047616,0.14285714285714285,all_gt
bl_rate_q990__ni_k3_r4,R4,136,40,12,5,0.125,0.4166666666666667,all_gt
bl_rate_q990__ni_k3_r4,R5,173,66,13,4,0.06060606060606061,0.3076923076923077,all_gt
bl_rate_q990__ni_k3_r4,L1,217,10,11,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r4,L2,188,6,8,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r4,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r4,L4,30,5,1,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r4,L5,225,18,10,1,0.05555555555555555,0.1,assigned_gt
bl_rate_q990__ni_k3_r4,R1,190,9,11,2,0.2222222222222222,0.18181818181818182,assigned_gt
bl_rate_q990__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k3_r4,R3,146,7,7,1,0.14285714285714285,0.14285714285714285,assigned_gt
bl_rate_q990__ni_k3_r4,R4,118,22,12,5,0.22727272727272727,0.4166666666666667,assigned_gt
bl_rate_q990__ni_k3_r4,R5,134,27,13,4,0.14814814814814814,0.3076923076923077,assigned_gt
bl_rate_q990__ni_k5_r1,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r1,L2,207,25,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r1,L3,110,20,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r1,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r1,L5,279,72,4,1,0.013888888888888888,0.25,all_gt
bl_rate_q990__ni_k5_r1,R1,252,71,4,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r1,R3,160,21,1,1,0.047619047619047616,1.0,all_gt
bl_rate_q990__ni_k5_r1,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q990__ni_k5_r1,R5,173,66,3,1,0.015151515151515152,0.3333333333333333,all_gt
bl_rate_q990__ni_k5_r1,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r1,L2,188,6,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r1,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r1,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r1,L5,225,18,4,1,0.05555555555555555,0.25,assigned_gt
bl_rate_q990__ni_k5_r1,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r1,R3,146,7,1,1,0.14285714285714285,1.0,assigned_gt
bl_rate_q990__ni_k5_r1,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q990__ni_k5_r1,R5,134,27,3,1,0.037037037037037035,0.3333333333333333,assigned_gt
bl_rate_q990__ni_k5_r2,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r2,L2,207,25,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r2,L3,110,20,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r2,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r2,L5,279,72,4,1,0.013888888888888888,0.25,all_gt
bl_rate_q990__ni_k5_r2,R1,252,71,4,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r2,R3,160,21,1,1,0.047619047619047616,1.0,all_gt
bl_rate_q990__ni_k5_r2,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q990__ni_k5_r2,R5,173,66,3,1,0.015151515151515152,0.3333333333333333,all_gt
bl_rate_q990__ni_k5_r2,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r2,L2,188,6,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r2,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r2,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r2,L5,225,18,4,1,0.05555555555555555,0.25,assigned_gt
bl_rate_q990__ni_k5_r2,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r2,R3,146,7,1,1,0.14285714285714285,1.0,assigned_gt
bl_rate_q990__ni_k5_r2,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q990__ni_k5_r2,R5,134,27,3,1,0.037037037037037035,0.3333333333333333,assigned_gt
bl_rate_q990__ni_k5_r4,L1,252,45,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r4,L2,207,25,2,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r4,L3,110,20,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r4,L4,38,13,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r4,L5,279,72,4,1,0.013888888888888888,0.25,all_gt
bl_rate_q990__ni_k5_r4,R1,252,71,4,0,0.0,0.0,all_gt
bl_rate_q990__ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
bl_rate_q990__ni_k5_r4,R3,160,21,1,1,0.047619047619047616,1.0,all_gt
bl_rate_q990__ni_k5_r4,R4,136,40,1,1,0.025,1.0,all_gt
bl_rate_q990__ni_k5_r4,R5,173,66,3,1,0.015151515151515152,0.3333333333333333,all_gt
bl_rate_q990__ni_k5_r4,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r4,L2,188,6,2,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r4,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r4,L4,30,5,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r4,L5,225,18,4,1,0.05555555555555555,0.25,assigned_gt
bl_rate_q990__ni_k5_r4,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_rate_q990__ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q990__ni_k5_r4,R3,146,7,1,1,0.14285714285714285,1.0,assigned_gt
bl_rate_q990__ni_k5_r4,R4,118,22,1,1,0.045454545454545456,1.0,assigned_gt
bl_rate_q990__ni_k5_r4,R5,134,27,3,1,0.037037037037037035,0.3333333333333333,assigned_gt
bl_rate_q975__ni_k2_r1,L1,252,45,16,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r1,L2,207,25,9,1,0.04,0.1111111111111111,all_gt
bl_rate_q975__ni_k2_r1,L3,110,20,2,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r1,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975__ni_k2_r1,L5,279,72,15,1,0.013888888888888888,0.06666666666666667,all_gt
bl_rate_q975__ni_k2_r1,R1,252,71,24,1,0.014084507042253521,0.041666666666666664,all_gt
bl_rate_q975__ni_k2_r1,R2,193,19,6,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r1,R3,160,21,13,1,0.047619047619047616,0.07692307692307693,all_gt
bl_rate_q975__ni_k2_r1,R4,136,40,9,2,0.05,0.2222222222222222,all_gt
bl_rate_q975__ni_k2_r1,R5,173,66,17,5,0.07575757575757576,0.29411764705882354,all_gt
bl_rate_q975__ni_k2_r1,L1,217,10,16,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r1,L2,188,6,9,1,0.16666666666666666,0.1111111111111111,assigned_gt
bl_rate_q975__ni_k2_r1,L3,95,5,2,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r1,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975__ni_k2_r1,L5,225,18,15,1,0.05555555555555555,0.06666666666666667,assigned_gt
bl_rate_q975__ni_k2_r1,R1,190,9,24,1,0.1111111111111111,0.041666666666666664,assigned_gt
bl_rate_q975__ni_k2_r1,R2,182,8,6,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r1,R3,146,7,13,1,0.14285714285714285,0.07692307692307693,assigned_gt
bl_rate_q975__ni_k2_r1,R4,118,22,9,2,0.09090909090909091,0.2222222222222222,assigned_gt
bl_rate_q975__ni_k2_r1,R5,134,27,17,5,0.18518518518518517,0.29411764705882354,assigned_gt
bl_rate_q975__ni_k2_r2,L1,252,45,20,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r2,L2,207,25,17,1,0.04,0.058823529411764705,all_gt
bl_rate_q975__ni_k2_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r2,L4,38,13,2,1,0.07692307692307693,0.5,all_gt
bl_rate_q975__ni_k2_r2,L5,279,72,21,3,0.041666666666666664,0.14285714285714285,all_gt
bl_rate_q975__ni_k2_r2,R1,252,71,31,3,0.04225352112676056,0.0967741935483871,all_gt
bl_rate_q975__ni_k2_r2,R2,193,19,13,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r2,R3,160,21,19,1,0.047619047619047616,0.05263157894736842,all_gt
bl_rate_q975__ni_k2_r2,R4,136,40,15,4,0.1,0.26666666666666666,all_gt
bl_rate_q975__ni_k2_r2,R5,173,66,22,6,0.09090909090909091,0.2727272727272727,all_gt
bl_rate_q975__ni_k2_r2,L1,217,10,20,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r2,L2,188,6,17,1,0.16666666666666666,0.058823529411764705,assigned_gt
bl_rate_q975__ni_k2_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r2,L4,30,5,2,1,0.2,0.5,assigned_gt
bl_rate_q975__ni_k2_r2,L5,225,18,21,3,0.16666666666666666,0.14285714285714285,assigned_gt
bl_rate_q975__ni_k2_r2,R1,190,9,31,3,0.3333333333333333,0.0967741935483871,assigned_gt
bl_rate_q975__ni_k2_r2,R2,182,8,13,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r2,R3,146,7,19,1,0.14285714285714285,0.05263157894736842,assigned_gt
bl_rate_q975__ni_k2_r2,R4,118,22,15,4,0.18181818181818182,0.26666666666666666,assigned_gt
bl_rate_q975__ni_k2_r2,R5,134,27,22,6,0.2222222222222222,0.2727272727272727,assigned_gt
bl_rate_q975__ni_k2_r4,L1,252,45,34,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r4,L2,207,25,31,1,0.04,0.03225806451612903,all_gt
bl_rate_q975__ni_k2_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r4,L4,38,13,3,1,0.07692307692307693,0.3333333333333333,all_gt
bl_rate_q975__ni_k2_r4,L5,279,72,32,4,0.05555555555555555,0.125,all_gt
bl_rate_q975__ni_k2_r4,R1,252,71,38,3,0.04225352112676056,0.07894736842105263,all_gt
bl_rate_q975__ni_k2_r4,R2,193,19,20,0,0.0,0.0,all_gt
bl_rate_q975__ni_k2_r4,R3,160,21,30,1,0.047619047619047616,0.03333333333333333,all_gt
bl_rate_q975__ni_k2_r4,R4,136,40,30,8,0.2,0.26666666666666666,all_gt
bl_rate_q975__ni_k2_r4,R5,173,66,31,8,0.12121212121212122,0.25806451612903225,all_gt
bl_rate_q975__ni_k2_r4,L1,217,10,34,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r4,L2,188,6,31,1,0.16666666666666666,0.03225806451612903,assigned_gt
bl_rate_q975__ni_k2_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r4,L4,30,5,3,1,0.2,0.3333333333333333,assigned_gt
bl_rate_q975__ni_k2_r4,L5,225,18,32,4,0.2222222222222222,0.125,assigned_gt
bl_rate_q975__ni_k2_r4,R1,190,9,38,3,0.3333333333333333,0.07894736842105263,assigned_gt
bl_rate_q975__ni_k2_r4,R2,182,8,20,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k2_r4,R3,146,7,30,1,0.14285714285714285,0.03333333333333333,assigned_gt
bl_rate_q975__ni_k2_r4,R4,118,22,30,8,0.36363636363636365,0.26666666666666666,assigned_gt
bl_rate_q975__ni_k2_r4,R5,134,27,31,8,0.2962962962962963,0.25806451612903225,assigned_gt
bl_rate_q975__ni_k3_r1,L1,252,45,7,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r1,L2,207,25,3,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r1,L3,110,20,0,0,0.0,,all_gt
bl_rate_q975__ni_k3_r1,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975__ni_k3_r1,L5,279,72,10,1,0.013888888888888888,0.1,all_gt
bl_rate_q975__ni_k3_r1,R1,252,71,14,1,0.014084507042253521,0.07142857142857142,all_gt
bl_rate_q975__ni_k3_r1,R2,193,19,1,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r1,R3,160,21,6,1,0.047619047619047616,0.16666666666666666,all_gt
bl_rate_q975__ni_k3_r1,R4,136,40,6,2,0.05,0.3333333333333333,all_gt
bl_rate_q975__ni_k3_r1,R5,173,66,10,2,0.030303030303030304,0.2,all_gt
bl_rate_q975__ni_k3_r1,L1,217,10,7,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r1,L2,188,6,3,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r1,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k3_r1,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975__ni_k3_r1,L5,225,18,10,1,0.05555555555555555,0.1,assigned_gt
bl_rate_q975__ni_k3_r1,R1,190,9,14,1,0.1111111111111111,0.07142857142857142,assigned_gt
bl_rate_q975__ni_k3_r1,R2,182,8,1,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r1,R3,146,7,6,1,0.14285714285714285,0.16666666666666666,assigned_gt
bl_rate_q975__ni_k3_r1,R4,118,22,6,2,0.09090909090909091,0.3333333333333333,assigned_gt
bl_rate_q975__ni_k3_r1,R5,134,27,10,2,0.07407407407407407,0.2,assigned_gt
bl_rate_q975__ni_k3_r2,L1,252,45,10,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r2,L2,207,25,4,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r2,L3,110,20,3,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r2,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975__ni_k3_r2,L5,279,72,10,1,0.013888888888888888,0.1,all_gt
bl_rate_q975__ni_k3_r2,R1,252,71,17,2,0.028169014084507043,0.11764705882352941,all_gt
bl_rate_q975__ni_k3_r2,R2,193,19,2,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r2,R3,160,21,6,1,0.047619047619047616,0.16666666666666666,all_gt
bl_rate_q975__ni_k3_r2,R4,136,40,7,3,0.075,0.42857142857142855,all_gt
bl_rate_q975__ni_k3_r2,R5,173,66,14,3,0.045454545454545456,0.21428571428571427,all_gt
bl_rate_q975__ni_k3_r2,L1,217,10,10,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r2,L2,188,6,4,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r2,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r2,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975__ni_k3_r2,L5,225,18,10,1,0.05555555555555555,0.1,assigned_gt
bl_rate_q975__ni_k3_r2,R1,190,9,17,2,0.2222222222222222,0.11764705882352941,assigned_gt
bl_rate_q975__ni_k3_r2,R2,182,8,2,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r2,R3,146,7,6,1,0.14285714285714285,0.16666666666666666,assigned_gt
bl_rate_q975__ni_k3_r2,R4,118,22,7,3,0.13636363636363635,0.42857142857142855,assigned_gt
bl_rate_q975__ni_k3_r2,R5,134,27,14,3,0.1111111111111111,0.21428571428571427,assigned_gt
bl_rate_q975__ni_k3_r4,L1,252,45,15,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r4,L2,207,25,9,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r4,L3,110,20,3,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r4,L4,38,13,2,1,0.07692307692307693,0.5,all_gt
bl_rate_q975__ni_k3_r4,L5,279,72,14,1,0.013888888888888888,0.07142857142857142,all_gt
bl_rate_q975__ni_k3_r4,R1,252,71,18,2,0.028169014084507043,0.1111111111111111,all_gt
bl_rate_q975__ni_k3_r4,R2,193,19,4,0,0.0,0.0,all_gt
bl_rate_q975__ni_k3_r4,R3,160,21,9,1,0.047619047619047616,0.1111111111111111,all_gt
bl_rate_q975__ni_k3_r4,R4,136,40,13,6,0.15,0.46153846153846156,all_gt
bl_rate_q975__ni_k3_r4,R5,173,66,16,4,0.06060606060606061,0.25,all_gt
bl_rate_q975__ni_k3_r4,L1,217,10,15,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r4,L2,188,6,9,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r4,L3,95,5,3,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r4,L4,30,5,2,1,0.2,0.5,assigned_gt
bl_rate_q975__ni_k3_r4,L5,225,18,14,1,0.05555555555555555,0.07142857142857142,assigned_gt
bl_rate_q975__ni_k3_r4,R1,190,9,18,2,0.2222222222222222,0.1111111111111111,assigned_gt
bl_rate_q975__ni_k3_r4,R2,182,8,4,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k3_r4,R3,146,7,9,1,0.14285714285714285,0.1111111111111111,assigned_gt
bl_rate_q975__ni_k3_r4,R4,118,22,13,6,0.2727272727272727,0.46153846153846156,assigned_gt
bl_rate_q975__ni_k3_r4,R5,134,27,16,4,0.14814814814814814,0.25,assigned_gt
bl_rate_q975__ni_k5_r1,L1,252,45,6,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r1,L2,207,25,3,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r1,L3,110,20,0,0,0.0,,all_gt
bl_rate_q975__ni_k5_r1,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975__ni_k5_r1,L5,279,72,8,1,0.013888888888888888,0.125,all_gt
bl_rate_q975__ni_k5_r1,R1,252,71,11,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r1,R2,193,19,0,0,0.0,,all_gt
bl_rate_q975__ni_k5_r1,R3,160,21,3,1,0.047619047619047616,0.3333333333333333,all_gt
bl_rate_q975__ni_k5_r1,R4,136,40,3,2,0.05,0.6666666666666666,all_gt
bl_rate_q975__ni_k5_r1,R5,173,66,7,1,0.015151515151515152,0.14285714285714285,all_gt
bl_rate_q975__ni_k5_r1,L1,217,10,6,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r1,L2,188,6,3,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r1,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k5_r1,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975__ni_k5_r1,L5,225,18,8,1,0.05555555555555555,0.125,assigned_gt
bl_rate_q975__ni_k5_r1,R1,190,9,11,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r1,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k5_r1,R3,146,7,3,1,0.14285714285714285,0.3333333333333333,assigned_gt
bl_rate_q975__ni_k5_r1,R4,118,22,3,2,0.09090909090909091,0.6666666666666666,assigned_gt
bl_rate_q975__ni_k5_r1,R5,134,27,7,1,0.037037037037037035,0.14285714285714285,assigned_gt
bl_rate_q975__ni_k5_r2,L1,252,45,6,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r2,L2,207,25,3,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r2,L3,110,20,0,0,0.0,,all_gt
bl_rate_q975__ni_k5_r2,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975__ni_k5_r2,L5,279,72,8,1,0.013888888888888888,0.125,all_gt
bl_rate_q975__ni_k5_r2,R1,252,71,11,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r2,R2,193,19,0,0,0.0,,all_gt
bl_rate_q975__ni_k5_r2,R3,160,21,3,1,0.047619047619047616,0.3333333333333333,all_gt
bl_rate_q975__ni_k5_r2,R4,136,40,3,2,0.05,0.6666666666666666,all_gt
bl_rate_q975__ni_k5_r2,R5,173,66,7,1,0.015151515151515152,0.14285714285714285,all_gt
bl_rate_q975__ni_k5_r2,L1,217,10,6,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r2,L2,188,6,3,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r2,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k5_r2,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975__ni_k5_r2,L5,225,18,8,1,0.05555555555555555,0.125,assigned_gt
bl_rate_q975__ni_k5_r2,R1,190,9,11,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r2,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k5_r2,R3,146,7,3,1,0.14285714285714285,0.3333333333333333,assigned_gt
bl_rate_q975__ni_k5_r2,R4,118,22,3,2,0.09090909090909091,0.6666666666666666,assigned_gt
bl_rate_q975__ni_k5_r2,R5,134,27,7,1,0.037037037037037035,0.14285714285714285,assigned_gt
bl_rate_q975__ni_k5_r4,L1,252,45,6,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r4,L2,207,25,3,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r4,L3,110,20,0,0,0.0,,all_gt
bl_rate_q975__ni_k5_r4,L4,38,13,1,1,0.07692307692307693,1.0,all_gt
bl_rate_q975__ni_k5_r4,L5,279,72,8,1,0.013888888888888888,0.125,all_gt
bl_rate_q975__ni_k5_r4,R1,252,71,11,0,0.0,0.0,all_gt
bl_rate_q975__ni_k5_r4,R2,193,19,0,0,0.0,,all_gt
bl_rate_q975__ni_k5_r4,R3,160,21,3,1,0.047619047619047616,0.3333333333333333,all_gt
bl_rate_q975__ni_k5_r4,R4,136,40,3,2,0.05,0.6666666666666666,all_gt
bl_rate_q975__ni_k5_r4,R5,173,66,7,1,0.015151515151515152,0.14285714285714285,all_gt
bl_rate_q975__ni_k5_r4,L1,217,10,6,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r4,L2,188,6,3,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r4,L3,95,5,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k5_r4,L4,30,5,1,1,0.2,1.0,assigned_gt
bl_rate_q975__ni_k5_r4,L5,225,18,8,1,0.05555555555555555,0.125,assigned_gt
bl_rate_q975__ni_k5_r4,R1,190,9,11,0,0.0,0.0,assigned_gt
bl_rate_q975__ni_k5_r4,R2,182,8,0,0,0.0,,assigned_gt
bl_rate_q975__ni_k5_r4,R3,146,7,3,1,0.14285714285714285,0.3333333333333333,assigned_gt
bl_rate_q975__ni_k5_r4,R4,118,22,3,2,0.09090909090909091,0.6666666666666666,assigned_gt
bl_rate_q975__ni_k5_r4,R5,134,27,7,1,0.037037037037037035,0.14285714285714285,assigned_gt
bl_hmm_disagreement__ni_k2_r1,L1,252,45,94,7,0.15555555555555556,0.07446808510638298,all_gt
bl_hmm_disagreement__ni_k2_r1,L2,207,25,158,4,0.16,0.02531645569620253,all_gt
bl_hmm_disagreement__ni_k2_r1,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k2_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k2_r1,L5,279,72,129,17,0.2361111111111111,0.13178294573643412,all_gt
bl_hmm_disagreement__ni_k2_r1,R1,252,71,89,4,0.056338028169014086,0.0449438202247191,all_gt
bl_hmm_disagreement__ni_k2_r1,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k2_r1,R3,160,21,128,5,0.23809523809523808,0.0390625,all_gt
bl_hmm_disagreement__ni_k2_r1,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
bl_hmm_disagreement__ni_k2_r1,R5,173,66,85,25,0.3787878787878788,0.29411764705882354,all_gt
bl_hmm_disagreement__ni_k2_r1,L1,217,10,94,7,0.7,0.07446808510638298,assigned_gt
bl_hmm_disagreement__ni_k2_r1,L2,188,6,158,4,0.6666666666666666,0.02531645569620253,assigned_gt
bl_hmm_disagreement__ni_k2_r1,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k2_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k2_r1,L5,225,18,129,17,0.9444444444444444,0.13178294573643412,assigned_gt
bl_hmm_disagreement__ni_k2_r1,R1,190,9,89,4,0.4444444444444444,0.0449438202247191,assigned_gt
bl_hmm_disagreement__ni_k2_r1,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k2_r1,R3,146,7,128,5,0.7142857142857143,0.0390625,assigned_gt
bl_hmm_disagreement__ni_k2_r1,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
bl_hmm_disagreement__ni_k2_r1,R5,134,27,85,25,0.9259259259259259,0.29411764705882354,assigned_gt
bl_hmm_disagreement__ni_k2_r2,L1,252,45,94,7,0.15555555555555556,0.07446808510638298,all_gt
bl_hmm_disagreement__ni_k2_r2,L2,207,25,158,4,0.16,0.02531645569620253,all_gt
bl_hmm_disagreement__ni_k2_r2,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k2_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k2_r2,L5,279,72,131,17,0.2361111111111111,0.1297709923664122,all_gt
bl_hmm_disagreement__ni_k2_r2,R1,252,71,94,5,0.07042253521126761,0.05319148936170213,all_gt
bl_hmm_disagreement__ni_k2_r2,R2,193,19,122,7,0.3684210526315789,0.05737704918032787,all_gt
bl_hmm_disagreement__ni_k2_r2,R3,160,21,128,5,0.23809523809523808,0.0390625,all_gt
bl_hmm_disagreement__ni_k2_r2,R4,136,40,110,19,0.475,0.17272727272727273,all_gt
bl_hmm_disagreement__ni_k2_r2,R5,173,66,88,25,0.3787878787878788,0.2840909090909091,all_gt
bl_hmm_disagreement__ni_k2_r2,L1,217,10,94,7,0.7,0.07446808510638298,assigned_gt
bl_hmm_disagreement__ni_k2_r2,L2,188,6,158,4,0.6666666666666666,0.02531645569620253,assigned_gt
bl_hmm_disagreement__ni_k2_r2,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k2_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k2_r2,L5,225,18,131,17,0.9444444444444444,0.1297709923664122,assigned_gt
bl_hmm_disagreement__ni_k2_r2,R1,190,9,94,5,0.5555555555555556,0.05319148936170213,assigned_gt
bl_hmm_disagreement__ni_k2_r2,R2,182,8,122,7,0.875,0.05737704918032787,assigned_gt
bl_hmm_disagreement__ni_k2_r2,R3,146,7,128,5,0.7142857142857143,0.0390625,assigned_gt
bl_hmm_disagreement__ni_k2_r2,R4,118,22,110,19,0.8636363636363636,0.17272727272727273,assigned_gt
bl_hmm_disagreement__ni_k2_r2,R5,134,27,88,25,0.9259259259259259,0.2840909090909091,assigned_gt
bl_hmm_disagreement__ni_k2_r4,L1,252,45,103,7,0.15555555555555556,0.06796116504854369,all_gt
bl_hmm_disagreement__ni_k2_r4,L2,207,25,158,4,0.16,0.02531645569620253,all_gt
bl_hmm_disagreement__ni_k2_r4,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k2_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k2_r4,L5,279,72,134,17,0.2361111111111111,0.12686567164179105,all_gt
bl_hmm_disagreement__ni_k2_r4,R1,252,71,99,5,0.07042253521126761,0.050505050505050504,all_gt
bl_hmm_disagreement__ni_k2_r4,R2,193,19,124,7,0.3684210526315789,0.056451612903225805,all_gt
bl_hmm_disagreement__ni_k2_r4,R3,160,21,130,5,0.23809523809523808,0.038461538461538464,all_gt
bl_hmm_disagreement__ni_k2_r4,R4,136,40,112,19,0.475,0.16964285714285715,all_gt
bl_hmm_disagreement__ni_k2_r4,R5,173,66,91,25,0.3787878787878788,0.27472527472527475,all_gt
bl_hmm_disagreement__ni_k2_r4,L1,217,10,103,7,0.7,0.06796116504854369,assigned_gt
bl_hmm_disagreement__ni_k2_r4,L2,188,6,158,4,0.6666666666666666,0.02531645569620253,assigned_gt
bl_hmm_disagreement__ni_k2_r4,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k2_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k2_r4,L5,225,18,134,17,0.9444444444444444,0.12686567164179105,assigned_gt
bl_hmm_disagreement__ni_k2_r4,R1,190,9,99,5,0.5555555555555556,0.050505050505050504,assigned_gt
bl_hmm_disagreement__ni_k2_r4,R2,182,8,124,7,0.875,0.056451612903225805,assigned_gt
bl_hmm_disagreement__ni_k2_r4,R3,146,7,130,5,0.7142857142857143,0.038461538461538464,assigned_gt
bl_hmm_disagreement__ni_k2_r4,R4,118,22,112,19,0.8636363636363636,0.16964285714285715,assigned_gt
bl_hmm_disagreement__ni_k2_r4,R5,134,27,91,25,0.9259259259259259,0.27472527472527475,assigned_gt
bl_hmm_disagreement__ni_k3_r1,L1,252,45,88,7,0.15555555555555556,0.07954545454545454,all_gt
bl_hmm_disagreement__ni_k3_r1,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement__ni_k3_r1,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k3_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k3_r1,L5,279,72,126,17,0.2361111111111111,0.1349206349206349,all_gt
bl_hmm_disagreement__ni_k3_r1,R1,252,71,82,4,0.056338028169014086,0.04878048780487805,all_gt
bl_hmm_disagreement__ni_k3_r1,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k3_r1,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement__ni_k3_r1,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
bl_hmm_disagreement__ni_k3_r1,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
bl_hmm_disagreement__ni_k3_r1,L1,217,10,88,7,0.7,0.07954545454545454,assigned_gt
bl_hmm_disagreement__ni_k3_r1,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement__ni_k3_r1,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k3_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k3_r1,L5,225,18,126,17,0.9444444444444444,0.1349206349206349,assigned_gt
bl_hmm_disagreement__ni_k3_r1,R1,190,9,82,4,0.4444444444444444,0.04878048780487805,assigned_gt
bl_hmm_disagreement__ni_k3_r1,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k3_r1,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement__ni_k3_r1,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
bl_hmm_disagreement__ni_k3_r1,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
bl_hmm_disagreement__ni_k3_r2,L1,252,45,90,7,0.15555555555555556,0.07777777777777778,all_gt
bl_hmm_disagreement__ni_k3_r2,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement__ni_k3_r2,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k3_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k3_r2,L5,279,72,126,17,0.2361111111111111,0.1349206349206349,all_gt
bl_hmm_disagreement__ni_k3_r2,R1,252,71,84,5,0.07042253521126761,0.05952380952380952,all_gt
bl_hmm_disagreement__ni_k3_r2,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k3_r2,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement__ni_k3_r2,R4,136,40,110,19,0.475,0.17272727272727273,all_gt
bl_hmm_disagreement__ni_k3_r2,R5,173,66,84,25,0.3787878787878788,0.2976190476190476,all_gt
bl_hmm_disagreement__ni_k3_r2,L1,217,10,90,7,0.7,0.07777777777777778,assigned_gt
bl_hmm_disagreement__ni_k3_r2,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement__ni_k3_r2,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k3_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k3_r2,L5,225,18,126,17,0.9444444444444444,0.1349206349206349,assigned_gt
bl_hmm_disagreement__ni_k3_r2,R1,190,9,84,5,0.5555555555555556,0.05952380952380952,assigned_gt
bl_hmm_disagreement__ni_k3_r2,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k3_r2,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement__ni_k3_r2,R4,118,22,110,19,0.8636363636363636,0.17272727272727273,assigned_gt
bl_hmm_disagreement__ni_k3_r2,R5,134,27,84,25,0.9259259259259259,0.2976190476190476,assigned_gt
bl_hmm_disagreement__ni_k3_r4,L1,252,45,93,7,0.15555555555555556,0.07526881720430108,all_gt
bl_hmm_disagreement__ni_k3_r4,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement__ni_k3_r4,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k3_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k3_r4,L5,279,72,128,17,0.2361111111111111,0.1328125,all_gt
bl_hmm_disagreement__ni_k3_r4,R1,252,71,84,5,0.07042253521126761,0.05952380952380952,all_gt
bl_hmm_disagreement__ni_k3_r4,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k3_r4,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement__ni_k3_r4,R4,136,40,111,19,0.475,0.17117117117117117,all_gt
bl_hmm_disagreement__ni_k3_r4,R5,173,66,84,25,0.3787878787878788,0.2976190476190476,all_gt
bl_hmm_disagreement__ni_k3_r4,L1,217,10,93,7,0.7,0.07526881720430108,assigned_gt
bl_hmm_disagreement__ni_k3_r4,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement__ni_k3_r4,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k3_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k3_r4,L5,225,18,128,17,0.9444444444444444,0.1328125,assigned_gt
bl_hmm_disagreement__ni_k3_r4,R1,190,9,84,5,0.5555555555555556,0.05952380952380952,assigned_gt
bl_hmm_disagreement__ni_k3_r4,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k3_r4,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement__ni_k3_r4,R4,118,22,111,19,0.8636363636363636,0.17117117117117117,assigned_gt
bl_hmm_disagreement__ni_k3_r4,R5,134,27,84,25,0.9259259259259259,0.2976190476190476,assigned_gt
bl_hmm_disagreement__ni_k5_r1,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
bl_hmm_disagreement__ni_k5_r1,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement__ni_k5_r1,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k5_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k5_r1,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
bl_hmm_disagreement__ni_k5_r1,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
bl_hmm_disagreement__ni_k5_r1,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k5_r1,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement__ni_k5_r1,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
bl_hmm_disagreement__ni_k5_r1,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
bl_hmm_disagreement__ni_k5_r1,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
bl_hmm_disagreement__ni_k5_r1,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement__ni_k5_r1,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k5_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k5_r1,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
bl_hmm_disagreement__ni_k5_r1,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
bl_hmm_disagreement__ni_k5_r1,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k5_r1,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement__ni_k5_r1,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
bl_hmm_disagreement__ni_k5_r1,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
bl_hmm_disagreement__ni_k5_r2,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
bl_hmm_disagreement__ni_k5_r2,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement__ni_k5_r2,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k5_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k5_r2,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
bl_hmm_disagreement__ni_k5_r2,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
bl_hmm_disagreement__ni_k5_r2,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k5_r2,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement__ni_k5_r2,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
bl_hmm_disagreement__ni_k5_r2,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
bl_hmm_disagreement__ni_k5_r2,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
bl_hmm_disagreement__ni_k5_r2,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement__ni_k5_r2,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k5_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k5_r2,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
bl_hmm_disagreement__ni_k5_r2,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
bl_hmm_disagreement__ni_k5_r2,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k5_r2,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement__ni_k5_r2,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
bl_hmm_disagreement__ni_k5_r2,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
bl_hmm_disagreement__ni_k5_r4,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
bl_hmm_disagreement__ni_k5_r4,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
bl_hmm_disagreement__ni_k5_r4,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
bl_hmm_disagreement__ni_k5_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
bl_hmm_disagreement__ni_k5_r4,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
bl_hmm_disagreement__ni_k5_r4,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
bl_hmm_disagreement__ni_k5_r4,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
bl_hmm_disagreement__ni_k5_r4,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
bl_hmm_disagreement__ni_k5_r4,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
bl_hmm_disagreement__ni_k5_r4,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
bl_hmm_disagreement__ni_k5_r4,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
bl_hmm_disagreement__ni_k5_r4,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
bl_hmm_disagreement__ni_k5_r4,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
bl_hmm_disagreement__ni_k5_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
bl_hmm_disagreement__ni_k5_r4,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
bl_hmm_disagreement__ni_k5_r4,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
bl_hmm_disagreement__ni_k5_r4,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
bl_hmm_disagreement__ni_k5_r4,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
bl_hmm_disagreement__ni_k5_r4,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
bl_hmm_disagreement__ni_k5_r4,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
bl_practical_or_rate995__ni_k2_r1,L1,252,45,25,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r1,L2,207,25,15,1,0.04,0.06666666666666667,all_gt
bl_practical_or_rate995__ni_k2_r1,L3,110,20,8,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r1,L5,279,72,64,6,0.08333333333333333,0.09375,all_gt
bl_practical_or_rate995__ni_k2_r1,R1,252,71,21,1,0.014084507042253521,0.047619047619047616,all_gt
bl_practical_or_rate995__ni_k2_r1,R2,193,19,16,1,0.05263157894736842,0.0625,all_gt
bl_practical_or_rate995__ni_k2_r1,R3,160,21,13,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r1,R4,136,40,14,1,0.025,0.07142857142857142,all_gt
bl_practical_or_rate995__ni_k2_r1,R5,173,66,22,10,0.15151515151515152,0.45454545454545453,all_gt
bl_practical_or_rate995__ni_k2_r1,L1,217,10,25,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r1,L2,188,6,15,1,0.16666666666666666,0.06666666666666667,assigned_gt
bl_practical_or_rate995__ni_k2_r1,L3,95,5,8,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r1,L5,225,18,64,6,0.3333333333333333,0.09375,assigned_gt
bl_practical_or_rate995__ni_k2_r1,R1,190,9,21,1,0.1111111111111111,0.047619047619047616,assigned_gt
bl_practical_or_rate995__ni_k2_r1,R2,182,8,16,1,0.125,0.0625,assigned_gt
bl_practical_or_rate995__ni_k2_r1,R3,146,7,13,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r1,R4,118,22,14,1,0.045454545454545456,0.07142857142857142,assigned_gt
bl_practical_or_rate995__ni_k2_r1,R5,134,27,22,10,0.37037037037037035,0.45454545454545453,assigned_gt
bl_practical_or_rate995__ni_k2_r2,L1,252,45,29,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r2,L2,207,25,23,1,0.04,0.043478260869565216,all_gt
bl_practical_or_rate995__ni_k2_r2,L3,110,20,13,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r2,L4,38,13,5,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r2,L5,279,72,67,7,0.09722222222222222,0.1044776119402985,all_gt
bl_practical_or_rate995__ni_k2_r2,R1,252,71,28,3,0.04225352112676056,0.10714285714285714,all_gt
bl_practical_or_rate995__ni_k2_r2,R2,193,19,22,1,0.05263157894736842,0.045454545454545456,all_gt
bl_practical_or_rate995__ni_k2_r2,R3,160,21,18,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r2,R4,136,40,19,3,0.075,0.15789473684210525,all_gt
bl_practical_or_rate995__ni_k2_r2,R5,173,66,26,10,0.15151515151515152,0.38461538461538464,all_gt
bl_practical_or_rate995__ni_k2_r2,L1,217,10,29,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r2,L2,188,6,23,1,0.16666666666666666,0.043478260869565216,assigned_gt
bl_practical_or_rate995__ni_k2_r2,L3,95,5,13,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r2,L5,225,18,67,7,0.3888888888888889,0.1044776119402985,assigned_gt
bl_practical_or_rate995__ni_k2_r2,R1,190,9,28,3,0.3333333333333333,0.10714285714285714,assigned_gt
bl_practical_or_rate995__ni_k2_r2,R2,182,8,22,1,0.125,0.045454545454545456,assigned_gt
bl_practical_or_rate995__ni_k2_r2,R3,146,7,18,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r2,R4,118,22,19,3,0.13636363636363635,0.15789473684210525,assigned_gt
bl_practical_or_rate995__ni_k2_r2,R5,134,27,26,10,0.37037037037037035,0.38461538461538464,assigned_gt
bl_practical_or_rate995__ni_k2_r4,L1,252,45,44,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r4,L2,207,25,34,1,0.04,0.029411764705882353,all_gt
bl_practical_or_rate995__ni_k2_r4,L3,110,20,15,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r4,L4,38,13,6,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r4,L5,279,72,77,8,0.1111111111111111,0.1038961038961039,all_gt
bl_practical_or_rate995__ni_k2_r4,R1,252,71,34,3,0.04225352112676056,0.08823529411764706,all_gt
bl_practical_or_rate995__ni_k2_r4,R2,193,19,28,1,0.05263157894736842,0.03571428571428571,all_gt
bl_practical_or_rate995__ni_k2_r4,R3,160,21,28,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k2_r4,R4,136,40,33,7,0.175,0.21212121212121213,all_gt
bl_practical_or_rate995__ni_k2_r4,R5,173,66,36,11,0.16666666666666666,0.3055555555555556,all_gt
bl_practical_or_rate995__ni_k2_r4,L1,217,10,44,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r4,L2,188,6,34,1,0.16666666666666666,0.029411764705882353,assigned_gt
bl_practical_or_rate995__ni_k2_r4,L3,95,5,15,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r4,L4,30,5,6,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r4,L5,225,18,77,8,0.4444444444444444,0.1038961038961039,assigned_gt
bl_practical_or_rate995__ni_k2_r4,R1,190,9,34,3,0.3333333333333333,0.08823529411764706,assigned_gt
bl_practical_or_rate995__ni_k2_r4,R2,182,8,28,1,0.125,0.03571428571428571,assigned_gt
bl_practical_or_rate995__ni_k2_r4,R3,146,7,28,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k2_r4,R4,118,22,33,7,0.3181818181818182,0.21212121212121213,assigned_gt
bl_practical_or_rate995__ni_k2_r4,R5,134,27,36,11,0.4074074074074074,0.3055555555555556,assigned_gt
bl_practical_or_rate995__ni_k3_r1,L1,252,45,16,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r1,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r1,L5,279,72,59,6,0.08333333333333333,0.1016949152542373,all_gt
bl_practical_or_rate995__ni_k3_r1,R1,252,71,11,1,0.014084507042253521,0.09090909090909091,all_gt
bl_practical_or_rate995__ni_k3_r1,R2,193,19,12,1,0.05263157894736842,0.08333333333333333,all_gt
bl_practical_or_rate995__ni_k3_r1,R3,160,21,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r1,R4,136,40,12,1,0.025,0.08333333333333333,all_gt
bl_practical_or_rate995__ni_k3_r1,R5,173,66,16,7,0.10606060606060606,0.4375,all_gt
bl_practical_or_rate995__ni_k3_r1,L1,217,10,16,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r1,L5,225,18,59,6,0.3333333333333333,0.1016949152542373,assigned_gt
bl_practical_or_rate995__ni_k3_r1,R1,190,9,11,1,0.1111111111111111,0.09090909090909091,assigned_gt
bl_practical_or_rate995__ni_k3_r1,R2,182,8,12,1,0.125,0.08333333333333333,assigned_gt
bl_practical_or_rate995__ni_k3_r1,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r1,R4,118,22,12,1,0.045454545454545456,0.08333333333333333,assigned_gt
bl_practical_or_rate995__ni_k3_r1,R5,134,27,16,7,0.25925925925925924,0.4375,assigned_gt
bl_practical_or_rate995__ni_k3_r2,L1,252,45,19,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r2,L2,207,25,11,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r2,L3,110,20,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r2,L5,279,72,59,6,0.08333333333333333,0.1016949152542373,all_gt
bl_practical_or_rate995__ni_k3_r2,R1,252,71,14,2,0.028169014084507043,0.14285714285714285,all_gt
bl_practical_or_rate995__ni_k3_r2,R2,193,19,13,1,0.05263157894736842,0.07692307692307693,all_gt
bl_practical_or_rate995__ni_k3_r2,R3,160,21,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r2,R4,136,40,13,2,0.05,0.15384615384615385,all_gt
bl_practical_or_rate995__ni_k3_r2,R5,173,66,19,7,0.10606060606060606,0.3684210526315789,all_gt
bl_practical_or_rate995__ni_k3_r2,L1,217,10,19,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r2,L2,188,6,11,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r2,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r2,L5,225,18,59,6,0.3333333333333333,0.1016949152542373,assigned_gt
bl_practical_or_rate995__ni_k3_r2,R1,190,9,14,2,0.2222222222222222,0.14285714285714285,assigned_gt
bl_practical_or_rate995__ni_k3_r2,R2,182,8,13,1,0.125,0.07692307692307693,assigned_gt
bl_practical_or_rate995__ni_k3_r2,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r2,R4,118,22,13,2,0.09090909090909091,0.15384615384615385,assigned_gt
bl_practical_or_rate995__ni_k3_r2,R5,134,27,19,7,0.25925925925925924,0.3684210526315789,assigned_gt
bl_practical_or_rate995__ni_k3_r4,L1,252,45,24,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r4,L2,207,25,15,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r4,L4,38,13,5,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r4,L5,279,72,61,6,0.08333333333333333,0.09836065573770492,all_gt
bl_practical_or_rate995__ni_k3_r4,R1,252,71,15,2,0.028169014084507043,0.13333333333333333,all_gt
bl_practical_or_rate995__ni_k3_r4,R2,193,19,14,1,0.05263157894736842,0.07142857142857142,all_gt
bl_practical_or_rate995__ni_k3_r4,R3,160,21,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k3_r4,R4,136,40,20,5,0.125,0.25,all_gt
bl_practical_or_rate995__ni_k3_r4,R5,173,66,22,8,0.12121212121212122,0.36363636363636365,all_gt
bl_practical_or_rate995__ni_k3_r4,L1,217,10,24,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r4,L2,188,6,15,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r4,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r4,L5,225,18,61,6,0.3333333333333333,0.09836065573770492,assigned_gt
bl_practical_or_rate995__ni_k3_r4,R1,190,9,15,2,0.2222222222222222,0.13333333333333333,assigned_gt
bl_practical_or_rate995__ni_k3_r4,R2,182,8,14,1,0.125,0.07142857142857142,assigned_gt
bl_practical_or_rate995__ni_k3_r4,R3,146,7,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k3_r4,R4,118,22,20,5,0.22727272727272727,0.25,assigned_gt
bl_practical_or_rate995__ni_k3_r4,R5,134,27,22,8,0.2962962962962963,0.36363636363636365,assigned_gt
bl_practical_or_rate995__ni_k5_r1,L1,252,45,15,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r1,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r1,L5,279,72,57,6,0.08333333333333333,0.10526315789473684,all_gt
bl_practical_or_rate995__ni_k5_r1,R1,252,71,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r1,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_rate995__ni_k5_r1,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r1,R4,136,40,10,1,0.025,0.1,all_gt
bl_practical_or_rate995__ni_k5_r1,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_practical_or_rate995__ni_k5_r1,L1,217,10,15,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r1,L5,225,18,57,6,0.3333333333333333,0.10526315789473684,assigned_gt
bl_practical_or_rate995__ni_k5_r1,R1,190,9,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r1,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_rate995__ni_k5_r1,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r1,R4,118,22,10,1,0.045454545454545456,0.1,assigned_gt
bl_practical_or_rate995__ni_k5_r1,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
bl_practical_or_rate995__ni_k5_r2,L1,252,45,15,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r2,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r2,L5,279,72,57,6,0.08333333333333333,0.10526315789473684,all_gt
bl_practical_or_rate995__ni_k5_r2,R1,252,71,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r2,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_rate995__ni_k5_r2,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r2,R4,136,40,10,1,0.025,0.1,all_gt
bl_practical_or_rate995__ni_k5_r2,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_practical_or_rate995__ni_k5_r2,L1,217,10,15,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r2,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r2,L5,225,18,57,6,0.3333333333333333,0.10526315789473684,assigned_gt
bl_practical_or_rate995__ni_k5_r2,R1,190,9,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r2,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_rate995__ni_k5_r2,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r2,R4,118,22,10,1,0.045454545454545456,0.1,assigned_gt
bl_practical_or_rate995__ni_k5_r2,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
bl_practical_or_rate995__ni_k5_r4,L1,252,45,15,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r4,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r4,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r4,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r4,L5,279,72,57,6,0.08333333333333333,0.10526315789473684,all_gt
bl_practical_or_rate995__ni_k5_r4,R1,252,71,7,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r4,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_rate995__ni_k5_r4,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_rate995__ni_k5_r4,R4,136,40,10,1,0.025,0.1,all_gt
bl_practical_or_rate995__ni_k5_r4,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_practical_or_rate995__ni_k5_r4,L1,217,10,15,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r4,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r4,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r4,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r4,L5,225,18,57,6,0.3333333333333333,0.10526315789473684,assigned_gt
bl_practical_or_rate995__ni_k5_r4,R1,190,9,7,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r4,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_rate995__ni_k5_r4,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_rate995__ni_k5_r4,R4,118,22,10,1,0.045454545454545456,0.1,assigned_gt
bl_practical_or_rate995__ni_k5_r4,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
bl_practical_or_crossing__ni_k2_r1,L1,252,45,23,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r1,L2,207,25,15,1,0.04,0.06666666666666667,all_gt
bl_practical_or_crossing__ni_k2_r1,L3,110,20,8,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r1,L5,279,72,63,6,0.08333333333333333,0.09523809523809523,all_gt
bl_practical_or_crossing__ni_k2_r1,R1,252,71,18,1,0.014084507042253521,0.05555555555555555,all_gt
bl_practical_or_crossing__ni_k2_r1,R2,193,19,16,1,0.05263157894736842,0.0625,all_gt
bl_practical_or_crossing__ni_k2_r1,R3,160,21,13,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r1,R4,136,40,13,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r1,R5,173,66,21,9,0.13636363636363635,0.42857142857142855,all_gt
bl_practical_or_crossing__ni_k2_r1,L1,217,10,23,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r1,L2,188,6,15,1,0.16666666666666666,0.06666666666666667,assigned_gt
bl_practical_or_crossing__ni_k2_r1,L3,95,5,8,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r1,L5,225,18,63,6,0.3333333333333333,0.09523809523809523,assigned_gt
bl_practical_or_crossing__ni_k2_r1,R1,190,9,18,1,0.1111111111111111,0.05555555555555555,assigned_gt
bl_practical_or_crossing__ni_k2_r1,R2,182,8,16,1,0.125,0.0625,assigned_gt
bl_practical_or_crossing__ni_k2_r1,R3,146,7,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r1,R4,118,22,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r1,R5,134,27,21,9,0.3333333333333333,0.42857142857142855,assigned_gt
bl_practical_or_crossing__ni_k2_r2,L1,252,45,27,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r2,L2,207,25,23,1,0.04,0.043478260869565216,all_gt
bl_practical_or_crossing__ni_k2_r2,L3,110,20,13,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r2,L4,38,13,5,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r2,L5,279,72,66,7,0.09722222222222222,0.10606060606060606,all_gt
bl_practical_or_crossing__ni_k2_r2,R1,252,71,25,3,0.04225352112676056,0.12,all_gt
bl_practical_or_crossing__ni_k2_r2,R2,193,19,22,1,0.05263157894736842,0.045454545454545456,all_gt
bl_practical_or_crossing__ni_k2_r2,R3,160,21,18,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r2,R4,136,40,18,2,0.05,0.1111111111111111,all_gt
bl_practical_or_crossing__ni_k2_r2,R5,173,66,26,10,0.15151515151515152,0.38461538461538464,all_gt
bl_practical_or_crossing__ni_k2_r2,L1,217,10,27,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r2,L2,188,6,23,1,0.16666666666666666,0.043478260869565216,assigned_gt
bl_practical_or_crossing__ni_k2_r2,L3,95,5,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r2,L5,225,18,66,7,0.3888888888888889,0.10606060606060606,assigned_gt
bl_practical_or_crossing__ni_k2_r2,R1,190,9,25,3,0.3333333333333333,0.12,assigned_gt
bl_practical_or_crossing__ni_k2_r2,R2,182,8,22,1,0.125,0.045454545454545456,assigned_gt
bl_practical_or_crossing__ni_k2_r2,R3,146,7,18,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r2,R4,118,22,18,2,0.09090909090909091,0.1111111111111111,assigned_gt
bl_practical_or_crossing__ni_k2_r2,R5,134,27,26,10,0.37037037037037035,0.38461538461538464,assigned_gt
bl_practical_or_crossing__ni_k2_r4,L1,252,45,42,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r4,L2,207,25,34,1,0.04,0.029411764705882353,all_gt
bl_practical_or_crossing__ni_k2_r4,L3,110,20,15,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r4,L4,38,13,6,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r4,L5,279,72,76,8,0.1111111111111111,0.10526315789473684,all_gt
bl_practical_or_crossing__ni_k2_r4,R1,252,71,31,3,0.04225352112676056,0.0967741935483871,all_gt
bl_practical_or_crossing__ni_k2_r4,R2,193,19,28,1,0.05263157894736842,0.03571428571428571,all_gt
bl_practical_or_crossing__ni_k2_r4,R3,160,21,28,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k2_r4,R4,136,40,32,6,0.15,0.1875,all_gt
bl_practical_or_crossing__ni_k2_r4,R5,173,66,36,11,0.16666666666666666,0.3055555555555556,all_gt
bl_practical_or_crossing__ni_k2_r4,L1,217,10,42,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r4,L2,188,6,34,1,0.16666666666666666,0.029411764705882353,assigned_gt
bl_practical_or_crossing__ni_k2_r4,L3,95,5,15,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r4,L4,30,5,6,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r4,L5,225,18,76,8,0.4444444444444444,0.10526315789473684,assigned_gt
bl_practical_or_crossing__ni_k2_r4,R1,190,9,31,3,0.3333333333333333,0.0967741935483871,assigned_gt
bl_practical_or_crossing__ni_k2_r4,R2,182,8,28,1,0.125,0.03571428571428571,assigned_gt
bl_practical_or_crossing__ni_k2_r4,R3,146,7,28,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k2_r4,R4,118,22,32,6,0.2727272727272727,0.1875,assigned_gt
bl_practical_or_crossing__ni_k2_r4,R5,134,27,36,11,0.4074074074074074,0.3055555555555556,assigned_gt
bl_practical_or_crossing__ni_k3_r1,L1,252,45,14,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r1,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r1,L5,279,72,58,6,0.08333333333333333,0.10344827586206896,all_gt
bl_practical_or_crossing__ni_k3_r1,R1,252,71,8,1,0.014084507042253521,0.125,all_gt
bl_practical_or_crossing__ni_k3_r1,R2,193,19,12,1,0.05263157894736842,0.08333333333333333,all_gt
bl_practical_or_crossing__ni_k3_r1,R3,160,21,7,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r1,R4,136,40,11,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r1,R5,173,66,15,6,0.09090909090909091,0.4,all_gt
bl_practical_or_crossing__ni_k3_r1,L1,217,10,14,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r1,L5,225,18,58,6,0.3333333333333333,0.10344827586206896,assigned_gt
bl_practical_or_crossing__ni_k3_r1,R1,190,9,8,1,0.1111111111111111,0.125,assigned_gt
bl_practical_or_crossing__ni_k3_r1,R2,182,8,12,1,0.125,0.08333333333333333,assigned_gt
bl_practical_or_crossing__ni_k3_r1,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r1,R4,118,22,11,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r1,R5,134,27,15,6,0.2222222222222222,0.4,assigned_gt
bl_practical_or_crossing__ni_k3_r2,L1,252,45,17,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r2,L2,207,25,11,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r2,L3,110,20,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r2,L5,279,72,58,6,0.08333333333333333,0.10344827586206896,all_gt
bl_practical_or_crossing__ni_k3_r2,R1,252,71,11,2,0.028169014084507043,0.18181818181818182,all_gt
bl_practical_or_crossing__ni_k3_r2,R2,193,19,13,1,0.05263157894736842,0.07692307692307693,all_gt
bl_practical_or_crossing__ni_k3_r2,R3,160,21,7,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r2,R4,136,40,12,1,0.025,0.08333333333333333,all_gt
bl_practical_or_crossing__ni_k3_r2,R5,173,66,18,6,0.09090909090909091,0.3333333333333333,all_gt
bl_practical_or_crossing__ni_k3_r2,L1,217,10,17,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r2,L2,188,6,11,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r2,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r2,L5,225,18,58,6,0.3333333333333333,0.10344827586206896,assigned_gt
bl_practical_or_crossing__ni_k3_r2,R1,190,9,11,2,0.2222222222222222,0.18181818181818182,assigned_gt
bl_practical_or_crossing__ni_k3_r2,R2,182,8,13,1,0.125,0.07692307692307693,assigned_gt
bl_practical_or_crossing__ni_k3_r2,R3,146,7,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r2,R4,118,22,12,1,0.045454545454545456,0.08333333333333333,assigned_gt
bl_practical_or_crossing__ni_k3_r2,R5,134,27,18,6,0.2222222222222222,0.3333333333333333,assigned_gt
bl_practical_or_crossing__ni_k3_r4,L1,252,45,22,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r4,L2,207,25,15,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r4,L4,38,13,5,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r4,L5,279,72,60,6,0.08333333333333333,0.1,all_gt
bl_practical_or_crossing__ni_k3_r4,R1,252,71,12,2,0.028169014084507043,0.16666666666666666,all_gt
bl_practical_or_crossing__ni_k3_r4,R2,193,19,14,1,0.05263157894736842,0.07142857142857142,all_gt
bl_practical_or_crossing__ni_k3_r4,R3,160,21,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k3_r4,R4,136,40,19,4,0.1,0.21052631578947367,all_gt
bl_practical_or_crossing__ni_k3_r4,R5,173,66,21,7,0.10606060606060606,0.3333333333333333,all_gt
bl_practical_or_crossing__ni_k3_r4,L1,217,10,22,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r4,L2,188,6,15,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r4,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r4,L5,225,18,60,6,0.3333333333333333,0.1,assigned_gt
bl_practical_or_crossing__ni_k3_r4,R1,190,9,12,2,0.2222222222222222,0.16666666666666666,assigned_gt
bl_practical_or_crossing__ni_k3_r4,R2,182,8,14,1,0.125,0.07142857142857142,assigned_gt
bl_practical_or_crossing__ni_k3_r4,R3,146,7,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k3_r4,R4,118,22,19,4,0.18181818181818182,0.21052631578947367,assigned_gt
bl_practical_or_crossing__ni_k3_r4,R5,134,27,21,7,0.25925925925925924,0.3333333333333333,assigned_gt
bl_practical_or_crossing__ni_k5_r1,L1,252,45,13,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_practical_or_crossing__ni_k5_r1,R1,252,71,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_crossing__ni_k5_r1,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,R4,136,40,9,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r1,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_practical_or_crossing__ni_k5_r1,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_practical_or_crossing__ni_k5_r1,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_crossing__ni_k5_r1,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r1,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_practical_or_crossing__ni_k5_r2,L1,252,45,13,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_practical_or_crossing__ni_k5_r2,R1,252,71,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_crossing__ni_k5_r2,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,R4,136,40,9,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r2,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_practical_or_crossing__ni_k5_r2,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_practical_or_crossing__ni_k5_r2,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_crossing__ni_k5_r2,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r2,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_practical_or_crossing__ni_k5_r4,L1,252,45,13,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,L2,207,25,10,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,L3,110,20,7,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,L4,38,13,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,L5,279,72,56,6,0.08333333333333333,0.10714285714285714,all_gt
bl_practical_or_crossing__ni_k5_r4,R1,252,71,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
bl_practical_or_crossing__ni_k5_r4,R3,160,21,4,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,R4,136,40,9,0,0.0,0.0,all_gt
bl_practical_or_crossing__ni_k5_r4,R5,173,66,13,6,0.09090909090909091,0.46153846153846156,all_gt
bl_practical_or_crossing__ni_k5_r4,L1,217,10,13,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,L5,225,18,56,6,0.3333333333333333,0.10714285714285714,assigned_gt
bl_practical_or_crossing__ni_k5_r4,R1,190,9,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
bl_practical_or_crossing__ni_k5_r4,R3,146,7,4,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_practical_or_crossing__ni_k5_r4,R5,134,27,13,6,0.2222222222222222,0.46153846153846156,assigned_gt
bl_two_signal_strict__ni_k2_r1,L1,252,45,11,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r1,L2,207,25,14,1,0.04,0.07142857142857142,all_gt
bl_two_signal_strict__ni_k2_r1,L3,110,20,8,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r1,L5,279,72,35,7,0.09722222222222222,0.2,all_gt
bl_two_signal_strict__ni_k2_r1,R1,252,71,15,1,0.014084507042253521,0.06666666666666667,all_gt
bl_two_signal_strict__ni_k2_r1,R2,193,19,12,1,0.05263157894736842,0.08333333333333333,all_gt
bl_two_signal_strict__ni_k2_r1,R3,160,21,14,1,0.047619047619047616,0.07142857142857142,all_gt
bl_two_signal_strict__ni_k2_r1,R4,136,40,13,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r1,R5,173,66,23,10,0.15151515151515152,0.43478260869565216,all_gt
bl_two_signal_strict__ni_k2_r1,L1,217,10,11,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r1,L2,188,6,14,1,0.16666666666666666,0.07142857142857142,assigned_gt
bl_two_signal_strict__ni_k2_r1,L3,95,5,8,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r1,L5,225,18,35,7,0.3888888888888889,0.2,assigned_gt
bl_two_signal_strict__ni_k2_r1,R1,190,9,15,1,0.1111111111111111,0.06666666666666667,assigned_gt
bl_two_signal_strict__ni_k2_r1,R2,182,8,12,1,0.125,0.08333333333333333,assigned_gt
bl_two_signal_strict__ni_k2_r1,R3,146,7,14,1,0.14285714285714285,0.07142857142857142,assigned_gt
bl_two_signal_strict__ni_k2_r1,R4,118,22,13,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r1,R5,134,27,23,10,0.37037037037037035,0.43478260869565216,assigned_gt
bl_two_signal_strict__ni_k2_r2,L1,252,45,15,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r2,L2,207,25,22,1,0.04,0.045454545454545456,all_gt
bl_two_signal_strict__ni_k2_r2,L3,110,20,13,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r2,L4,38,13,5,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r2,L5,279,72,39,8,0.1111111111111111,0.20512820512820512,all_gt
bl_two_signal_strict__ni_k2_r2,R1,252,71,22,3,0.04225352112676056,0.13636363636363635,all_gt
bl_two_signal_strict__ni_k2_r2,R2,193,19,19,1,0.05263157894736842,0.05263157894736842,all_gt
bl_two_signal_strict__ni_k2_r2,R3,160,21,19,1,0.047619047619047616,0.05263157894736842,all_gt
bl_two_signal_strict__ni_k2_r2,R4,136,40,18,2,0.05,0.1111111111111111,all_gt
bl_two_signal_strict__ni_k2_r2,R5,173,66,27,10,0.15151515151515152,0.37037037037037035,all_gt
bl_two_signal_strict__ni_k2_r2,L1,217,10,15,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r2,L2,188,6,22,1,0.16666666666666666,0.045454545454545456,assigned_gt
bl_two_signal_strict__ni_k2_r2,L3,95,5,13,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r2,L5,225,18,39,8,0.4444444444444444,0.20512820512820512,assigned_gt
bl_two_signal_strict__ni_k2_r2,R1,190,9,22,3,0.3333333333333333,0.13636363636363635,assigned_gt
bl_two_signal_strict__ni_k2_r2,R2,182,8,19,1,0.125,0.05263157894736842,assigned_gt
bl_two_signal_strict__ni_k2_r2,R3,146,7,19,1,0.14285714285714285,0.05263157894736842,assigned_gt
bl_two_signal_strict__ni_k2_r2,R4,118,22,18,2,0.09090909090909091,0.1111111111111111,assigned_gt
bl_two_signal_strict__ni_k2_r2,R5,134,27,27,10,0.37037037037037035,0.37037037037037035,assigned_gt
bl_two_signal_strict__ni_k2_r4,L1,252,45,30,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r4,L2,207,25,33,1,0.04,0.030303030303030304,all_gt
bl_two_signal_strict__ni_k2_r4,L3,110,20,15,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r4,L4,38,13,6,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k2_r4,L5,279,72,51,9,0.125,0.17647058823529413,all_gt
bl_two_signal_strict__ni_k2_r4,R1,252,71,29,3,0.04225352112676056,0.10344827586206896,all_gt
bl_two_signal_strict__ni_k2_r4,R2,193,19,25,1,0.05263157894736842,0.04,all_gt
bl_two_signal_strict__ni_k2_r4,R3,160,21,29,1,0.047619047619047616,0.034482758620689655,all_gt
bl_two_signal_strict__ni_k2_r4,R4,136,40,32,6,0.15,0.1875,all_gt
bl_two_signal_strict__ni_k2_r4,R5,173,66,36,11,0.16666666666666666,0.3055555555555556,all_gt
bl_two_signal_strict__ni_k2_r4,L1,217,10,30,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r4,L2,188,6,33,1,0.16666666666666666,0.030303030303030304,assigned_gt
bl_two_signal_strict__ni_k2_r4,L3,95,5,15,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r4,L4,30,5,6,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k2_r4,L5,225,18,51,9,0.5,0.17647058823529413,assigned_gt
bl_two_signal_strict__ni_k2_r4,R1,190,9,29,3,0.3333333333333333,0.10344827586206896,assigned_gt
bl_two_signal_strict__ni_k2_r4,R2,182,8,25,1,0.125,0.04,assigned_gt
bl_two_signal_strict__ni_k2_r4,R3,146,7,29,1,0.14285714285714285,0.034482758620689655,assigned_gt
bl_two_signal_strict__ni_k2_r4,R4,118,22,32,6,0.2727272727272727,0.1875,assigned_gt
bl_two_signal_strict__ni_k2_r4,R5,134,27,36,11,0.4074074074074074,0.3055555555555556,assigned_gt
bl_two_signal_strict__ni_k3_r1,L1,252,45,2,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r1,L2,207,25,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r1,L5,279,72,30,7,0.09722222222222222,0.23333333333333334,all_gt
bl_two_signal_strict__ni_k3_r1,R1,252,71,5,1,0.014084507042253521,0.2,all_gt
bl_two_signal_strict__ni_k3_r1,R2,193,19,8,1,0.05263157894736842,0.125,all_gt
bl_two_signal_strict__ni_k3_r1,R3,160,21,8,1,0.047619047619047616,0.125,all_gt
bl_two_signal_strict__ni_k3_r1,R4,136,40,11,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r1,R5,173,66,16,7,0.10606060606060606,0.4375,all_gt
bl_two_signal_strict__ni_k3_r1,L1,217,10,2,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r1,L2,188,6,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r1,L5,225,18,30,7,0.3888888888888889,0.23333333333333334,assigned_gt
bl_two_signal_strict__ni_k3_r1,R1,190,9,5,1,0.1111111111111111,0.2,assigned_gt
bl_two_signal_strict__ni_k3_r1,R2,182,8,8,1,0.125,0.125,assigned_gt
bl_two_signal_strict__ni_k3_r1,R3,146,7,8,1,0.14285714285714285,0.125,assigned_gt
bl_two_signal_strict__ni_k3_r1,R4,118,22,11,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r1,R5,134,27,16,7,0.25925925925925924,0.4375,assigned_gt
bl_two_signal_strict__ni_k3_r2,L1,252,45,5,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r2,L2,207,25,10,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r2,L3,110,20,10,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r2,L5,279,72,30,7,0.09722222222222222,0.23333333333333334,all_gt
bl_two_signal_strict__ni_k3_r2,R1,252,71,8,2,0.028169014084507043,0.25,all_gt
bl_two_signal_strict__ni_k3_r2,R2,193,19,9,1,0.05263157894736842,0.1111111111111111,all_gt
bl_two_signal_strict__ni_k3_r2,R3,160,21,8,1,0.047619047619047616,0.125,all_gt
bl_two_signal_strict__ni_k3_r2,R4,136,40,12,1,0.025,0.08333333333333333,all_gt
bl_two_signal_strict__ni_k3_r2,R5,173,66,19,7,0.10606060606060606,0.3684210526315789,all_gt
bl_two_signal_strict__ni_k3_r2,L1,217,10,5,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r2,L2,188,6,10,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r2,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r2,L5,225,18,30,7,0.3888888888888889,0.23333333333333334,assigned_gt
bl_two_signal_strict__ni_k3_r2,R1,190,9,8,2,0.2222222222222222,0.25,assigned_gt
bl_two_signal_strict__ni_k3_r2,R2,182,8,9,1,0.125,0.1111111111111111,assigned_gt
bl_two_signal_strict__ni_k3_r2,R3,146,7,8,1,0.14285714285714285,0.125,assigned_gt
bl_two_signal_strict__ni_k3_r2,R4,118,22,12,1,0.045454545454545456,0.08333333333333333,assigned_gt
bl_two_signal_strict__ni_k3_r2,R5,134,27,19,7,0.25925925925925924,0.3684210526315789,assigned_gt
bl_two_signal_strict__ni_k3_r4,L1,252,45,10,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r4,L2,207,25,14,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r4,L3,110,20,10,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r4,L4,38,13,5,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k3_r4,L5,279,72,34,7,0.09722222222222222,0.20588235294117646,all_gt
bl_two_signal_strict__ni_k3_r4,R1,252,71,9,2,0.028169014084507043,0.2222222222222222,all_gt
bl_two_signal_strict__ni_k3_r4,R2,193,19,10,1,0.05263157894736842,0.1,all_gt
bl_two_signal_strict__ni_k3_r4,R3,160,21,11,1,0.047619047619047616,0.09090909090909091,all_gt
bl_two_signal_strict__ni_k3_r4,R4,136,40,19,4,0.1,0.21052631578947367,all_gt
bl_two_signal_strict__ni_k3_r4,R5,173,66,22,8,0.12121212121212122,0.36363636363636365,all_gt
bl_two_signal_strict__ni_k3_r4,L1,217,10,10,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r4,L2,188,6,14,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r4,L4,30,5,5,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k3_r4,L5,225,18,34,7,0.3888888888888889,0.20588235294117646,assigned_gt
bl_two_signal_strict__ni_k3_r4,R1,190,9,9,2,0.2222222222222222,0.2222222222222222,assigned_gt
bl_two_signal_strict__ni_k3_r4,R2,182,8,10,1,0.125,0.1,assigned_gt
bl_two_signal_strict__ni_k3_r4,R3,146,7,11,1,0.14285714285714285,0.09090909090909091,assigned_gt
bl_two_signal_strict__ni_k3_r4,R4,118,22,19,4,0.18181818181818182,0.21052631578947367,assigned_gt
bl_two_signal_strict__ni_k3_r4,R5,134,27,22,8,0.2962962962962963,0.36363636363636365,assigned_gt
bl_two_signal_strict__ni_k5_r1,L1,252,45,1,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r1,L2,207,25,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r1,L3,110,20,7,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r1,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r1,L5,279,72,28,7,0.09722222222222222,0.25,all_gt
bl_two_signal_strict__ni_k5_r1,R1,252,71,2,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r1,R2,193,19,7,1,0.05263157894736842,0.14285714285714285,all_gt
bl_two_signal_strict__ni_k5_r1,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
bl_two_signal_strict__ni_k5_r1,R4,136,40,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r1,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_two_signal_strict__ni_k5_r1,L1,217,10,1,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r1,L2,188,6,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r1,L5,225,18,28,7,0.3888888888888889,0.25,assigned_gt
bl_two_signal_strict__ni_k5_r1,R1,190,9,2,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r1,R2,182,8,7,1,0.125,0.14285714285714285,assigned_gt
bl_two_signal_strict__ni_k5_r1,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
bl_two_signal_strict__ni_k5_r1,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r1,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
bl_two_signal_strict__ni_k5_r2,L1,252,45,1,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r2,L2,207,25,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r2,L3,110,20,7,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r2,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r2,L5,279,72,28,7,0.09722222222222222,0.25,all_gt
bl_two_signal_strict__ni_k5_r2,R1,252,71,2,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r2,R2,193,19,7,1,0.05263157894736842,0.14285714285714285,all_gt
bl_two_signal_strict__ni_k5_r2,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
bl_two_signal_strict__ni_k5_r2,R4,136,40,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r2,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_two_signal_strict__ni_k5_r2,L1,217,10,1,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r2,L2,188,6,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r2,L5,225,18,28,7,0.3888888888888889,0.25,assigned_gt
bl_two_signal_strict__ni_k5_r2,R1,190,9,2,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r2,R2,182,8,7,1,0.125,0.14285714285714285,assigned_gt
bl_two_signal_strict__ni_k5_r2,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
bl_two_signal_strict__ni_k5_r2,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r2,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
bl_two_signal_strict__ni_k5_r4,L1,252,45,1,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r4,L2,207,25,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r4,L3,110,20,7,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r4,L4,38,13,4,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r4,L5,279,72,28,7,0.09722222222222222,0.25,all_gt
bl_two_signal_strict__ni_k5_r4,R1,252,71,2,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r4,R2,193,19,7,1,0.05263157894736842,0.14285714285714285,all_gt
bl_two_signal_strict__ni_k5_r4,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
bl_two_signal_strict__ni_k5_r4,R4,136,40,9,0,0.0,0.0,all_gt
bl_two_signal_strict__ni_k5_r4,R5,173,66,14,7,0.10606060606060606,0.5,all_gt
bl_two_signal_strict__ni_k5_r4,L1,217,10,1,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r4,L2,188,6,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r4,L3,95,5,7,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r4,L4,30,5,4,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r4,L5,225,18,28,7,0.3888888888888889,0.25,assigned_gt
bl_two_signal_strict__ni_k5_r4,R1,190,9,2,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r4,R2,182,8,7,1,0.125,0.14285714285714285,assigned_gt
bl_two_signal_strict__ni_k5_r4,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
bl_two_signal_strict__ni_k5_r4,R4,118,22,9,0,0.0,0.0,assigned_gt
bl_two_signal_strict__ni_k5_r4,R5,134,27,14,7,0.25925925925925924,0.5,assigned_gt
wl_model_agreement__ni_k2_r1,L1,252,45,94,7,0.15555555555555556,0.07446808510638298,all_gt
wl_model_agreement__ni_k2_r1,L2,207,25,158,4,0.16,0.02531645569620253,all_gt
wl_model_agreement__ni_k2_r1,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k2_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k2_r1,L5,279,72,129,17,0.2361111111111111,0.13178294573643412,all_gt
wl_model_agreement__ni_k2_r1,R1,252,71,89,4,0.056338028169014086,0.0449438202247191,all_gt
wl_model_agreement__ni_k2_r1,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k2_r1,R3,160,21,128,5,0.23809523809523808,0.0390625,all_gt
wl_model_agreement__ni_k2_r1,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
wl_model_agreement__ni_k2_r1,R5,173,66,85,25,0.3787878787878788,0.29411764705882354,all_gt
wl_model_agreement__ni_k2_r1,L1,217,10,94,7,0.7,0.07446808510638298,assigned_gt
wl_model_agreement__ni_k2_r1,L2,188,6,158,4,0.6666666666666666,0.02531645569620253,assigned_gt
wl_model_agreement__ni_k2_r1,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k2_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k2_r1,L5,225,18,129,17,0.9444444444444444,0.13178294573643412,assigned_gt
wl_model_agreement__ni_k2_r1,R1,190,9,89,4,0.4444444444444444,0.0449438202247191,assigned_gt
wl_model_agreement__ni_k2_r1,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k2_r1,R3,146,7,128,5,0.7142857142857143,0.0390625,assigned_gt
wl_model_agreement__ni_k2_r1,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
wl_model_agreement__ni_k2_r1,R5,134,27,85,25,0.9259259259259259,0.29411764705882354,assigned_gt
wl_model_agreement__ni_k2_r2,L1,252,45,94,7,0.15555555555555556,0.07446808510638298,all_gt
wl_model_agreement__ni_k2_r2,L2,207,25,158,4,0.16,0.02531645569620253,all_gt
wl_model_agreement__ni_k2_r2,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k2_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k2_r2,L5,279,72,131,17,0.2361111111111111,0.1297709923664122,all_gt
wl_model_agreement__ni_k2_r2,R1,252,71,94,5,0.07042253521126761,0.05319148936170213,all_gt
wl_model_agreement__ni_k2_r2,R2,193,19,122,7,0.3684210526315789,0.05737704918032787,all_gt
wl_model_agreement__ni_k2_r2,R3,160,21,128,5,0.23809523809523808,0.0390625,all_gt
wl_model_agreement__ni_k2_r2,R4,136,40,110,19,0.475,0.17272727272727273,all_gt
wl_model_agreement__ni_k2_r2,R5,173,66,88,25,0.3787878787878788,0.2840909090909091,all_gt
wl_model_agreement__ni_k2_r2,L1,217,10,94,7,0.7,0.07446808510638298,assigned_gt
wl_model_agreement__ni_k2_r2,L2,188,6,158,4,0.6666666666666666,0.02531645569620253,assigned_gt
wl_model_agreement__ni_k2_r2,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k2_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k2_r2,L5,225,18,131,17,0.9444444444444444,0.1297709923664122,assigned_gt
wl_model_agreement__ni_k2_r2,R1,190,9,94,5,0.5555555555555556,0.05319148936170213,assigned_gt
wl_model_agreement__ni_k2_r2,R2,182,8,122,7,0.875,0.05737704918032787,assigned_gt
wl_model_agreement__ni_k2_r2,R3,146,7,128,5,0.7142857142857143,0.0390625,assigned_gt
wl_model_agreement__ni_k2_r2,R4,118,22,110,19,0.8636363636363636,0.17272727272727273,assigned_gt
wl_model_agreement__ni_k2_r2,R5,134,27,88,25,0.9259259259259259,0.2840909090909091,assigned_gt
wl_model_agreement__ni_k2_r4,L1,252,45,103,7,0.15555555555555556,0.06796116504854369,all_gt
wl_model_agreement__ni_k2_r4,L2,207,25,158,4,0.16,0.02531645569620253,all_gt
wl_model_agreement__ni_k2_r4,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k2_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k2_r4,L5,279,72,134,17,0.2361111111111111,0.12686567164179105,all_gt
wl_model_agreement__ni_k2_r4,R1,252,71,99,5,0.07042253521126761,0.050505050505050504,all_gt
wl_model_agreement__ni_k2_r4,R2,193,19,124,7,0.3684210526315789,0.056451612903225805,all_gt
wl_model_agreement__ni_k2_r4,R3,160,21,130,5,0.23809523809523808,0.038461538461538464,all_gt
wl_model_agreement__ni_k2_r4,R4,136,40,112,19,0.475,0.16964285714285715,all_gt
wl_model_agreement__ni_k2_r4,R5,173,66,91,25,0.3787878787878788,0.27472527472527475,all_gt
wl_model_agreement__ni_k2_r4,L1,217,10,103,7,0.7,0.06796116504854369,assigned_gt
wl_model_agreement__ni_k2_r4,L2,188,6,158,4,0.6666666666666666,0.02531645569620253,assigned_gt
wl_model_agreement__ni_k2_r4,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k2_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k2_r4,L5,225,18,134,17,0.9444444444444444,0.12686567164179105,assigned_gt
wl_model_agreement__ni_k2_r4,R1,190,9,99,5,0.5555555555555556,0.050505050505050504,assigned_gt
wl_model_agreement__ni_k2_r4,R2,182,8,124,7,0.875,0.056451612903225805,assigned_gt
wl_model_agreement__ni_k2_r4,R3,146,7,130,5,0.7142857142857143,0.038461538461538464,assigned_gt
wl_model_agreement__ni_k2_r4,R4,118,22,112,19,0.8636363636363636,0.16964285714285715,assigned_gt
wl_model_agreement__ni_k2_r4,R5,134,27,91,25,0.9259259259259259,0.27472527472527475,assigned_gt
wl_model_agreement__ni_k3_r1,L1,252,45,88,7,0.15555555555555556,0.07954545454545454,all_gt
wl_model_agreement__ni_k3_r1,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement__ni_k3_r1,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k3_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k3_r1,L5,279,72,126,17,0.2361111111111111,0.1349206349206349,all_gt
wl_model_agreement__ni_k3_r1,R1,252,71,82,4,0.056338028169014086,0.04878048780487805,all_gt
wl_model_agreement__ni_k3_r1,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k3_r1,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement__ni_k3_r1,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
wl_model_agreement__ni_k3_r1,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
wl_model_agreement__ni_k3_r1,L1,217,10,88,7,0.7,0.07954545454545454,assigned_gt
wl_model_agreement__ni_k3_r1,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement__ni_k3_r1,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k3_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k3_r1,L5,225,18,126,17,0.9444444444444444,0.1349206349206349,assigned_gt
wl_model_agreement__ni_k3_r1,R1,190,9,82,4,0.4444444444444444,0.04878048780487805,assigned_gt
wl_model_agreement__ni_k3_r1,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k3_r1,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement__ni_k3_r1,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
wl_model_agreement__ni_k3_r1,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
wl_model_agreement__ni_k3_r2,L1,252,45,90,7,0.15555555555555556,0.07777777777777778,all_gt
wl_model_agreement__ni_k3_r2,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement__ni_k3_r2,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k3_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k3_r2,L5,279,72,126,17,0.2361111111111111,0.1349206349206349,all_gt
wl_model_agreement__ni_k3_r2,R1,252,71,84,5,0.07042253521126761,0.05952380952380952,all_gt
wl_model_agreement__ni_k3_r2,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k3_r2,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement__ni_k3_r2,R4,136,40,110,19,0.475,0.17272727272727273,all_gt
wl_model_agreement__ni_k3_r2,R5,173,66,84,25,0.3787878787878788,0.2976190476190476,all_gt
wl_model_agreement__ni_k3_r2,L1,217,10,90,7,0.7,0.07777777777777778,assigned_gt
wl_model_agreement__ni_k3_r2,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement__ni_k3_r2,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k3_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k3_r2,L5,225,18,126,17,0.9444444444444444,0.1349206349206349,assigned_gt
wl_model_agreement__ni_k3_r2,R1,190,9,84,5,0.5555555555555556,0.05952380952380952,assigned_gt
wl_model_agreement__ni_k3_r2,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k3_r2,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement__ni_k3_r2,R4,118,22,110,19,0.8636363636363636,0.17272727272727273,assigned_gt
wl_model_agreement__ni_k3_r2,R5,134,27,84,25,0.9259259259259259,0.2976190476190476,assigned_gt
wl_model_agreement__ni_k3_r4,L1,252,45,93,7,0.15555555555555556,0.07526881720430108,all_gt
wl_model_agreement__ni_k3_r4,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement__ni_k3_r4,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k3_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k3_r4,L5,279,72,128,17,0.2361111111111111,0.1328125,all_gt
wl_model_agreement__ni_k3_r4,R1,252,71,84,5,0.07042253521126761,0.05952380952380952,all_gt
wl_model_agreement__ni_k3_r4,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k3_r4,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement__ni_k3_r4,R4,136,40,111,19,0.475,0.17117117117117117,all_gt
wl_model_agreement__ni_k3_r4,R5,173,66,84,25,0.3787878787878788,0.2976190476190476,all_gt
wl_model_agreement__ni_k3_r4,L1,217,10,93,7,0.7,0.07526881720430108,assigned_gt
wl_model_agreement__ni_k3_r4,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement__ni_k3_r4,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k3_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k3_r4,L5,225,18,128,17,0.9444444444444444,0.1328125,assigned_gt
wl_model_agreement__ni_k3_r4,R1,190,9,84,5,0.5555555555555556,0.05952380952380952,assigned_gt
wl_model_agreement__ni_k3_r4,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k3_r4,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement__ni_k3_r4,R4,118,22,111,19,0.8636363636363636,0.17117117117117117,assigned_gt
wl_model_agreement__ni_k3_r4,R5,134,27,84,25,0.9259259259259259,0.2976190476190476,assigned_gt
wl_model_agreement__ni_k5_r1,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
wl_model_agreement__ni_k5_r1,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement__ni_k5_r1,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k5_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k5_r1,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
wl_model_agreement__ni_k5_r1,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
wl_model_agreement__ni_k5_r1,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k5_r1,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement__ni_k5_r1,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
wl_model_agreement__ni_k5_r1,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
wl_model_agreement__ni_k5_r1,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
wl_model_agreement__ni_k5_r1,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement__ni_k5_r1,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k5_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k5_r1,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
wl_model_agreement__ni_k5_r1,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
wl_model_agreement__ni_k5_r1,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k5_r1,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement__ni_k5_r1,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
wl_model_agreement__ni_k5_r1,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
wl_model_agreement__ni_k5_r2,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
wl_model_agreement__ni_k5_r2,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement__ni_k5_r2,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k5_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k5_r2,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
wl_model_agreement__ni_k5_r2,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
wl_model_agreement__ni_k5_r2,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k5_r2,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement__ni_k5_r2,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
wl_model_agreement__ni_k5_r2,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
wl_model_agreement__ni_k5_r2,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
wl_model_agreement__ni_k5_r2,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement__ni_k5_r2,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k5_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k5_r2,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
wl_model_agreement__ni_k5_r2,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
wl_model_agreement__ni_k5_r2,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k5_r2,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement__ni_k5_r2,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
wl_model_agreement__ni_k5_r2,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
wl_model_agreement__ni_k5_r4,L1,252,45,87,7,0.15555555555555556,0.08045977011494253,all_gt
wl_model_agreement__ni_k5_r4,L2,207,25,157,3,0.12,0.01910828025477707,all_gt
wl_model_agreement__ni_k5_r4,L3,110,20,90,4,0.2,0.044444444444444446,all_gt
wl_model_agreement__ni_k5_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_model_agreement__ni_k5_r4,L5,279,72,124,17,0.2361111111111111,0.13709677419354838,all_gt
wl_model_agreement__ni_k5_r4,R1,252,71,81,4,0.056338028169014086,0.04938271604938271,all_gt
wl_model_agreement__ni_k5_r4,R2,193,19,121,7,0.3684210526315789,0.05785123966942149,all_gt
wl_model_agreement__ni_k5_r4,R3,160,21,127,5,0.23809523809523808,0.03937007874015748,all_gt
wl_model_agreement__ni_k5_r4,R4,136,40,109,18,0.45,0.1651376146788991,all_gt
wl_model_agreement__ni_k5_r4,R5,173,66,83,25,0.3787878787878788,0.30120481927710846,all_gt
wl_model_agreement__ni_k5_r4,L1,217,10,87,7,0.7,0.08045977011494253,assigned_gt
wl_model_agreement__ni_k5_r4,L2,188,6,157,3,0.5,0.01910828025477707,assigned_gt
wl_model_agreement__ni_k5_r4,L3,95,5,90,4,0.8,0.044444444444444446,assigned_gt
wl_model_agreement__ni_k5_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_model_agreement__ni_k5_r4,L5,225,18,124,17,0.9444444444444444,0.13709677419354838,assigned_gt
wl_model_agreement__ni_k5_r4,R1,190,9,81,4,0.4444444444444444,0.04938271604938271,assigned_gt
wl_model_agreement__ni_k5_r4,R2,182,8,121,7,0.875,0.05785123966942149,assigned_gt
wl_model_agreement__ni_k5_r4,R3,146,7,127,5,0.7142857142857143,0.03937007874015748,assigned_gt
wl_model_agreement__ni_k5_r4,R4,118,22,109,18,0.8181818181818182,0.1651376146788991,assigned_gt
wl_model_agreement__ni_k5_r4,R5,134,27,83,25,0.9259259259259259,0.30120481927710846,assigned_gt
wl_strict_obvious__ni_k2_r1,L1,252,45,154,7,0.15555555555555556,0.045454545454545456,all_gt
wl_strict_obvious__ni_k2_r1,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k2_r1,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k2_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k2_r1,L5,279,72,190,17,0.2361111111111111,0.08947368421052632,all_gt
wl_strict_obvious__ni_k2_r1,R1,252,71,119,8,0.11267605633802817,0.06722689075630252,all_gt
wl_strict_obvious__ni_k2_r1,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k2_r1,R3,160,21,131,7,0.3333333333333333,0.05343511450381679,all_gt
wl_strict_obvious__ni_k2_r1,R4,136,40,113,20,0.5,0.17699115044247787,all_gt
wl_strict_obvious__ni_k2_r1,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k2_r1,L1,217,10,154,7,0.7,0.045454545454545456,assigned_gt
wl_strict_obvious__ni_k2_r1,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k2_r1,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k2_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k2_r1,L5,225,18,190,17,0.9444444444444444,0.08947368421052632,assigned_gt
wl_strict_obvious__ni_k2_r1,R1,190,9,119,8,0.8888888888888888,0.06722689075630252,assigned_gt
wl_strict_obvious__ni_k2_r1,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k2_r1,R3,146,7,131,7,1.0,0.05343511450381679,assigned_gt
wl_strict_obvious__ni_k2_r1,R4,118,22,113,20,0.9090909090909091,0.17699115044247787,assigned_gt
wl_strict_obvious__ni_k2_r1,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
wl_strict_obvious__ni_k2_r2,L1,252,45,154,7,0.15555555555555556,0.045454545454545456,all_gt
wl_strict_obvious__ni_k2_r2,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k2_r2,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k2_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k2_r2,L5,279,72,190,17,0.2361111111111111,0.08947368421052632,all_gt
wl_strict_obvious__ni_k2_r2,R1,252,71,123,8,0.11267605633802817,0.06504065040650407,all_gt
wl_strict_obvious__ni_k2_r2,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k2_r2,R3,160,21,131,7,0.3333333333333333,0.05343511450381679,all_gt
wl_strict_obvious__ni_k2_r2,R4,136,40,114,21,0.525,0.18421052631578946,all_gt
wl_strict_obvious__ni_k2_r2,R5,173,66,117,27,0.4090909090909091,0.23076923076923078,all_gt
wl_strict_obvious__ni_k2_r2,L1,217,10,154,7,0.7,0.045454545454545456,assigned_gt
wl_strict_obvious__ni_k2_r2,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k2_r2,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k2_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k2_r2,L5,225,18,190,17,0.9444444444444444,0.08947368421052632,assigned_gt
wl_strict_obvious__ni_k2_r2,R1,190,9,123,8,0.8888888888888888,0.06504065040650407,assigned_gt
wl_strict_obvious__ni_k2_r2,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k2_r2,R3,146,7,131,7,1.0,0.05343511450381679,assigned_gt
wl_strict_obvious__ni_k2_r2,R4,118,22,114,21,0.9545454545454546,0.18421052631578946,assigned_gt
wl_strict_obvious__ni_k2_r2,R5,134,27,117,27,1.0,0.23076923076923078,assigned_gt
wl_strict_obvious__ni_k2_r4,L1,252,45,159,7,0.15555555555555556,0.0440251572327044,all_gt
wl_strict_obvious__ni_k2_r4,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k2_r4,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k2_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k2_r4,L5,279,72,190,17,0.2361111111111111,0.08947368421052632,all_gt
wl_strict_obvious__ni_k2_r4,R1,252,71,127,8,0.11267605633802817,0.06299212598425197,all_gt
wl_strict_obvious__ni_k2_r4,R2,193,19,139,8,0.42105263157894735,0.05755395683453238,all_gt
wl_strict_obvious__ni_k2_r4,R3,160,21,133,7,0.3333333333333333,0.05263157894736842,all_gt
wl_strict_obvious__ni_k2_r4,R4,136,40,115,21,0.525,0.1826086956521739,all_gt
wl_strict_obvious__ni_k2_r4,R5,173,66,119,27,0.4090909090909091,0.226890756302521,all_gt
wl_strict_obvious__ni_k2_r4,L1,217,10,159,7,0.7,0.0440251572327044,assigned_gt
wl_strict_obvious__ni_k2_r4,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k2_r4,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k2_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k2_r4,L5,225,18,190,17,0.9444444444444444,0.08947368421052632,assigned_gt
wl_strict_obvious__ni_k2_r4,R1,190,9,127,8,0.8888888888888888,0.06299212598425197,assigned_gt
wl_strict_obvious__ni_k2_r4,R2,182,8,139,8,1.0,0.05755395683453238,assigned_gt
wl_strict_obvious__ni_k2_r4,R3,146,7,133,7,1.0,0.05263157894736842,assigned_gt
wl_strict_obvious__ni_k2_r4,R4,118,22,115,21,0.9545454545454546,0.1826086956521739,assigned_gt
wl_strict_obvious__ni_k2_r4,R5,134,27,119,27,1.0,0.226890756302521,assigned_gt
wl_strict_obvious__ni_k3_r1,L1,252,45,151,7,0.15555555555555556,0.046357615894039736,all_gt
wl_strict_obvious__ni_k3_r1,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k3_r1,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k3_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k3_r1,L5,279,72,189,17,0.2361111111111111,0.08994708994708994,all_gt
wl_strict_obvious__ni_k3_r1,R1,252,71,114,8,0.11267605633802817,0.07017543859649122,all_gt
wl_strict_obvious__ni_k3_r1,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k3_r1,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious__ni_k3_r1,R4,136,40,113,20,0.5,0.17699115044247787,all_gt
wl_strict_obvious__ni_k3_r1,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k3_r1,L1,217,10,151,7,0.7,0.046357615894039736,assigned_gt
wl_strict_obvious__ni_k3_r1,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k3_r1,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k3_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k3_r1,L5,225,18,189,17,0.9444444444444444,0.08994708994708994,assigned_gt
wl_strict_obvious__ni_k3_r1,R1,190,9,114,8,0.8888888888888888,0.07017543859649122,assigned_gt
wl_strict_obvious__ni_k3_r1,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k3_r1,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious__ni_k3_r1,R4,118,22,113,20,0.9090909090909091,0.17699115044247787,assigned_gt
wl_strict_obvious__ni_k3_r1,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
wl_strict_obvious__ni_k3_r2,L1,252,45,152,7,0.15555555555555556,0.046052631578947366,all_gt
wl_strict_obvious__ni_k3_r2,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k3_r2,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k3_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k3_r2,L5,279,72,189,17,0.2361111111111111,0.08994708994708994,all_gt
wl_strict_obvious__ni_k3_r2,R1,252,71,115,8,0.11267605633802817,0.06956521739130435,all_gt
wl_strict_obvious__ni_k3_r2,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k3_r2,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious__ni_k3_r2,R4,136,40,114,21,0.525,0.18421052631578946,all_gt
wl_strict_obvious__ni_k3_r2,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k3_r2,L1,217,10,152,7,0.7,0.046052631578947366,assigned_gt
wl_strict_obvious__ni_k3_r2,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k3_r2,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k3_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k3_r2,L5,225,18,189,17,0.9444444444444444,0.08994708994708994,assigned_gt
wl_strict_obvious__ni_k3_r2,R1,190,9,115,8,0.8888888888888888,0.06956521739130435,assigned_gt
wl_strict_obvious__ni_k3_r2,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k3_r2,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious__ni_k3_r2,R4,118,22,114,21,0.9545454545454546,0.18421052631578946,assigned_gt
wl_strict_obvious__ni_k3_r2,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
wl_strict_obvious__ni_k3_r4,L1,252,45,154,7,0.15555555555555556,0.045454545454545456,all_gt
wl_strict_obvious__ni_k3_r4,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k3_r4,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k3_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k3_r4,L5,279,72,189,17,0.2361111111111111,0.08994708994708994,all_gt
wl_strict_obvious__ni_k3_r4,R1,252,71,115,8,0.11267605633802817,0.06956521739130435,all_gt
wl_strict_obvious__ni_k3_r4,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k3_r4,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious__ni_k3_r4,R4,136,40,115,21,0.525,0.1826086956521739,all_gt
wl_strict_obvious__ni_k3_r4,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k3_r4,L1,217,10,154,7,0.7,0.045454545454545456,assigned_gt
wl_strict_obvious__ni_k3_r4,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k3_r4,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k3_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k3_r4,L5,225,18,189,17,0.9444444444444444,0.08994708994708994,assigned_gt
wl_strict_obvious__ni_k3_r4,R1,190,9,115,8,0.8888888888888888,0.06956521739130435,assigned_gt
wl_strict_obvious__ni_k3_r4,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k3_r4,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious__ni_k3_r4,R4,118,22,115,21,0.9545454545454546,0.1826086956521739,assigned_gt
wl_strict_obvious__ni_k3_r4,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
wl_strict_obvious__ni_k5_r1,L1,252,45,150,7,0.15555555555555556,0.04666666666666667,all_gt
wl_strict_obvious__ni_k5_r1,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k5_r1,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k5_r1,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k5_r1,L5,279,72,188,17,0.2361111111111111,0.09042553191489362,all_gt
wl_strict_obvious__ni_k5_r1,R1,252,71,114,8,0.11267605633802817,0.07017543859649122,all_gt
wl_strict_obvious__ni_k5_r1,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k5_r1,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious__ni_k5_r1,R4,136,40,113,20,0.5,0.17699115044247787,all_gt
wl_strict_obvious__ni_k5_r1,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k5_r1,L1,217,10,150,7,0.7,0.04666666666666667,assigned_gt
wl_strict_obvious__ni_k5_r1,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k5_r1,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k5_r1,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k5_r1,L5,225,18,188,17,0.9444444444444444,0.09042553191489362,assigned_gt
wl_strict_obvious__ni_k5_r1,R1,190,9,114,8,0.8888888888888888,0.07017543859649122,assigned_gt
wl_strict_obvious__ni_k5_r1,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k5_r1,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious__ni_k5_r1,R4,118,22,113,20,0.9090909090909091,0.17699115044247787,assigned_gt
wl_strict_obvious__ni_k5_r1,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
wl_strict_obvious__ni_k5_r2,L1,252,45,150,7,0.15555555555555556,0.04666666666666667,all_gt
wl_strict_obvious__ni_k5_r2,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k5_r2,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k5_r2,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k5_r2,L5,279,72,188,17,0.2361111111111111,0.09042553191489362,all_gt
wl_strict_obvious__ni_k5_r2,R1,252,71,114,8,0.11267605633802817,0.07017543859649122,all_gt
wl_strict_obvious__ni_k5_r2,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k5_r2,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious__ni_k5_r2,R4,136,40,113,20,0.5,0.17699115044247787,all_gt
wl_strict_obvious__ni_k5_r2,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k5_r2,L1,217,10,150,7,0.7,0.04666666666666667,assigned_gt
wl_strict_obvious__ni_k5_r2,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k5_r2,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k5_r2,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k5_r2,L5,225,18,188,17,0.9444444444444444,0.09042553191489362,assigned_gt
wl_strict_obvious__ni_k5_r2,R1,190,9,114,8,0.8888888888888888,0.07017543859649122,assigned_gt
wl_strict_obvious__ni_k5_r2,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k5_r2,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious__ni_k5_r2,R4,118,22,113,20,0.9090909090909091,0.17699115044247787,assigned_gt
wl_strict_obvious__ni_k5_r2,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
wl_strict_obvious__ni_k5_r4,L1,252,45,150,7,0.15555555555555556,0.04666666666666667,all_gt
wl_strict_obvious__ni_k5_r4,L2,207,25,170,5,0.2,0.029411764705882353,all_gt
wl_strict_obvious__ni_k5_r4,L3,110,20,92,5,0.25,0.05434782608695652,all_gt
wl_strict_obvious__ni_k5_r4,L4,38,13,30,5,0.38461538461538464,0.16666666666666666,all_gt
wl_strict_obvious__ni_k5_r4,L5,279,72,188,17,0.2361111111111111,0.09042553191489362,all_gt
wl_strict_obvious__ni_k5_r4,R1,252,71,114,8,0.11267605633802817,0.07017543859649122,all_gt
wl_strict_obvious__ni_k5_r4,R2,193,19,137,8,0.42105263157894735,0.058394160583941604,all_gt
wl_strict_obvious__ni_k5_r4,R3,160,21,130,7,0.3333333333333333,0.05384615384615385,all_gt
wl_strict_obvious__ni_k5_r4,R4,136,40,113,20,0.5,0.17699115044247787,all_gt
wl_strict_obvious__ni_k5_r4,R5,173,66,115,27,0.4090909090909091,0.23478260869565218,all_gt
wl_strict_obvious__ni_k5_r4,L1,217,10,150,7,0.7,0.04666666666666667,assigned_gt
wl_strict_obvious__ni_k5_r4,L2,188,6,170,5,0.8333333333333334,0.029411764705882353,assigned_gt
wl_strict_obvious__ni_k5_r4,L3,95,5,92,5,1.0,0.05434782608695652,assigned_gt
wl_strict_obvious__ni_k5_r4,L4,30,5,30,5,1.0,0.16666666666666666,assigned_gt
wl_strict_obvious__ni_k5_r4,L5,225,18,188,17,0.9444444444444444,0.09042553191489362,assigned_gt
wl_strict_obvious__ni_k5_r4,R1,190,9,114,8,0.8888888888888888,0.07017543859649122,assigned_gt
wl_strict_obvious__ni_k5_r4,R2,182,8,137,8,1.0,0.058394160583941604,assigned_gt
wl_strict_obvious__ni_k5_r4,R3,146,7,130,7,1.0,0.05384615384615385,assigned_gt
wl_strict_obvious__ni_k5_r4,R4,118,22,113,20,0.9090909090909091,0.17699115044247787,assigned_gt
wl_strict_obvious__ni_k5_r4,R5,134,27,115,27,1.0,0.23478260869565218,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,L1,252,45,24,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r1,L2,207,25,15,1,0.04,0.06666666666666667,all_gt
hy_direct_plus_corroborated__ni_k2_r1,L3,110,20,8,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r1,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r1,L5,279,72,64,7,0.09722222222222222,0.109375,all_gt
hy_direct_plus_corroborated__ni_k2_r1,R1,252,71,19,1,0.014084507042253521,0.05263157894736842,all_gt
hy_direct_plus_corroborated__ni_k2_r1,R2,193,19,16,1,0.05263157894736842,0.0625,all_gt
hy_direct_plus_corroborated__ni_k2_r1,R3,160,21,14,1,0.047619047619047616,0.07142857142857142,all_gt
hy_direct_plus_corroborated__ni_k2_r1,R4,136,40,13,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r1,R5,173,66,24,10,0.15151515151515152,0.4166666666666667,all_gt
hy_direct_plus_corroborated__ni_k2_r1,L1,217,10,24,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,L2,188,6,15,1,0.16666666666666666,0.06666666666666667,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,L3,95,5,8,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,L5,225,18,64,7,0.3888888888888889,0.109375,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,R1,190,9,19,1,0.1111111111111111,0.05263157894736842,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,R2,182,8,16,1,0.125,0.0625,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,R3,146,7,14,1,0.14285714285714285,0.07142857142857142,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,R4,118,22,13,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r1,R5,134,27,24,10,0.37037037037037035,0.4166666666666667,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,L1,252,45,28,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r2,L2,207,25,23,1,0.04,0.043478260869565216,all_gt
hy_direct_plus_corroborated__ni_k2_r2,L3,110,20,13,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r2,L4,38,13,5,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r2,L5,279,72,67,8,0.1111111111111111,0.11940298507462686,all_gt
hy_direct_plus_corroborated__ni_k2_r2,R1,252,71,26,3,0.04225352112676056,0.11538461538461539,all_gt
hy_direct_plus_corroborated__ni_k2_r2,R2,193,19,22,1,0.05263157894736842,0.045454545454545456,all_gt
hy_direct_plus_corroborated__ni_k2_r2,R3,160,21,19,1,0.047619047619047616,0.05263157894736842,all_gt
hy_direct_plus_corroborated__ni_k2_r2,R4,136,40,18,2,0.05,0.1111111111111111,all_gt
hy_direct_plus_corroborated__ni_k2_r2,R5,173,66,28,10,0.15151515151515152,0.35714285714285715,all_gt
hy_direct_plus_corroborated__ni_k2_r2,L1,217,10,28,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,L2,188,6,23,1,0.16666666666666666,0.043478260869565216,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,L3,95,5,13,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,L4,30,5,5,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,L5,225,18,67,8,0.4444444444444444,0.11940298507462686,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,R1,190,9,26,3,0.3333333333333333,0.11538461538461539,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,R2,182,8,22,1,0.125,0.045454545454545456,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,R3,146,7,19,1,0.14285714285714285,0.05263157894736842,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,R4,118,22,18,2,0.09090909090909091,0.1111111111111111,assigned_gt
hy_direct_plus_corroborated__ni_k2_r2,R5,134,27,28,10,0.37037037037037035,0.35714285714285715,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,L1,252,45,43,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r4,L2,207,25,34,1,0.04,0.029411764705882353,all_gt
hy_direct_plus_corroborated__ni_k2_r4,L3,110,20,15,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r4,L4,38,13,6,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k2_r4,L5,279,72,77,9,0.125,0.11688311688311688,all_gt
hy_direct_plus_corroborated__ni_k2_r4,R1,252,71,32,3,0.04225352112676056,0.09375,all_gt
hy_direct_plus_corroborated__ni_k2_r4,R2,193,19,28,1,0.05263157894736842,0.03571428571428571,all_gt
hy_direct_plus_corroborated__ni_k2_r4,R3,160,21,29,1,0.047619047619047616,0.034482758620689655,all_gt
hy_direct_plus_corroborated__ni_k2_r4,R4,136,40,32,6,0.15,0.1875,all_gt
hy_direct_plus_corroborated__ni_k2_r4,R5,173,66,37,11,0.16666666666666666,0.2972972972972973,all_gt
hy_direct_plus_corroborated__ni_k2_r4,L1,217,10,43,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,L2,188,6,34,1,0.16666666666666666,0.029411764705882353,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,L3,95,5,15,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,L4,30,5,6,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,L5,225,18,77,9,0.5,0.11688311688311688,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,R1,190,9,32,3,0.3333333333333333,0.09375,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,R2,182,8,28,1,0.125,0.03571428571428571,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,R3,146,7,29,1,0.14285714285714285,0.034482758620689655,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,R4,118,22,32,6,0.2727272727272727,0.1875,assigned_gt
hy_direct_plus_corroborated__ni_k2_r4,R5,134,27,37,11,0.4074074074074074,0.2972972972972973,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,L1,252,45,15,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r1,L2,207,25,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r1,L3,110,20,7,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r1,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r1,L5,279,72,59,7,0.09722222222222222,0.11864406779661017,all_gt
hy_direct_plus_corroborated__ni_k3_r1,R1,252,71,9,1,0.014084507042253521,0.1111111111111111,all_gt
hy_direct_plus_corroborated__ni_k3_r1,R2,193,19,12,1,0.05263157894736842,0.08333333333333333,all_gt
hy_direct_plus_corroborated__ni_k3_r1,R3,160,21,8,1,0.047619047619047616,0.125,all_gt
hy_direct_plus_corroborated__ni_k3_r1,R4,136,40,11,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r1,R5,173,66,18,7,0.10606060606060606,0.3888888888888889,all_gt
hy_direct_plus_corroborated__ni_k3_r1,L1,217,10,15,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,L5,225,18,59,7,0.3888888888888889,0.11864406779661017,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,R1,190,9,9,1,0.1111111111111111,0.1111111111111111,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,R2,182,8,12,1,0.125,0.08333333333333333,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,R3,146,7,8,1,0.14285714285714285,0.125,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,R4,118,22,11,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r1,R5,134,27,18,7,0.25925925925925924,0.3888888888888889,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,L1,252,45,18,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r2,L2,207,25,11,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r2,L3,110,20,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r2,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r2,L5,279,72,59,7,0.09722222222222222,0.11864406779661017,all_gt
hy_direct_plus_corroborated__ni_k3_r2,R1,252,71,12,2,0.028169014084507043,0.16666666666666666,all_gt
hy_direct_plus_corroborated__ni_k3_r2,R2,193,19,13,1,0.05263157894736842,0.07692307692307693,all_gt
hy_direct_plus_corroborated__ni_k3_r2,R3,160,21,8,1,0.047619047619047616,0.125,all_gt
hy_direct_plus_corroborated__ni_k3_r2,R4,136,40,12,1,0.025,0.08333333333333333,all_gt
hy_direct_plus_corroborated__ni_k3_r2,R5,173,66,21,7,0.10606060606060606,0.3333333333333333,all_gt
hy_direct_plus_corroborated__ni_k3_r2,L1,217,10,18,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,L2,188,6,11,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,L3,95,5,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,L5,225,18,59,7,0.3888888888888889,0.11864406779661017,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,R1,190,9,12,2,0.2222222222222222,0.16666666666666666,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,R2,182,8,13,1,0.125,0.07692307692307693,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,R3,146,7,8,1,0.14285714285714285,0.125,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,R4,118,22,12,1,0.045454545454545456,0.08333333333333333,assigned_gt
hy_direct_plus_corroborated__ni_k3_r2,R5,134,27,21,7,0.25925925925925924,0.3333333333333333,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,L1,252,45,23,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r4,L2,207,25,15,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r4,L3,110,20,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r4,L4,38,13,5,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k3_r4,L5,279,72,61,7,0.09722222222222222,0.11475409836065574,all_gt
hy_direct_plus_corroborated__ni_k3_r4,R1,252,71,13,2,0.028169014084507043,0.15384615384615385,all_gt
hy_direct_plus_corroborated__ni_k3_r4,R2,193,19,14,1,0.05263157894736842,0.07142857142857142,all_gt
hy_direct_plus_corroborated__ni_k3_r4,R3,160,21,11,1,0.047619047619047616,0.09090909090909091,all_gt
hy_direct_plus_corroborated__ni_k3_r4,R4,136,40,19,4,0.1,0.21052631578947367,all_gt
hy_direct_plus_corroborated__ni_k3_r4,R5,173,66,24,8,0.12121212121212122,0.3333333333333333,all_gt
hy_direct_plus_corroborated__ni_k3_r4,L1,217,10,23,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,L2,188,6,15,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,L3,95,5,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,L4,30,5,5,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,L5,225,18,61,7,0.3888888888888889,0.11475409836065574,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,R1,190,9,13,2,0.2222222222222222,0.15384615384615385,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,R2,182,8,14,1,0.125,0.07142857142857142,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,R3,146,7,11,1,0.14285714285714285,0.09090909090909091,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,R4,118,22,19,4,0.18181818181818182,0.21052631578947367,assigned_gt
hy_direct_plus_corroborated__ni_k3_r4,R5,134,27,24,8,0.2962962962962963,0.3333333333333333,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,L1,252,45,14,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r1,L2,207,25,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r1,L3,110,20,7,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r1,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r1,L5,279,72,57,7,0.09722222222222222,0.12280701754385964,all_gt
hy_direct_plus_corroborated__ni_k5_r1,R1,252,71,6,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r1,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
hy_direct_plus_corroborated__ni_k5_r1,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
hy_direct_plus_corroborated__ni_k5_r1,R4,136,40,9,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r1,R5,173,66,16,7,0.10606060606060606,0.4375,all_gt
hy_direct_plus_corroborated__ni_k5_r1,L1,217,10,14,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,L2,188,6,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,L3,95,5,7,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,L5,225,18,57,7,0.3888888888888889,0.12280701754385964,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,R1,190,9,6,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,R4,118,22,9,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r1,R5,134,27,16,7,0.25925925925925924,0.4375,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,L1,252,45,14,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r2,L2,207,25,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r2,L3,110,20,7,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r2,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r2,L5,279,72,57,7,0.09722222222222222,0.12280701754385964,all_gt
hy_direct_plus_corroborated__ni_k5_r2,R1,252,71,6,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r2,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
hy_direct_plus_corroborated__ni_k5_r2,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
hy_direct_plus_corroborated__ni_k5_r2,R4,136,40,9,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r2,R5,173,66,16,7,0.10606060606060606,0.4375,all_gt
hy_direct_plus_corroborated__ni_k5_r2,L1,217,10,14,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,L2,188,6,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,L3,95,5,7,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,L5,225,18,57,7,0.3888888888888889,0.12280701754385964,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,R1,190,9,6,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,R4,118,22,9,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r2,R5,134,27,16,7,0.25925925925925924,0.4375,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,L1,252,45,14,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r4,L2,207,25,10,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r4,L3,110,20,7,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r4,L4,38,13,4,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r4,L5,279,72,57,7,0.09722222222222222,0.12280701754385964,all_gt
hy_direct_plus_corroborated__ni_k5_r4,R1,252,71,6,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r4,R2,193,19,11,1,0.05263157894736842,0.09090909090909091,all_gt
hy_direct_plus_corroborated__ni_k5_r4,R3,160,21,5,1,0.047619047619047616,0.2,all_gt
hy_direct_plus_corroborated__ni_k5_r4,R4,136,40,9,0,0.0,0.0,all_gt
hy_direct_plus_corroborated__ni_k5_r4,R5,173,66,16,7,0.10606060606060606,0.4375,all_gt
hy_direct_plus_corroborated__ni_k5_r4,L1,217,10,14,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,L2,188,6,10,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,L3,95,5,7,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,L4,30,5,4,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,L5,225,18,57,7,0.3888888888888889,0.12280701754385964,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,R1,190,9,6,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,R2,182,8,11,1,0.125,0.09090909090909091,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,R3,146,7,5,1,0.14285714285714285,0.2,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,R4,118,22,9,0,0.0,0.0,assigned_gt
hy_direct_plus_corroborated__ni_k5_r4,R5,134,27,16,7,0.25925925925925924,0.4375,assigned_gt
hy_two_of_three_families__ni_k2_r1,L1,252,45,15,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r1,L2,207,25,24,1,0.04,0.041666666666666664,all_gt
hy_two_of_three_families__ni_k2_r1,L3,110,20,21,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r1,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r1,L5,279,72,43,10,0.1388888888888889,0.23255813953488372,all_gt
hy_two_of_three_families__ni_k2_r1,R1,252,71,16,1,0.014084507042253521,0.0625,all_gt
hy_two_of_three_families__ni_k2_r1,R2,193,19,25,1,0.05263157894736842,0.04,all_gt
hy_two_of_three_families__ni_k2_r1,R3,160,21,25,2,0.09523809523809523,0.08,all_gt
hy_two_of_three_families__ni_k2_r1,R4,136,40,23,2,0.05,0.08695652173913043,all_gt
hy_two_of_three_families__ni_k2_r1,R5,173,66,29,11,0.16666666666666666,0.3793103448275862,all_gt
hy_two_of_three_families__ni_k2_r1,L1,217,10,15,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r1,L2,188,6,24,1,0.16666666666666666,0.041666666666666664,assigned_gt
hy_two_of_three_families__ni_k2_r1,L3,95,5,21,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r1,L5,225,18,43,10,0.5555555555555556,0.23255813953488372,assigned_gt
hy_two_of_three_families__ni_k2_r1,R1,190,9,16,1,0.1111111111111111,0.0625,assigned_gt
hy_two_of_three_families__ni_k2_r1,R2,182,8,25,1,0.125,0.04,assigned_gt
hy_two_of_three_families__ni_k2_r1,R3,146,7,25,2,0.2857142857142857,0.08,assigned_gt
hy_two_of_three_families__ni_k2_r1,R4,118,22,23,2,0.09090909090909091,0.08695652173913043,assigned_gt
hy_two_of_three_families__ni_k2_r1,R5,134,27,29,11,0.4074074074074074,0.3793103448275862,assigned_gt
hy_two_of_three_families__ni_k2_r2,L1,252,45,19,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r2,L2,207,25,30,1,0.04,0.03333333333333333,all_gt
hy_two_of_three_families__ni_k2_r2,L3,110,20,25,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r2,L4,38,13,9,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r2,L5,279,72,47,11,0.1527777777777778,0.23404255319148937,all_gt
hy_two_of_three_families__ni_k2_r2,R1,252,71,23,3,0.04225352112676056,0.13043478260869565,all_gt
hy_two_of_three_families__ni_k2_r2,R2,193,19,32,1,0.05263157894736842,0.03125,all_gt
hy_two_of_three_families__ni_k2_r2,R3,160,21,30,2,0.09523809523809523,0.06666666666666667,all_gt
hy_two_of_three_families__ni_k2_r2,R4,136,40,28,4,0.1,0.14285714285714285,all_gt
hy_two_of_three_families__ni_k2_r2,R5,173,66,33,11,0.16666666666666666,0.3333333333333333,all_gt
hy_two_of_three_families__ni_k2_r2,L1,217,10,19,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r2,L2,188,6,30,1,0.16666666666666666,0.03333333333333333,assigned_gt
hy_two_of_three_families__ni_k2_r2,L3,95,5,25,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r2,L4,30,5,9,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r2,L5,225,18,47,11,0.6111111111111112,0.23404255319148937,assigned_gt
hy_two_of_three_families__ni_k2_r2,R1,190,9,23,3,0.3333333333333333,0.13043478260869565,assigned_gt
hy_two_of_three_families__ni_k2_r2,R2,182,8,32,1,0.125,0.03125,assigned_gt
hy_two_of_three_families__ni_k2_r2,R3,146,7,30,2,0.2857142857142857,0.06666666666666667,assigned_gt
hy_two_of_three_families__ni_k2_r2,R4,118,22,28,4,0.18181818181818182,0.14285714285714285,assigned_gt
hy_two_of_three_families__ni_k2_r2,R5,134,27,33,11,0.4074074074074074,0.3333333333333333,assigned_gt
hy_two_of_three_families__ni_k2_r4,L1,252,45,33,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r4,L2,207,25,38,1,0.04,0.02631578947368421,all_gt
hy_two_of_three_families__ni_k2_r4,L3,110,20,26,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r4,L4,38,13,10,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k2_r4,L5,279,72,57,12,0.16666666666666666,0.21052631578947367,all_gt
hy_two_of_three_families__ni_k2_r4,R1,252,71,30,3,0.04225352112676056,0.1,all_gt
hy_two_of_three_families__ni_k2_r4,R2,193,19,38,1,0.05263157894736842,0.02631578947368421,all_gt
hy_two_of_three_families__ni_k2_r4,R3,160,21,40,2,0.09523809523809523,0.05,all_gt
hy_two_of_three_families__ni_k2_r4,R4,136,40,42,8,0.2,0.19047619047619047,all_gt
hy_two_of_three_families__ni_k2_r4,R5,173,66,42,12,0.18181818181818182,0.2857142857142857,all_gt
hy_two_of_three_families__ni_k2_r4,L1,217,10,33,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r4,L2,188,6,38,1,0.16666666666666666,0.02631578947368421,assigned_gt
hy_two_of_three_families__ni_k2_r4,L3,95,5,26,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r4,L4,30,5,10,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k2_r4,L5,225,18,57,12,0.6666666666666666,0.21052631578947367,assigned_gt
hy_two_of_three_families__ni_k2_r4,R1,190,9,30,3,0.3333333333333333,0.1,assigned_gt
hy_two_of_three_families__ni_k2_r4,R2,182,8,38,1,0.125,0.02631578947368421,assigned_gt
hy_two_of_three_families__ni_k2_r4,R3,146,7,40,2,0.2857142857142857,0.05,assigned_gt
hy_two_of_three_families__ni_k2_r4,R4,118,22,42,8,0.36363636363636365,0.19047619047619047,assigned_gt
hy_two_of_three_families__ni_k2_r4,R5,134,27,42,12,0.4444444444444444,0.2857142857142857,assigned_gt
hy_two_of_three_families__ni_k3_r1,L1,252,45,6,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r1,L2,207,25,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r1,L3,110,20,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r1,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r1,L5,279,72,38,10,0.1388888888888889,0.2631578947368421,all_gt
hy_two_of_three_families__ni_k3_r1,R1,252,71,6,1,0.014084507042253521,0.16666666666666666,all_gt
hy_two_of_three_families__ni_k3_r1,R2,193,19,21,1,0.05263157894736842,0.047619047619047616,all_gt
hy_two_of_three_families__ni_k3_r1,R3,160,21,19,2,0.09523809523809523,0.10526315789473684,all_gt
hy_two_of_three_families__ni_k3_r1,R4,136,40,21,2,0.05,0.09523809523809523,all_gt
hy_two_of_three_families__ni_k3_r1,R5,173,66,23,9,0.13636363636363635,0.391304347826087,all_gt
hy_two_of_three_families__ni_k3_r1,L1,217,10,6,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r1,L2,188,6,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r1,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r1,L5,225,18,38,10,0.5555555555555556,0.2631578947368421,assigned_gt
hy_two_of_three_families__ni_k3_r1,R1,190,9,6,1,0.1111111111111111,0.16666666666666666,assigned_gt
hy_two_of_three_families__ni_k3_r1,R2,182,8,21,1,0.125,0.047619047619047616,assigned_gt
hy_two_of_three_families__ni_k3_r1,R3,146,7,19,2,0.2857142857142857,0.10526315789473684,assigned_gt
hy_two_of_three_families__ni_k3_r1,R4,118,22,21,2,0.09090909090909091,0.09523809523809523,assigned_gt
hy_two_of_three_families__ni_k3_r1,R5,134,27,23,9,0.3333333333333333,0.391304347826087,assigned_gt
hy_two_of_three_families__ni_k3_r2,L1,252,45,9,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r2,L2,207,25,21,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r2,L3,110,20,22,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r2,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r2,L5,279,72,38,10,0.1388888888888889,0.2631578947368421,all_gt
hy_two_of_three_families__ni_k3_r2,R1,252,71,9,2,0.028169014084507043,0.2222222222222222,all_gt
hy_two_of_three_families__ni_k3_r2,R2,193,19,22,1,0.05263157894736842,0.045454545454545456,all_gt
hy_two_of_three_families__ni_k3_r2,R3,160,21,19,2,0.09523809523809523,0.10526315789473684,all_gt
hy_two_of_three_families__ni_k3_r2,R4,136,40,22,3,0.075,0.13636363636363635,all_gt
hy_two_of_three_families__ni_k3_r2,R5,173,66,26,9,0.13636363636363635,0.34615384615384615,all_gt
hy_two_of_three_families__ni_k3_r2,L1,217,10,9,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r2,L2,188,6,21,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r2,L3,95,5,22,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r2,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r2,L5,225,18,38,10,0.5555555555555556,0.2631578947368421,assigned_gt
hy_two_of_three_families__ni_k3_r2,R1,190,9,9,2,0.2222222222222222,0.2222222222222222,assigned_gt
hy_two_of_three_families__ni_k3_r2,R2,182,8,22,1,0.125,0.045454545454545456,assigned_gt
hy_two_of_three_families__ni_k3_r2,R3,146,7,19,2,0.2857142857142857,0.10526315789473684,assigned_gt
hy_two_of_three_families__ni_k3_r2,R4,118,22,22,3,0.13636363636363635,0.13636363636363635,assigned_gt
hy_two_of_three_families__ni_k3_r2,R5,134,27,26,9,0.3333333333333333,0.34615384615384615,assigned_gt
hy_two_of_three_families__ni_k3_r4,L1,252,45,14,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r4,L2,207,25,23,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r4,L3,110,20,22,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r4,L4,38,13,9,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k3_r4,L5,279,72,42,10,0.1388888888888889,0.23809523809523808,all_gt
hy_two_of_three_families__ni_k3_r4,R1,252,71,10,2,0.028169014084507043,0.2,all_gt
hy_two_of_three_families__ni_k3_r4,R2,193,19,23,1,0.05263157894736842,0.043478260869565216,all_gt
hy_two_of_three_families__ni_k3_r4,R3,160,21,22,2,0.09523809523809523,0.09090909090909091,all_gt
hy_two_of_three_families__ni_k3_r4,R4,136,40,29,6,0.15,0.20689655172413793,all_gt
hy_two_of_three_families__ni_k3_r4,R5,173,66,29,10,0.15151515151515152,0.3448275862068966,all_gt
hy_two_of_three_families__ni_k3_r4,L1,217,10,14,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r4,L2,188,6,23,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r4,L3,95,5,22,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r4,L4,30,5,9,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k3_r4,L5,225,18,42,10,0.5555555555555556,0.23809523809523808,assigned_gt
hy_two_of_three_families__ni_k3_r4,R1,190,9,10,2,0.2222222222222222,0.2,assigned_gt
hy_two_of_three_families__ni_k3_r4,R2,182,8,23,1,0.125,0.043478260869565216,assigned_gt
hy_two_of_three_families__ni_k3_r4,R3,146,7,22,2,0.2857142857142857,0.09090909090909091,assigned_gt
hy_two_of_three_families__ni_k3_r4,R4,118,22,29,6,0.2727272727272727,0.20689655172413793,assigned_gt
hy_two_of_three_families__ni_k3_r4,R5,134,27,29,10,0.37037037037037035,0.3448275862068966,assigned_gt
hy_two_of_three_families__ni_k5_r1,L1,252,45,5,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r1,L2,207,25,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r1,L3,110,20,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r1,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r1,L5,279,72,36,10,0.1388888888888889,0.2777777777777778,all_gt
hy_two_of_three_families__ni_k5_r1,R1,252,71,3,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r1,R2,193,19,20,1,0.05263157894736842,0.05,all_gt
hy_two_of_three_families__ni_k5_r1,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_two_of_three_families__ni_k5_r1,R4,136,40,19,2,0.05,0.10526315789473684,all_gt
hy_two_of_three_families__ni_k5_r1,R5,173,66,21,9,0.13636363636363635,0.42857142857142855,all_gt
hy_two_of_three_families__ni_k5_r1,L1,217,10,5,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r1,L2,188,6,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r1,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r1,L5,225,18,36,10,0.5555555555555556,0.2777777777777778,assigned_gt
hy_two_of_three_families__ni_k5_r1,R1,190,9,3,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r1,R2,182,8,20,1,0.125,0.05,assigned_gt
hy_two_of_three_families__ni_k5_r1,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_two_of_three_families__ni_k5_r1,R4,118,22,19,2,0.09090909090909091,0.10526315789473684,assigned_gt
hy_two_of_three_families__ni_k5_r1,R5,134,27,21,9,0.3333333333333333,0.42857142857142855,assigned_gt
hy_two_of_three_families__ni_k5_r2,L1,252,45,5,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r2,L2,207,25,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r2,L3,110,20,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r2,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r2,L5,279,72,36,10,0.1388888888888889,0.2777777777777778,all_gt
hy_two_of_three_families__ni_k5_r2,R1,252,71,3,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r2,R2,193,19,20,1,0.05263157894736842,0.05,all_gt
hy_two_of_three_families__ni_k5_r2,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_two_of_three_families__ni_k5_r2,R4,136,40,19,2,0.05,0.10526315789473684,all_gt
hy_two_of_three_families__ni_k5_r2,R5,173,66,21,9,0.13636363636363635,0.42857142857142855,all_gt
hy_two_of_three_families__ni_k5_r2,L1,217,10,5,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r2,L2,188,6,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r2,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r2,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r2,L5,225,18,36,10,0.5555555555555556,0.2777777777777778,assigned_gt
hy_two_of_three_families__ni_k5_r2,R1,190,9,3,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r2,R2,182,8,20,1,0.125,0.05,assigned_gt
hy_two_of_three_families__ni_k5_r2,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_two_of_three_families__ni_k5_r2,R4,118,22,19,2,0.09090909090909091,0.10526315789473684,assigned_gt
hy_two_of_three_families__ni_k5_r2,R5,134,27,21,9,0.3333333333333333,0.42857142857142855,assigned_gt
hy_two_of_three_families__ni_k5_r4,L1,252,45,5,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r4,L2,207,25,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r4,L3,110,20,20,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r4,L4,38,13,8,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r4,L5,279,72,36,10,0.1388888888888889,0.2777777777777778,all_gt
hy_two_of_three_families__ni_k5_r4,R1,252,71,3,0,0.0,0.0,all_gt
hy_two_of_three_families__ni_k5_r4,R2,193,19,20,1,0.05263157894736842,0.05,all_gt
hy_two_of_three_families__ni_k5_r4,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_two_of_three_families__ni_k5_r4,R4,136,40,19,2,0.05,0.10526315789473684,all_gt
hy_two_of_three_families__ni_k5_r4,R5,173,66,21,9,0.13636363636363635,0.42857142857142855,all_gt
hy_two_of_three_families__ni_k5_r4,L1,217,10,5,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r4,L2,188,6,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r4,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r4,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r4,L5,225,18,36,10,0.5555555555555556,0.2777777777777778,assigned_gt
hy_two_of_three_families__ni_k5_r4,R1,190,9,3,0,0.0,0.0,assigned_gt
hy_two_of_three_families__ni_k5_r4,R2,182,8,20,1,0.125,0.05,assigned_gt
hy_two_of_three_families__ni_k5_r4,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_two_of_three_families__ni_k5_r4,R4,118,22,19,2,0.09090909090909091,0.10526315789473684,assigned_gt
hy_two_of_three_families__ni_k5_r4,R5,134,27,21,9,0.3333333333333333,0.42857142857142855,assigned_gt
hy_hierarchical__ni_k2_r1,L1,252,45,28,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r1,L2,207,25,25,1,0.04,0.04,all_gt
hy_hierarchical__ni_k2_r1,L3,110,20,21,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r1,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r1,L5,279,72,72,10,0.1388888888888889,0.1388888888888889,all_gt
hy_hierarchical__ni_k2_r1,R1,252,71,20,1,0.014084507042253521,0.05,all_gt
hy_hierarchical__ni_k2_r1,R2,193,19,29,1,0.05263157894736842,0.034482758620689655,all_gt
hy_hierarchical__ni_k2_r1,R3,160,21,25,2,0.09523809523809523,0.08,all_gt
hy_hierarchical__ni_k2_r1,R4,136,40,22,1,0.025,0.045454545454545456,all_gt
hy_hierarchical__ni_k2_r1,R5,173,66,31,11,0.16666666666666666,0.3548387096774194,all_gt
hy_hierarchical__ni_k2_r1,L1,217,10,28,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r1,L2,188,6,25,1,0.16666666666666666,0.04,assigned_gt
hy_hierarchical__ni_k2_r1,L3,95,5,21,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r1,L5,225,18,72,10,0.5555555555555556,0.1388888888888889,assigned_gt
hy_hierarchical__ni_k2_r1,R1,190,9,20,1,0.1111111111111111,0.05,assigned_gt
hy_hierarchical__ni_k2_r1,R2,182,8,29,1,0.125,0.034482758620689655,assigned_gt
hy_hierarchical__ni_k2_r1,R3,146,7,25,2,0.2857142857142857,0.08,assigned_gt
hy_hierarchical__ni_k2_r1,R4,118,22,22,1,0.045454545454545456,0.045454545454545456,assigned_gt
hy_hierarchical__ni_k2_r1,R5,134,27,31,11,0.4074074074074074,0.3548387096774194,assigned_gt
hy_hierarchical__ni_k2_r2,L1,252,45,32,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r2,L2,207,25,31,1,0.04,0.03225806451612903,all_gt
hy_hierarchical__ni_k2_r2,L3,110,20,25,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r2,L4,38,13,9,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r2,L5,279,72,75,11,0.1527777777777778,0.14666666666666667,all_gt
hy_hierarchical__ni_k2_r2,R1,252,71,27,3,0.04225352112676056,0.1111111111111111,all_gt
hy_hierarchical__ni_k2_r2,R2,193,19,35,1,0.05263157894736842,0.02857142857142857,all_gt
hy_hierarchical__ni_k2_r2,R3,160,21,30,2,0.09523809523809523,0.06666666666666667,all_gt
hy_hierarchical__ni_k2_r2,R4,136,40,27,3,0.075,0.1111111111111111,all_gt
hy_hierarchical__ni_k2_r2,R5,173,66,35,11,0.16666666666666666,0.3142857142857143,all_gt
hy_hierarchical__ni_k2_r2,L1,217,10,32,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r2,L2,188,6,31,1,0.16666666666666666,0.03225806451612903,assigned_gt
hy_hierarchical__ni_k2_r2,L3,95,5,25,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r2,L4,30,5,9,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r2,L5,225,18,75,11,0.6111111111111112,0.14666666666666667,assigned_gt
hy_hierarchical__ni_k2_r2,R1,190,9,27,3,0.3333333333333333,0.1111111111111111,assigned_gt
hy_hierarchical__ni_k2_r2,R2,182,8,35,1,0.125,0.02857142857142857,assigned_gt
hy_hierarchical__ni_k2_r2,R3,146,7,30,2,0.2857142857142857,0.06666666666666667,assigned_gt
hy_hierarchical__ni_k2_r2,R4,118,22,27,3,0.13636363636363635,0.1111111111111111,assigned_gt
hy_hierarchical__ni_k2_r2,R5,134,27,35,11,0.4074074074074074,0.3142857142857143,assigned_gt
hy_hierarchical__ni_k2_r4,L1,252,45,46,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r4,L2,207,25,39,1,0.04,0.02564102564102564,all_gt
hy_hierarchical__ni_k2_r4,L3,110,20,26,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r4,L4,38,13,10,0,0.0,0.0,all_gt
hy_hierarchical__ni_k2_r4,L5,279,72,83,12,0.16666666666666666,0.14457831325301204,all_gt
hy_hierarchical__ni_k2_r4,R1,252,71,33,3,0.04225352112676056,0.09090909090909091,all_gt
hy_hierarchical__ni_k2_r4,R2,193,19,41,1,0.05263157894736842,0.024390243902439025,all_gt
hy_hierarchical__ni_k2_r4,R3,160,21,40,2,0.09523809523809523,0.05,all_gt
hy_hierarchical__ni_k2_r4,R4,136,40,41,7,0.175,0.17073170731707318,all_gt
hy_hierarchical__ni_k2_r4,R5,173,66,44,12,0.18181818181818182,0.2727272727272727,all_gt
hy_hierarchical__ni_k2_r4,L1,217,10,46,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r4,L2,188,6,39,1,0.16666666666666666,0.02564102564102564,assigned_gt
hy_hierarchical__ni_k2_r4,L3,95,5,26,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r4,L4,30,5,10,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k2_r4,L5,225,18,83,12,0.6666666666666666,0.14457831325301204,assigned_gt
hy_hierarchical__ni_k2_r4,R1,190,9,33,3,0.3333333333333333,0.09090909090909091,assigned_gt
hy_hierarchical__ni_k2_r4,R2,182,8,41,1,0.125,0.024390243902439025,assigned_gt
hy_hierarchical__ni_k2_r4,R3,146,7,40,2,0.2857142857142857,0.05,assigned_gt
hy_hierarchical__ni_k2_r4,R4,118,22,41,7,0.3181818181818182,0.17073170731707318,assigned_gt
hy_hierarchical__ni_k2_r4,R5,134,27,44,12,0.4444444444444444,0.2727272727272727,assigned_gt
hy_hierarchical__ni_k3_r1,L1,252,45,19,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r1,L2,207,25,21,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r1,L3,110,20,20,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r1,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r1,L5,279,72,67,10,0.1388888888888889,0.14925373134328357,all_gt
hy_hierarchical__ni_k3_r1,R1,252,71,10,1,0.014084507042253521,0.1,all_gt
hy_hierarchical__ni_k3_r1,R2,193,19,25,1,0.05263157894736842,0.04,all_gt
hy_hierarchical__ni_k3_r1,R3,160,21,19,2,0.09523809523809523,0.10526315789473684,all_gt
hy_hierarchical__ni_k3_r1,R4,136,40,20,1,0.025,0.05,all_gt
hy_hierarchical__ni_k3_r1,R5,173,66,26,9,0.13636363636363635,0.34615384615384615,all_gt
hy_hierarchical__ni_k3_r1,L1,217,10,19,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r1,L2,188,6,21,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r1,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r1,L5,225,18,67,10,0.5555555555555556,0.14925373134328357,assigned_gt
hy_hierarchical__ni_k3_r1,R1,190,9,10,1,0.1111111111111111,0.1,assigned_gt
hy_hierarchical__ni_k3_r1,R2,182,8,25,1,0.125,0.04,assigned_gt
hy_hierarchical__ni_k3_r1,R3,146,7,19,2,0.2857142857142857,0.10526315789473684,assigned_gt
hy_hierarchical__ni_k3_r1,R4,118,22,20,1,0.045454545454545456,0.05,assigned_gt
hy_hierarchical__ni_k3_r1,R5,134,27,26,9,0.3333333333333333,0.34615384615384615,assigned_gt
hy_hierarchical__ni_k3_r2,L1,252,45,22,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r2,L2,207,25,22,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r2,L3,110,20,22,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r2,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r2,L5,279,72,67,10,0.1388888888888889,0.14925373134328357,all_gt
hy_hierarchical__ni_k3_r2,R1,252,71,13,2,0.028169014084507043,0.15384615384615385,all_gt
hy_hierarchical__ni_k3_r2,R2,193,19,26,1,0.05263157894736842,0.038461538461538464,all_gt
hy_hierarchical__ni_k3_r2,R3,160,21,19,2,0.09523809523809523,0.10526315789473684,all_gt
hy_hierarchical__ni_k3_r2,R4,136,40,21,2,0.05,0.09523809523809523,all_gt
hy_hierarchical__ni_k3_r2,R5,173,66,29,9,0.13636363636363635,0.3103448275862069,all_gt
hy_hierarchical__ni_k3_r2,L1,217,10,22,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r2,L2,188,6,22,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r2,L3,95,5,22,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r2,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r2,L5,225,18,67,10,0.5555555555555556,0.14925373134328357,assigned_gt
hy_hierarchical__ni_k3_r2,R1,190,9,13,2,0.2222222222222222,0.15384615384615385,assigned_gt
hy_hierarchical__ni_k3_r2,R2,182,8,26,1,0.125,0.038461538461538464,assigned_gt
hy_hierarchical__ni_k3_r2,R3,146,7,19,2,0.2857142857142857,0.10526315789473684,assigned_gt
hy_hierarchical__ni_k3_r2,R4,118,22,21,2,0.09090909090909091,0.09523809523809523,assigned_gt
hy_hierarchical__ni_k3_r2,R5,134,27,29,9,0.3333333333333333,0.3103448275862069,assigned_gt
hy_hierarchical__ni_k3_r4,L1,252,45,27,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r4,L2,207,25,24,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r4,L3,110,20,22,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r4,L4,38,13,9,0,0.0,0.0,all_gt
hy_hierarchical__ni_k3_r4,L5,279,72,69,10,0.1388888888888889,0.14492753623188406,all_gt
hy_hierarchical__ni_k3_r4,R1,252,71,14,2,0.028169014084507043,0.14285714285714285,all_gt
hy_hierarchical__ni_k3_r4,R2,193,19,27,1,0.05263157894736842,0.037037037037037035,all_gt
hy_hierarchical__ni_k3_r4,R3,160,21,22,2,0.09523809523809523,0.09090909090909091,all_gt
hy_hierarchical__ni_k3_r4,R4,136,40,28,5,0.125,0.17857142857142858,all_gt
hy_hierarchical__ni_k3_r4,R5,173,66,32,10,0.15151515151515152,0.3125,all_gt
hy_hierarchical__ni_k3_r4,L1,217,10,27,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r4,L2,188,6,24,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r4,L3,95,5,22,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r4,L4,30,5,9,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k3_r4,L5,225,18,69,10,0.5555555555555556,0.14492753623188406,assigned_gt
hy_hierarchical__ni_k3_r4,R1,190,9,14,2,0.2222222222222222,0.14285714285714285,assigned_gt
hy_hierarchical__ni_k3_r4,R2,182,8,27,1,0.125,0.037037037037037035,assigned_gt
hy_hierarchical__ni_k3_r4,R3,146,7,22,2,0.2857142857142857,0.09090909090909091,assigned_gt
hy_hierarchical__ni_k3_r4,R4,118,22,28,5,0.22727272727272727,0.17857142857142858,assigned_gt
hy_hierarchical__ni_k3_r4,R5,134,27,32,10,0.37037037037037035,0.3125,assigned_gt
hy_hierarchical__ni_k5_r1,L1,252,45,18,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r1,L2,207,25,21,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r1,L3,110,20,20,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r1,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r1,L5,279,72,65,10,0.1388888888888889,0.15384615384615385,all_gt
hy_hierarchical__ni_k5_r1,R1,252,71,7,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r1,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
hy_hierarchical__ni_k5_r1,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_hierarchical__ni_k5_r1,R4,136,40,18,1,0.025,0.05555555555555555,all_gt
hy_hierarchical__ni_k5_r1,R5,173,66,24,9,0.13636363636363635,0.375,all_gt
hy_hierarchical__ni_k5_r1,L1,217,10,18,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r1,L2,188,6,21,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r1,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r1,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r1,L5,225,18,65,10,0.5555555555555556,0.15384615384615385,assigned_gt
hy_hierarchical__ni_k5_r1,R1,190,9,7,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r1,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
hy_hierarchical__ni_k5_r1,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_hierarchical__ni_k5_r1,R4,118,22,18,1,0.045454545454545456,0.05555555555555555,assigned_gt
hy_hierarchical__ni_k5_r1,R5,134,27,24,9,0.3333333333333333,0.375,assigned_gt
hy_hierarchical__ni_k5_r2,L1,252,45,18,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r2,L2,207,25,21,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r2,L3,110,20,20,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r2,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r2,L5,279,72,65,10,0.1388888888888889,0.15384615384615385,all_gt
hy_hierarchical__ni_k5_r2,R1,252,71,7,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r2,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
hy_hierarchical__ni_k5_r2,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_hierarchical__ni_k5_r2,R4,136,40,18,1,0.025,0.05555555555555555,all_gt
hy_hierarchical__ni_k5_r2,R5,173,66,24,9,0.13636363636363635,0.375,all_gt
hy_hierarchical__ni_k5_r2,L1,217,10,18,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r2,L2,188,6,21,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r2,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r2,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r2,L5,225,18,65,10,0.5555555555555556,0.15384615384615385,assigned_gt
hy_hierarchical__ni_k5_r2,R1,190,9,7,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r2,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
hy_hierarchical__ni_k5_r2,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_hierarchical__ni_k5_r2,R4,118,22,18,1,0.045454545454545456,0.05555555555555555,assigned_gt
hy_hierarchical__ni_k5_r2,R5,134,27,24,9,0.3333333333333333,0.375,assigned_gt
hy_hierarchical__ni_k5_r4,L1,252,45,18,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r4,L2,207,25,21,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r4,L3,110,20,20,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r4,L4,38,13,8,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r4,L5,279,72,65,10,0.1388888888888889,0.15384615384615385,all_gt
hy_hierarchical__ni_k5_r4,R1,252,71,7,0,0.0,0.0,all_gt
hy_hierarchical__ni_k5_r4,R2,193,19,24,1,0.05263157894736842,0.041666666666666664,all_gt
hy_hierarchical__ni_k5_r4,R3,160,21,17,2,0.09523809523809523,0.11764705882352941,all_gt
hy_hierarchical__ni_k5_r4,R4,136,40,18,1,0.025,0.05555555555555555,all_gt
hy_hierarchical__ni_k5_r4,R5,173,66,24,9,0.13636363636363635,0.375,all_gt
hy_hierarchical__ni_k5_r4,L1,217,10,18,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r4,L2,188,6,21,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r4,L3,95,5,20,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r4,L4,30,5,8,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r4,L5,225,18,65,10,0.5555555555555556,0.15384615384615385,assigned_gt
hy_hierarchical__ni_k5_r4,R1,190,9,7,0,0.0,0.0,assigned_gt
hy_hierarchical__ni_k5_r4,R2,182,8,24,1,0.125,0.041666666666666664,assigned_gt
hy_hierarchical__ni_k5_r4,R3,146,7,17,2,0.2857142857142857,0.11764705882352941,assigned_gt
hy_hierarchical__ni_k5_r4,R4,118,22,18,1,0.045454545454545456,0.05555555555555555,assigned_gt
hy_hierarchical__ni_k5_r4,R5,134,27,24,9,0.3333333333333333,0.375,assigned_gt
```

### Predicted-finger workload table (all ten fingers)

```csv
set_id,predicted_finger_id,eligible_notes,hard_count,hard_percentage
ni_k2_r1,L1,56961,2716,0.047681747160337774
ni_k2_r1,L2,42151,2047,0.048563497900405685
ni_k2_r1,L3,32568,1613,0.04952714320805699
ni_k2_r1,L4,17383,1076,0.061899557038485875
ni_k2_r1,L5,41112,2042,0.04966919634170072
ni_k2_r1,R1,65498,3555,0.05427646645699105
ni_k2_r1,R2,58395,2597,0.04447298570083055
ni_k2_r1,R3,48125,2195,0.04561038961038961
ni_k2_r1,R4,31856,1758,0.05518583626318433
ni_k2_r1,R5,40324,2052,0.050887808749132034
ni_k2_r1,NA,0,0,
ni_k2_r2,L1,56961,5129,0.09004406523761872
ni_k2_r2,L2,42151,3856,0.09148062916656781
ni_k2_r2,L3,32568,3038,0.09328174895603046
ni_k2_r2,L4,17383,1974,0.11355922452971294
ni_k2_r2,L5,41112,3818,0.092868262307842
ni_k2_r2,R1,65498,6397,0.09766710433906378
ni_k2_r2,R2,58395,5025,0.08605188800410994
ni_k2_r2,R3,48125,4174,0.08673246753246754
ni_k2_r2,R4,31856,3259,0.1023041185334003
ni_k2_r2,R5,40324,3859,0.09569983136593592
ni_k2_r2,NA,0,0,
ni_k2_r4,L1,56961,9185,0.16125068029002299
ni_k2_r4,L2,42151,6924,0.16426656544328722
ni_k2_r4,L3,32568,5461,0.16767993122083028
ni_k2_r4,L4,17383,3426,0.1970891100500489
ni_k2_r4,L5,41112,6852,0.16666666666666666
ni_k2_r4,R1,65498,11175,0.17061589666860058
ni_k2_r4,R2,58395,9042,0.1548420241459029
ni_k2_r4,R3,48125,7639,0.15873246753246753
ni_k2_r4,R4,31856,5742,0.18024861878453038
ni_k2_r4,R5,40324,6927,0.1717835532189267
ni_k2_r4,NA,0,0,
ni_k3_r1,L1,56961,636,0.011165534312950967
ni_k3_r1,L2,42151,534,0.012668738582714527
ni_k3_r1,L3,32568,417,0.012803979366249078
ni_k3_r1,L4,17383,311,0.017891042973019616
ni_k3_r1,L5,41112,530,0.012891613154310177
ni_k3_r1,R1,65498,899,0.013725609942288315
ni_k3_r1,R2,58395,608,0.010411850329651511
ni_k3_r1,R3,48125,522,0.010846753246753247
ni_k3_r1,R4,31856,427,0.013404068307383225
ni_k3_r1,R5,40324,513,0.012721952187283008
ni_k3_r1,NA,0,0,
ni_k3_r2,L1,56961,1299,0.022805077158055513
ni_k3_r2,L2,42151,1016,0.024103817228535503
ni_k3_r2,L3,32568,831,0.02551584377302874
ni_k3_r2,L4,17383,585,0.0336535695794742
ni_k3_r2,L5,41112,988,0.02403191282350652
ni_k3_r2,R1,65498,1648,0.025161073620568568
ni_k3_r2,R2,58395,1218,0.02085795016696635
ni_k3_r2,R3,48125,1045,0.021714285714285714
ni_k3_r2,R4,31856,832,0.026117528879959818
ni_k3_r2,R5,40324,1014,0.02514631484971729
ni_k3_r2,NA,0,0,
ni_k3_r4,L1,56961,2472,0.04339811449939432
ni_k3_r4,L2,42151,1929,0.04576403881283955
ni_k3_r4,L3,32568,1528,0.046917219356423484
ni_k3_r4,L4,17383,1077,0.0619570845078525
ni_k3_r4,L5,41112,1859,0.045217941233703055
ni_k3_r4,R1,65498,3054,0.04662737793520413
ni_k3_r4,R2,58395,2312,0.039592430858806406
ni_k3_r4,R3,48125,2082,0.04326233766233766
ni_k3_r4,R4,31856,1599,0.050194625816172775
ni_k3_r4,R5,40324,1912,0.04741593095923023
ni_k3_r4,NA,0,0,
ni_k5_r1,L1,56961,71,0.0012464668808482998
ni_k5_r1,L2,42151,51,0.0012099357073379043
ni_k5_r1,L3,32568,60,0.0018422991893883567
ni_k5_r1,L4,17383,44,0.002531208652131393
ni_k5_r1,L5,41112,61,0.0014837517026658883
ni_k5_r1,R1,65498,85,0.001297749549604568
ni_k5_r1,R2,58395,55,0.0009418614607415018
ni_k5_r1,R3,48125,56,0.0011636363636363637
ni_k5_r1,R4,31856,39,0.0012242591662481165
ni_k5_r1,R5,40324,58,0.0014383493701021724
ni_k5_r1,NA,0,0,
ni_k5_r2,L1,56961,149,0.002615824862625305
ni_k5_r2,L2,42151,106,0.002514768332898389
ni_k5_r2,L3,32568,113,0.0034696634733480717
ni_k5_r2,L4,17383,80,0.004602197549329805
ni_k5_r2,L5,41112,109,0.002651294026075112
ni_k5_r2,R1,65498,172,0.0026260343827292434
ni_k5_r2,R2,58395,116,0.001986471444472986
ni_k5_r2,R3,48125,115,0.0023896103896103894
ni_k5_r2,R4,31856,83,0.002605474635861376
ni_k5_r2,R5,40324,106,0.0026287074694970737
ni_k5_r2,NA,0,0,
ni_k5_r4,L1,56961,257,0.0045118589912396204
ni_k5_r4,L2,42151,196,0.004649948992906455
ni_k5_r4,L3,32568,199,0.006110292311471383
ni_k5_r4,L4,17383,133,0.007651153425760801
ni_k5_r4,L5,41112,225,0.005472854640980736
ni_k5_r4,R1,65498,333,0.005084124706097896
ni_k5_r4,R2,58395,242,0.004144190427262608
ni_k5_r4,R3,48125,257,0.005340259740259741
ni_k5_r4,R4,31856,187,0.005870165745856353
ni_k5_r4,R5,40324,218,0.0054062097014185104
ni_k5_r4,NA,0,0,
ni_w5_q995,L1,56961,92,0.0016151401836344166
ni_w5_q995,L2,42151,77,0.0018267656757846788
ni_w5_q995,L3,32568,56,0.0017194792434291329
ni_w5_q995,L4,17383,36,0.002070988897198412
ni_w5_q995,L5,41112,72,0.0017513134851138354
ni_w5_q995,R1,65498,141,0.0021527374881675776
ni_w5_q995,R2,58395,94,0.0016097268601763849
ni_w5_q995,R3,48125,77,0.0016
ni_w5_q995,R4,31856,74,0.0023229532898041186
ni_w5_q995,R5,40324,66,0.0016367423866679894
ni_w5_q995,NA,0,0,
ni_w5_q990,L1,56961,92,0.0016151401836344166
ni_w5_q990,L2,42151,77,0.0018267656757846788
ni_w5_q990,L3,32568,56,0.0017194792434291329
ni_w5_q990,L4,17383,36,0.002070988897198412
ni_w5_q990,L5,41112,72,0.0017513134851138354
ni_w5_q990,R1,65498,141,0.0021527374881675776
ni_w5_q990,R2,58395,94,0.0016097268601763849
ni_w5_q990,R3,48125,77,0.0016
ni_w5_q990,R4,31856,74,0.0023229532898041186
ni_w5_q990,R5,40324,66,0.0016367423866679894
ni_w5_q990,NA,0,0,
ni_w5_q975,L1,56961,1058,0.01857411211179579
ni_w5_q975,L2,42151,811,0.019240350169628242
ni_w5_q975,L3,32568,651,0.019988946204863668
ni_w5_q975,L4,17383,456,0.026232526031179888
ni_w5_q975,L5,41112,836,0.020334695466043978
ni_w5_q975,R1,65498,1456,0.022229686402638248
ni_w5_q975,R2,58395,1009,0.017278876616148645
ni_w5_q975,R3,48125,864,0.017953246753246752
ni_w5_q975,R4,31856,691,0.021691361125062782
ni_w5_q975,R5,40324,795,0.019715306021228052
ni_w5_q975,NA,0,0,
ni_w9_q995,L1,56961,121,0.0021242604589104827
ni_w9_q995,L2,42151,113,0.002680837939787905
ni_w9_q995,L3,32568,87,0.002671333824613117
ni_w9_q995,L4,17383,59,0.003394120692630731
ni_w9_q995,L5,41112,100,0.0024323798404358825
ni_w9_q995,R1,65498,175,0.0026718373080094047
ni_w9_q995,R2,58395,126,0.0021577189827896226
ni_w9_q995,R3,48125,121,0.002514285714285714
ni_w9_q995,R4,31856,88,0.0027624309392265192
ni_w9_q995,R5,40324,104,0.0025791092153556196
ni_w9_q995,NA,0,0,
ni_w9_q990,L1,56961,615,0.01079686101016485
ni_w9_q990,L2,42151,491,0.011648596711821784
ni_w9_q990,L3,32568,415,0.012742569393269468
ni_w9_q990,L4,17383,296,0.01702813093252028
ni_w9_q990,L5,41112,468,0.01138353765323993
ni_w9_q990,R1,65498,813,0.012412592750923693
ni_w9_q990,R2,58395,593,0.010154979022176555
ni_w9_q990,R3,48125,465,0.009662337662337662
ni_w9_q990,R4,31856,395,0.012399547965846308
ni_w9_q990,R5,40324,509,0.012622755679000099
ni_w9_q990,NA,0,0,
ni_w9_q975,L1,56961,615,0.01079686101016485
ni_w9_q975,L2,42151,491,0.011648596711821784
ni_w9_q975,L3,32568,415,0.012742569393269468
ni_w9_q975,L4,17383,296,0.01702813093252028
ni_w9_q975,L5,41112,468,0.01138353765323993
ni_w9_q975,R1,65498,813,0.012412592750923693
ni_w9_q975,R2,58395,593,0.010154979022176555
ni_w9_q975,R3,48125,465,0.009662337662337662
ni_w9_q975,R4,31856,395,0.012399547965846308
ni_w9_q975,R5,40324,509,0.012622755679000099
ni_w9_q975,NA,0,0,
ni_w17_q995,L1,56961,115,0.002018925229543021
ni_w17_q995,L2,42151,91,0.0021589048895637114
ni_w17_q995,L3,32568,85,0.002609923851633505
ni_w17_q995,L4,17383,62,0.003566703100730599
ni_w17_q995,L5,41112,81,0.001970227670753065
ni_w17_q995,R1,65498,154,0.0023512168310482764
ni_w17_q995,R2,58395,116,0.001986471444472986
ni_w17_q995,R3,48125,90,0.0018701298701298702
ni_w17_q995,R4,31856,90,0.0028252134605725766
ni_w17_q995,R5,40324,120,0.0029758952484872534
ni_w17_q995,NA,0,0,
ni_w17_q990,L1,56961,313,0.005494987798669265
ni_w17_q990,L2,42151,231,0.005480297027354037
ni_w17_q990,L3,32568,202,0.006202407270940801
ni_w17_q990,L4,17383,151,0.008686647874360007
ni_w17_q990,L5,41112,232,0.005643121229811247
ni_w17_q990,R1,65498,389,0.005939112644660905
ni_w17_q990,R2,58395,272,0.004657933042212519
ni_w17_q990,R3,48125,239,0.004966233766233766
ni_w17_q990,R4,31856,211,0.0066235560020090406
ni_w17_q990,R5,40324,256,0.00634857653010614
ni_w17_q990,NA,0,0,
ni_w17_q975,L1,56961,744,0.013061568441565282
ni_w17_q975,L2,42151,580,0.013760053142274204
ni_w17_q975,L3,32568,464,0.014247113731269958
ni_w17_q975,L4,17383,355,0.02042225162515101
ni_w17_q975,L5,41112,574,0.013961860284101965
ni_w17_q975,R1,65498,949,0.014488992030291001
ni_w17_q975,R2,58395,680,0.011644832605531296
ni_w17_q975,R3,48125,595,0.012363636363636363
ni_w17_q975,R4,31856,478,0.015005022601707685
ni_w17_q975,R5,40324,587,0.014557087590516814
ni_w17_q975,NA,0,0,
mandatory_missing__ni_k2_r1,L1,56961,2716,0.047681747160337774
mandatory_missing__ni_k2_r1,L2,42151,2047,0.048563497900405685
mandatory_missing__ni_k2_r1,L3,32568,1613,0.04952714320805699
mandatory_missing__ni_k2_r1,L4,17383,1076,0.061899557038485875
mandatory_missing__ni_k2_r1,L5,41112,2042,0.04966919634170072
mandatory_missing__ni_k2_r1,R1,65498,3555,0.05427646645699105
mandatory_missing__ni_k2_r1,R2,58395,2597,0.04447298570083055
mandatory_missing__ni_k2_r1,R3,48125,2195,0.04561038961038961
mandatory_missing__ni_k2_r1,R4,31856,1758,0.05518583626318433
mandatory_missing__ni_k2_r1,R5,40324,2052,0.050887808749132034
mandatory_missing__ni_k2_r1,NA,0,0,
mandatory_missing__ni_k2_r2,L1,56961,5129,0.09004406523761872
mandatory_missing__ni_k2_r2,L2,42151,3856,0.09148062916656781
mandatory_missing__ni_k2_r2,L3,32568,3038,0.09328174895603046
mandatory_missing__ni_k2_r2,L4,17383,1974,0.11355922452971294
mandatory_missing__ni_k2_r2,L5,41112,3818,0.092868262307842
mandatory_missing__ni_k2_r2,R1,65498,6397,0.09766710433906378
mandatory_missing__ni_k2_r2,R2,58395,5025,0.08605188800410994
mandatory_missing__ni_k2_r2,R3,48125,4174,0.08673246753246754
mandatory_missing__ni_k2_r2,R4,31856,3259,0.1023041185334003
mandatory_missing__ni_k2_r2,R5,40324,3859,0.09569983136593592
mandatory_missing__ni_k2_r2,NA,0,0,
mandatory_missing__ni_k2_r4,L1,56961,9185,0.16125068029002299
mandatory_missing__ni_k2_r4,L2,42151,6924,0.16426656544328722
mandatory_missing__ni_k2_r4,L3,32568,5461,0.16767993122083028
mandatory_missing__ni_k2_r4,L4,17383,3426,0.1970891100500489
mandatory_missing__ni_k2_r4,L5,41112,6852,0.16666666666666666
mandatory_missing__ni_k2_r4,R1,65498,11175,0.17061589666860058
mandatory_missing__ni_k2_r4,R2,58395,9042,0.1548420241459029
mandatory_missing__ni_k2_r4,R3,48125,7639,0.15873246753246753
mandatory_missing__ni_k2_r4,R4,31856,5742,0.18024861878453038
mandatory_missing__ni_k2_r4,R5,40324,6927,0.1717835532189267
mandatory_missing__ni_k2_r4,NA,0,0,
mandatory_missing__ni_k3_r1,L1,56961,636,0.011165534312950967
mandatory_missing__ni_k3_r1,L2,42151,534,0.012668738582714527
mandatory_missing__ni_k3_r1,L3,32568,417,0.012803979366249078
mandatory_missing__ni_k3_r1,L4,17383,311,0.017891042973019616
mandatory_missing__ni_k3_r1,L5,41112,530,0.012891613154310177
mandatory_missing__ni_k3_r1,R1,65498,899,0.013725609942288315
mandatory_missing__ni_k3_r1,R2,58395,608,0.010411850329651511
mandatory_missing__ni_k3_r1,R3,48125,522,0.010846753246753247
mandatory_missing__ni_k3_r1,R4,31856,427,0.013404068307383225
mandatory_missing__ni_k3_r1,R5,40324,513,0.012721952187283008
mandatory_missing__ni_k3_r1,NA,0,0,
mandatory_missing__ni_k3_r2,L1,56961,1299,0.022805077158055513
mandatory_missing__ni_k3_r2,L2,42151,1016,0.024103817228535503
mandatory_missing__ni_k3_r2,L3,32568,831,0.02551584377302874
mandatory_missing__ni_k3_r2,L4,17383,585,0.0336535695794742
mandatory_missing__ni_k3_r2,L5,41112,988,0.02403191282350652
mandatory_missing__ni_k3_r2,R1,65498,1648,0.025161073620568568
mandatory_missing__ni_k3_r2,R2,58395,1218,0.02085795016696635
mandatory_missing__ni_k3_r2,R3,48125,1045,0.021714285714285714
mandatory_missing__ni_k3_r2,R4,31856,832,0.026117528879959818
mandatory_missing__ni_k3_r2,R5,40324,1014,0.02514631484971729
mandatory_missing__ni_k3_r2,NA,0,0,
mandatory_missing__ni_k3_r4,L1,56961,2472,0.04339811449939432
mandatory_missing__ni_k3_r4,L2,42151,1929,0.04576403881283955
mandatory_missing__ni_k3_r4,L3,32568,1528,0.046917219356423484
mandatory_missing__ni_k3_r4,L4,17383,1077,0.0619570845078525
mandatory_missing__ni_k3_r4,L5,41112,1859,0.045217941233703055
mandatory_missing__ni_k3_r4,R1,65498,3054,0.04662737793520413
mandatory_missing__ni_k3_r4,R2,58395,2312,0.039592430858806406
mandatory_missing__ni_k3_r4,R3,48125,2082,0.04326233766233766
mandatory_missing__ni_k3_r4,R4,31856,1599,0.050194625816172775
mandatory_missing__ni_k3_r4,R5,40324,1912,0.04741593095923023
mandatory_missing__ni_k3_r4,NA,0,0,
mandatory_missing__ni_k5_r1,L1,56961,71,0.0012464668808482998
mandatory_missing__ni_k5_r1,L2,42151,51,0.0012099357073379043
mandatory_missing__ni_k5_r1,L3,32568,60,0.0018422991893883567
mandatory_missing__ni_k5_r1,L4,17383,44,0.002531208652131393
mandatory_missing__ni_k5_r1,L5,41112,61,0.0014837517026658883
mandatory_missing__ni_k5_r1,R1,65498,85,0.001297749549604568
mandatory_missing__ni_k5_r1,R2,58395,55,0.0009418614607415018
mandatory_missing__ni_k5_r1,R3,48125,56,0.0011636363636363637
mandatory_missing__ni_k5_r1,R4,31856,39,0.0012242591662481165
mandatory_missing__ni_k5_r1,R5,40324,58,0.0014383493701021724
mandatory_missing__ni_k5_r1,NA,0,0,
mandatory_missing__ni_k5_r2,L1,56961,149,0.002615824862625305
mandatory_missing__ni_k5_r2,L2,42151,106,0.002514768332898389
mandatory_missing__ni_k5_r2,L3,32568,113,0.0034696634733480717
mandatory_missing__ni_k5_r2,L4,17383,80,0.004602197549329805
mandatory_missing__ni_k5_r2,L5,41112,109,0.002651294026075112
mandatory_missing__ni_k5_r2,R1,65498,172,0.0026260343827292434
mandatory_missing__ni_k5_r2,R2,58395,116,0.001986471444472986
mandatory_missing__ni_k5_r2,R3,48125,115,0.0023896103896103894
mandatory_missing__ni_k5_r2,R4,31856,83,0.002605474635861376
mandatory_missing__ni_k5_r2,R5,40324,106,0.0026287074694970737
mandatory_missing__ni_k5_r2,NA,0,0,
mandatory_missing__ni_k5_r4,L1,56961,257,0.0045118589912396204
mandatory_missing__ni_k5_r4,L2,42151,196,0.004649948992906455
mandatory_missing__ni_k5_r4,L3,32568,199,0.006110292311471383
mandatory_missing__ni_k5_r4,L4,17383,133,0.007651153425760801
mandatory_missing__ni_k5_r4,L5,41112,225,0.005472854640980736
mandatory_missing__ni_k5_r4,R1,65498,333,0.005084124706097896
mandatory_missing__ni_k5_r4,R2,58395,242,0.004144190427262608
mandatory_missing__ni_k5_r4,R3,48125,257,0.005340259740259741
mandatory_missing__ni_k5_r4,R4,31856,187,0.005870165745856353
mandatory_missing__ni_k5_r4,R5,40324,218,0.0054062097014185104
mandatory_missing__ni_k5_r4,NA,0,0,
legacy_current_default__ni_k2_r1,L1,56961,3784,0.066431417987746
legacy_current_default__ni_k2_r1,L2,42151,5016,0.11900073545111622
legacy_current_default__ni_k2_r1,L3,32568,4762,0.1462171456644559
legacy_current_default__ni_k2_r1,L4,17383,3093,0.17793246275096358
legacy_current_default__ni_k2_r1,L5,41112,5194,0.12633780891223972
legacy_current_default__ni_k2_r1,R1,65498,4357,0.06652111514855416
legacy_current_default__ni_k2_r1,R2,58395,7228,0.123777720695265
legacy_current_default__ni_k2_r1,R3,48125,6854,0.14242077922077923
legacy_current_default__ni_k2_r1,R4,31856,4747,0.1490143144148669
legacy_current_default__ni_k2_r1,R5,40324,4396,0.10901696260291638
legacy_current_default__ni_k2_r1,NA,0,0,
legacy_current_default__ni_k2_r2,L1,56961,5580,0.09796176331173961
legacy_current_default__ni_k2_r2,L2,42151,6306,0.1496049915778985
legacy_current_default__ni_k2_r2,L3,32568,5706,0.17520265291083273
legacy_current_default__ni_k2_r2,L4,17383,3656,0.2103204280043721
legacy_current_default__ni_k2_r2,L5,41112,6444,0.15674255691768826
legacy_current_default__ni_k2_r2,R1,65498,6531,0.09971296833491099
legacy_current_default__ni_k2_r2,R2,58395,8926,0.15285555270142992
legacy_current_default__ni_k2_r2,R3,48125,8187,0.17011948051948053
legacy_current_default__ni_k2_r2,R4,31856,5756,0.18068809643395278
legacy_current_default__ni_k2_r2,R5,40324,5670,0.1406110504910227
legacy_current_default__ni_k2_r2,NA,0,0,
legacy_current_default__ni_k2_r4,L1,56961,9578,0.16815013781359175
legacy_current_default__ni_k2_r4,L2,42151,9134,0.21669711276126308
legacy_current_default__ni_k2_r4,L3,32568,7878,0.24189388356669123
legacy_current_default__ni_k2_r4,L4,17383,4878,0.28061899557038483
legacy_current_default__ni_k2_r4,L5,41112,9167,0.22297625997275736
legacy_current_default__ni_k2_r4,R1,65498,11289,0.1723564078292467
legacy_current_default__ni_k2_r4,R2,58395,12588,0.21556640123298226
legacy_current_default__ni_k2_r4,R3,48125,11261,0.2339948051948052
legacy_current_default__ni_k2_r4,R4,31856,7961,0.2499058262179809
legacy_current_default__ni_k2_r4,R5,40324,8527,0.21146215653209008
legacy_current_default__ni_k2_r4,NA,0,0,
legacy_current_default__ni_k3_r1,L1,56961,1796,0.03153034532399361
legacy_current_default__ni_k3_r1,L2,42151,3694,0.0876373039785533
legacy_current_default__ni_k3_r1,L3,32568,3734,0.1146524195529354
legacy_current_default__ni_k3_r1,L4,17383,2484,0.14289823390669046
legacy_current_default__ni_k3_r1,L5,41112,3892,0.09466822338976455
legacy_current_default__ni_k3_r1,R1,65498,1801,0.02749702280985679
legacy_current_default__ni_k3_r1,R2,58395,5509,0.09434026885863515
legacy_current_default__ni_k3_r1,R3,48125,5447,0.11318441558441558
legacy_current_default__ni_k3_r1,R4,31856,3638,0.11420140632847815
legacy_current_default__ni_k3_r1,R5,40324,3005,0.07452137684753497
legacy_current_default__ni_k3_r1,NA,0,0,
legacy_current_default__ni_k3_r2,L1,56961,1796,0.03153034532399361
legacy_current_default__ni_k3_r2,L2,42151,3694,0.0876373039785533
legacy_current_default__ni_k3_r2,L3,32568,3734,0.1146524195529354
legacy_current_default__ni_k3_r2,L4,17383,2484,0.14289823390669046
legacy_current_default__ni_k3_r2,L5,41112,3892,0.09466822338976455
legacy_current_default__ni_k3_r2,R1,65498,1801,0.02749702280985679
legacy_current_default__ni_k3_r2,R2,58395,5509,0.09434026885863515
legacy_current_default__ni_k3_r2,R3,48125,5447,0.11318441558441558
legacy_current_default__ni_k3_r2,R4,31856,3638,0.11420140632847815
legacy_current_default__ni_k3_r2,R5,40324,3005,0.07452137684753497
legacy_current_default__ni_k3_r2,NA,0,0,
legacy_current_default__ni_k3_r4,L1,56961,2950,0.05178982110566879
legacy_current_default__ni_k3_r4,L2,42151,4521,0.10725724182107187
legacy_current_default__ni_k3_r4,L3,32568,4351,0.13359739621714567
legacy_current_default__ni_k3_r4,L4,17383,2886,0.16602427659207272
legacy_current_default__ni_k3_r4,L5,41112,4670,0.11359213854835572
legacy_current_default__ni_k3_r4,R1,65498,3201,0.04887172127393203
legacy_current_default__ni_k3_r4,R2,58395,6477,0.11091703056768559
legacy_current_default__ni_k3_r4,R3,48125,6343,0.1318025974025974
legacy_current_default__ni_k3_r4,R4,31856,4315,0.13545328980411853
legacy_current_default__ni_k3_r4,R5,40324,3856,0.09562543398472374
legacy_current_default__ni_k3_r4,NA,0,0,
legacy_current_default__ni_k5_r1,L1,56961,1796,0.03153034532399361
legacy_current_default__ni_k5_r1,L2,42151,3694,0.0876373039785533
legacy_current_default__ni_k5_r1,L3,32568,3734,0.1146524195529354
legacy_current_default__ni_k5_r1,L4,17383,2484,0.14289823390669046
legacy_current_default__ni_k5_r1,L5,41112,3892,0.09466822338976455
legacy_current_default__ni_k5_r1,R1,65498,1801,0.02749702280985679
legacy_current_default__ni_k5_r1,R2,58395,5509,0.09434026885863515
legacy_current_default__ni_k5_r1,R3,48125,5447,0.11318441558441558
legacy_current_default__ni_k5_r1,R4,31856,3638,0.11420140632847815
legacy_current_default__ni_k5_r1,R5,40324,3005,0.07452137684753497
legacy_current_default__ni_k5_r1,NA,0,0,
legacy_current_default__ni_k5_r2,L1,56961,1796,0.03153034532399361
legacy_current_default__ni_k5_r2,L2,42151,3694,0.0876373039785533
legacy_current_default__ni_k5_r2,L3,32568,3734,0.1146524195529354
legacy_current_default__ni_k5_r2,L4,17383,2484,0.14289823390669046
legacy_current_default__ni_k5_r2,L5,41112,3892,0.09466822338976455
legacy_current_default__ni_k5_r2,R1,65498,1801,0.02749702280985679
legacy_current_default__ni_k5_r2,R2,58395,5509,0.09434026885863515
legacy_current_default__ni_k5_r2,R3,48125,5447,0.11318441558441558
legacy_current_default__ni_k5_r2,R4,31856,3638,0.11420140632847815
legacy_current_default__ni_k5_r2,R5,40324,3005,0.07452137684753497
legacy_current_default__ni_k5_r2,NA,0,0,
legacy_current_default__ni_k5_r4,L1,56961,1889,0.03316304137918927
legacy_current_default__ni_k5_r4,L2,42151,3770,0.08944034542478232
legacy_current_default__ni_k5_r4,L3,32568,3793,0.11646401375583394
legacy_current_default__ni_k5_r4,L4,17383,2518,0.14485416786515562
legacy_current_default__ni_k5_r4,L5,41112,3981,0.09683304144775248
legacy_current_default__ni_k5_r4,R1,65498,1937,0.0295734220892241
legacy_current_default__ni_k5_r4,R2,58395,5603,0.09594999571881155
legacy_current_default__ni_k5_r4,R3,48125,5551,0.11534545454545454
legacy_current_default__ni_k5_r4,R4,31856,3718,0.11671270718232044
legacy_current_default__ni_k5_r4,R5,40324,3100,0.07687729391925405
legacy_current_default__ni_k5_r4,NA,0,0,
bl_span_practical__ni_k2_r1,L1,56961,6927,0.12160952230473482
bl_span_practical__ni_k2_r1,L2,42151,5577,0.13231002823183316
bl_span_practical__ni_k2_r1,L3,32568,5281,0.1621530336526652
bl_span_practical__ni_k2_r1,L4,17383,4216,0.24253581084968073
bl_span_practical__ni_k2_r1,L5,41112,9702,0.23598949211908932
bl_span_practical__ni_k2_r1,R1,65498,5332,0.08140706586460655
bl_span_practical__ni_k2_r1,R2,58395,5438,0.09312441133658704
bl_span_practical__ni_k2_r1,R3,48125,5375,0.11168831168831168
bl_span_practical__ni_k2_r1,R4,31856,4496,0.1411351079859367
bl_span_practical__ni_k2_r1,R5,40324,5288,0.13113778395000497
bl_span_practical__ni_k2_r1,NA,0,0,
bl_span_practical__ni_k2_r2,L1,56961,9117,0.16005688102385843
bl_span_practical__ni_k2_r2,L2,42151,7119,0.16889279020663803
bl_span_practical__ni_k2_r2,L3,32568,6464,0.19847703267010564
bl_span_practical__ni_k2_r2,L4,17383,4864,0.2798136109992521
bl_span_practical__ni_k2_r2,L5,41112,11107,0.2701644288772135
bl_span_practical__ni_k2_r2,R1,65498,8040,0.12275183975083209
bl_span_practical__ni_k2_r2,R2,58395,7647,0.1309529925507321
bl_span_practical__ni_k2_r2,R3,48125,7159,0.14875844155844156
bl_span_practical__ni_k2_r2,R4,31856,5779,0.18141009542943246
bl_span_practical__ni_k2_r2,R5,40324,6876,0.1705187977383196
bl_span_practical__ni_k2_r2,NA,0,0,
bl_span_practical__ni_k2_r4,L1,56961,12813,0.22494338231421498
bl_span_practical__ni_k2_r4,L2,42151,9819,0.23294821000688
bl_span_practical__ni_k2_r4,L3,32568,8512,0.2613608450012282
bl_span_practical__ni_k2_r4,L4,17383,5966,0.3432088822412702
bl_span_practical__ni_k2_r4,L5,41112,13441,0.32693617435298694
bl_span_practical__ni_k2_r4,R1,65498,12641,0.19299825948883936
bl_span_practical__ni_k2_r4,R2,58395,11397,0.19517081941947084
bl_span_practical__ni_k2_r4,R3,48125,10293,0.21388051948051948
bl_span_practical__ni_k2_r4,R4,31856,7988,0.2507533902561527
bl_span_practical__ni_k2_r4,R5,40324,9611,0.23834441027675826
bl_span_practical__ni_k2_r4,NA,0,0,
bl_span_practical__ni_k3_r1,L1,56961,5101,0.0895525008339039
bl_span_practical__ni_k3_r1,L2,42151,4343,0.10303432896016702
bl_span_practical__ni_k3_r1,L3,32568,4352,0.13362810120363547
bl_span_practical__ni_k3_r1,L4,17383,3687,0.2121037795547374
bl_span_practical__ni_k3_r1,L5,41112,8556,0.2081144191476941
bl_span_practical__ni_k3_r1,R1,65498,2794,0.04265779107759016
bl_span_practical__ni_k3_r1,R2,58395,3701,0.06337871393098725
bl_span_practical__ni_k3_r1,R3,48125,3945,0.08197402597402598
bl_span_practical__ni_k3_r1,R4,31856,3394,0.10654193872425917
bl_span_practical__ni_k3_r1,R5,40324,3959,0.09817974407300863
bl_span_practical__ni_k3_r1,NA,0,0,
bl_span_practical__ni_k3_r2,L1,56961,5679,0.09969979459630274
bl_span_practical__ni_k3_r2,L2,42151,4754,0.1127849873075372
bl_span_practical__ni_k3_r2,L3,32568,4686,0.14388356669123065
bl_span_practical__ni_k3_r2,L4,17383,3882,0.2233216360812288
bl_span_practical__ni_k3_r2,L5,41112,8906,0.21662774858921968
bl_span_practical__ni_k3_r2,R1,65498,3500,0.0534367461601881
bl_span_practical__ni_k3_r2,R2,58395,4250,0.07278020378457059
bl_span_practical__ni_k3_r2,R3,48125,4407,0.09157402597402597
bl_span_practical__ni_k3_r2,R4,31856,3720,0.1167754897036665
bl_span_practical__ni_k3_r2,R5,40324,4388,0.10881856958635056
bl_span_practical__ni_k3_r2,NA,0,0,
bl_span_practical__ni_k3_r4,L1,56961,6728,0.11811590386404733
bl_span_practical__ni_k3_r4,L2,42151,5527,0.1311238167540509
bl_span_practical__ni_k3_r4,L3,32568,5261,0.16153893392286908
bl_span_practical__ni_k3_r4,L4,17383,4235,0.24362883276764655
bl_span_practical__ni_k3_r4,L5,41112,9584,0.23311928390737496
bl_span_practical__ni_k3_r4,R1,65498,4831,0.07375797734281962
bl_span_practical__ni_k3_r4,R2,58395,5256,0.09000770613922425
bl_span_practical__ni_k3_r4,R3,48125,5321,0.11056623376623377
bl_span_practical__ni_k3_r4,R4,31856,4387,0.1377134605725766
bl_span_practical__ni_k3_r4,R5,40324,5190,0.1287074694970737
bl_span_practical__ni_k3_r4,NA,0,0,
bl_span_practical__ni_k5_r1,L1,56961,4609,0.08091501202577202
bl_span_practical__ni_k5_r1,L2,42151,3953,0.0937818794334654
bl_span_practical__ni_k5_r1,L3,32568,4089,0.1255526897568165
bl_span_practical__ni_k5_r1,L4,17383,3510,0.2019214174768452
bl_span_practical__ni_k5_r1,L5,41112,8220,0.19994162288382955
bl_span_practical__ni_k5_r1,R1,65498,2033,0.03103911569818926
bl_span_practical__ni_k5_r1,R2,58395,3230,0.055312954876273655
bl_span_practical__ni_k5_r1,R3,48125,3555,0.07387012987012986
bl_span_practical__ni_k5_r1,R4,31856,3086,0.09687343043696635
bl_span_practical__ni_k5_r1,R5,40324,3581,0.08880567404027379
bl_span_practical__ni_k5_r1,NA,0,0,
bl_span_practical__ni_k5_r2,L1,56961,4674,0.08205614367725286
bl_span_practical__ni_k5_r2,L2,42151,3993,0.0947308486156912
bl_span_practical__ni_k5_r2,L3,32568,4133,0.12690370916236796
bl_span_practical__ni_k5_r2,L4,17383,3536,0.20341713168037737
bl_span_practical__ni_k5_r2,L5,41112,8253,0.20074430823117337
bl_span_practical__ni_k5_r2,R1,65498,2112,0.0322452593972335
bl_span_practical__ni_k5_r2,R2,58395,3285,0.05625481633701516
bl_span_practical__ni_k5_r2,R3,48125,3608,0.07497142857142858
bl_span_practical__ni_k5_r2,R4,31856,3119,0.0979093420391763
bl_span_practical__ni_k5_r2,R5,40324,3623,0.08984723737724432
bl_span_practical__ni_k5_r2,NA,0,0,
bl_span_practical__ni_k5_r4,L1,56961,4763,0.08361861624620354
bl_span_practical__ni_k5_r4,L2,42151,4066,0.0964627173732533
bl_span_practical__ni_k5_r4,L3,32568,4204,0.1290837632031442
bl_span_practical__ni_k5_r4,L4,17383,3568,0.2052580107001093
bl_span_practical__ni_k5_r4,L5,41112,8339,0.20283615489394824
bl_span_practical__ni_k5_r4,R1,65498,2257,0.034459067452441296
bl_span_practical__ni_k5_r4,R2,58395,3395,0.058138539258498156
bl_span_practical__ni_k5_r4,R3,48125,3731,0.07752727272727272
bl_span_practical__ni_k5_r4,R4,31856,3206,0.10064038171772978
bl_span_practical__ni_k5_r4,R5,40324,3718,0.09220315444896339
bl_span_practical__ni_k5_r4,NA,0,0,
bl_span_comfortable__ni_k2_r1,L1,56961,9023,0.15840662909710151
bl_span_comfortable__ni_k2_r1,L2,42151,8469,0.20092050010675902
bl_span_comfortable__ni_k2_r1,L3,32568,8684,0.26664210267747485
bl_span_comfortable__ni_k2_r1,L4,17383,5624,0.3235344877178853
bl_span_comfortable__ni_k2_r1,L5,41112,12558,0.3054582603619381
bl_span_comfortable__ni_k2_r1,R1,65498,7477,0.11415615743992183
bl_span_comfortable__ni_k2_r1,R2,58395,9173,0.15708536689785085
bl_span_comfortable__ni_k2_r1,R3,48125,9235,0.1918961038961039
bl_span_comfortable__ni_k2_r1,R4,31856,7400,0.23229532898041186
bl_span_comfortable__ni_k2_r1,R5,40324,7970,0.19764904275369508
bl_span_comfortable__ni_k2_r1,NA,0,0,
bl_span_comfortable__ni_k2_r2,L1,56961,11125,0.1953090711188357
bl_span_comfortable__ni_k2_r2,L2,42151,9858,0.2338734549595502
bl_span_comfortable__ni_k2_r2,L3,32568,9693,0.29762343404568903
bl_span_comfortable__ni_k2_r2,L4,17383,6180,0.35551976068572744
bl_span_comfortable__ni_k2_r2,L5,41112,13781,0.335206265810469
bl_span_comfortable__ni_k2_r2,R1,65498,10039,0.15327185562917953
bl_span_comfortable__ni_k2_r2,R2,58395,11129,0.19058138539258498
bl_span_comfortable__ni_k2_r2,R3,48125,10766,0.2237090909090909
bl_span_comfortable__ni_k2_r2,R4,31856,8530,0.26776745354093423
bl_span_comfortable__ni_k2_r2,R5,40324,9407,0.23328538835432994
bl_span_comfortable__ni_k2_r2,NA,0,0,
bl_span_comfortable__ni_k2_r4,L1,56961,14612,0.25652639525289234
bl_span_comfortable__ni_k2_r4,L2,42151,12293,0.2916419539275462
bl_span_comfortable__ni_k2_r4,L3,32568,11424,0.3507737656595431
bl_span_comfortable__ni_k2_r4,L4,17383,7142,0.4108611862164183
bl_span_comfortable__ni_k2_r4,L5,41112,15828,0.38499708114419146
bl_span_comfortable__ni_k2_r4,R1,65498,14402,0.219884576628294
bl_span_comfortable__ni_k2_r4,R2,58395,14484,0.2480349344978166
bl_span_comfortable__ni_k2_r4,R3,48125,13450,0.2794805194805195
bl_span_comfortable__ni_k2_r4,R4,31856,10429,0.3273794575590156
bl_span_comfortable__ni_k2_r4,R5,40324,11887,0.2947872234897332
bl_span_comfortable__ni_k2_r4,NA,0,0,
bl_span_comfortable__ni_k3_r1,L1,56961,7308,0.12829830936956865
bl_span_comfortable__ni_k3_r1,L2,42151,7383,0.17515598680932837
bl_span_comfortable__ni_k3_r1,L3,32568,7902,0.24263080324244657
bl_span_comfortable__ni_k3_r1,L4,17383,5161,0.29689926940113903
bl_span_comfortable__ni_k3_r1,L5,41112,11552,0.2809885191671531
bl_span_comfortable__ni_k3_r1,R1,65498,5083,0.07760542306635317
bl_span_comfortable__ni_k3_r1,R2,58395,7617,0.13043924993578218
bl_span_comfortable__ni_k3_r1,R3,48125,7998,0.16619220779220778
bl_span_comfortable__ni_k3_r1,R4,31856,6456,0.20266197890507281
bl_span_comfortable__ni_k3_r1,R5,40324,6786,0.16828687630195416
bl_span_comfortable__ni_k3_r1,NA,0,0,
bl_span_comfortable__ni_k3_r2,L1,56961,7862,0.13802426221449765
bl_span_comfortable__ni_k3_r2,L2,42151,7754,0.18395767597447274
bl_span_comfortable__ni_k3_r2,L3,32568,8182,0.25122819945959224
bl_span_comfortable__ni_k3_r2,L4,17383,5326,0.30639130184663177
bl_span_comfortable__ni_k3_r2,L5,41112,11848,0.28818836349484334
bl_span_comfortable__ni_k3_r2,R1,65498,5758,0.08791108125438944
bl_span_comfortable__ni_k3_r2,R2,58395,8105,0.13879612980563405
bl_span_comfortable__ni_k3_r2,R3,48125,8399,0.17452467532467533
bl_span_comfortable__ni_k3_r2,R4,31856,6742,0.21163987945755902
bl_span_comfortable__ni_k3_r2,R5,40324,7176,0.17795853585953775
bl_span_comfortable__ni_k3_r2,NA,0,0,
bl_span_comfortable__ni_k3_r4,L1,56961,8842,0.1552290163445164
bl_span_comfortable__ni_k3_r4,L2,42151,8445,0.20035111859742355
bl_span_comfortable__ni_k3_r4,L3,32568,8664,0.2660280029476787
bl_span_comfortable__ni_k3_r4,L4,17383,5636,0.32422481735028474
bl_span_comfortable__ni_k3_r4,L5,41112,12450,0.30283129013426735
bl_span_comfortable__ni_k3_r4,R1,65498,7021,0.10719411279733733
bl_span_comfortable__ni_k3_r4,R2,58395,8998,0.1540885349773097
bl_span_comfortable__ni_k3_r4,R3,48125,9182,0.19079480519480518
bl_span_comfortable__ni_k3_r4,R4,31856,7319,0.22975263686589653
bl_span_comfortable__ni_k3_r4,R5,40324,7891,0.19568991171510763
bl_span_comfortable__ni_k3_r4,NA,0,0,
bl_span_comfortable__ni_k5_r1,L1,56961,6847,0.12020505257983533
bl_span_comfortable__ni_k5_r1,L2,42151,7036,0.1669236791535195
bl_span_comfortable__ni_k5_r1,L3,32568,7675,0.23566077130926064
bl_span_comfortable__ni_k5_r1,L4,17383,5008,0.2880975665880458
bl_span_comfortable__ni_k5_r1,L5,41112,11254,0.2737400272426542
bl_span_comfortable__ni_k5_r1,R1,65498,4364,0.06662798864087453
bl_span_comfortable__ni_k5_r1,R2,58395,7203,0.12334960184947341
bl_span_comfortable__ni_k5_r1,R3,48125,7664,0.15925194805194806
bl_span_comfortable__ni_k5_r1,R4,31856,6179,0.1939665996986439
bl_span_comfortable__ni_k5_r1,R5,40324,6451,0.1599791687332606
bl_span_comfortable__ni_k5_r1,NA,0,0,
bl_span_comfortable__ni_k5_r2,L1,56961,6908,0.12127596074507119
bl_span_comfortable__ni_k5_r2,L2,42151,7074,0.167825199876634
bl_span_comfortable__ni_k5_r2,L3,32568,7714,0.23685826578236305
bl_span_comfortable__ni_k5_r2,L4,17383,5031,0.2894206983834781
bl_span_comfortable__ni_k5_r2,L5,41112,11282,0.2744210935979763
bl_span_comfortable__ni_k5_r2,R1,65498,4443,0.06783413233991878
bl_span_comfortable__ni_k5_r2,R2,58395,7252,0.12418871478722493
bl_span_comfortable__ni_k5_r2,R3,48125,7713,0.16027012987012987
bl_span_comfortable__ni_k5_r2,R4,31856,6208,0.19487694625816174
bl_span_comfortable__ni_k5_r2,R5,40324,6488,0.16089673643487748
bl_span_comfortable__ni_k5_r2,NA,0,0,
bl_span_comfortable__ni_k5_r4,L1,56961,6991,0.12273309808465441
bl_span_comfortable__ni_k5_r4,L2,42151,7139,0.16936727479775093
bl_span_comfortable__ni_k5_r4,L3,32568,7777,0.23879267993122083
bl_span_comfortable__ni_k5_r4,L4,17383,5059,0.29103146752574355
bl_span_comfortable__ni_k5_r4,L5,41112,11360,0.2763183498735162
bl_span_comfortable__ni_k5_r4,R1,65498,4583,0.0699716021863263
bl_span_comfortable__ni_k5_r4,R2,58395,7351,0.12588406541655964
bl_span_comfortable__ni_k5_r4,R3,48125,7819,0.16247272727272727
bl_span_comfortable__ni_k5_r4,R4,31856,6283,0.19723129080863888
bl_span_comfortable__ni_k5_r4,R5,40324,6577,0.16310385874417221
bl_span_comfortable__ni_k5_r4,NA,0,0,
bl_span_relative__ni_k2_r1,L1,56961,24155,0.42406207756184056
bl_span_relative__ni_k2_r1,L2,42151,19563,0.4641171027970867
bl_span_relative__ni_k2_r1,L3,32568,14832,0.45541635961680177
bl_span_relative__ni_k2_r1,L4,17383,8614,0.49554162112408673
bl_span_relative__ni_k2_r1,L5,41112,23966,0.5829441525588636
bl_span_relative__ni_k2_r1,R1,65498,26576,0.40575284741518824
bl_span_relative__ni_k2_r1,R2,58395,21641,0.3705967976710335
bl_span_relative__ni_k2_r1,R3,48125,19610,0.4074805194805195
bl_span_relative__ni_k2_r1,R4,31856,13762,0.43200652938222
bl_span_relative__ni_k2_r1,R5,40324,20821,0.5163426247396091
bl_span_relative__ni_k2_r1,NA,0,0,
bl_span_relative__ni_k2_r2,L1,56961,25548,0.44851740664665296
bl_span_relative__ni_k2_r2,L2,42151,20469,0.4856112547745012
bl_span_relative__ni_k2_r2,L3,32568,15537,0.47706337509211494
bl_span_relative__ni_k2_r2,L4,17383,9000,0.5177472242996031
bl_span_relative__ni_k2_r2,L5,41112,24654,0.5996789258610624
bl_span_relative__ni_k2_r2,R1,65498,28161,0.4299520596048734
bl_span_relative__ni_k2_r2,R2,58395,22992,0.3937323400976111
bl_span_relative__ni_k2_r2,R3,48125,20667,0.42944415584415585
bl_span_relative__ni_k2_r2,R4,31856,14549,0.4567114515318935
bl_span_relative__ni_k2_r2,R5,40324,21653,0.5369754984624541
bl_span_relative__ni_k2_r2,NA,0,0,
bl_span_relative__ni_k2_r4,L1,56961,27773,0.4875792208704201
bl_span_relative__ni_k2_r4,L2,42151,22000,0.521933050224194
bl_span_relative__ni_k2_r4,L3,32568,16731,0.5137251289609432
bl_span_relative__ni_k2_r4,L4,17383,9661,0.5557728815509406
bl_span_relative__ni_k2_r4,L5,41112,25834,0.6283810079782058
bl_span_relative__ni_k2_r4,R1,65498,30859,0.47114415707349844
bl_span_relative__ni_k2_r4,R2,58395,25397,0.43491737306276224
bl_span_relative__ni_k2_r4,R3,48125,22497,0.46747012987012987
bl_span_relative__ni_k2_r4,R4,31856,15898,0.49905826217980914
bl_span_relative__ni_k2_r4,R5,40324,23077,0.5722894554111695
bl_span_relative__ni_k2_r4,NA,0,0,
bl_span_relative__ni_k3_r1,L1,56961,23048,0.40462772774354383
bl_span_relative__ni_k3_r1,L2,42151,18883,0.44798462669924793
bl_span_relative__ni_k3_r1,L3,32568,14293,0.43886637189879635
bl_span_relative__ni_k3_r1,L4,17383,8281,0.47638497382500145
bl_span_relative__ni_k3_r1,L5,41112,23368,0.568398521113057
bl_span_relative__ni_k3_r1,R1,65498,24997,0.3816452410760634
bl_span_relative__ni_k3_r1,R2,58395,20569,0.35223906156349005
bl_span_relative__ni_k3_r1,R3,48125,18725,0.3890909090909091
bl_span_relative__ni_k3_r1,R4,31856,13079,0.41056629834254144
bl_span_relative__ni_k3_r1,R5,40324,20100,0.49846245412161494
bl_span_relative__ni_k3_r1,NA,0,0,
bl_span_relative__ni_k3_r2,L1,56961,23414,0.411053176734959
bl_span_relative__ni_k3_r2,L2,42151,19130,0.4538445113994923
bl_span_relative__ni_k3_r2,L3,32568,14484,0.4447310243183493
bl_span_relative__ni_k3_r2,L4,17383,8389,0.48259794051659666
bl_span_relative__ni_k3_r2,L5,41112,23546,0.5727281572290329
bl_span_relative__ni_k3_r2,R1,65498,25386,0.3875843537207243
bl_span_relative__ni_k3_r2,R2,58395,20890,0.3577361075434541
bl_span_relative__ni_k3_r2,R3,48125,19002,0.3948467532467532
bl_span_relative__ni_k3_r2,R4,31856,13273,0.416656202913109
bl_span_relative__ni_k3_r2,R5,40324,20333,0.5042406507290943
bl_span_relative__ni_k3_r2,NA,0,0,
bl_span_relative__ni_k3_r4,L1,56961,24038,0.42200804058917507
bl_span_relative__ni_k3_r4,L2,42151,19583,0.46459158738819956
bl_span_relative__ni_k3_r4,L3,32568,14816,0.4549250798329649
bl_span_relative__ni_k3_r4,L4,17383,8597,0.49456365414485415
bl_span_relative__ni_k3_r4,L5,41112,23899,0.5813144580657715
bl_span_relative__ni_k3_r4,R1,65498,26160,0.39940150844300587
bl_span_relative__ni_k3_r4,R2,58395,21513,0.36840482918058054
bl_span_relative__ni_k3_r4,R3,48125,19534,0.4059012987012987
bl_span_relative__ni_k3_r4,R4,31856,13663,0.42889879457559016
bl_span_relative__ni_k3_r4,R5,40324,20734,0.5141851006844559
bl_span_relative__ni_k3_r4,NA,0,0,
bl_span_relative__ni_k5_r1,L1,56961,22754,0.3994663015045382
bl_span_relative__ni_k5_r1,L2,42151,18651,0.44248060544233825
bl_span_relative__ni_k5_r1,L3,32568,14133,0.4339535740604274
bl_span_relative__ni_k5_r1,L4,17383,8175,0.47028706207213944
bl_span_relative__ni_k5_r1,L5,41112,23192,0.5641175325938899
bl_span_relative__ni_k5_r1,R1,65498,24539,0.37465266114995877
bl_span_relative__ni_k5_r1,R2,58395,20270,0.34711876016782256
bl_span_relative__ni_k5_r1,R3,48125,18492,0.3842493506493507
bl_span_relative__ni_k5_r1,R4,31856,12892,0.4046961325966851
bl_span_relative__ni_k5_r1,R5,40324,19892,0.4933042356909037
bl_span_relative__ni_k5_r1,NA,0,0,
bl_span_relative__ni_k5_r2,L1,56961,22797,0.40022120398167166
bl_span_relative__ni_k5_r2,L2,42151,18678,0.44312115964034066
bl_span_relative__ni_k5_r2,L3,32568,14159,0.4347519037091624
bl_span_relative__ni_k5_r2,L4,17383,8189,0.47109244664327216
bl_span_relative__ni_k5_r2,L5,41112,23211,0.5645796847635727
bl_span_relative__ni_k5_r2,R1,65498,24583,0.3753244373874011
bl_span_relative__ni_k5_r2,R2,58395,20308,0.3477695008134258
bl_span_relative__ni_k5_r2,R3,48125,18529,0.3850181818181818
bl_span_relative__ni_k5_r2,R4,31856,12911,0.4052925665494726
bl_span_relative__ni_k5_r2,R5,40324,19917,0.4939242138676719
bl_span_relative__ni_k5_r2,NA,0,0,
bl_span_relative__ni_k5_r4,L1,56961,22851,0.40116922104597885
bl_span_relative__ni_k5_r4,L2,42151,18718,0.44407012882256647
bl_span_relative__ni_k5_r4,L3,32568,14199,0.43598010316875463
bl_span_relative__ni_k5_r4,L4,17383,8209,0.4722429960306046
bl_span_relative__ni_k5_r4,L5,41112,23257,0.5656985794901732
bl_span_relative__ni_k5_r4,R1,65498,24664,0.3765611163699655
bl_span_relative__ni_k5_r4,R2,58395,20376,0.3489339840739789
bl_span_relative__ni_k5_r4,R3,48125,18607,0.38663896103896106
bl_span_relative__ni_k5_r4,R4,31856,12960,0.40683073832245104
bl_span_relative__ni_k5_r4,R5,40324,19976,0.49538736236484476
bl_span_relative__ni_k5_r4,NA,0,0,
bl_crossing__ni_k2_r1,L1,56961,2716,0.047681747160337774
bl_crossing__ni_k2_r1,L2,42151,2751,0.06526535550757989
bl_crossing__ni_k2_r1,L3,32568,2693,0.08268852861704741
bl_crossing__ni_k2_r1,L4,17383,1905,0.10958982914341599
bl_crossing__ni_k2_r1,L5,41112,3110,0.07564701303755594
bl_crossing__ni_k2_r1,R1,65498,3555,0.05427646645699105
bl_crossing__ni_k2_r1,R2,58395,3521,0.06029625824128778
bl_crossing__ni_k2_r1,R3,48125,3402,0.0706909090909091
bl_crossing__ni_k2_r1,R4,31856,2691,0.08447388247112005
bl_crossing__ni_k2_r1,R5,40324,2871,0.07119829382005753
bl_crossing__ni_k2_r1,NA,0,0,
bl_crossing__ni_k2_r2,L1,56961,5129,0.09004406523761872
bl_crossing__ni_k2_r2,L2,42151,4504,0.1068539299186259
bl_crossing__ni_k2_r2,L3,32568,4052,0.1244166052566937
bl_crossing__ni_k2_r2,L4,17383,2740,0.15762526606454583
bl_crossing__ni_k2_r2,L5,41112,4813,0.11707044172017902
bl_crossing__ni_k2_r2,R1,65498,6397,0.09766710433906378
bl_crossing__ni_k2_r2,R2,58395,5890,0.10086480006849902
bl_crossing__ni_k2_r2,R3,48125,5315,0.11044155844155844
bl_crossing__ni_k2_r2,R4,31856,4130,0.12964590657960823
bl_crossing__ni_k2_r2,R5,40324,4625,0.11469596270211288
bl_crossing__ni_k2_r2,NA,0,0,
bl_crossing__ni_k2_r4,L1,56961,9185,0.16125068029002299
bl_crossing__ni_k2_r4,L2,42151,7498,0.17788427320822756
bl_crossing__ni_k2_r4,L3,32568,6379,0.1958671088184721
bl_crossing__ni_k2_r4,L4,17383,4116,0.23678306391301845
bl_crossing__ni_k2_r4,L5,41112,7706,0.1874391905039891
bl_crossing__ni_k2_r4,R1,65498,11175,0.17061589666860058
bl_crossing__ni_k2_r4,R2,58395,9814,0.16806233410394725
bl_crossing__ni_k2_r4,R3,48125,8669,0.18013506493506493
bl_crossing__ni_k2_r4,R4,31856,6528,0.2049221496735309
bl_crossing__ni_k2_r4,R5,40324,7603,0.18854776311873822
bl_crossing__ni_k2_r4,NA,0,0,
bl_crossing__ni_k3_r1,L1,56961,636,0.011165534312950967
bl_crossing__ni_k3_r1,L2,42151,1285,0.030485634979004056
bl_crossing__ni_k3_r1,L3,32568,1560,0.047899778924097275
bl_crossing__ni_k3_r1,L4,17383,1202,0.06914801817868033
bl_crossing__ni_k3_r1,L5,41112,1665,0.04049912434325744
bl_crossing__ni_k3_r1,R1,65498,899,0.013725609942288315
bl_crossing__ni_k3_r1,R2,58395,1611,0.02758797842281017
bl_crossing__ni_k3_r1,R3,48125,1817,0.037755844155844154
bl_crossing__ni_k3_r1,R4,31856,1419,0.04454419889502762
bl_crossing__ni_k3_r1,R5,40324,1388,0.03442118837416923
bl_crossing__ni_k3_r1,NA,0,0,
bl_crossing__ni_k3_r2,L1,56961,1299,0.022805077158055513
bl_crossing__ni_k3_r2,L2,42151,1753,0.041588574411046
bl_crossing__ni_k3_r2,L3,32568,1948,0.05981331368214198
bl_crossing__ni_k3_r2,L4,17383,1451,0.08347235805096934
bl_crossing__ni_k3_r2,L5,41112,2099,0.05105565285074917
bl_crossing__ni_k3_r2,R1,65498,1648,0.025161073620568568
bl_crossing__ni_k3_r2,R2,58395,2199,0.03765733367582841
bl_crossing__ni_k3_r2,R3,48125,2314,0.04808311688311688
bl_crossing__ni_k3_r2,R4,31856,1803,0.05659844299347062
bl_crossing__ni_k3_r2,R5,40324,1869,0.04634956849518897
bl_crossing__ni_k3_r2,NA,0,0,
bl_crossing__ni_k3_r4,L1,56961,2472,0.04339811449939432
bl_crossing__ni_k3_r4,L2,42151,2634,0.06248962064956941
bl_crossing__ni_k3_r4,L3,32568,2612,0.08020142471137312
bl_crossing__ni_k3_r4,L4,17383,1914,0.11010757636771558
bl_crossing__ni_k3_r4,L5,41112,2924,0.0711227865343452
bl_crossing__ni_k3_r4,R1,65498,3054,0.04662737793520413
bl_crossing__ni_k3_r4,R2,58395,3268,0.055963695521876876
bl_crossing__ni_k3_r4,R3,48125,3310,0.06877922077922077
bl_crossing__ni_k3_r4,R4,31856,2535,0.07957684580612757
bl_crossing__ni_k3_r4,R5,40324,2751,0.06822239857157028
bl_crossing__ni_k3_r4,NA,0,0,
bl_crossing__ni_k5_r1,L1,56961,71,0.0012464668808482998
bl_crossing__ni_k5_r1,L2,42151,829,0.019667386301629855
bl_crossing__ni_k5_r1,L3,32568,1230,0.037767133382461314
bl_crossing__ni_k5_r1,L4,17383,959,0.05516884312259104
bl_crossing__ni_k5_r1,L5,41112,1231,0.029942595835765713
bl_crossing__ni_k5_r1,R1,65498,85,0.001297749549604568
bl_crossing__ni_k5_r1,R2,58395,1084,0.018563233153523418
bl_crossing__ni_k5_r1,R3,48125,1374,0.02855064935064935
bl_crossing__ni_k5_r1,R4,31856,1058,0.03321195379206429
bl_crossing__ni_k5_r1,R5,40324,956,0.023707965479615116
bl_crossing__ni_k5_r1,NA,0,0,
bl_crossing__ni_k5_r2,L1,56961,149,0.002615824862625305
bl_crossing__ni_k5_r2,L2,42151,880,0.020877322008967757
bl_crossing__ni_k5_r2,L3,32568,1281,0.03933308769344142
bl_crossing__ni_k5_r2,L4,17383,992,0.05706724961168958
bl_crossing__ni_k5_r2,L5,41112,1277,0.03106149056236622
bl_crossing__ni_k5_r2,R1,65498,172,0.0026260343827292434
bl_crossing__ni_k5_r2,R2,58395,1141,0.01953934412192825
bl_crossing__ni_k5_r2,R3,48125,1431,0.029735064935064934
bl_crossing__ni_k5_r2,R4,31856,1100,0.034530386740331494
bl_crossing__ni_k5_r2,R5,40324,1002,0.024848725324868565
bl_crossing__ni_k5_r2,NA,0,0,
bl_crossing__ni_k5_r4,L1,56961,257,0.0045118589912396204
bl_crossing__ni_k5_r4,L2,42151,966,0.022917605750753245
bl_crossing__ni_k5_r4,L3,32568,1360,0.04175878162613608
bl_crossing__ni_k5_r4,L4,17383,1041,0.05988609561065409
bl_crossing__ni_k5_r4,L5,41112,1385,0.03368846079003697
bl_crossing__ni_k5_r4,R1,65498,333,0.005084124706097896
bl_crossing__ni_k5_r4,R2,58395,1261,0.021594314581727888
bl_crossing__ni_k5_r4,R3,48125,1565,0.03251948051948052
bl_crossing__ni_k5_r4,R4,31856,1198,0.0376067302862883
bl_crossing__ni_k5_r4,R5,40324,1109,0.027502231921436367
bl_crossing__ni_k5_r4,NA,0,0,
bl_step_crossing__ni_k2_r1,L1,56961,2716,0.047681747160337774
bl_step_crossing__ni_k2_r1,L2,42151,2273,0.053925173779981496
bl_step_crossing__ni_k2_r1,L3,32568,2014,0.061839842790469175
bl_step_crossing__ni_k2_r1,L4,17383,1319,0.07587873209457516
bl_step_crossing__ni_k2_r1,L5,41112,2309,0.05616365051566453
bl_step_crossing__ni_k2_r1,R1,65498,3555,0.05427646645699105
bl_step_crossing__ni_k2_r1,R2,58395,3010,0.051545509033307645
bl_step_crossing__ni_k2_r1,R3,48125,2809,0.05836883116883117
bl_step_crossing__ni_k2_r1,R4,31856,2263,0.07103842290306378
bl_step_crossing__ni_k2_r1,R5,40324,2478,0.06145223688126178
bl_step_crossing__ni_k2_r1,NA,0,0,
bl_step_crossing__ni_k2_r2,L1,56961,5129,0.09004406523761872
bl_step_crossing__ni_k2_r2,L2,42151,4072,0.09660506275058718
bl_step_crossing__ni_k2_r2,L3,32568,3426,0.10519528371407516
bl_step_crossing__ni_k2_r2,L4,17383,2199,0.12650290513720303
bl_step_crossing__ni_k2_r2,L5,41112,4066,0.09890056431212298
bl_step_crossing__ni_k2_r2,R1,65498,6397,0.09766710433906378
bl_step_crossing__ni_k2_r2,R2,58395,5419,0.09279904101378543
bl_step_crossing__ni_k2_r2,R3,48125,4762,0.09895064935064934
bl_step_crossing__ni_k2_r2,R4,31856,3732,0.11715218483174285
bl_step_crossing__ni_k2_r2,R5,40324,4261,0.10566908044836822
bl_step_crossing__ni_k2_r2,NA,0,0,
bl_step_crossing__ni_k2_r4,L1,56961,9185,0.16125068029002299
bl_step_crossing__ni_k2_r4,L2,42151,7117,0.16884534174752674
bl_step_crossing__ni_k2_r4,L3,32568,5821,0.1787337263571604
bl_step_crossing__ni_k2_r4,L4,17383,3638,0.20928493355577288
bl_step_crossing__ni_k2_r4,L5,41112,7061,0.17175034053317767
bl_step_crossing__ni_k2_r4,R1,65498,11175,0.17061589666860058
bl_step_crossing__ni_k2_r4,R2,58395,9401,0.16098981077147015
bl_step_crossing__ni_k2_r4,R3,48125,8185,0.17007792207792208
bl_step_crossing__ni_k2_r4,R4,31856,6178,0.19393520843797088
bl_step_crossing__ni_k2_r4,R5,40324,7294,0.18088483285388354
bl_step_crossing__ni_k2_r4,NA,0,0,
bl_step_crossing__ni_k3_r1,L1,56961,636,0.011165534312950967
bl_step_crossing__ni_k3_r1,L2,42151,768,0.0182202082987355
bl_step_crossing__ni_k3_r1,L3,32568,832,0.025546548759518548
bl_step_crossing__ni_k3_r1,L4,17383,566,0.03256054766150837
bl_step_crossing__ni_k3_r1,L5,41112,812,0.019750924304339366
bl_step_crossing__ni_k3_r1,R1,65498,899,0.013725609942288315
bl_step_crossing__ni_k3_r1,R2,58395,1048,0.017946742015583526
bl_step_crossing__ni_k3_r1,R3,48125,1173,0.024374025974025975
bl_step_crossing__ni_k3_r1,R4,31856,956,0.03001004520341537
bl_step_crossing__ni_k3_r1,R5,40324,971,0.024079952385676024
bl_step_crossing__ni_k3_r1,NA,0,0,
bl_step_crossing__ni_k3_r2,L1,56961,1299,0.022805077158055513
bl_step_crossing__ni_k3_r2,L2,42151,1247,0.02958411425588954
bl_step_crossing__ni_k3_r2,L3,32568,1241,0.03810488823384918
bl_step_crossing__ni_k3_r2,L4,17383,833,0.04792038198239659
bl_step_crossing__ni_k3_r2,L5,41112,1261,0.03067230978789648
bl_step_crossing__ni_k3_r2,R1,65498,1648,0.025161073620568568
bl_step_crossing__ni_k3_r2,R2,58395,1652,0.028290093329908384
bl_step_crossing__ni_k3_r2,R3,48125,1688,0.035075324675324676
bl_step_crossing__ni_k3_r2,R4,31856,1353,0.04247237569060774
bl_step_crossing__ni_k3_r2,R5,40324,1458,0.036157127269120126
bl_step_crossing__ni_k3_r2,NA,0,0,
bl_step_crossing__ni_k3_r4,L1,56961,2472,0.04339811449939432
bl_step_crossing__ni_k3_r4,L2,42151,2149,0.05098336931508149
bl_step_crossing__ni_k3_r4,L3,32568,1929,0.05922991893883567
bl_step_crossing__ni_k3_r4,L4,17383,1320,0.07593625956394179
bl_step_crossing__ni_k3_r4,L5,41112,2117,0.05149348122202763
bl_step_crossing__ni_k3_r4,R1,65498,3054,0.04662737793520413
bl_step_crossing__ni_k3_r4,R2,58395,2737,0.046870451237263466
bl_step_crossing__ni_k3_r4,R3,48125,2709,0.05629090909090909
bl_step_crossing__ni_k3_r4,R4,31856,2107,0.06614138623807132
bl_step_crossing__ni_k3_r4,R5,40324,2350,0.05827794861620871
bl_step_crossing__ni_k3_r4,NA,0,0,
bl_step_crossing__ni_k5_r1,L1,56961,71,0.0012464668808482998
bl_step_crossing__ni_k5_r1,L2,42151,288,0.006832578112025812
bl_step_crossing__ni_k5_r1,L3,32568,478,0.014676983542127242
bl_step_crossing__ni_k5_r1,L4,17383,303,0.017430823218086637
bl_step_crossing__ni_k5_r1,L5,41112,346,0.008416034247908153
bl_step_crossing__ni_k5_r1,R1,65498,85,0.001297749549604568
bl_step_crossing__ni_k5_r1,R2,58395,504,0.00863087593115849
bl_step_crossing__ni_k5_r1,R3,48125,717,0.014898701298701298
bl_step_crossing__ni_k5_r1,R4,31856,581,0.018238322451029635
bl_step_crossing__ni_k5_r1,R5,40324,526,0.01304434083920246
bl_step_crossing__ni_k5_r1,NA,0,0,
bl_step_crossing__ni_k5_r2,L1,56961,149,0.002615824862625305
bl_step_crossing__ni_k5_r2,L2,42151,342,0.008113686508030652
bl_step_crossing__ni_k5_r2,L3,32568,531,0.016304347826086956
bl_step_crossing__ni_k5_r2,L4,17383,337,0.019386757176551802
bl_step_crossing__ni_k5_r2,L5,41112,393,0.009559252772913017
bl_step_crossing__ni_k5_r2,R1,65498,172,0.0026260343827292434
bl_step_crossing__ni_k5_r2,R2,58395,563,0.009641236407226646
bl_step_crossing__ni_k5_r2,R3,48125,776,0.016124675324675326
bl_step_crossing__ni_k5_r2,R4,31856,625,0.019619537920642895
bl_step_crossing__ni_k5_r2,R5,40324,573,0.014209899811526634
bl_step_crossing__ni_k5_r2,NA,0,0,
bl_step_crossing__ni_k5_r4,L1,56961,257,0.0045118589912396204
bl_step_crossing__ni_k5_r4,L2,42151,430,0.010201418708927427
bl_step_crossing__ni_k5_r4,L3,32568,615,0.018883566691230657
bl_step_crossing__ni_k5_r4,L4,17383,390,0.022435713052982798
bl_step_crossing__ni_k5_r4,L5,41112,507,0.012332165791009923
bl_step_crossing__ni_k5_r4,R1,65498,333,0.005084124706097896
bl_step_crossing__ni_k5_r4,R2,58395,685,0.011730456374689615
bl_step_crossing__ni_k5_r4,R3,48125,916,0.019033766233766233
bl_step_crossing__ni_k5_r4,R4,31856,726,0.022790055248618785
bl_step_crossing__ni_k5_r4,R5,40324,684,0.016962602916377342
bl_step_crossing__ni_k5_r4,NA,0,0,
bl_rate_q995__ni_k2_r1,L1,56961,3612,0.0634118080792121
bl_rate_q995__ni_k2_r1,L2,42151,2206,0.05233565039975327
bl_rate_q995__ni_k2_r1,L3,32568,1721,0.05284328174895603
bl_rate_q995__ni_k2_r1,L4,17383,1188,0.06834263360754761
bl_rate_q995__ni_k2_r1,L5,41112,2505,0.060931115002918854
bl_rate_q995__ni_k2_r1,R1,65498,4558,0.06958991114232496
bl_rate_q995__ni_k2_r1,R2,58395,2816,0.04822330678996489
bl_rate_q995__ni_k2_r1,R3,48125,2476,0.05144935064935065
bl_rate_q995__ni_k2_r1,R4,31856,2149,0.06745981918633852
bl_rate_q995__ni_k2_r1,R5,40324,3017,0.0748189663723837
bl_rate_q995__ni_k2_r1,NA,0,0,
bl_rate_q995__ni_k2_r2,L1,56961,5977,0.10493144432155334
bl_rate_q995__ni_k2_r2,L2,42151,4006,0.09503926359991459
bl_rate_q995__ni_k2_r2,L3,32568,3140,0.09641365757799067
bl_rate_q995__ni_k2_r2,L4,17383,2080,0.11965713628257493
bl_rate_q995__ni_k2_r2,L5,41112,4258,0.10357073360575987
bl_rate_q995__ni_k2_r2,R1,65498,7339,0.11204922287703441
bl_rate_q995__ni_k2_r2,R2,58395,5234,0.08963096155492765
bl_rate_q995__ni_k2_r2,R3,48125,4437,0.0921974025974026
bl_rate_q995__ni_k2_r2,R4,31856,3621,0.11366775489703666
bl_rate_q995__ni_k2_r2,R5,40324,4776,0.11844063088979268
bl_rate_q995__ni_k2_r2,NA,0,0,
bl_rate_q995__ni_k2_r4,L1,56961,9958,0.17482136900686435
bl_rate_q995__ni_k2_r4,L2,42151,7056,0.1673981637446324
bl_rate_q995__ni_k2_r4,L3,32568,5550,0.170412675018423
bl_rate_q995__ni_k2_r4,L4,17383,3513,0.20209399988494506
bl_rate_q995__ni_k2_r4,L5,41112,7256,0.17649348122202763
bl_rate_q995__ni_k2_r4,R1,65498,12013,0.1834101804635256
bl_rate_q995__ni_k2_r4,R2,58395,9233,0.15811285212775067
bl_rate_q995__ni_k2_r4,R3,48125,7863,0.163387012987013
bl_rate_q995__ni_k2_r4,R4,31856,6056,0.1901054746358614
bl_rate_q995__ni_k2_r4,R5,40324,7745,0.19206923916278146
bl_rate_q995__ni_k2_r4,NA,0,0,
bl_rate_q995__ni_k3_r1,L1,56961,1559,0.027369603763978862
bl_rate_q995__ni_k3_r1,L2,42151,694,0.016464615311617754
bl_rate_q995__ni_k3_r1,L3,32568,526,0.016150822893637926
bl_rate_q995__ni_k3_r1,L4,17383,427,0.024564229419547834
bl_rate_q995__ni_k3_r1,L5,41112,999,0.024299474605954465
bl_rate_q995__ni_k3_r1,R1,65498,1934,0.029527619163943936
bl_rate_q995__ni_k3_r1,R2,58395,834,0.014282044695607501
bl_rate_q995__ni_k3_r1,R3,48125,812,0.016872727272727272
bl_rate_q995__ni_k3_r1,R4,31856,823,0.02583500753390256
bl_rate_q995__ni_k3_r1,R5,40324,1493,0.037025096716595575
bl_rate_q995__ni_k3_r1,NA,0,0,
bl_rate_q995__ni_k3_r2,L1,56961,2212,0.038833587893470974
bl_rate_q995__ni_k3_r2,L2,42151,1172,0.02780479703921615
bl_rate_q995__ni_k3_r2,L3,32568,940,0.028862687300417588
bl_rate_q995__ni_k3_r2,L4,17383,699,0.040211701087269174
bl_rate_q995__ni_k3_r2,L5,41112,1453,0.03534247908153337
bl_rate_q995__ni_k3_r2,R1,65498,2666,0.04070353293230328
bl_rate_q995__ni_k3_r2,R2,58395,1436,0.02459114650226903
bl_rate_q995__ni_k3_r2,R3,48125,1331,0.027657142857142856
bl_rate_q995__ni_k3_r2,R4,31856,1223,0.038391511803114016
bl_rate_q995__ni_k3_r2,R5,40324,1983,0.04917666898125186
bl_rate_q995__ni_k3_r2,NA,0,0,
bl_rate_q995__ni_k3_r4,L1,56961,3357,0.05893506083109496
bl_rate_q995__ni_k3_r4,L2,42151,2080,0.049346397475741974
bl_rate_q995__ni_k3_r4,L3,32568,1635,0.05020265291083272
bl_rate_q995__ni_k3_r4,L4,17383,1181,0.06793994132198125
bl_rate_q995__ni_k3_r4,L5,41112,2313,0.05626094570928196
bl_rate_q995__ni_k3_r4,R1,65498,4043,0.06172707563589728
bl_rate_q995__ni_k3_r4,R2,58395,2526,0.04325712817878243
bl_rate_q995__ni_k3_r4,R3,48125,2353,0.048893506493506496
bl_rate_q995__ni_k3_r4,R4,31856,1975,0.06199773982923154
bl_rate_q995__ni_k3_r4,R5,40324,2853,0.07075190953278444
bl_rate_q995__ni_k3_r4,NA,0,0,
bl_rate_q995__ni_k5_r1,L1,56961,1000,0.017555871561243656
bl_rate_q995__ni_k5_r1,L2,42151,213,0.005053260895352424
bl_rate_q995__ni_k5_r1,L3,32568,169,0.005189142716777204
bl_rate_q995__ni_k5_r1,L4,17383,160,0.00920439509865961
bl_rate_q995__ni_k5_r1,L5,41112,535,0.01301323214633197
bl_rate_q995__ni_k5_r1,R1,65498,1129,0.017237167547100675
bl_rate_q995__ni_k5_r1,R2,58395,283,0.004846305334360819
bl_rate_q995__ni_k5_r1,R3,48125,350,0.007272727272727273
bl_rate_q995__ni_k5_r1,R4,31856,440,0.013812154696132596
bl_rate_q995__ni_k5_r1,R5,40324,1044,0.025890288661839102
bl_rate_q995__ni_k5_r1,NA,0,0,
bl_rate_q995__ni_k5_r2,L1,56961,1077,0.01890767367145942
bl_rate_q995__ni_k5_r2,L2,42151,268,0.006358093520912909
bl_rate_q995__ni_k5_r2,L3,32568,222,0.0068165070007369195
bl_rate_q995__ni_k5_r2,L4,17383,196,0.011275383995858023
bl_rate_q995__ni_k5_r2,L5,41112,583,0.014180774469741194
bl_rate_q995__ni_k5_r2,R1,65498,1215,0.018550184738465297
bl_rate_q995__ni_k5_r2,R2,58395,344,0.0058909153180923025
bl_rate_q995__ni_k5_r2,R3,48125,409,0.008498701298701299
bl_rate_q995__ni_k5_r2,R4,31856,484,0.015193370165745856
bl_rate_q995__ni_k5_r2,R5,40324,1089,0.027006249380021823
bl_rate_q995__ni_k5_r2,NA,0,0,
bl_rate_q995__ni_k5_r4,L1,56961,1184,0.020786151928512492
bl_rate_q995__ni_k5_r4,L2,42151,357,0.00846954995136533
bl_rate_q995__ni_k5_r4,L3,32568,308,0.009457135838860231
bl_rate_q995__ni_k5_r4,L4,17383,248,0.014266812402922395
bl_rate_q995__ni_k5_r4,L5,41112,699,0.017002335084646818
bl_rate_q995__ni_k5_r4,R1,65498,1373,0.02096247213655379
bl_rate_q995__ni_k5_r4,R2,58395,470,0.008048634300881924
bl_rate_q995__ni_k5_r4,R3,48125,551,0.01144935064935065
bl_rate_q995__ni_k5_r4,R4,31856,586,0.018395278754394777
bl_rate_q995__ni_k5_r4,R5,40324,1196,0.029659755976589625
bl_rate_q995__ni_k5_r4,NA,0,0,
bl_rate_q990__ni_k2_r1,L1,56961,3867,0.06788855532732922
bl_rate_q990__ni_k2_r1,L2,42151,2266,0.05375910417309198
bl_rate_q990__ni_k2_r1,L3,32568,1757,0.05394866126258904
bl_rate_q990__ni_k2_r1,L4,17383,1239,0.07127653454524535
bl_rate_q990__ni_k2_r1,L5,41112,2691,0.0654553415061296
bl_rate_q990__ni_k2_r1,R1,65498,4989,0.07617026474090811
bl_rate_q990__ni_k2_r1,R2,58395,2927,0.05012415446527956
bl_rate_q990__ni_k2_r1,R3,48125,2558,0.05315324675324675
bl_rate_q990__ni_k2_r1,R4,31856,2260,0.0709442491210447
bl_rate_q990__ni_k2_r1,R5,40324,3240,0.08034917170915584
bl_rate_q990__ni_k2_r1,NA,0,0,
bl_rate_q990__ni_k2_r2,L1,56961,6222,0.10923263285405804
bl_rate_q990__ni_k2_r2,L2,42151,4063,0.09639154468458637
bl_rate_q990__ni_k2_r2,L3,32568,3174,0.09745762711864407
bl_rate_q990__ni_k2_r2,L4,17383,2129,0.12247598228153944
bl_rate_q990__ni_k2_r2,L5,41112,4432,0.10780307452811831
bl_rate_q990__ni_k2_r2,R1,65498,7739,0.11815627958105591
bl_rate_q990__ni_k2_r2,R2,58395,5338,0.09141193595342068
bl_rate_q990__ni_k2_r2,R3,48125,4514,0.0937974025974026
bl_rate_q990__ni_k2_r2,R4,31856,3723,0.11686966348568559
bl_rate_q990__ni_k2_r2,R5,40324,4986,0.12364844757464537
bl_rate_q990__ni_k2_r2,NA,0,0,
bl_rate_q990__ni_k2_r4,L1,56961,10180,0.17871877249346044
bl_rate_q990__ni_k2_r4,L2,42151,7108,0.16863182368152593
bl_rate_q990__ni_k2_r4,L3,32568,5583,0.1714259395725866
bl_rate_q990__ni_k2_r4,L4,17383,3555,0.2045101535983432
bl_rate_q990__ni_k2_r4,L5,41112,7410,0.1802393461762989
bl_rate_q990__ni_k2_r4,R1,65498,12365,0.18878439036306452
bl_rate_q990__ni_k2_r4,R2,58395,9326,0.15970545423409538
bl_rate_q990__ni_k2_r4,R3,48125,7933,0.16484155844155843
bl_rate_q990__ni_k2_r4,R4,31856,6150,0.19305625313912606
bl_rate_q990__ni_k2_r4,R5,40324,7934,0.1967562741791489
bl_rate_q990__ni_k2_r4,NA,0,0,
bl_rate_q990__ni_k3_r1,L1,56961,1821,0.0319692421130247
bl_rate_q990__ni_k3_r1,L2,42151,754,0.017888069084956465
bl_rate_q990__ni_k3_r1,L3,32568,562,0.01725620240727094
bl_rate_q990__ni_k3_r1,L4,17383,481,0.027670712765345454
bl_rate_q990__ni_k3_r1,L5,41112,1190,0.028945320101187
bl_rate_q990__ni_k3_r1,R1,65498,2378,0.0363064521054078
bl_rate_q990__ni_k3_r1,R2,58395,949,0.01625139138624882
bl_rate_q990__ni_k3_r1,R3,48125,896,0.01861818181818182
bl_rate_q990__ni_k3_r1,R4,31856,938,0.029445002511300854
bl_rate_q990__ni_k3_r1,R5,40324,1721,0.042679297688721356
bl_rate_q990__ni_k3_r1,NA,0,0,
bl_rate_q990__ni_k3_r2,L1,56961,2472,0.04339811449939432
bl_rate_q990__ni_k3_r2,L2,42151,1232,0.02922825081255486
bl_rate_q990__ni_k3_r2,L3,32568,976,0.029968066814050603
bl_rate_q990__ni_k3_r2,L4,17383,753,0.043318184433066786
bl_rate_q990__ni_k3_r2,L5,41112,1643,0.03996400077836155
bl_rate_q990__ni_k3_r2,R1,65498,3104,0.047390760023206815
bl_rate_q990__ni_k3_r2,R2,58395,1549,0.026526243685247024
bl_rate_q990__ni_k3_r2,R3,48125,1413,0.029361038961038963
bl_rate_q990__ni_k3_r2,R4,31856,1334,0.041875941737820194
bl_rate_q990__ni_k3_r2,R5,40324,2209,0.05478127169923619
bl_rate_q990__ni_k3_r2,NA,0,0,
bl_rate_q990__ni_k3_r4,L1,56961,3612,0.0634118080792121
bl_rate_q990__ni_k3_r4,L2,42151,2138,0.050722402789969395
bl_rate_q990__ni_k3_r4,L3,32568,1670,0.051277327437975924
bl_rate_q990__ni_k3_r4,L4,17383,1231,0.07081631479031238
bl_rate_q990__ni_k3_r4,L5,41112,2498,0.060760848414088346
bl_rate_q990__ni_k3_r4,R1,65498,4461,0.06810894989159974
bl_rate_q990__ni_k3_r4,R2,58395,2632,0.04507235208493878
bl_rate_q990__ni_k3_r4,R3,48125,2432,0.05053506493506493
bl_rate_q990__ni_k3_r4,R4,31856,2080,0.06529382219989954
bl_rate_q990__ni_k3_r4,R5,40324,3072,0.07618291836127368
bl_rate_q990__ni_k3_r4,NA,0,0,
bl_rate_q990__ni_k5_r1,L1,56961,1264,0.022190621653411985
bl_rate_q990__ni_k5_r1,L2,42151,274,0.00650043889824678
bl_rate_q990__ni_k5_r1,L3,32568,206,0.006325227216900025
bl_rate_q990__ni_k5_r1,L4,17383,217,0.012483460852557096
bl_rate_q990__ni_k5_r1,L5,41112,728,0.017707725238373224
bl_rate_q990__ni_k5_r1,R1,65498,1578,0.024092338697364806
bl_rate_q990__ni_k5_r1,R2,58395,398,0.00681565202500214
bl_rate_q990__ni_k5_r1,R3,48125,434,0.009018181818181818
bl_rate_q990__ni_k5_r1,R4,31856,557,0.017484932194876946
bl_rate_q990__ni_k5_r1,R5,40324,1272,0.03154448963396488
bl_rate_q990__ni_k5_r1,NA,0,0,
bl_rate_q990__ni_k5_r2,L1,56961,1341,0.023542423763627744
bl_rate_q990__ni_k5_r2,L2,42151,329,0.007805271523807264
bl_rate_q990__ni_k5_r2,L3,32568,259,0.00795259150085974
bl_rate_q990__ni_k5_r2,L4,17383,253,0.014554449749755508
bl_rate_q990__ni_k5_r2,L5,41112,776,0.018875267561782448
bl_rate_q990__ni_k5_r2,R1,65498,1664,0.025405355888729428
bl_rate_q990__ni_k5_r2,R2,58395,459,0.007860262008733625
bl_rate_q990__ni_k5_r2,R3,48125,493,0.010244155844155844
bl_rate_q990__ni_k5_r2,R4,31856,601,0.018866147664490206
bl_rate_q990__ni_k5_r2,R5,40324,1317,0.0326604503521476
bl_rate_q990__ni_k5_r2,NA,0,0,
bl_rate_q990__ni_k5_r4,L1,56961,1448,0.025420902020680817
bl_rate_q990__ni_k5_r4,L2,42151,418,0.009916727954259685
bl_rate_q990__ni_k5_r4,L3,32568,345,0.01059322033898305
bl_rate_q990__ni_k5_r4,L4,17383,305,0.01754587815681988
bl_rate_q990__ni_k5_r4,L5,41112,892,0.021696828176688072
bl_rate_q990__ni_k5_r4,R1,65498,1819,0.027771840361537757
bl_rate_q990__ni_k5_r4,R2,58395,585,0.010017980991523246
bl_rate_q990__ni_k5_r4,R3,48125,635,0.013194805194805195
bl_rate_q990__ni_k5_r4,R4,31856,702,0.022036664992466096
bl_rate_q990__ni_k5_r4,R5,40324,1423,0.03528915782164468
bl_rate_q990__ni_k5_r4,NA,0,0,
bl_rate_q975__ni_k2_r1,L1,56961,4974,0.08732290514562595
bl_rate_q975__ni_k2_r1,L2,42151,2723,0.06460107708002183
bl_rate_q975__ni_k2_r1,L3,32568,2127,0.06530950626381725
bl_rate_q975__ni_k2_r1,L4,17383,1447,0.08324224817350284
bl_rate_q975__ni_k2_r1,L5,41112,3441,0.08369819030939872
bl_rate_q975__ni_k2_r1,R1,65498,6781,0.10352987877492442
bl_rate_q975__ni_k2_r1,R2,58395,4031,0.06902988269543625
bl_rate_q975__ni_k2_r1,R3,48125,3292,0.0684051948051948
bl_rate_q975__ni_k2_r1,R4,31856,2777,0.0871735308890005
bl_rate_q975__ni_k2_r1,R5,40324,4283,0.10621466124392422
bl_rate_q975__ni_k2_r1,NA,0,0,
bl_rate_q975__ni_k2_r2,L1,56961,7268,0.1275960745071189
bl_rate_q975__ni_k2_r2,L2,42151,4493,0.10659296339351379
bl_rate_q975__ni_k2_r2,L3,32568,3532,0.10845001228199459
bl_rate_q975__ni_k2_r2,L4,17383,2320,0.13346372893056435
bl_rate_q975__ni_k2_r2,L5,41112,5136,0.12492702860478692
bl_rate_q975__ni_k2_r2,R1,65498,9424,0.14388225594674647
bl_rate_q975__ni_k2_r2,R2,58395,6379,0.10923880469218256
bl_rate_q975__ni_k2_r2,R3,48125,5211,0.10828051948051948
bl_rate_q975__ni_k2_r2,R4,31856,4212,0.13221998995479659
bl_rate_q975__ni_k2_r2,R5,40324,5979,0.1482739807558774
bl_rate_q975__ni_k2_r2,NA,0,0,
bl_rate_q975__ni_k2_r4,L1,56961,11125,0.1953090711188357
bl_rate_q975__ni_k2_r4,L2,42151,7503,0.17800289435600578
bl_rate_q975__ni_k2_r4,L3,32568,5912,0.18152788012773274
bl_rate_q975__ni_k2_r4,L4,17383,3721,0.21405971351320255
bl_rate_q975__ni_k2_r4,L5,41112,8050,0.19580657715508853
bl_rate_q975__ni_k2_r4,R1,65498,13848,0.21142630309322422
bl_rate_q975__ni_k2_r4,R2,58395,10265,0.17578559808202757
bl_rate_q975__ni_k2_r4,R3,48125,8559,0.17784935064935065
bl_rate_q975__ni_k2_r4,R4,31856,6585,0.20671145153189352
bl_rate_q975__ni_k2_r4,R5,40324,8846,0.21937307806765202
bl_rate_q975__ni_k2_r4,NA,0,0,
bl_rate_q975__ni_k3_r1,L1,56961,2955,0.05187760046347501
bl_rate_q975__ni_k3_r1,L2,42151,1216,0.02884866313966454
bl_rate_q975__ni_k3_r1,L3,32568,938,0.028801277327437976
bl_rate_q975__ni_k3_r1,L4,17383,692,0.039809008801702815
bl_rate_q975__ni_k3_r1,L5,41112,1956,0.04757734967892586
bl_rate_q975__ni_k3_r1,R1,65498,4219,0.06441418058566674
bl_rate_q975__ni_k3_r1,R2,58395,2069,0.035431115677712136
bl_rate_q975__ni_k3_r1,R3,48125,1638,0.034036363636363635
bl_rate_q975__ni_k3_r1,R4,31856,1462,0.045894023103967854
bl_rate_q975__ni_k3_r1,R5,40324,2780,0.06894157325662137
bl_rate_q975__ni_k3_r1,NA,0,0,
bl_rate_q975__ni_k3_r2,L1,56961,3591,0.06304313477642598
bl_rate_q975__ni_k3_r2,L2,42151,1685,0.03997532680126213
bl_rate_q975__ni_k3_r2,L3,32568,1348,0.04139032178825841
bl_rate_q975__ni_k3_r2,L4,17383,959,0.05516884312259104
bl_rate_q975__ni_k3_r2,L5,41112,2399,0.05835279237205682
bl_rate_q975__ni_k3_r2,R1,65498,4920,0.07511679745946441
bl_rate_q975__ni_k3_r2,R2,58395,2648,0.0453463481462454
bl_rate_q975__ni_k3_r2,R3,48125,2147,0.044612987012987014
bl_rate_q975__ni_k3_r2,R4,31856,1855,0.05823078854846811
bl_rate_q975__ni_k3_r2,R5,40324,3254,0.08069635948814602
bl_rate_q975__ni_k3_r2,NA,0,0,
bl_rate_q975__ni_k3_r4,L1,56961,4693,0.08238970523691648
bl_rate_q975__ni_k3_r4,L2,42151,2582,0.061255960712675854
bl_rate_q975__ni_k3_r4,L3,32568,2030,0.06233112257430607
bl_rate_q975__ni_k3_r4,L4,17383,1429,0.08220675372490364
bl_rate_q975__ni_k3_r4,L5,41112,3235,0.0786874878381008
bl_rate_q975__ni_k3_r4,R1,65498,6209,0.09479678768817368
bl_rate_q975__ni_k3_r4,R2,58395,3696,0.06329309016182892
bl_rate_q975__ni_k3_r4,R3,48125,3150,0.06545454545454546
bl_rate_q975__ni_k3_r4,R4,31856,2584,0.08111501757910598
bl_rate_q975__ni_k3_r4,R5,40324,4091,0.10145322884634461
bl_rate_q975__ni_k3_r4,NA,0,0,
bl_rate_q975__ni_k5_r1,L1,56961,2401,0.04215164761854602
bl_rate_q975__ni_k5_r1,L2,42151,741,0.017579654100733078
bl_rate_q975__ni_k5_r1,L3,32568,584,0.017931712110046672
bl_rate_q975__ni_k5_r1,L4,17383,430,0.0247368118276477
bl_rate_q975__ni_k5_r1,L5,41112,1500,0.036485697606538234
bl_rate_q975__ni_k5_r1,R1,65498,3434,0.05242908180402455
bl_rate_q975__ni_k5_r1,R2,58395,1524,0.026098124839455433
bl_rate_q975__ni_k5_r1,R3,48125,1182,0.02456103896103896
bl_rate_q975__ni_k5_r1,R4,31856,1087,0.03412230035158212
bl_rate_q975__ni_k5_r1,R5,40324,2342,0.058079555599642896
bl_rate_q975__ni_k5_r1,NA,0,0,
bl_rate_q975__ni_k5_r2,L1,56961,2472,0.04339811449939432
bl_rate_q975__ni_k5_r2,L2,42151,795,0.018860762496737918
bl_rate_q975__ni_k5_r2,L3,32568,637,0.019559076394006388
bl_rate_q975__ni_k5_r2,L4,17383,464,0.02669274578611287
bl_rate_q975__ni_k5_r2,L5,41112,1547,0.0376289161315431
bl_rate_q975__ni_k5_r2,R1,65498,3516,0.05368102842834896
bl_rate_q975__ni_k5_r2,R2,58395,1584,0.027125610069355252
bl_rate_q975__ni_k5_r2,R3,48125,1241,0.025787012987012987
bl_rate_q975__ni_k5_r2,R4,31856,1131,0.03550351582119538
bl_rate_q975__ni_k5_r2,R5,40324,2387,0.05919551631782561
bl_rate_q975__ni_k5_r2,NA,0,0,
bl_rate_q975__ni_k5_r4,L1,56961,2577,0.045241481013324904
bl_rate_q975__ni_k5_r4,L2,42151,884,0.02097221892719034
bl_rate_q975__ni_k5_r4,L3,32568,723,0.022199705232129698
bl_rate_q975__ni_k5_r4,L4,17383,516,0.029684174193177242
bl_rate_q975__ni_k5_r4,L5,41112,1660,0.040377505351235646
bl_rate_q975__ni_k5_r4,R1,65498,3663,0.055925371767076855
bl_rate_q975__ni_k5_r4,R2,58395,1707,0.029231954790649883
bl_rate_q975__ni_k5_r4,R3,48125,1380,0.028675324675324677
bl_rate_q975__ni_k5_r4,R4,31856,1228,0.038548468106479154
bl_rate_q975__ni_k5_r4,R5,40324,2493,0.061824223787322684
bl_rate_q975__ni_k5_r4,NA,0,0,
bl_hmm_disagreement__ni_k2_r1,L1,56961,23038,0.4044521690279314
bl_hmm_disagreement__ni_k2_r1,L2,42151,33984,0.8062442172190458
bl_hmm_disagreement__ni_k2_r1,L3,32568,29706,0.9121223286661754
bl_hmm_disagreement__ni_k2_r1,L4,17383,16926,0.9737099464994535
bl_hmm_disagreement__ni_k2_r1,L5,41112,23054,0.5607608484140884
bl_hmm_disagreement__ni_k2_r1,R1,65498,31737,0.48454914653882564
bl_hmm_disagreement__ni_k2_r1,R2,58395,38755,0.6636698347461255
bl_hmm_disagreement__ni_k2_r1,R3,48125,40151,0.8343064935064936
bl_hmm_disagreement__ni_k2_r1,R4,31856,28866,0.9061401305876444
bl_hmm_disagreement__ni_k2_r1,R5,40324,20900,0.5183017557781966
bl_hmm_disagreement__ni_k2_r1,NA,0,0,
bl_hmm_disagreement__ni_k2_r2,L1,56961,24633,0.43245378416811503
bl_hmm_disagreement__ni_k2_r2,L2,42151,34340,0.8146900429408555
bl_hmm_disagreement__ni_k2_r2,L3,32568,29830,0.9159297469909113
bl_hmm_disagreement__ni_k2_r2,L4,17383,16963,0.9758384628660185
bl_hmm_disagreement__ni_k2_r2,L5,41112,23991,0.5835522475189726
bl_hmm_disagreement__ni_k2_r2,R1,65498,33402,0.5099697700693151
bl_hmm_disagreement__ni_k2_r2,R2,58395,39563,0.6775066358421098
bl_hmm_disagreement__ni_k2_r2,R3,48125,40450,0.8405194805194806
bl_hmm_disagreement__ni_k2_r2,R4,31856,29026,0.911162732295329
bl_hmm_disagreement__ni_k2_r2,R5,40324,21796,0.5405217736335681
bl_hmm_disagreement__ni_k2_r2,NA,0,0,
bl_hmm_disagreement__ni_k2_r4,L1,56961,27310,0.4794508523375643
bl_hmm_disagreement__ni_k2_r4,L2,42151,34963,0.8294702379540224
bl_hmm_disagreement__ni_k2_r4,L3,32568,30029,0.9220400393023827
bl_hmm_disagreement__ni_k2_r4,L4,17383,17004,0.9781970891100501
bl_hmm_disagreement__ni_k2_r4,L5,41112,25506,0.6204028021015762
bl_hmm_disagreement__ni_k2_r4,R1,65498,36189,0.5525206876545848
bl_hmm_disagreement__ni_k2_r4,R2,58395,40942,0.7011216713759739
bl_hmm_disagreement__ni_k2_r4,R3,48125,40975,0.8514285714285714
bl_hmm_disagreement__ni_k2_r4,R4,31856,29317,0.9202975891511803
bl_hmm_disagreement__ni_k2_r4,R5,40324,23358,0.5792580101180439
bl_hmm_disagreement__ni_k2_r4,NA,0,0,
bl_hmm_disagreement__ni_k3_r1,L1,56961,21717,0.3812608626955285
bl_hmm_disagreement__ni_k3_r1,L2,42151,33733,0.8002894356005789
bl_hmm_disagreement__ni_k3_r1,L3,32568,29602,0.9089290100712356
bl_hmm_disagreement__ni_k3_r1,L4,17383,16901,0.9722717597652879
bl_hmm_disagreement__ni_k3_r1,L5,41112,22295,0.54229908542518
bl_hmm_disagreement__ni_k3_r1,R1,65498,30259,0.4619835720174662
bl_hmm_disagreement__ni_k3_r1,R2,58395,38139,0.6531209863858207
bl_hmm_disagreement__ni_k3_r1,R3,48125,39898,0.8290493506493507
bl_hmm_disagreement__ni_k3_r1,R4,31856,28728,0.9018081366147664
bl_hmm_disagreement__ni_k3_r1,R5,40324,20143,0.4995288165856562
bl_hmm_disagreement__ni_k3_r1,NA,0,0,
bl_hmm_disagreement__ni_k3_r2,L1,56961,22168,0.3891785607696494
bl_hmm_disagreement__ni_k3_r2,L2,42151,33819,0.8023297193423644
bl_hmm_disagreement__ni_k3_r2,L3,32568,29639,0.9100650945713584
bl_hmm_disagreement__ni_k3_r2,L4,17383,16912,0.9729045619283208
bl_hmm_disagreement__ni_k3_r2,L5,41112,22535,0.5481367970422261
bl_hmm_disagreement__ni_k3_r2,R1,65498,30726,0.4691135607194113
bl_hmm_disagreement__ni_k3_r2,R2,58395,38348,0.6567000599366384
bl_hmm_disagreement__ni_k3_r2,R3,48125,39964,0.8304207792207792
bl_hmm_disagreement__ni_k3_r2,R4,31856,28773,0.9032207433450528
bl_hmm_disagreement__ni_k3_r2,R5,40324,20385,0.5055302053367722
bl_hmm_disagreement__ni_k3_r2,NA,0,0,
bl_hmm_disagreement__ni_k3_r4,L1,56961,22964,0.40315303453239937
bl_hmm_disagreement__ni_k3_r4,L2,42151,34013,0.8069322198761595
bl_hmm_disagreement__ni_k3_r4,L3,32568,29703,0.912030213706706
bl_hmm_disagreement__ni_k3_r4,L4,17383,16927,0.9737674739688201
bl_hmm_disagreement__ni_k3_r4,L5,41112,22967,0.5586446779529092
bl_hmm_disagreement__ni_k3_r4,R1,65498,31535,0.48146508290329476
bl_hmm_disagreement__ni_k3_r4,R2,58395,38731,0.6632588406541656
bl_hmm_disagreement__ni_k3_r4,R3,48125,40118,0.8336207792207793
bl_hmm_disagreement__ni_k3_r4,R4,31856,28855,0.905794826720241
bl_hmm_disagreement__ni_k3_r4,R5,40324,20853,0.5171361968058724
bl_hmm_disagreement__ni_k3_r4,NA,0,0,
bl_hmm_disagreement__ni_k5_r1,L1,56961,21360,0.37499341654816454
bl_hmm_disagreement__ni_k5_r1,L2,42151,33639,0.7980593580223482
bl_hmm_disagreement__ni_k5_r1,L3,32568,29577,0.9081613854089904
bl_hmm_disagreement__ni_k5_r1,L4,17383,16886,0.9714088477247886
bl_hmm_disagreement__ni_k5_r1,L5,41112,22039,0.5360721930336642
bl_hmm_disagreement__ni_k5_r1,R1,65498,29810,0.4551284008672021
bl_hmm_disagreement__ni_k5_r1,R2,58395,37964,0.6501241544652796
bl_hmm_disagreement__ni_k5_r1,R3,48125,39829,0.8276155844155845
bl_hmm_disagreement__ni_k5_r1,R4,31856,28693,0.9007094424912104
bl_hmm_disagreement__ni_k5_r1,R5,40324,19941,0.4945193929173693
bl_hmm_disagreement__ni_k5_r1,NA,0,0,
bl_hmm_disagreement__ni_k5_r2,L1,56961,21415,0.37595898948403295
bl_hmm_disagreement__ni_k5_r2,L2,42151,33651,0.7983440487770159
bl_hmm_disagreement__ni_k5_r2,L3,32568,29583,0.9083456153279292
bl_hmm_disagreement__ni_k5_r2,L4,17383,16887,0.9714663751941552
bl_hmm_disagreement__ni_k5_r2,L5,41112,22067,0.5367532593889862
bl_hmm_disagreement__ni_k5_r2,R1,65498,29859,0.4558765153134447
bl_hmm_disagreement__ni_k5_r2,R2,58395,37984,0.6504666495419128
bl_hmm_disagreement__ni_k5_r2,R3,48125,39838,0.8278025974025974
bl_hmm_disagreement__ni_k5_r2,R4,31856,28695,0.9007722250125565
bl_hmm_disagreement__ni_k5_r2,R5,40324,19963,0.4950649737129253
bl_hmm_disagreement__ni_k5_r2,NA,0,0,
bl_hmm_disagreement__ni_k5_r4,L1,56961,21483,0.3771527887501975
bl_hmm_disagreement__ni_k5_r4,L2,42151,33670,0.7987948091385733
bl_hmm_disagreement__ni_k5_r4,L3,32568,29592,0.9086219602063376
bl_hmm_disagreement__ni_k5_r4,L4,17383,16888,0.9715239026635218
bl_hmm_disagreement__ni_k5_r4,L5,41112,22125,0.538164039696439
bl_hmm_disagreement__ni_k5_r4,R1,65498,29955,0.45734220892240984
bl_hmm_disagreement__ni_k5_r4,R2,58395,38024,0.6511516396951794
bl_hmm_disagreement__ni_k5_r4,R3,48125,39860,0.8282597402597403
bl_hmm_disagreement__ni_k5_r4,R4,31856,28702,0.9009919638372677
bl_hmm_disagreement__ni_k5_r4,R5,40324,20033,0.4968009126078762
bl_hmm_disagreement__ni_k5_r4,NA,0,0,
bl_practical_or_rate995__ni_k2_r1,L1,56961,7823,0.13733958322360915
bl_practical_or_rate995__ni_k2_r1,L2,42151,5735,0.1360584565016251
bl_practical_or_rate995__ni_k2_r1,L3,32568,5389,0.16546917219356425
bl_practical_or_rate995__ni_k2_r1,L4,17383,4328,0.24897888741874244
bl_practical_or_rate995__ni_k2_r1,L5,41112,10165,0.24725141078030746
bl_practical_or_rate995__ni_k2_r1,R1,65498,6333,0.09668997526642036
bl_practical_or_rate995__ni_k2_r1,R2,58395,5654,0.0968233581642264
bl_practical_or_rate995__ni_k2_r1,R3,48125,5656,0.11752727272727273
bl_practical_or_rate995__ni_k2_r1,R4,31856,4880,0.1531893520843797
bl_practical_or_rate995__ni_k2_r1,R5,40324,6252,0.1550441424461859
bl_practical_or_rate995__ni_k2_r1,NA,0,0,
bl_practical_or_rate995__ni_k2_r2,L1,56961,9965,0.17494426010779304
bl_practical_or_rate995__ni_k2_r2,L2,42151,7268,0.17242770041042918
bl_practical_or_rate995__ni_k2_r2,L3,32568,6566,0.20160894129206583
bl_practical_or_rate995__ni_k2_r2,L4,17383,4970,0.2859115227521141
bl_practical_or_rate995__ni_k2_r2,L5,41112,11547,0.28086690017513133
bl_practical_or_rate995__ni_k2_r2,R1,65498,8980,0.1371034230052826
bl_practical_or_rate995__ni_k2_r2,R2,58395,7853,0.1344806918400548
bl_practical_or_rate995__ni_k2_r2,R3,48125,7422,0.15422337662337662
bl_practical_or_rate995__ni_k2_r2,R4,31856,6134,0.1925539929683576
bl_practical_or_rate995__ni_k2_r2,R5,40324,7792,0.19323479813510563
bl_practical_or_rate995__ni_k2_r2,NA,0,0,
bl_practical_or_rate995__ni_k2_r4,L1,56961,13586,0.23851407103105635
bl_practical_or_rate995__ni_k2_r4,L2,42151,9950,0.23605608407866954
bl_practical_or_rate995__ni_k2_r4,L3,32568,8601,0.26409358879882094
bl_practical_or_rate995__ni_k2_r4,L4,17383,6053,0.3482137720761664
bl_practical_or_rate995__ni_k2_r4,L5,41112,13845,0.33676298890834794
bl_practical_or_rate995__ni_k2_r4,R1,65498,13477,0.20576200800024427
bl_practical_or_rate995__ni_k2_r4,R2,58395,11585,0.1983902731398236
bl_practical_or_rate995__ni_k2_r4,R3,48125,10517,0.21853506493506494
bl_practical_or_rate995__ni_k2_r4,R4,31856,8295,0.26039050728277247
bl_practical_or_rate995__ni_k2_r4,R5,40324,10428,0.2586052970935423
bl_practical_or_rate995__ni_k2_r4,NA,0,0,
bl_practical_or_rate995__ni_k3_r1,L1,56961,6024,0.1057565702849318
bl_practical_or_rate995__ni_k3_r1,L2,42151,4502,0.1068064814595146
bl_practical_or_rate995__ni_k3_r1,L3,32568,4461,0.1369749447310243
bl_practical_or_rate995__ni_k3_r1,L4,17383,3803,0.21877696600126562
bl_practical_or_rate995__ni_k3_r1,L5,41112,9025,0.2195222805993384
bl_practical_or_rate995__ni_k3_r1,R1,65498,3827,0.05842926501572567
bl_practical_or_rate995__ni_k3_r1,R2,58395,3924,0.06719753403544824
bl_practical_or_rate995__ni_k3_r1,R3,48125,4235,0.088
bl_practical_or_rate995__ni_k3_r1,R4,31856,3783,0.1187531391260673
bl_practical_or_rate995__ni_k3_r1,R5,40324,4938,0.12245808947525047
bl_practical_or_rate995__ni_k3_r1,NA,0,0,
bl_practical_or_rate995__ni_k3_r2,L1,56961,6592,0.1157283053317182
bl_practical_or_rate995__ni_k3_r2,L2,42151,4909,0.11646224288866219
bl_practical_or_rate995__ni_k3_r2,L3,32568,4795,0.14723041021861952
bl_practical_or_rate995__ni_k3_r2,L4,17383,3996,0.22987976758902376
bl_practical_or_rate995__ni_k3_r2,L5,41112,9371,0.22793831484724655
bl_practical_or_rate995__ni_k3_r2,R1,65498,4516,0.0689486701884027
bl_practical_or_rate995__ni_k3_r2,R2,58395,4465,0.07646202585837829
bl_practical_or_rate995__ni_k3_r2,R3,48125,4693,0.09751688311688311
bl_practical_or_rate995__ni_k3_r2,R4,31856,4104,0.1288297338021095
bl_practical_or_rate995__ni_k3_r2,R5,40324,5356,0.1328241245908144
bl_practical_or_rate995__ni_k3_r2,NA,0,0,
bl_practical_or_rate995__ni_k3_r4,L1,56961,7613,0.13365285019574796
bl_practical_or_rate995__ni_k3_r4,L2,42151,5677,0.1346824511873977
bl_practical_or_rate995__ni_k3_r4,L3,32568,5368,0.1648243674772783
bl_practical_or_rate995__ni_k3_r4,L4,17383,4339,0.2496116895817753
bl_practical_or_rate995__ni_k3_r4,L5,41112,10038,0.2441622883829539
bl_practical_or_rate995__ni_k3_r4,R1,65498,5818,0.08882713975999267
bl_practical_or_rate995__ni_k3_r4,R2,58395,5467,0.09362102919770528
bl_practical_or_rate995__ni_k3_r4,R3,48125,5592,0.11619740259740259
bl_practical_or_rate995__ni_k3_r4,R4,31856,4756,0.14929683576092415
bl_practical_or_rate995__ni_k3_r4,R5,40324,6130,0.15201864894355718
bl_practical_or_rate995__ni_k3_r4,NA,0,0,
bl_practical_or_rate995__ni_k5_r1,L1,56961,5538,0.09722441670616738
bl_practical_or_rate995__ni_k5_r1,L2,42151,4114,0.09760148039192428
bl_practical_or_rate995__ni_k5_r1,L3,32568,4198,0.12889953328420536
bl_practical_or_rate995__ni_k5_r1,L4,17383,3626,0.20859460392337342
bl_practical_or_rate995__ni_k5_r1,L5,41112,8694,0.21147110332749564
bl_practical_or_rate995__ni_k5_r1,R1,65498,3074,0.046932730770405204
bl_practical_or_rate995__ni_k5_r1,R2,58395,3455,0.05916602448839798
bl_practical_or_rate995__ni_k5_r1,R3,48125,3848,0.07995844155844156
bl_practical_or_rate995__ni_k5_r1,R4,31856,3480,0.10924158714213963
bl_practical_or_rate995__ni_k5_r1,R5,40324,4566,0.11323281420493998
bl_practical_or_rate995__ni_k5_r1,NA,0,0,
bl_practical_or_rate995__ni_k5_r2,L1,56961,5602,0.09834799248608697
bl_practical_or_rate995__ni_k5_r2,L2,42151,4154,0.09855044957415007
bl_practical_or_rate995__ni_k5_r2,L3,32568,4242,0.13025055268975683
bl_practical_or_rate995__ni_k5_r2,L4,17383,3652,0.2100903181269056
bl_practical_or_rate995__ni_k5_r2,L5,41112,8727,0.21227378867483945
bl_practical_or_rate995__ni_k5_r2,R1,65498,3152,0.048123606827689396
bl_practical_or_rate995__ni_k5_r2,R2,58395,3510,0.06010788594913948
bl_practical_or_rate995__ni_k5_r2,R3,48125,3901,0.08105974025974026
bl_practical_or_rate995__ni_k5_r2,R4,31856,3513,0.11027749874434957
bl_practical_or_rate995__ni_k5_r2,R5,40324,4605,0.11419998016069835
bl_practical_or_rate995__ni_k5_r2,NA,0,0,
bl_practical_or_rate995__ni_k5_r4,L1,56961,5690,0.09989290918347642
bl_practical_or_rate995__ni_k5_r4,L2,42151,4226,0.10025859410215653
bl_practical_or_rate995__ni_k5_r4,L3,32568,4313,0.13243060673053303
bl_practical_or_rate995__ni_k5_r4,L4,17383,3683,0.2118736696772709
bl_practical_or_rate995__ni_k5_r4,L5,41112,8813,0.21436563533761432
bl_practical_or_rate995__ni_k5_r4,R1,65498,3294,0.05029161195761703
bl_practical_or_rate995__ni_k5_r4,R2,58395,3620,0.06199160887062249
bl_practical_or_rate995__ni_k5_r4,R3,48125,4024,0.08361558441558442
bl_practical_or_rate995__ni_k5_r4,R4,31856,3598,0.112945755901557
bl_practical_or_rate995__ni_k5_r4,R5,40324,4695,0.11643190159706378
bl_practical_or_rate995__ni_k5_r4,NA,0,0,
bl_practical_or_crossing__ni_k2_r1,L1,56961,6927,0.12160952230473482
bl_practical_or_crossing__ni_k2_r1,L2,42151,5577,0.13231002823183316
bl_practical_or_crossing__ni_k2_r1,L3,32568,5281,0.1621530336526652
bl_practical_or_crossing__ni_k2_r1,L4,17383,4216,0.24253581084968073
bl_practical_or_crossing__ni_k2_r1,L5,41112,9702,0.23598949211908932
bl_practical_or_crossing__ni_k2_r1,R1,65498,5332,0.08140706586460655
bl_practical_or_crossing__ni_k2_r1,R2,58395,5438,0.09312441133658704
bl_practical_or_crossing__ni_k2_r1,R3,48125,5375,0.11168831168831168
bl_practical_or_crossing__ni_k2_r1,R4,31856,4496,0.1411351079859367
bl_practical_or_crossing__ni_k2_r1,R5,40324,5288,0.13113778395000497
bl_practical_or_crossing__ni_k2_r1,NA,0,0,
bl_practical_or_crossing__ni_k2_r2,L1,56961,9117,0.16005688102385843
bl_practical_or_crossing__ni_k2_r2,L2,42151,7119,0.16889279020663803
bl_practical_or_crossing__ni_k2_r2,L3,32568,6464,0.19847703267010564
bl_practical_or_crossing__ni_k2_r2,L4,17383,4864,0.2798136109992521
bl_practical_or_crossing__ni_k2_r2,L5,41112,11107,0.2701644288772135
bl_practical_or_crossing__ni_k2_r2,R1,65498,8040,0.12275183975083209
bl_practical_or_crossing__ni_k2_r2,R2,58395,7647,0.1309529925507321
bl_practical_or_crossing__ni_k2_r2,R3,48125,7159,0.14875844155844156
bl_practical_or_crossing__ni_k2_r2,R4,31856,5779,0.18141009542943246
bl_practical_or_crossing__ni_k2_r2,R5,40324,6876,0.1705187977383196
bl_practical_or_crossing__ni_k2_r2,NA,0,0,
bl_practical_or_crossing__ni_k2_r4,L1,56961,12813,0.22494338231421498
bl_practical_or_crossing__ni_k2_r4,L2,42151,9819,0.23294821000688
bl_practical_or_crossing__ni_k2_r4,L3,32568,8512,0.2613608450012282
bl_practical_or_crossing__ni_k2_r4,L4,17383,5966,0.3432088822412702
bl_practical_or_crossing__ni_k2_r4,L5,41112,13441,0.32693617435298694
bl_practical_or_crossing__ni_k2_r4,R1,65498,12641,0.19299825948883936
bl_practical_or_crossing__ni_k2_r4,R2,58395,11397,0.19517081941947084
bl_practical_or_crossing__ni_k2_r4,R3,48125,10293,0.21388051948051948
bl_practical_or_crossing__ni_k2_r4,R4,31856,7988,0.2507533902561527
bl_practical_or_crossing__ni_k2_r4,R5,40324,9611,0.23834441027675826
bl_practical_or_crossing__ni_k2_r4,NA,0,0,
bl_practical_or_crossing__ni_k3_r1,L1,56961,5101,0.0895525008339039
bl_practical_or_crossing__ni_k3_r1,L2,42151,4343,0.10303432896016702
bl_practical_or_crossing__ni_k3_r1,L3,32568,4352,0.13362810120363547
bl_practical_or_crossing__ni_k3_r1,L4,17383,3687,0.2121037795547374
bl_practical_or_crossing__ni_k3_r1,L5,41112,8556,0.2081144191476941
bl_practical_or_crossing__ni_k3_r1,R1,65498,2794,0.04265779107759016
bl_practical_or_crossing__ni_k3_r1,R2,58395,3701,0.06337871393098725
bl_practical_or_crossing__ni_k3_r1,R3,48125,3945,0.08197402597402598
bl_practical_or_crossing__ni_k3_r1,R4,31856,3394,0.10654193872425917
bl_practical_or_crossing__ni_k3_r1,R5,40324,3959,0.09817974407300863
bl_practical_or_crossing__ni_k3_r1,NA,0,0,
bl_practical_or_crossing__ni_k3_r2,L1,56961,5679,0.09969979459630274
bl_practical_or_crossing__ni_k3_r2,L2,42151,4754,0.1127849873075372
bl_practical_or_crossing__ni_k3_r2,L3,32568,4686,0.14388356669123065
bl_practical_or_crossing__ni_k3_r2,L4,17383,3882,0.2233216360812288
bl_practical_or_crossing__ni_k3_r2,L5,41112,8906,0.21662774858921968
bl_practical_or_crossing__ni_k3_r2,R1,65498,3500,0.0534367461601881
bl_practical_or_crossing__ni_k3_r2,R2,58395,4250,0.07278020378457059
bl_practical_or_crossing__ni_k3_r2,R3,48125,4407,0.09157402597402597
bl_practical_or_crossing__ni_k3_r2,R4,31856,3720,0.1167754897036665
bl_practical_or_crossing__ni_k3_r2,R5,40324,4388,0.10881856958635056
bl_practical_or_crossing__ni_k3_r2,NA,0,0,
bl_practical_or_crossing__ni_k3_r4,L1,56961,6728,0.11811590386404733
bl_practical_or_crossing__ni_k3_r4,L2,42151,5527,0.1311238167540509
bl_practical_or_crossing__ni_k3_r4,L3,32568,5261,0.16153893392286908
bl_practical_or_crossing__ni_k3_r4,L4,17383,4235,0.24362883276764655
bl_practical_or_crossing__ni_k3_r4,L5,41112,9584,0.23311928390737496
bl_practical_or_crossing__ni_k3_r4,R1,65498,4831,0.07375797734281962
bl_practical_or_crossing__ni_k3_r4,R2,58395,5256,0.09000770613922425
bl_practical_or_crossing__ni_k3_r4,R3,48125,5321,0.11056623376623377
bl_practical_or_crossing__ni_k3_r4,R4,31856,4387,0.1377134605725766
bl_practical_or_crossing__ni_k3_r4,R5,40324,5190,0.1287074694970737
bl_practical_or_crossing__ni_k3_r4,NA,0,0,
bl_practical_or_crossing__ni_k5_r1,L1,56961,4609,0.08091501202577202
bl_practical_or_crossing__ni_k5_r1,L2,42151,3953,0.0937818794334654
bl_practical_or_crossing__ni_k5_r1,L3,32568,4089,0.1255526897568165
bl_practical_or_crossing__ni_k5_r1,L4,17383,3510,0.2019214174768452
bl_practical_or_crossing__ni_k5_r1,L5,41112,8220,0.19994162288382955
bl_practical_or_crossing__ni_k5_r1,R1,65498,2033,0.03103911569818926
bl_practical_or_crossing__ni_k5_r1,R2,58395,3230,0.055312954876273655
bl_practical_or_crossing__ni_k5_r1,R3,48125,3555,0.07387012987012986
bl_practical_or_crossing__ni_k5_r1,R4,31856,3086,0.09687343043696635
bl_practical_or_crossing__ni_k5_r1,R5,40324,3581,0.08880567404027379
bl_practical_or_crossing__ni_k5_r1,NA,0,0,
bl_practical_or_crossing__ni_k5_r2,L1,56961,4674,0.08205614367725286
bl_practical_or_crossing__ni_k5_r2,L2,42151,3993,0.0947308486156912
bl_practical_or_crossing__ni_k5_r2,L3,32568,4133,0.12690370916236796
bl_practical_or_crossing__ni_k5_r2,L4,17383,3536,0.20341713168037737
bl_practical_or_crossing__ni_k5_r2,L5,41112,8253,0.20074430823117337
bl_practical_or_crossing__ni_k5_r2,R1,65498,2112,0.0322452593972335
bl_practical_or_crossing__ni_k5_r2,R2,58395,3285,0.05625481633701516
bl_practical_or_crossing__ni_k5_r2,R3,48125,3608,0.07497142857142858
bl_practical_or_crossing__ni_k5_r2,R4,31856,3119,0.0979093420391763
bl_practical_or_crossing__ni_k5_r2,R5,40324,3623,0.08984723737724432
bl_practical_or_crossing__ni_k5_r2,NA,0,0,
bl_practical_or_crossing__ni_k5_r4,L1,56961,4763,0.08361861624620354
bl_practical_or_crossing__ni_k5_r4,L2,42151,4066,0.0964627173732533
bl_practical_or_crossing__ni_k5_r4,L3,32568,4204,0.1290837632031442
bl_practical_or_crossing__ni_k5_r4,L4,17383,3568,0.2052580107001093
bl_practical_or_crossing__ni_k5_r4,L5,41112,8339,0.20283615489394824
bl_practical_or_crossing__ni_k5_r4,R1,65498,2257,0.034459067452441296
bl_practical_or_crossing__ni_k5_r4,R2,58395,3395,0.058138539258498156
bl_practical_or_crossing__ni_k5_r4,R3,48125,3731,0.07752727272727272
bl_practical_or_crossing__ni_k5_r4,R4,31856,3206,0.10064038171772978
bl_practical_or_crossing__ni_k5_r4,R5,40324,3718,0.09220315444896339
bl_practical_or_crossing__ni_k5_r4,NA,0,0,
bl_two_signal_strict__ni_k2_r1,L1,56961,4284,0.07520935376836783
bl_two_signal_strict__ni_k2_r1,L2,42151,5290,0.125501174349363
bl_two_signal_strict__ni_k2_r1,L3,32568,5266,0.1616924588553181
bl_two_signal_strict__ni_k2_r1,L4,17383,4299,0.2473105908071104
bl_two_signal_strict__ni_k2_r1,L5,41112,6197,0.15073457871181165
bl_two_signal_strict__ni_k2_r1,R1,65498,4565,0.06969678463464533
bl_two_signal_strict__ni_k2_r1,R2,58395,5020,0.08596626423495163
bl_two_signal_strict__ni_k2_r1,R3,48125,5461,0.11347532467532467
bl_two_signal_strict__ni_k2_r1,R4,31856,4763,0.14951657458563536
bl_two_signal_strict__ni_k2_r1,R5,40324,4597,0.11400158714413253
bl_two_signal_strict__ni_k2_r1,NA,0,0,
bl_two_signal_strict__ni_k2_r2,L1,56961,6616,0.11614964624918804
bl_two_signal_strict__ni_k2_r2,L2,42151,6848,0.16246352399705818
bl_two_signal_strict__ni_k2_r2,L3,32568,6457,0.198262097764677
bl_two_signal_strict__ni_k2_r2,L4,17383,4945,0.28447333601794855
bl_two_signal_strict__ni_k2_r2,L5,41112,7777,0.18916618019069859
bl_two_signal_strict__ni_k2_r2,R1,65498,7325,0.11183547589239366
bl_two_signal_strict__ni_k2_r2,R2,58395,7272,0.12453120986385821
bl_two_signal_strict__ni_k2_r2,R3,48125,7241,0.15046233766233766
bl_two_signal_strict__ni_k2_r2,R4,31856,6027,0.18919512807634353
bl_two_signal_strict__ni_k2_r2,R5,40324,6256,0.1551433389544688
bl_two_signal_strict__ni_k2_r2,NA,0,0,
bl_two_signal_strict__ni_k2_r4,L1,56961,10519,0.18467021295272204
bl_two_signal_strict__ni_k2_r4,L2,42151,9576,0.22718322222485826
bl_two_signal_strict__ni_k2_r4,L3,32568,8505,0.26114591009579957
bl_two_signal_strict__ni_k2_r4,L4,17383,6033,0.34706322268883394
bl_two_signal_strict__ni_k2_r4,L5,41112,10411,0.2532350651877797
bl_two_signal_strict__ni_k2_r4,R1,65498,12011,0.1833796451800055
bl_two_signal_strict__ni_k2_r4,R2,58395,11064,0.18946827639352684
bl_two_signal_strict__ni_k2_r4,R3,48125,10362,0.2153142857142857
bl_two_signal_strict__ni_k2_r4,R4,31856,8204,0.25753390256152686
bl_two_signal_strict__ni_k2_r4,R5,40324,9061,0.22470489038785835
bl_two_signal_strict__ni_k2_r4,NA,0,0,
bl_two_signal_strict__ni_k3_r1,L1,56961,2309,0.040536507434911606
bl_two_signal_strict__ni_k3_r1,L2,42151,4027,0.09553747242058314
bl_two_signal_strict__ni_k3_r1,L3,32568,4328,0.13289118152788013
bl_two_signal_strict__ni_k3_r1,L4,17383,3769,0.21682103204280043
bl_two_signal_strict__ni_k3_r1,L5,41112,4885,0.11882175520529285
bl_two_signal_strict__ni_k3_r1,R1,65498,1974,0.030138324834346086
bl_two_signal_strict__ni_k3_r1,R2,58395,3229,0.05529583012244199
bl_two_signal_strict__ni_k3_r1,R3,48125,4017,0.08347012987012987
bl_two_signal_strict__ni_k3_r1,R4,31856,3653,0.11467227523857358
bl_two_signal_strict__ni_k3_r1,R5,40324,3189,0.07908441622854875
bl_two_signal_strict__ni_k3_r1,NA,0,0,
bl_two_signal_strict__ni_k3_r2,L1,56961,2939,0.051596706518495114
bl_two_signal_strict__ni_k3_r2,L2,42151,4443,0.10540675191573154
bl_two_signal_strict__ni_k3_r2,L3,32568,4666,0.14326946696143453
bl_two_signal_strict__ni_k3_r2,L4,17383,3963,0.2279813610999252
bl_two_signal_strict__ni_k3_r2,L5,41112,5285,0.12855127456703638
bl_two_signal_strict__ni_k3_r2,R1,65498,2703,0.04126843567742527
bl_two_signal_strict__ni_k3_r2,R2,58395,3781,0.06474869423752033
bl_two_signal_strict__ni_k3_r2,R3,48125,4475,0.09298701298701299
bl_two_signal_strict__ni_k3_r2,R4,31856,3978,0.12487443495730789
bl_two_signal_strict__ni_k3_r2,R5,40324,3646,0.09041761729987105
bl_two_signal_strict__ni_k3_r2,NA,0,0,
bl_two_signal_strict__ni_k3_r4,L1,56961,4069,0.07143484138270044
bl_two_signal_strict__ni_k3_r4,L2,42151,5234,0.12417261749424688
bl_two_signal_strict__ni_k3_r4,L3,32568,5246,0.16107835912552199
bl_two_signal_strict__ni_k3_r4,L4,17383,4306,0.24771328309267676
bl_two_signal_strict__ni_k3_r4,L5,41112,6036,0.14681844716870987
bl_two_signal_strict__ni_k3_r4,R1,65498,4074,0.062200372530458944
bl_two_signal_strict__ni_k3_r4,R2,58395,4803,0.0822501926534806
bl_two_signal_strict__ni_k3_r4,R3,48125,5382,0.11183376623376623
bl_two_signal_strict__ni_k3_r4,R4,31856,4631,0.1453729281767956
bl_two_signal_strict__ni_k3_r4,R5,40324,4467,0.110777700624938
bl_two_signal_strict__ni_k3_r4,NA,0,0,
bl_two_signal_strict__ni_k5_r1,L1,56961,1768,0.031038780920278786
bl_two_signal_strict__ni_k5_r1,L2,42151,3632,0.0861664017461033
bl_two_signal_strict__ni_k5_r1,L3,32568,4064,0.12478506509457137
bl_two_signal_strict__ni_k5_r1,L4,17383,3591,0.20658114249554163
bl_two_signal_strict__ni_k5_r1,L5,41112,4489,0.10918953103716676
bl_two_signal_strict__ni_k5_r1,R1,65498,1190,0.018168493694463952
bl_two_signal_strict__ni_k5_r1,R2,58395,2738,0.04688757599109513
bl_two_signal_strict__ni_k5_r1,R3,48125,3623,0.07528311688311688
bl_two_signal_strict__ni_k5_r1,R4,31856,3347,0.10506654947262682
bl_two_signal_strict__ni_k5_r1,R5,40324,2789,0.06916476540025791
bl_two_signal_strict__ni_k5_r1,NA,0,0,
bl_two_signal_strict__ni_k5_r2,L1,56961,1843,0.03235547128737206
bl_two_signal_strict__ni_k5_r2,L2,42151,3674,0.08716281938744039
bl_two_signal_strict__ni_k5_r2,L3,32568,4110,0.12619749447310244
bl_two_signal_strict__ni_k5_r2,L4,17383,3617,0.2080768566990738
bl_two_signal_strict__ni_k5_r2,L5,41112,4533,0.11025977816695855
bl_two_signal_strict__ni_k5_r2,R1,65498,1272,0.01942044031878836
bl_two_signal_strict__ni_k5_r2,R2,58395,2794,0.04784656220566829
bl_two_signal_strict__ni_k5_r2,R3,48125,3677,0.07640519480519481
bl_two_signal_strict__ni_k5_r2,R4,31856,3380,0.10610246107483677
bl_two_signal_strict__ni_k5_r2,R5,40324,2833,0.0702559269913699
bl_two_signal_strict__ni_k5_r2,NA,0,0,
bl_two_signal_strict__ni_k5_r4,L1,56961,1941,0.03407594670037394
bl_two_signal_strict__ni_k5_r4,L2,42151,3748,0.08891841237455814
bl_two_signal_strict__ni_k5_r4,L3,32568,4181,0.12837754851387864
bl_two_signal_strict__ni_k5_r4,L4,17383,3648,0.2098602082494391
bl_two_signal_strict__ni_k5_r4,L5,41112,4631,0.11264351041058572
bl_two_signal_strict__ni_k5_r4,R1,65498,1427,0.02178692479159669
bl_two_signal_strict__ni_k5_r4,R2,58395,2907,0.04978165938864629
bl_two_signal_strict__ni_k5_r4,R3,48125,3800,0.07896103896103897
bl_two_signal_strict__ni_k5_r4,R4,31856,3464,0.10873932697137118
bl_two_signal_strict__ni_k5_r4,R5,40324,2934,0.07276063882551334
bl_two_signal_strict__ni_k5_r4,NA,0,0,
wl_model_agreement__ni_k2_r1,L1,56961,23038,0.4044521690279314
wl_model_agreement__ni_k2_r1,L2,42151,33984,0.8062442172190458
wl_model_agreement__ni_k2_r1,L3,32568,29706,0.9121223286661754
wl_model_agreement__ni_k2_r1,L4,17383,16926,0.9737099464994535
wl_model_agreement__ni_k2_r1,L5,41112,23054,0.5607608484140884
wl_model_agreement__ni_k2_r1,R1,65498,31737,0.48454914653882564
wl_model_agreement__ni_k2_r1,R2,58395,38755,0.6636698347461255
wl_model_agreement__ni_k2_r1,R3,48125,40151,0.8343064935064936
wl_model_agreement__ni_k2_r1,R4,31856,28866,0.9061401305876444
wl_model_agreement__ni_k2_r1,R5,40324,20900,0.5183017557781966
wl_model_agreement__ni_k2_r1,NA,0,0,
wl_model_agreement__ni_k2_r2,L1,56961,24633,0.43245378416811503
wl_model_agreement__ni_k2_r2,L2,42151,34340,0.8146900429408555
wl_model_agreement__ni_k2_r2,L3,32568,29830,0.9159297469909113
wl_model_agreement__ni_k2_r2,L4,17383,16963,0.9758384628660185
wl_model_agreement__ni_k2_r2,L5,41112,23991,0.5835522475189726
wl_model_agreement__ni_k2_r2,R1,65498,33402,0.5099697700693151
wl_model_agreement__ni_k2_r2,R2,58395,39563,0.6775066358421098
wl_model_agreement__ni_k2_r2,R3,48125,40450,0.8405194805194806
wl_model_agreement__ni_k2_r2,R4,31856,29026,0.911162732295329
wl_model_agreement__ni_k2_r2,R5,40324,21796,0.5405217736335681
wl_model_agreement__ni_k2_r2,NA,0,0,
wl_model_agreement__ni_k2_r4,L1,56961,27310,0.4794508523375643
wl_model_agreement__ni_k2_r4,L2,42151,34963,0.8294702379540224
wl_model_agreement__ni_k2_r4,L3,32568,30029,0.9220400393023827
wl_model_agreement__ni_k2_r4,L4,17383,17004,0.9781970891100501
wl_model_agreement__ni_k2_r4,L5,41112,25506,0.6204028021015762
wl_model_agreement__ni_k2_r4,R1,65498,36189,0.5525206876545848
wl_model_agreement__ni_k2_r4,R2,58395,40942,0.7011216713759739
wl_model_agreement__ni_k2_r4,R3,48125,40975,0.8514285714285714
wl_model_agreement__ni_k2_r4,R4,31856,29317,0.9202975891511803
wl_model_agreement__ni_k2_r4,R5,40324,23358,0.5792580101180439
wl_model_agreement__ni_k2_r4,NA,0,0,
wl_model_agreement__ni_k3_r1,L1,56961,21717,0.3812608626955285
wl_model_agreement__ni_k3_r1,L2,42151,33733,0.8002894356005789
wl_model_agreement__ni_k3_r1,L3,32568,29602,0.9089290100712356
wl_model_agreement__ni_k3_r1,L4,17383,16901,0.9722717597652879
wl_model_agreement__ni_k3_r1,L5,41112,22295,0.54229908542518
wl_model_agreement__ni_k3_r1,R1,65498,30259,0.4619835720174662
wl_model_agreement__ni_k3_r1,R2,58395,38139,0.6531209863858207
wl_model_agreement__ni_k3_r1,R3,48125,39898,0.8290493506493507
wl_model_agreement__ni_k3_r1,R4,31856,28728,0.9018081366147664
wl_model_agreement__ni_k3_r1,R5,40324,20143,0.4995288165856562
wl_model_agreement__ni_k3_r1,NA,0,0,
wl_model_agreement__ni_k3_r2,L1,56961,22168,0.3891785607696494
wl_model_agreement__ni_k3_r2,L2,42151,33819,0.8023297193423644
wl_model_agreement__ni_k3_r2,L3,32568,29639,0.9100650945713584
wl_model_agreement__ni_k3_r2,L4,17383,16912,0.9729045619283208
wl_model_agreement__ni_k3_r2,L5,41112,22535,0.5481367970422261
wl_model_agreement__ni_k3_r2,R1,65498,30726,0.4691135607194113
wl_model_agreement__ni_k3_r2,R2,58395,38348,0.6567000599366384
wl_model_agreement__ni_k3_r2,R3,48125,39964,0.8304207792207792
wl_model_agreement__ni_k3_r2,R4,31856,28773,0.9032207433450528
wl_model_agreement__ni_k3_r2,R5,40324,20385,0.5055302053367722
wl_model_agreement__ni_k3_r2,NA,0,0,
wl_model_agreement__ni_k3_r4,L1,56961,22964,0.40315303453239937
wl_model_agreement__ni_k3_r4,L2,42151,34013,0.8069322198761595
wl_model_agreement__ni_k3_r4,L3,32568,29703,0.912030213706706
wl_model_agreement__ni_k3_r4,L4,17383,16927,0.9737674739688201
wl_model_agreement__ni_k3_r4,L5,41112,22967,0.5586446779529092
wl_model_agreement__ni_k3_r4,R1,65498,31535,0.48146508290329476
wl_model_agreement__ni_k3_r4,R2,58395,38731,0.6632588406541656
wl_model_agreement__ni_k3_r4,R3,48125,40118,0.8336207792207793
wl_model_agreement__ni_k3_r4,R4,31856,28855,0.905794826720241
wl_model_agreement__ni_k3_r4,R5,40324,20853,0.5171361968058724
wl_model_agreement__ni_k3_r4,NA,0,0,
wl_model_agreement__ni_k5_r1,L1,56961,21360,0.37499341654816454
wl_model_agreement__ni_k5_r1,L2,42151,33639,0.7980593580223482
wl_model_agreement__ni_k5_r1,L3,32568,29577,0.9081613854089904
wl_model_agreement__ni_k5_r1,L4,17383,16886,0.9714088477247886
wl_model_agreement__ni_k5_r1,L5,41112,22039,0.5360721930336642
wl_model_agreement__ni_k5_r1,R1,65498,29810,0.4551284008672021
wl_model_agreement__ni_k5_r1,R2,58395,37964,0.6501241544652796
wl_model_agreement__ni_k5_r1,R3,48125,39829,0.8276155844155845
wl_model_agreement__ni_k5_r1,R4,31856,28693,0.9007094424912104
wl_model_agreement__ni_k5_r1,R5,40324,19941,0.4945193929173693
wl_model_agreement__ni_k5_r1,NA,0,0,
wl_model_agreement__ni_k5_r2,L1,56961,21415,0.37595898948403295
wl_model_agreement__ni_k5_r2,L2,42151,33651,0.7983440487770159
wl_model_agreement__ni_k5_r2,L3,32568,29583,0.9083456153279292
wl_model_agreement__ni_k5_r2,L4,17383,16887,0.9714663751941552
wl_model_agreement__ni_k5_r2,L5,41112,22067,0.5367532593889862
wl_model_agreement__ni_k5_r2,R1,65498,29859,0.4558765153134447
wl_model_agreement__ni_k5_r2,R2,58395,37984,0.6504666495419128
wl_model_agreement__ni_k5_r2,R3,48125,39838,0.8278025974025974
wl_model_agreement__ni_k5_r2,R4,31856,28695,0.9007722250125565
wl_model_agreement__ni_k5_r2,R5,40324,19963,0.4950649737129253
wl_model_agreement__ni_k5_r2,NA,0,0,
wl_model_agreement__ni_k5_r4,L1,56961,21483,0.3771527887501975
wl_model_agreement__ni_k5_r4,L2,42151,33670,0.7987948091385733
wl_model_agreement__ni_k5_r4,L3,32568,29592,0.9086219602063376
wl_model_agreement__ni_k5_r4,L4,17383,16888,0.9715239026635218
wl_model_agreement__ni_k5_r4,L5,41112,22125,0.538164039696439
wl_model_agreement__ni_k5_r4,R1,65498,29955,0.45734220892240984
wl_model_agreement__ni_k5_r4,R2,58395,38024,0.6511516396951794
wl_model_agreement__ni_k5_r4,R3,48125,39860,0.8282597402597403
wl_model_agreement__ni_k5_r4,R4,31856,28702,0.9009919638372677
wl_model_agreement__ni_k5_r4,R5,40324,20033,0.4968009126078762
wl_model_agreement__ni_k5_r4,NA,0,0,
wl_strict_obvious__ni_k2_r1,L1,56961,37802,0.6636470567581327
wl_strict_obvious__ni_k2_r1,L2,42151,37129,0.88085691917155
wl_strict_obvious__ni_k2_r1,L3,32568,30965,0.9507799066568411
wl_strict_obvious__ni_k2_r1,L4,17383,17179,0.988264396249209
wl_strict_obvious__ni_k2_r1,L5,41112,35232,0.8569760653823701
wl_strict_obvious__ni_k2_r1,R1,65498,47197,0.7205868881492564
wl_strict_obvious__ni_k2_r1,R2,58395,45395,0.7773782001883723
wl_strict_obvious__ni_k2_r1,R3,48125,43184,0.8973298701298701
wl_strict_obvious__ni_k2_r1,R4,31856,30195,0.9478591160220995
wl_strict_obvious__ni_k2_r1,R5,40324,31088,0.7709552623747644
wl_strict_obvious__ni_k2_r1,NA,0,0,
wl_strict_obvious__ni_k2_r2,L1,56961,38672,0.6789206650164148
wl_strict_obvious__ni_k2_r2,L2,42151,37328,0.8855780408531233
wl_strict_obvious__ni_k2_r2,L3,32568,31031,0.9528064357651682
wl_strict_obvious__ni_k2_r2,L4,17383,17192,0.9890122533509751
wl_strict_obvious__ni_k2_r2,L5,41112,35542,0.8645164428877213
wl_strict_obvious__ni_k2_r2,R1,65498,48023,0.7331979602430608
wl_strict_obvious__ni_k2_r2,R2,58395,45845,0.7850843394126209
wl_strict_obvious__ni_k2_r2,R3,48125,43329,0.9003428571428571
wl_strict_obvious__ni_k2_r2,R4,31856,30267,0.9501192867905575
wl_strict_obvious__ni_k2_r2,R5,40324,31474,0.7805277254240651
wl_strict_obvious__ni_k2_r2,NA,0,0,
wl_strict_obvious__ni_k2_r4,L1,56961,40053,0.7031653236424922
wl_strict_obvious__ni_k2_r4,L2,42151,37677,0.8938577969680435
wl_strict_obvious__ni_k2_r4,L3,32568,31123,0.9556312945222304
wl_strict_obvious__ni_k2_r4,L4,17383,17204,0.9897025829833745
wl_strict_obvious__ni_k2_r4,L5,41112,36010,0.8758999805409613
wl_strict_obvious__ni_k2_r4,R1,65498,49362,0.7536413325597728
wl_strict_obvious__ni_k2_r4,R2,58395,46667,0.7991608870622485
wl_strict_obvious__ni_k2_r4,R3,48125,43593,0.9058285714285714
wl_strict_obvious__ni_k2_r4,R4,31856,30401,0.9543257157207433
wl_strict_obvious__ni_k2_r4,R5,40324,32154,0.7973911318321595
wl_strict_obvious__ni_k2_r4,NA,0,0,
wl_strict_obvious__ni_k3_r1,L1,56961,37123,0.6517266199680483
wl_strict_obvious__ni_k3_r1,L2,42151,37011,0.8780574600839838
wl_strict_obvious__ni_k3_r1,L3,32568,30925,0.9495517071972488
wl_strict_obvious__ni_k3_r1,L4,17383,17172,0.9878617039636426
wl_strict_obvious__ni_k3_r1,L5,41112,34982,0.8508951157812804
wl_strict_obvious__ni_k3_r1,R1,65498,46399,0.7084033100247336
wl_strict_obvious__ni_k3_r1,R2,58395,45078,0.7719496532237349
wl_strict_obvious__ni_k3_r1,R3,48125,43050,0.8945454545454545
wl_strict_obvious__ni_k3_r1,R4,31856,30132,0.9458814665996986
wl_strict_obvious__ni_k3_r1,R5,40324,30739,0.7623003670270806
wl_strict_obvious__ni_k3_r1,NA,0,0,
wl_strict_obvious__ni_k3_r2,L1,56961,37365,0.6559751408858693
wl_strict_obvious__ni_k3_r2,L2,42151,37064,0.8793148442504329
wl_strict_obvious__ni_k3_r2,L3,32568,30941,0.9500429869810857
wl_strict_obvious__ni_k3_r2,L4,17383,17177,0.9881493413104757
wl_strict_obvious__ni_k3_r2,L5,41112,35070,0.853035610040864
wl_strict_obvious__ni_k3_r2,R1,65498,46620,0.7117774588537055
wl_strict_obvious__ni_k3_r2,R2,58395,45193,0.7739189999143762
wl_strict_obvious__ni_k3_r2,R3,48125,43087,0.8953142857142857
wl_strict_obvious__ni_k3_r2,R4,31856,30151,0.9464779005524862
wl_strict_obvious__ni_k3_r2,R5,40324,30834,0.7646562840987997
wl_strict_obvious__ni_k3_r2,NA,0,0,
wl_strict_obvious__ni_k3_r4,L1,56961,37763,0.6629623777672442
wl_strict_obvious__ni_k3_r4,L2,42151,37169,0.8818058883537757
wl_strict_obvious__ni_k3_r4,L3,32568,30961,0.9506570867108819
wl_strict_obvious__ni_k3_r4,L4,17383,17181,0.9883794511879422
wl_strict_obvious__ni_k3_r4,L5,41112,35210,0.8564409418174742
wl_strict_obvious__ni_k3_r4,R1,65498,46990,0.7174264863049253
wl_strict_obvious__ni_k3_r4,R2,58395,45411,0.777652196249679
wl_strict_obvious__ni_k3_r4,R3,48125,43163,0.8968935064935065
wl_strict_obvious__ni_k3_r4,R4,31856,30188,0.9476393771973882
wl_strict_obvious__ni_k3_r4,R5,40324,31038,0.7697153060212281
wl_strict_obvious__ni_k3_r4,NA,0,0,
wl_strict_obvious__ni_k5_r1,L1,56961,36947,0.6486367865732694
wl_strict_obvious__ni_k5_r1,L2,42151,36960,0.8768475243766458
wl_strict_obvious__ni_k5_r1,L3,32568,30917,0.9493060673053304
wl_strict_obvious__ni_k5_r1,L4,17383,17168,0.9876315940861762
wl_strict_obvious__ni_k5_r1,L5,41112,34902,0.8489492119089317
wl_strict_obvious__ni_k5_r1,R1,65498,46166,0.7048459494946411
wl_strict_obvious__ni_k5_r1,R2,58395,44987,0.7703913006250536
wl_strict_obvious__ni_k5_r1,R3,48125,43013,0.8937766233766233
wl_strict_obvious__ni_k5_r1,R4,31856,30116,0.9453792064289301
wl_strict_obvious__ni_k5_r1,R5,40324,30651,0.7601180438448567
wl_strict_obvious__ni_k5_r1,NA,0,0,
wl_strict_obvious__ni_k5_r2,L1,56961,36974,0.649110795105423
wl_strict_obvious__ni_k5_r2,L2,42151,36968,0.877037318213091
wl_strict_obvious__ni_k5_r2,L3,32568,30918,0.9493367722918202
wl_strict_obvious__ni_k5_r2,L4,17383,17169,0.9876891215555428
wl_strict_obvious__ni_k5_r2,L5,41112,34910,0.8491438022961666
wl_strict_obvious__ni_k5_r2,R1,65498,46187,0.7051665699716022
wl_strict_obvious__ni_k5_r2,R2,58395,45002,0.7706481719325284
wl_strict_obvious__ni_k5_r2,R3,48125,43016,0.893838961038961
wl_strict_obvious__ni_k5_r2,R4,31856,30117,0.9454105976896032
wl_strict_obvious__ni_k5_r2,R5,40324,30661,0.7603660351155639
wl_strict_obvious__ni_k5_r2,NA,0,0,
wl_strict_obvious__ni_k5_r4,L1,56961,37006,0.6496725829953828
wl_strict_obvious__ni_k5_r4,L2,42151,36978,0.8772745605086475
wl_strict_obvious__ni_k5_r4,L3,32568,30919,0.94936747727831
wl_strict_obvious__ni_k5_r4,L4,17383,17170,0.9877466490249094
wl_strict_obvious__ni_k5_r4,L5,41112,34927,0.8495573068690406
wl_strict_obvious__ni_k5_r4,R1,65498,46226,0.7057620080002442
wl_strict_obvious__ni_k5_r4,R2,58395,45025,0.7710420412706568
wl_strict_obvious__ni_k5_r4,R3,48125,43031,0.8941506493506494
wl_strict_obvious__ni_k5_r4,R4,31856,30120,0.9455047714716223
wl_strict_obvious__ni_k5_r4,R5,40324,30693,0.7611596071818272
wl_strict_obvious__ni_k5_r4,NA,0,0,
hy_direct_plus_corroborated__ni_k2_r1,L1,56961,7178,0.126016046066607
hy_direct_plus_corroborated__ni_k2_r1,L2,42151,5715,0.13558397191051222
hy_direct_plus_corroborated__ni_k2_r1,L3,32568,5410,0.16611397690985016
hy_direct_plus_corroborated__ni_k2_r1,L4,17383,4372,0.2515100960708738
hy_direct_plus_corroborated__ni_k2_r1,L5,41112,9824,0.2389569955244211
hy_direct_plus_corroborated__ni_k2_r1,R1,65498,5469,0.08349873278573391
hy_direct_plus_corroborated__ni_k2_r1,R2,58395,5642,0.09661786111824643
hy_direct_plus_corroborated__ni_k2_r1,R3,48125,5685,0.11812987012987013
hy_direct_plus_corroborated__ni_k2_r1,R4,31856,4906,0.15400552486187846
hy_direct_plus_corroborated__ni_k2_r1,R5,40324,5831,0.14460370994940977
hy_direct_plus_corroborated__ni_k2_r1,NA,0,0,
hy_direct_plus_corroborated__ni_k2_r2,L1,56961,9356,0.16425273432699566
hy_direct_plus_corroborated__ni_k2_r2,L2,42151,7246,0.17190576736020496
hy_direct_plus_corroborated__ni_k2_r2,L3,32568,6587,0.20225374600835175
hy_direct_plus_corroborated__ni_k2_r2,L4,17383,5012,0.2883276764655123
hy_direct_plus_corroborated__ni_k2_r2,L5,41112,11226,0.27305896088733217
hy_direct_plus_corroborated__ni_k2_r2,R1,65498,8165,0.1246602949708388
hy_direct_plus_corroborated__ni_k2_r2,R2,58395,7844,0.13432656905556983
hy_direct_plus_corroborated__ni_k2_r2,R3,48125,7450,0.1548051948051948
hy_direct_plus_corroborated__ni_k2_r2,R4,31856,6159,0.19333877448518333
hy_direct_plus_corroborated__ni_k2_r2,R5,40324,7397,0.18343914294216843
hy_direct_plus_corroborated__ni_k2_r2,NA,0,0,
hy_direct_plus_corroborated__ni_k2_r4,L1,56961,13027,0.22870033882832114
hy_direct_plus_corroborated__ni_k2_r4,L2,42151,9934,0.23567649640577923
hy_direct_plus_corroborated__ni_k2_r4,L3,32568,8621,0.26470768852861704
hy_direct_plus_corroborated__ni_k2_r4,L4,17383,6089,0.3502847609733648
hy_direct_plus_corroborated__ni_k2_r4,L5,41112,13557,0.32975773496789257
hy_direct_plus_corroborated__ni_k2_r4,R1,65498,12751,0.19467770008244525
hy_direct_plus_corroborated__ni_k2_r4,R2,58395,11574,0.1982019008476753
hy_direct_plus_corroborated__ni_k2_r4,R3,48125,10547,0.21915844155844155
hy_direct_plus_corroborated__ni_k2_r4,R4,31856,8320,0.26117528879959817
hy_direct_plus_corroborated__ni_k2_r4,R5,40324,10084,0.2500743973812122
hy_direct_plus_corroborated__ni_k2_r4,NA,0,0,
hy_direct_plus_corroborated__ni_k3_r1,L1,56961,5359,0.09408191569670477
hy_direct_plus_corroborated__ni_k3_r1,L2,42151,4482,0.1063319968684017
hy_direct_plus_corroborated__ni_k3_r1,L3,32568,4482,0.13761974944731023
hy_direct_plus_corroborated__ni_k3_r1,L4,17383,3849,0.22142322959213023
hy_direct_plus_corroborated__ni_k3_r1,L5,41112,8681,0.21115489394823897
hy_direct_plus_corroborated__ni_k3_r1,R1,65498,2939,0.04487159913279795
hy_direct_plus_corroborated__ni_k3_r1,R2,58395,3912,0.06699203698946828
hy_direct_plus_corroborated__ni_k3_r1,R3,48125,4266,0.08864415584415584
hy_direct_plus_corroborated__ni_k3_r1,R4,31856,3812,0.11966348568558513
hy_direct_plus_corroborated__ni_k3_r1,R5,40324,4510,0.11184406308897926
hy_direct_plus_corroborated__ni_k3_r1,NA,0,0,
hy_direct_plus_corroborated__ni_k3_r2,L1,56961,5935,0.10419409771598111
hy_direct_plus_corroborated__ni_k3_r2,L2,42151,4890,0.11601148252710493
hy_direct_plus_corroborated__ni_k3_r2,L3,32568,4816,0.14787521493490544
hy_direct_plus_corroborated__ni_k3_r2,L4,17383,4042,0.2325260311798884
hy_direct_plus_corroborated__ni_k3_r2,L5,41112,9031,0.21966822338976455
hy_direct_plus_corroborated__ni_k3_r2,R1,65498,3642,0.05560475129011573
hy_direct_plus_corroborated__ni_k3_r2,R2,58395,4455,0.07629077832006165
hy_direct_plus_corroborated__ni_k3_r2,R3,48125,4722,0.09811948051948052
hy_direct_plus_corroborated__ni_k3_r2,R4,31856,4131,0.12967729784028126
hy_direct_plus_corroborated__ni_k3_r2,R5,40324,4937,0.12243329034817975
hy_direct_plus_corroborated__ni_k3_r2,NA,0,0,
hy_direct_plus_corroborated__ni_k3_r4,L1,56961,6978,0.12250487175435824
hy_direct_plus_corroborated__ni_k3_r4,L2,42151,5658,0.13423169082584044
hy_direct_plus_corroborated__ni_k3_r4,L3,32568,5388,0.16543846720707442
hy_direct_plus_corroborated__ni_k3_r4,L4,17383,4382,0.2520853707645401
hy_direct_plus_corroborated__ni_k3_r4,L5,41112,9709,0.23615975870791983
hy_direct_plus_corroborated__ni_k3_r4,R1,65498,4970,0.07588017954746709
hy_direct_plus_corroborated__ni_k3_r4,R2,58395,5455,0.09341553215172532
hy_direct_plus_corroborated__ni_k3_r4,R3,48125,5621,0.1168
hy_direct_plus_corroborated__ni_k3_r4,R4,31856,4781,0.15008161727774988
hy_direct_plus_corroborated__ni_k3_r4,R5,40324,5724,0.14195020335284197
hy_direct_plus_corroborated__ni_k3_r4,NA,0,0,
hy_direct_plus_corroborated__ni_k5_r1,L1,56961,4870,0.08549709450325661
hy_direct_plus_corroborated__ni_k5_r1,L2,42151,4094,0.09712699580081137
hy_direct_plus_corroborated__ni_k5_r1,L3,32568,4220,0.12957504298698108
hy_direct_plus_corroborated__ni_k5_r1,L4,17383,3675,0.2114134499223379
hy_direct_plus_corroborated__ni_k5_r1,L5,41112,8347,0.20303074528118312
hy_direct_plus_corroborated__ni_k5_r1,R1,65498,2178,0.03325292375339705
hy_direct_plus_corroborated__ni_k5_r1,R2,58395,3443,0.05896052744241802
hy_direct_plus_corroborated__ni_k5_r1,R3,48125,3879,0.0806025974025974
hy_direct_plus_corroborated__ni_k5_r1,R4,31856,3509,0.11015193370165746
hy_direct_plus_corroborated__ni_k5_r1,R5,40324,4134,0.10251959131038588
hy_direct_plus_corroborated__ni_k5_r1,NA,0,0,
hy_direct_plus_corroborated__ni_k5_r2,L1,56961,4935,0.08663822615473746
hy_direct_plus_corroborated__ni_k5_r2,L2,42151,4134,0.09807596498303718
hy_direct_plus_corroborated__ni_k5_r2,L3,32568,4264,0.13092606239253254
hy_direct_plus_corroborated__ni_k5_r2,L4,17383,3701,0.2129091641258701
hy_direct_plus_corroborated__ni_k5_r2,L5,41112,8380,0.20383343062852696
hy_direct_plus_corroborated__ni_k5_r2,R1,65498,2256,0.03444379981068124
hy_direct_plus_corroborated__ni_k5_r2,R2,58395,3498,0.05990238890315952
hy_direct_plus_corroborated__ni_k5_r2,R3,48125,3932,0.0817038961038961
hy_direct_plus_corroborated__ni_k5_r2,R4,31856,3542,0.1111878453038674
hy_direct_plus_corroborated__ni_k5_r2,R5,40324,4176,0.10356115464735641
hy_direct_plus_corroborated__ni_k5_r2,NA,0,0,
hy_direct_plus_corroborated__ni_k5_r4,L1,56961,5024,0.08820069872368813
hy_direct_plus_corroborated__ni_k5_r4,L2,42151,4206,0.09978410951104363
hy_direct_plus_corroborated__ni_k5_r4,L3,32568,4335,0.13310611643330877
hy_direct_plus_corroborated__ni_k5_r4,L4,17383,3732,0.2146925156762354
hy_direct_plus_corroborated__ni_k5_r4,L5,41112,8466,0.2059252772913018
hy_direct_plus_corroborated__ni_k5_r4,R1,65498,2401,0.036657607865889036
hy_direct_plus_corroborated__ni_k5_r4,R2,58395,3608,0.06178611182464252
hy_direct_plus_corroborated__ni_k5_r4,R3,48125,4055,0.08425974025974026
hy_direct_plus_corroborated__ni_k5_r4,R4,31856,3626,0.1138247112004018
hy_direct_plus_corroborated__ni_k5_r4,R5,40324,4268,0.1058426743378633
hy_direct_plus_corroborated__ni_k5_r4,NA,0,0,
hy_two_of_three_families__ni_k2_r1,L1,56961,4910,0.08619932936570636
hy_two_of_three_families__ni_k2_r1,L2,42151,7613,0.18061255960712677
hy_two_of_three_families__ni_k2_r1,L3,32568,8265,0.25377671333824614
hy_two_of_three_families__ni_k2_r1,L4,17383,5630,0.323879652534085
hy_two_of_three_families__ni_k2_r1,L5,41112,7277,0.17700428098851917
hy_two_of_three_families__ni_k2_r1,R1,65498,5496,0.08391095911325537
hy_two_of_three_families__ni_k2_r1,R2,58395,7352,0.1259011901703913
hy_two_of_three_families__ni_k2_r1,R3,48125,8485,0.1763116883116883
hy_two_of_three_families__ni_k2_r1,R4,31856,7193,0.22579733802109492
hy_two_of_three_families__ni_k2_r1,R5,40324,5399,0.13389048705485568
hy_two_of_three_families__ni_k2_r1,NA,0,0,
hy_two_of_three_families__ni_k2_r2,L1,56961,7215,0.12666561331437298
hy_two_of_three_families__ni_k2_r2,L2,42151,9049,0.21468055324903323
hy_two_of_three_families__ni_k2_r2,L3,32568,9297,0.28546425939572584
hy_two_of_three_families__ni_k2_r2,L4,17383,6191,0.3561525628487603
hy_two_of_three_families__ni_k2_r2,L5,41112,8798,0.21400077836154893
hy_two_of_three_families__ni_k2_r2,R1,65498,8181,0.12490457723899967
hy_two_of_three_families__ni_k2_r2,R2,58395,9433,0.1615378028940834
hy_two_of_three_families__ni_k2_r2,R3,48125,10054,0.20891428571428572
hy_two_of_three_families__ni_k2_r2,R4,31856,8329,0.26145781014565544
hy_two_of_three_families__ni_k2_r2,R5,40324,7005,0.1737178851304434
hy_two_of_three_families__ni_k2_r2,NA,0,0,
hy_two_of_three_families__ni_k2_r4,L1,56961,11069,0.19432594231140604
hy_two_of_three_families__ni_k2_r4,L2,42151,11573,0.27456050864748166
hy_two_of_three_families__ni_k2_r4,L3,32568,11058,0.33953574060427416
hy_two_of_three_families__ni_k2_r4,L4,17383,7148,0.4112063510326181
hy_two_of_three_families__ni_k2_r4,L5,41112,11336,0.2757345787118116
hy_two_of_three_families__ni_k2_r4,R1,65498,12759,0.1947998412165257
hy_two_of_three_families__ni_k2_r4,R2,58395,12981,0.2222964294888261
hy_two_of_three_families__ni_k2_r4,R3,48125,12811,0.2662025974025974
hy_two_of_three_families__ni_k2_r4,R4,31856,10257,0.32198016072325464
hy_two_of_three_families__ni_k2_r4,R5,40324,9747,0.24171709155837715
hy_two_of_three_families__ni_k2_r4,NA,0,0,
hy_two_of_three_families__ni_k3_r1,L1,56961,2975,0.052228717894699885
hy_two_of_three_families__ni_k3_r1,L2,42151,6460,0.15325852292946787
hy_two_of_three_families__ni_k3_r1,L3,32568,7445,0.22859862441660525
hy_two_of_three_families__ni_k3_r1,L4,17383,5160,0.29684174193177243
hy_two_of_three_families__ni_k3_r1,L5,41112,5997,0.14586981903093987
hy_two_of_three_families__ni_k3_r1,R1,65498,2966,0.0452838254603194
hy_two_of_three_families__ni_k3_r1,R2,58395,5657,0.09687473242572138
hy_two_of_three_families__ni_k3_r1,R3,48125,7195,0.1495064935064935
hy_two_of_three_families__ni_k3_r1,R4,31856,6212,0.19500251130085383
hy_two_of_three_families__ni_k3_r1,R5,40324,4033,0.10001487947624244
hy_two_of_three_families__ni_k3_r1,NA,0,0,
hy_two_of_three_families__ni_k3_r2,L1,56961,3597,0.06314847000579343
hy_two_of_three_families__ni_k3_r2,L2,42151,6839,0.1622500059310574
hy_two_of_three_families__ni_k3_r2,L3,32568,7736,0.2375337754851388
hy_two_of_three_families__ni_k3_r2,L4,17383,5325,0.30633377437726517
hy_two_of_three_families__ni_k3_r2,L5,41112,6386,0.15533177661023545
hy_two_of_three_families__ni_k3_r2,R1,65498,3680,0.05618492167699777
hy_two_of_three_families__ni_k3_r2,R2,58395,6164,0.10555698261837486
hy_two_of_three_families__ni_k3_r2,R3,48125,7604,0.15800519480519482
hy_two_of_three_families__ni_k3_r2,R4,31856,6501,0.20407458563535913
hy_two_of_three_families__ni_k3_r2,R5,40324,4482,0.11114968753099891
hy_two_of_three_families__ni_k3_r2,NA,0,0,
hy_two_of_three_families__ni_k3_r4,L1,56961,4713,0.08274082266814135
hy_two_of_three_families__ni_k3_r4,L2,42151,7570,0.17959241773623402
hy_two_of_three_families__ni_k3_r4,L3,32568,8235,0.25285556374355195
hy_two_of_three_families__ni_k3_r4,L4,17383,5628,0.32376459759535176
hy_two_of_three_families__ni_k3_r4,L5,41112,7105,0.17282058766296945
hy_two_of_three_families__ni_k3_r4,R1,65498,5019,0.07662829399370973
hy_two_of_three_families__ni_k3_r4,R2,58395,7113,0.12180837400462369
hy_two_of_three_families__ni_k3_r4,R3,48125,8408,0.17471168831168832
hy_two_of_three_families__ni_k3_r4,R4,31856,7080,0.2222501255650427
hy_two_of_three_families__ni_k3_r4,R5,40324,5266,0.13059220315444897
hy_two_of_three_families__ni_k3_r4,NA,0,0,
hy_two_of_three_families__ni_k5_r1,L1,56961,2440,0.042836326609434525
hy_two_of_three_families__ni_k5_r1,L2,42151,6095,0.14459917914165737
hy_two_of_three_families__ni_k5_r1,L3,32568,7213,0.22147506755097027
hy_two_of_three_families__ni_k5_r1,L4,17383,5001,0.2876948743024794
hy_two_of_three_families__ni_k5_r1,L5,41112,5600,0.13621327106440942
hy_two_of_three_families__ni_k5_r1,R1,65498,2205,0.0336651500809185
hy_two_of_three_families__ni_k5_r1,R2,58395,5197,0.08899734566315609
hy_two_of_three_families__ni_k5_r1,R3,48125,6847,0.14227532467532467
hy_two_of_three_families__ni_k5_r1,R4,31856,5932,0.18621295831240584
hy_two_of_three_families__ni_k5_r1,R5,40324,3651,0.09054161293522468
hy_two_of_three_families__ni_k5_r1,NA,0,0,
hy_two_of_three_families__ni_k5_r2,L1,56961,2513,0.044117905233405315
hy_two_of_three_families__ni_k5_r2,L2,42151,6135,0.1455481483238832
hy_two_of_three_families__ni_k5_r2,L3,32568,7256,0.22279538197003193
hy_two_of_three_families__ni_k5_r2,L4,17383,5024,0.28901800609791173
hy_two_of_three_families__ni_k5_r2,L5,41112,5643,0.13725919439579684
hy_two_of_three_families__ni_k5_r2,R1,65498,2287,0.03491709670524291
hy_two_of_three_families__ni_k5_r2,R2,58395,5249,0.08988783286240261
hy_two_of_three_families__ni_k5_r2,R3,48125,6900,0.14337662337662338
hy_two_of_three_families__ni_k5_r2,R4,31856,5962,0.1871546961325967
hy_two_of_three_families__ni_k5_r2,R5,40324,3691,0.09153357801805377
hy_two_of_three_families__ni_k5_r2,NA,0,0,
hy_two_of_three_families__ni_k5_r4,L1,56961,2610,0.04582082477484595
hy_two_of_three_families__ni_k5_r4,L2,42151,6205,0.14720884439277834
hy_two_of_three_families__ni_k5_r4,L3,32568,7323,0.22485261606484894
hy_two_of_three_families__ni_k5_r4,L4,17383,5051,0.29057124777081056
hy_two_of_three_families__ni_k5_r4,L5,41112,5737,0.1395456314458066
hy_two_of_three_families__ni_k5_r4,R1,65498,2439,0.037237778252771074
hy_two_of_three_families__ni_k5_r4,R2,58395,5356,0.09172018152239061
hy_two_of_three_families__ni_k5_r4,R3,48125,7010,0.14566233766233766
hy_two_of_three_families__ni_k5_r4,R4,31856,6035,0.18944625816172778
hy_two_of_three_families__ni_k5_r4,R5,40324,3793,0.09406308897926793
hy_two_of_three_families__ni_k5_r4,NA,0,0,
hy_hierarchical__ni_k2_r1,L1,56961,7776,0.13651445726023068
hy_hierarchical__ni_k2_r1,L2,42151,8143,0.1931864012716187
hy_hierarchical__ni_k2_r1,L3,32568,8491,0.2607160402849423
hy_hierarchical__ni_k2_r1,L4,17383,5724,0.32928723465454757
hy_hierarchical__ni_k2_r1,L5,41112,11273,0.27420217941233704
hy_hierarchical__ni_k2_r1,R1,65498,6353,0.09699532810162143
hy_hierarchical__ni_k2_r1,R2,58395,8207,0.14054285469646374
hy_hierarchical__ni_k2_r1,R3,48125,8867,0.18424935064935066
hy_hierarchical__ni_k2_r1,R4,31856,7389,0.23195002511300855
hy_hierarchical__ni_k2_r1,R5,40324,6859,0.17009721257811725
hy_hierarchical__ni_k2_r1,NA,0,0,
hy_hierarchical__ni_k2_r2,L1,56961,9930,0.17432980460314954
hy_hierarchical__ni_k2_r2,L2,42151,9545,0.22644777110863323
hy_hierarchical__ni_k2_r2,L3,32568,9506,0.2918816015720953
hy_hierarchical__ni_k2_r2,L4,17383,6277,0.3610999252142898
hy_hierarchical__ni_k2_r2,L5,41112,12591,0.306260945709282
hy_hierarchical__ni_k2_r2,R1,65498,8982,0.1371339582888027
hy_hierarchical__ni_k2_r2,R2,58395,10226,0.17511773268259267
hy_hierarchical__ni_k2_r2,R3,48125,10412,0.21635324675324674
hy_hierarchical__ni_k2_r2,R4,31856,8512,0.2672024108488197
hy_hierarchical__ni_k2_r2,R5,40324,8358,0.2072711040571372
hy_hierarchical__ni_k2_r2,NA,0,0,
hy_hierarchical__ni_k2_r4,L1,56961,13555,0.23796983901265778
hy_hierarchical__ni_k2_r4,L2,42151,12018,0.28511779079974375
hy_hierarchical__ni_k2_r4,L3,32568,11246,0.34530827806435765
hy_hierarchical__ni_k2_r4,L4,17383,7222,0.41546338376574815
hy_hierarchical__ni_k2_r4,L5,41112,14768,0.3592138548355711
hy_hierarchical__ni_k2_r4,R1,65498,13467,0.20560933158264375
hy_hierarchical__ni_k2_r4,R2,58395,13691,0.2344550047093073
hy_hierarchical__ni_k2_r4,R3,48125,13133,0.27289350649350647
hy_hierarchical__ni_k2_r4,R4,31856,10413,0.3268771973882471
hy_hierarchical__ni_k2_r4,R5,40324,10955,0.2716744370598155
hy_hierarchical__ni_k2_r4,NA,0,0,
hy_hierarchical__ni_k3_r1,L1,56961,5996,0.10526500588121697
hy_hierarchical__ni_k3_r1,L2,42151,7032,0.1668287822352969
hy_hierarchical__ni_k3_r1,L3,32568,7692,0.23618275607958733
hy_hierarchical__ni_k3_r1,L4,17383,5263,0.30276707127653457
hy_hierarchical__ni_k3_r1,L5,41112,10191,0.2478838295388208
hy_hierarchical__ni_k3_r1,R1,65498,3884,0.05929952059604873
hy_hierarchical__ni_k3_r1,R2,58395,6592,0.11288637725832691
hy_hierarchical__ni_k3_r1,R3,48125,7608,0.1580883116883117
hy_hierarchical__ni_k3_r1,R4,31856,6431,0.2018771973882471
hy_hierarchical__ni_k3_r1,R5,40324,5598,0.13882551334193036
hy_hierarchical__ni_k3_r1,NA,0,0,
hy_hierarchical__ni_k3_r2,L1,56961,6564,0.11523674092800337
hy_hierarchical__ni_k3_r2,L2,42151,7402,0.17560674717088562
hy_hierarchical__ni_k3_r2,L3,32568,7976,0.24490297224269222
hy_hierarchical__ni_k3_r2,L4,17383,5427,0.31220157625266065
hy_hierarchical__ni_k3_r2,L5,41112,10523,0.2559593306090679
hy_hierarchical__ni_k3_r2,R1,65498,4574,0.06983419341048581
hy_hierarchical__ni_k3_r2,R2,58395,7086,0.12134600565116876
hy_hierarchical__ni_k3_r2,R3,48125,8011,0.16646233766233767
hy_hierarchical__ni_k3_r2,R4,31856,6714,0.21076092415871422
hy_hierarchical__ni_k3_r2,R5,40324,6012,0.1490923519492114
hy_hierarchical__ni_k3_r2,NA,0,0,
hy_hierarchical__ni_k3_r4,L1,56961,7593,0.1333017327645231
hy_hierarchical__ni_k3_r4,L2,42151,8103,0.1922374320893929
hy_hierarchical__ni_k3_r4,L3,32568,8466,0.2599484156226971
hy_hierarchical__ni_k3_r4,L4,17383,5727,0.32945981706264743
hy_hierarchical__ni_k3_r4,L5,41112,11156,0.271356294999027
hy_hierarchical__ni_k3_r4,R1,65498,5871,0.08963632477327552
hy_hierarchical__ni_k3_r4,R2,58395,8007,0.137117903930131
hy_hierarchical__ni_k3_r4,R3,48125,8803,0.1829194805194805
hy_hierarchical__ni_k3_r4,R4,31856,7286,0.22871672526368658
hy_hierarchical__ni_k3_r4,R5,40324,6761,0.16766689812518598
hy_hierarchical__ni_k3_r4,NA,0,0,
hy_hierarchical__ni_k5_r1,L1,56961,5513,0.09678551991713628
hy_hierarchical__ni_k5_r1,L2,42151,6681,0.15850157766126546
hy_hierarchical__ni_k5_r1,L3,32568,7462,0.22912060918693195
hy_hierarchical__ni_k5_r1,L4,17383,5109,0.2939078409940747
hy_hierarchical__ni_k5_r1,L5,41112,9873,0.24014886164623467
hy_hierarchical__ni_k5_r1,R1,65498,3144,0.04800146569360896
hy_hierarchical__ni_k5_r1,R2,58395,6161,0.10550560835687987
hy_hierarchical__ni_k5_r1,R3,48125,7269,0.15104415584415584
hy_hierarchical__ni_k5_r1,R4,31856,6154,0.19318181818181818
hy_hierarchical__ni_k5_r1,R5,40324,5248,0.13014581886717588
hy_hierarchical__ni_k5_r1,NA,0,0,
hy_hierarchical__ni_k5_r2,L1,56961,5576,0.09789153982549463
hy_hierarchical__ni_k5_r2,L2,42151,6719,0.15940309838437997
hy_hierarchical__ni_k5_r2,L3,32568,7503,0.230379513633014
hy_hierarchical__ni_k5_r2,L4,17383,5132,0.295230972789507
hy_hierarchical__ni_k5_r2,L5,41112,9904,0.2409028993967698
hy_hierarchical__ni_k5_r2,R1,65498,3222,0.049192341750893154
hy_hierarchical__ni_k5_r2,R2,58395,6211,0.10636184604846305
hy_hierarchical__ni_k5_r2,R3,48125,7321,0.15212467532467533
hy_hierarchical__ni_k5_r2,R4,31856,6183,0.194092164741336
hy_hierarchical__ni_k5_r2,R5,40324,5287,0.13111298482293424
hy_hierarchical__ni_k5_r2,NA,0,0,
hy_hierarchical__ni_k5_r4,L1,56961,5664,0.09943645652288408
hy_hierarchical__ni_k5_r4,L2,42151,6786,0.16099262176460818
hy_hierarchical__ni_k5_r4,L3,32568,7569,0.2324060427413412
hy_hierarchical__ni_k5_r4,L4,17383,5159,0.2967842144624058
hy_hierarchical__ni_k5_r4,L5,41112,9983,0.24282447947071414
hy_hierarchical__ni_k5_r4,R1,65498,3364,0.051360346880820786
hy_hierarchical__ni_k5_r4,R2,58395,6314,0.10812569569312441
hy_hierarchical__ni_k5_r4,R3,48125,7430,0.15438961038961038
hy_hierarchical__ni_k5_r4,R4,31856,6255,0.19635233550979408
hy_hierarchical__ni_k5_r4,R5,40324,5378,0.1333697053863704
hy_hierarchical__ni_k5_r4,NA,0,0,
```

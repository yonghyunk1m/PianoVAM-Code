# PianoVAM Fingering Audit: Complete Research and Implementation Record

## Authoritative-offset rerun (2026-07-24)

The valid production run is `20260724T163450Z-authoritative-offset-audit-e015b28b`.
It uses original `key_offset` values from official `PianoVAM/PianoVAM_v1`
native TSVs at immutable revision `7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8`;
Vite `onset + 0.5`, inferred, nearest, and synthetic offsets are forbidden.
Acquisition verified 105/105 files and 508,621/508,621 exact joins, with zero
missing offsets, identity mismatches, and synthetic offsets; source fingering
TSV hashes are unchanged.

The earlier all-integrity run using the sidecar-less timing cache is invalid
and excluded from conclusions. The authoritative run is complete but its PIG
validity gate is truthfully closed because no checksum-verifiable PIG copy is
available; therefore no recommendation or Vite audit queue is published.

The exact generated physical, integrity, fixed/calibrated Noinfo, 189 strategy,
GT/assigned recall, precision/enrichment/incremental, methods, and all-ten-finger
tables are in the run artifacts at
`artifacts/fingering_audit/20260724T163450Z-authoritative-offset-audit-e015b28b/`.
Headline rows:

| set | hard notes | hard % | GT recall | assigned recall | precision | enrichment | incremental errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bl_two_signal_strict` | 31,066 | 6.11% | 4.08% | 13.68% | 17.78% | 1.33× | 26 |
| `bl_step_crossing` | 3,631 | 0.71% | 0.77% | 2.56% | 30.00% | 1.38× | 3 |
| `ni_k2_r1` | 21,651 | 4.26% | 1.53% | 5.13% | 8.45% | 0.39× | 6 |
| `ni_k5_r4` | 2,247 | 0.44% | 0.00% | 0.00% | — | — | 0 |

All ten fingers remain represented, including zero-recall fingers. Verify-only
passed; Python (124), Node audit-category (7), and Vite build checks passed.

**Study date:** 2026-07-23  
**Consolidated:** 2026-07-24  
**Run ID:** `20260723T122049Z-publication-audit-746e73d5`  
**Status:** computational study complete; publication recommendation gate
closed; physical and missing-context extension specified, implementation
pending

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
2. The closest set to 30,000 notes selects 31,066 notes but recalls only 4.08%
   of all authoritative errors (95% recording-clustered interval:
   1.68%–6.71%).
3. The corpus contains 74,248 notes without any hand/finger label. These
   account for 275 of the 392 ground-truth errors.
4. Filtering is strongly nonuniform by finger. In the 31,066-note set, the
   selected workload ranges from 1.70% of `R1` notes to 20.52% of `L4` notes.
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
- the legacy 500 ms cutoff is not treated as research-backed.

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
| Blacklist | `bl_step_crossing` | 3,631 | 0.71% | 0.77% | 30.00% | 2.56% |
| Blacklist | `bl_rate_q995` | 4,745 | 0.93% | 0.51% | 25.00% | 1.71% |
| Blacklist | `bl_rate_q990` | 6,351 | 1.25% | 1.02% | 23.53% | 3.42% |
| Blacklist | `bl_crossing` | 9,707 | 1.91% | 1.79% | 25.00% | 5.98% |
| Blacklist | `bl_rate_q975` | 14,655 | 2.88% | 1.53% | 14.29% | 5.13% |
| Blacklist | `bl_two_signal_strict` | 31,066 | 6.11% | 4.08% | 17.78% | 13.68% |
| Blacklist | `bl_span_practical` | 39,443 | 7.75% | 3.32% | 9.92% | 11.11% |
| Blacklist | `bl_practical_or_crossing` | 39,443 | 7.75% | 3.32% | 9.92% | 11.11% |
| Hybrid | `hy_direct_plus_corroborated` | 41,928 | 8.24% | 4.08% | 11.51% | 13.68% |
| Blacklist | `bl_practical_or_rate995` | 44,172 | 8.68% | 3.83% | 10.79% | 12.82% |
| Hybrid | `hy_two_of_three_families` | 49,748 | 9.78% | 6.12% | 14.20% | 20.51% |
| Legacy | `legacy_current_default` | 56,271 | 11.06% | 7.91% | 21.99% | 26.50% |
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
- non-thumb crossing;
- central time-conditioned position-change tail; and
- HMM disagreement.

It selects 31,066 notes (6.11%) but captures only 16 of the 392 authoritative
errors:

| Metric | Result |
|---|---:|
| All-GT exact-error recall | 4.08% |
| 95% recording-clustered interval | 1.68%–6.71% |
| Assigned-label error recall | 13.68% |
| GT precision | 17.78% |

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
| L2 | 42,151 | 3,620 | 8.59% |
| L3 | 32,568 | 4,035 | 12.39% |
| L4 | 17,383 | 3,567 | 20.52% |
| L5 | 41,112 | 4,544 | 11.05% |
| R1 | 65,498 | 1,113 | 1.70% |
| R2 | 58,395 | 2,733 | 4.68% |
| R3 | 48,125 | 3,602 | 7.48% |
| R4 | 31,856 | 3,341 | 10.49% |
| R5 | 40,324 | 2,810 | 6.97% |

The 12-fold difference between `R1` and `L4` confirms that aggregate workload
alone is misleading.

## 12. Missing labels and the workload constraint

The 74,248 missing predictions form a separate integrity problem:

- they are 14.60% of the corpus;
- they account for 275 of 392 GT errors (70.15%); and
- they already exceed the entire 30,000-note target.

Combining the missing-label queue with the nearest 31,066-note assigned-label
set would require 105,314 reviews before display context. It would capture 291
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

The 31,066-note candidate is rejected because of low and nonuniform error
recall. The 39,443-note practical-span candidate is more directly grounded in
published ergonomic research but performs even worse as an error detector.
The legacy set is larger and still recalls only 7.91% of all GT errors.

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

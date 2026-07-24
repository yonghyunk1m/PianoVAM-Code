# Physical-and-Pattern Fingering Audit Redesign

Date: 2026-07-24  
Branch: `260724-audit`  
Status: approved design

## 1. Purpose

PianoVAM contains too many assigned finger labels for exhaustive manual review.
The audit therefore needs a defensible queue of notes at elevated risk of an
incorrect assigned finger. The two co-primary outcomes are:

1. assigned-error recall: the proportion of the 117 known assigned-finger
   errors selected for review; and
2. hard-note percentage: the proportion of all assigned PianoVAM notes
   selected for review.

Neither outcome may be hidden inside a single optimization score. The study
will report their full trade-off and Pareto frontier. A queue near 30,000 notes
is useful, but validity and defensibility take priority over that target.

The redesign removes HMM and sequential ergonomic rules from the active
research scope. It evaluates only simultaneous physical conflicts and
observable detector-output patterns.

## 2. Definitions and denominators

An **assigned note** has a predicted hand in `{L, R}` and a predicted finger in
`{1, 2, 3, 4, 5}`. Notes without both values are `NoInfo` for this study.

The ground-truth set contains three distinct groups:

- 117 notes with an assigned but incorrect predicted finger;
- assigned notes whose predicted finger is correct; and
- 275 notes with no assigned predicted finger.

The primary recall denominator is exactly the first group: 117 known
assigned-finger errors. The 275 unassigned notes are not false negatives in
that metric. They remain a separate integrity-repair queue.

The primary workload denominator is all assigned PianoVAM notes. A pattern
rule may select an assigned note because it is near `NoInfo`, but the `NoInfo`
note itself does not enter the assigned-note audit queue.

The term **physical** means that the assigned fingering claims an impossible
simultaneous hand state. The term **pattern** means an observable context
associated with detector risk; it does not claim that a fingering is
physically invalid.

## 3. Scope

### 3.1 Active physical rules

The physical must-alert family contains only:

1. one simple finger assigned to two different simultaneously depressed keys
   in the same hand; and
2. a simultaneous simple-finger pair whose absolute pitch span exceeds the
   validated pair boundary.

Both endpoints of every violating pair are selected. Compound fingerings are
preserved but are not simplified into physical-invalidity claims.

Simultaneity uses the original MIDI key interval:

```text
earlier.key_offset > current.onset + 0.001 seconds
```

The comparison uses `key_offset`, never the sustain-extended frame offset.
Equal boundaries are permitted; only a strict excess triggers a rule.

### 3.2 Active pattern families

The predeclared pattern library contains:

1. context around a consecutive `NoInfo` run;
2. local `NoInfo` proportion in note-count and time windows;
3. distance to the nearest `NoInfo` note;
4. an assigned note isolated between `NoInfo` notes;
5. left/right hand pitch-region overlap or inversion;
6. simultaneous left/right pitch interleaving;
7. repeated-pitch hand/finger label instability; and
8. unusually high local note density or concurrency.

Pattern rules select assigned notes only. Every selected note retains all
triggering reason codes.

### 3.3 Excluded active rules

The following may remain in historical tables but cannot contribute to a new
recommended queue:

- PIG-trained HMM disagreement or agreement;
- sequential practical, comfortable, or relative spans;
- non-thumb crossing and step-crossing rules;
- large leaps, fast jumps, and position-change rates;
- any other sequential ergonomic transition rule;
- detector candidate uncertainty, because no complete detector-score artifact
  is available; and
- a feature or threshold invented after examining a held-out recording.

Historical results must be labeled superseded, not silently deleted.

## 4. PIG authority and validity gate

### 4.1 Local source

The user-provided PIG v1.02 archive is stored locally and excluded from Git.
Its provenance is:

```text
Google Drive file id:
1vXOQDb06mEOgW5s8JYfgjLraGaFcpjp7

SHA-256:
0da49da9c184c8612c8bd4749de7a32e66b719c606c045bbf95f3a1637d9def7
```

The extracted archive contains:

- 150 pieces;
- 309 fingering annotation files;
- 100,044 annotation rows; and
- 150 score PDFs.

The declared counts in `List.csv` must reconcile exactly with the discovered
files and annotation rows before the PIG gate can open.

### 4.2 Complete-dataset detection

PIG discovery must not infer completeness from the presence of one
`*_fingering.txt` file. It must require all of the following:

1. `List.csv` parses successfully and declares 150 pieces;
2. its declared fingering-file total is 309;
3. its declared annotation-row total is 100,044;
4. exactly 309 annotation files are present;
5. their parsed non-comment row total is 100,044; and
6. every declared piece and annotation file is represented.

This prevents the four-row test fixture from opening the production validity
gate.

### 4.3 Exact token preservation

Every PIG finger token is retained verbatim. Standard simple and compound
tokens retain all signed components.

PIG file `028-4_fingering.txt`, source line 87, contains the sole malformed
token `4_`. The canonical parser will:

- preserve `finger_token_raw="4_"`;
- parse the declared first component as finger 4, matching the repository's
  established PIG loader;
- store a `trailing_component_missing` anomaly flag;
- never fabricate a second finger; and
- include the exact file, line, note, and raw token in the anomaly report.

No source dataset file is edited.

### 4.4 Rule authority

PIG is an authoritative negative control for physical-invalidity claims. A
physical rule is eligible for must-alert status only when it triggers on zero
PIG annotations.

Pattern rules express review risk rather than invalidity. A pattern may occur
in a valid PIG performance and therefore does not fail the physical validity
gate. PIG results for applicable pattern features may be reported
descriptively but cannot be called errors.

If completeness, parsing, or physical validation fails, the pipeline may
produce diagnostic tables but must mark every recommendation
`recommendable=false`.

## 5. Physical span policy

For an unordered simple-finger pair, the published practical absolute limits
are:

| Finger pair | Practical absolute limit (semitones) |
|---|---:|
| 1-2 | 10 |
| 1-3 | 12 |
| 1-4 | 14 |
| 1-5 | 15 |
| 2-3 | 5 |
| 2-4 | 7 |
| 2-5 | 10 |
| 3-4 | 4 |
| 3-5 | 7 |
| 4-5 | 5 |

For each pair, the deployed physical boundary is:

```text
max(published practical absolute limit,
    largest valid simultaneous span observed anywhere in complete PIG)
```

A note is selected only when its span is strictly greater than that boundary.
The complete PIG scan must still record zero violations. The PIG-derived
maximum, final boundary, supporting source, observation count, and any
anomalies are written to the threshold-rationale table.

The same-finger simultaneous-key rule has no adjustable span threshold. If it
has any PIG violation at the 1 ms timing tolerance, it remains diagnostic and
is not enabled as a must-alert rule.

## 6. Pattern definitions and defensive anchors

All variants below are declared before inspecting held-out correctness labels.
The existing `ni_k3_r2` rule is retained as a comparator.

### 6.1 Consecutive NoInfo-run context

For each recording, sort notes by `(onset, note_idx)`. Around a consecutive
`NoInfo` run, select the nearest eligible assigned notes on each side.

- minimum run lengths: `{2, 3, 5}`;
- assigned-note radii per side: `{1, 2, 4}`;
- defensive anchor: `minimum run=3`, `radius=1`; and
- established comparator: `minimum run=3`, `radius=2`.

The `available_hand` ordering variant may be reported separately but cannot
replace recording-order context unless it meets the clear-improvement rule.

### 6.2 Local NoInfo density

For each assigned note, compute the proportion of `NoInfo` notes in:

- centered note windows of `{5, 9, 17}` notes; and
- centered time windows with half-widths `{0.25, 0.5, 1.0}` seconds.

Windows never cross a recording boundary. The proportion denominator is every
note present in the clipped window, including the assigned center note.
Test fixed proportions `{0.25, 0.50, 0.75}`. The defensive density anchor is
at least 50% `NoInfo` in the centered nine-note window. This encodes “many” as
a literal majority rather than as a fitted percentile.

### 6.3 Nearest NoInfo and isolation

Test nearest-`NoInfo` note distances `{1, 2, 4}` and time distances
`{0.10, 0.25, 0.50, 1.00}` seconds. The defensive anchor is direct adjacency
in recording order.

An isolated assigned note is one whose immediately preceding and following
notes in recording order are both `NoInfo`. This is reported as its own
interpretable rule even though it overlaps the distance family.

### 6.4 Hand-position overlap and interleaving

For each note, form left- and right-hand pitch regions within symmetric onset
half-windows `{0, 0.10, 0.20, 0.50}` seconds. Zero means an exact-onset group.
Test pitch tolerances `{0, 2}` semitones.

The basic overlap relation is:

```text
max(left pitches) >= min(right pitches) - tolerance
```

Strict inversion uses `max(left pitches) > min(right pitches)`. Interleaving
requires at least one left-hand pitch in the closed right-hand pitch range and
at least one right-hand pitch in the closed left-hand pitch range.

The defensive anchor is strict inversion at an exact onset with zero pitch
tolerance. The existing 200 ms, two-semitone rule remains a comparator.

### 6.5 Repeated-pitch label instability

For consecutive occurrences of the same MIDI pitch, test:

- hand-label changes; and
- finger-label changes within the same predicted hand;

at inter-onset limits `{0.10, 0.25, 0.50, 1.00}` seconds. These are detector
risk patterns, never physical-invalidity rules. Because finger substitution
can be valid, the defensive anchor for this family is disabled. A variant may
enter a recommended set only through the clear-improvement rule.

### 6.6 Local density and concurrency

Use original-onset local note counts and original-key-offset active-note
counts. Within each training fold, test upper-tail thresholds at the
`{97.5, 99.0, 99.5}` percentiles of the feature distribution. Correctness
labels are not used to calculate these percentiles.

The defensive anchor is disabled. The 99.5th-percentile variant is the most
conservative enabled candidate, but this family enters a recommended set only
through the clear-improvement rule.

## 7. Leakage-resistant evaluation

### 7.1 Leave-one-recording-out predictions

The 11 recordings define 11 folds. In each fold:

1. one entire recording is held out;
2. only the other ten recordings may be used for threshold and filter-set
   selection;
3. the selected rule set is frozen;
4. predictions are generated for the unseen recording; and
5. held-out predictions are stored before advancing to the next fold.

No row, neighbor-derived correctness label, or statistic calculated from the
held-out recording may influence correctness-driven selection. Label-free
features computed within a recording are permitted, but learned
distributional thresholds are fitted on training recordings only.

Pooled performance is calculated from the concatenated held-out predictions,
not from a filter refitted and evaluated on all 117 errors.

Confidence intervals use 2,000 paired recording-cluster bootstrap replicates,
resampling the 11 intact recordings with replacement and using random seed
`20260723`. Comparisons resample the same recording indices for the candidate
and anchor so that intervals describe their paired difference.

### 7.2 Defensive fallback and clear improvement

Every pattern family has an expert anchor or an explicitly disabled anchor.
The anchor remains the publication choice unless a single alternative passes
all of these conditions:

1. pooled held-out assigned-error recall is no lower than the anchor;
2. pooled held-out hard-note percentage is no higher than the anchor;
3. at least one of those two outcomes is strictly better;
4. in a paired recording-cluster bootstrap, the 95% interval supports
   non-inferiority on both outcomes and strict improvement on at least one;
5. it is Pareto-superior to the anchor in at least 8 of the 11
   training-fold comparisons; and
6. every applicable PIG physical gate passes.

For recall difference, non-inferiority requires the lower confidence bound to
be at least zero. For hard-note-percentage difference, where lower is better,
non-inferiority requires the upper confidence bound to be at most zero.
Strict improvement requires the relevant interval to exclude zero in the
beneficial direction.

If these conditions are not met, the expert or disabled anchor remains. A
recall/workload trade-off that does not dominate the anchor is still reported
on the Pareto frontier, but it cannot silently replace the defensive choice.

### 7.3 Filter-set combinations

Within each training fold:

1. choose at most one representative from each pattern family, using the
   defensive fallback rule;
2. retain the validated physical must-alert set as mandatory;
3. enumerate every deduplicated OR-combination of the retained pattern-family
   representatives;
4. compare those combinations with the physical-only, pattern-only, and
   expert-anchor unions; and
5. select a combination under the same clear-improvement rule.

All individual threshold variants are reported. Cross-products of every raw
threshold are not treated as independent discoveries; that would manufacture
hundreds of highly correlated opportunities to overfit 117 errors.

The deployed threshold and combination are refit on all 11 recordings only
after held-out performance has been finalized. Deployment results never
replace held-out estimates in the research tables.

## 8. Metrics

Each individual rule and filter set reports:

- assigned-error recall: caught errors divided by 117;
- hard-note count and percentage among all assigned PianoVAM notes;
- precision among ground-truth assigned notes selected for audit;
- enrichment over the assigned-error baseline;
- physical-only recall;
- pattern-only recall;
- union recall;
- physical-unique and pattern-unique caught errors;
- paired recording-cluster bootstrap intervals;
- Pareto-frontier membership; and
- secondary `assigned-error recall - hard-note percentage`, in percentage
  points.

The difference score is descriptive only. It does not override the two
co-primary outcomes.

### 8.1 Per-finger reporting

The report includes:

1. predicted-finger workload counts and rates for fingers 1 through 5;
2. predicted-finger error denominators, caught errors, and recall;
3. ground-truth-finger error denominators, caught errors, and recall;
4. a predicted-to-ground-truth error confusion matrix; and
5. macro recall, worst-finger recall, and missing-finger warnings.

All per-finger totals must reconcile with the pooled totals. Empty strata are
reported explicitly rather than converted to zero performance.

### 8.2 Physical contribution

The principal ablation compares:

```text
pattern-only
physical-only
pattern OR physical
```

It reports the incremental errors and incremental workload contributed by the
physical family. This allows the lab-meeting material to state precisely
whether physical availability is a weak audit filter without conflating that
claim with physical-rule correctness.

## 9. Components and data flow

The implementation keeps these responsibilities separate:

1. **PIG acquisition validation** verifies archive structure and completeness.
2. **Canonical parsing** preserves original PianoVAM and PIG fields and
   anomalies.
3. **Physical feature generation** computes simultaneous relationships only.
4. **Pattern feature generation** computes missingness, hand-region, label
   stability, density, and concurrency features without correctness labels.
5. **Fold selection** chooses thresholds and combinations using training
   recordings only.
6. **Evaluation** consumes frozen held-out masks and calculates metrics.
7. **Queue export** produces one row per selected assigned note with all
   physical and pattern reason codes.
8. **Reporting** materializes full tables, a consolidated report, and a
   lab-meeting summary.

The final audit mask is:

```text
validated physical must-alert
OR selected pattern-risk set
```

Data-integrity notes remain a separate queue and denominator.

## 10. Outputs

Every authoritative run produces:

- `filter_methods.csv`: exact method, family, definition, evidence class,
  thresholds, and active/superseded status;
- `threshold_sensitivity.csv`: every predeclared threshold result;
- `filter_sets.csv`: physical, pattern, union, and combination results;
- `pareto_frontier.csv`: nondominated recall/workload choices;
- `per_finger.csv`: both predicted- and ground-truth-finger metrics;
- `finger_confusion.csv`: assigned-error confusion matrix;
- `per_recording.csv`: held-out performance by recording;
- `error_pattern_statistics.csv`: prevalence of each predeclared pattern among
  the 117 errors and correct assigned notes;
- `pig_validation.csv`: rule-level PIG counts and violations;
- `pig_anomalies.csv`: exact raw-token anomalies;
- `physical_pattern_ablation.csv`: unique and incremental contributions;
- `audit_queue.tsv` or the equivalent Vite-compatible queue;
- one consolidated Markdown research document; and
- a shorter lab-meeting summary.

The consolidated document explains every filter set and threshold. It marks
previous sequential ergonomic and HMM results as historical and superseded.

## 11. Fail-closed behavior

The pipeline withholds a recommendation when any of these occurs:

- PIG structure or counts do not reconcile;
- an unrecognized PIG token is not covered by an explicit, tested anomaly
  policy;
- a must-alert physical rule has a PIG violation;
- ground-truth denominators differ from the declared 117 assigned errors and
  275 `NoInfo` notes;
- a held-out recording influences training selection;
- a per-finger or per-recording table does not reconcile with pooled totals;
- artifact hashes or row counts do not match the manifest; or
- a required output is missing.

A failed recommendation gate does not erase diagnostics. It emits the reason
and preserves non-recommendable tables for investigation.

## 12. Verification

Automated tests must cover:

- exact PIG completeness counts and fixture rejection;
- preservation and anomaly parsing of `4_`;
- signed and compound PIG tokens;
- use of original key offsets rather than sustain/frame offsets;
- 1 ms simultaneous-boundary behavior;
- strict span-boundary behavior;
- selection of both pair endpoints;
- same-pitch exclusion from different-key conflicts;
- compound-fingering exclusion from simple physical claims;
- each fixed pattern boundary and window edge;
- `NoInfo` exclusion from the assigned workload denominator;
- all 11 folds producing disjoint held-out predictions;
- training-only threshold fitting;
- defensive fallback when improvement is uncertain;
- paired cluster-bootstrap directionality;
- Pareto calculations with ties;
- predicted- and ground-truth-finger reconciliation;
- deterministic reruns from the same inputs and seed;
- artifact-manifest verification; and
- successful Vite production build with the exported queue.

The final run must execute the Python test suite, Node tests, Vite build,
artifact verifier, and an independent result-table reconciliation before any
completion claim.

## 13. Repository and publication policy

Implementation and documentation remain on branch `260724-audit`. The PIG
archive and extracted copyrighted dataset stay local and ignored by Git.
Only checksums, provenance, aggregate statistics, anomaly metadata, and
validation results may be committed.

After verification, code, tests, documentation, and aggregate result tables
will be committed and pushed to the existing remote branch.

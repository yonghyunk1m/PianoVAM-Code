# Physical-and-Pattern Fingering Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active HMM/sequential-ergonomic audit study with a fail-closed, PIG-validated physical-plus-pattern study that reports held-out assigned-error recall and assigned-note workload, selects defensive thresholds, and exports the final human-review queue.

**Architecture:** Keep PIG inventory/parsing, simultaneous physical rules, observable pattern features, held-out selection, evaluation, reporting, and Vite serialization as separate units. The study constructs masks from correctness-label-free features; a leave-one-recording-out selector evaluates frozen masks and retains expert anchors unless a candidate meets the strict paired-improvement policy. The pipeline validates complete PIG before enabling physical must-alert rules and writes reconciled research artifacts plus a Vite-compatible queue.

**Tech Stack:** Python 3.10, pandas, NumPy, PyYAML, PyArrow, pytest, React 18, Vite 5, Node's built-in test runner, Git.

## Global Constraints

- Work only on branch `260724-audit`; preserve unrelated user files and changes.
- PIG v1.02 is local-only and Git-ignored. Commit only provenance, checksum, aggregate validation, and anomaly metadata.
- PIG completeness is exactly 150 pieces, 309 annotation files, and 100,044 annotation rows.
- PIG ZIP SHA-256 is `0da49da9c184c8612c8bd4749de7a32e66b719c606c045bbf95f3a1637d9def7`.
- Preserve raw PIG token `4_`; parse first finger 4 and emit `trailing_component_missing` without fabricating another component.
- The authoritative study has 11 recordings, 117 assigned-finger errors, and 275 ground-truth notes with no assigned finger.
- Co-primary outcomes are assigned-error recall out of 117 and hard-note percentage among assigned PianoVAM notes. Never replace them with a single score.
- Use original MIDI `key_offset`, never sustain/frame offset, for simultaneity.
- Simultaneity is strict at `key_offset > onset + 0.001` seconds.
- Active rules are simultaneous physical rules and observable pattern rules only.
- HMM, sequential span, crossing, step crossing, leap, fast-jump, and movement-rate rules remain historical and cannot enter a recommended queue.
- Physical invalidity rules require zero complete-PIG violations. Pattern rules indicate audit risk and are not PIG invalidity claims.
- The expert or disabled anchor remains unless a candidate is held-out Pareto-superior, supported by paired 95% recording-cluster intervals, and superior in at least 8 of 11 training-fold comparisons.
- Use 2,000 paired recording-cluster bootstrap replicates with seed `20260723`.
- The target of roughly 30,000 notes is descriptive; defensibility and validity take priority.
- Run Python tests with `/home/junhyungp/autofinger/.venv/bin/python -m pytest`.
- Do not publish the PIG archive or extracted dataset.

---

### Task 1: Complete-PIG Inventory and Exact Token Preservation

**Files:**
- Create: `fingering_audit/pig_inventory.py`
- Modify: `fingering_audit/acquire.py`
- Modify: `fingering_audit/canonical.py`
- Modify: `tests/fingering_audit/test_evidence.py`
- Modify: `tests/fingering_audit/test_canonical.py`

**Interfaces:**
- Produces: `PigExpectedCounts(piece_count: int, fingering_file_count: int, annotation_row_count: int)`.
- Produces: `PigInventory(root: Path, piece_count: int, declared_file_count: int, observed_file_count: int, declared_row_count: int, observed_row_count: int)`.
- Produces: `inspect_pig_dataset(root: Path, expected: PigExpectedCounts = PIG_V102_COUNTS) -> PigInventory`.
- Produces: `PigFingerToken(raw: str, finger: int, finger_sign: int, components: tuple[int, ...], compound: bool, anomaly: str | None)`.
- Produces: `parse_pig_finger_token(raw: str) -> PigFingerToken | None`.
- Changes: `load_pig_canonical()` adds `finger_token_raw` and `finger_token_anomaly` while retaining `finger_token`.
- Consumes later: `pipeline.run_research()` uses the inventory before opening the PIG gate.

- [ ] **Step 1: Write inventory and parser failure tests**

Add tests that reject the four-row fixture as a complete production dataset,
validate a tiny dataset against explicit tiny expected counts, and preserve the
`4_` anomaly:

```python
from fingering_audit.pig_inventory import (
    PIG_V102_COUNTS,
    PigExpectedCounts,
    inspect_pig_dataset,
)
from fingering_audit.canonical import parse_pig_finger_token


def test_production_pig_inventory_rejects_partial_fixture():
    with pytest.raises(ValueError, match="PIG completeness"):
        inspect_pig_dataset(FIXTURES / "PIG")


def test_tiny_pig_inventory_reconciles_declared_and_observed_counts(tmp_path):
    root = tmp_path / "PianoFingeringDataset_v1.02"
    files = root / "FingeringFiles"
    files.mkdir(parents=True)
    (root / "List.csv").write_text(
        "Id,Composer,Piece,#Bars,#Notes,#Fingering,Fingering 1\n"
        "001,Bach,Fixture,1,2,1,AA\n",
        encoding="utf-8",
    )
    (files / "001-1_fingering.txt").write_text(
        "//Version: 1.00\n"
        "0\t0.0\t0.5\tC4\t60\t80\t0\t1\n"
        "1\t0.5\t1.0\tD4\t62\t80\t0\t2\n",
        encoding="utf-8",
    )
    result = inspect_pig_dataset(
        root,
        expected=PigExpectedCounts(1, 1, 2),
    )
    assert result.observed_file_count == 1
    assert result.observed_row_count == 2


def test_pig_trailing_separator_is_preserved_as_anomaly():
    token = parse_pig_finger_token("4_")
    assert token is not None
    assert token.raw == "4_"
    assert token.finger == 4
    assert token.components == (4,)
    assert token.compound is False
    assert token.anomaly == "trailing_component_missing"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_evidence.py \
  tests/fingering_audit/test_canonical.py -q
```

Expected: failures because `pig_inventory`, `PigFingerToken`, and strict
completeness validation do not exist and `4_` is rejected.

- [ ] **Step 3: Implement the inventory contract**

Create `fingering_audit/pig_inventory.py` with standard-library CSV parsing and
non-comment row counting:

```python
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PigExpectedCounts:
    piece_count: int
    fingering_file_count: int
    annotation_row_count: int


PIG_V102_COUNTS = PigExpectedCounts(150, 309, 100_044)
PIG_V102_ZIP_SHA256 = (
    "0da49da9c184c8612c8bd4749de7a32e66b719c606c045bbf95f3a1637d9def7"
)


@dataclass(frozen=True)
class PigInventory:
    root: Path
    piece_count: int
    declared_file_count: int
    observed_file_count: int
    declared_row_count: int
    observed_row_count: int
    archive_sha256: str | None


def _annotation_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(
            1
            for line in stream
            if line.strip() and not line.lstrip().startswith("//")
        )


def inspect_pig_dataset(
    root: Path,
    expected: PigExpectedCounts = PIG_V102_COUNTS,
) -> PigInventory:
    root = Path(root).resolve()
    dataset = (
        root / "PianoFingeringDataset_v1.02"
        if (root / "PianoFingeringDataset_v1.02").is_dir()
        else root
    )
    list_path = dataset / "List.csv"
    fingering_dir = dataset / "FingeringFiles"
    if not list_path.is_file() or not fingering_dir.is_dir():
        raise ValueError("PIG completeness: missing List.csv or FingeringFiles")
    with list_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    files = sorted(fingering_dir.glob("*_fingering.txt"))
    declared_files = sum(int(row["#Fingering"]) for row in rows)
    declared_rows = sum(
        int(row["#Notes"]) * int(row["#Fingering"]) for row in rows
    )
    observed_rows = sum(_annotation_rows(path) for path in files)
    observed_piece_ids = {path.name.split("-", 1)[0] for path in files}
    declared_piece_ids = {row["Id"].strip().zfill(3) for row in rows}
    observed_by_piece = Counter(
        path.name.split("-", 1)[0] for path in files
    )
    declared_by_piece = {
        row["Id"].strip().zfill(3): int(row["#Fingering"])
        for row in rows
    }
    actual = PigExpectedCounts(len(rows), len(files), observed_rows)
    if actual != expected:
        raise ValueError(
            f"PIG completeness: expected {expected}, observed {actual}"
        )
    if declared_files != expected.fingering_file_count:
        raise ValueError("PIG completeness: declared fingering count mismatch")
    if declared_rows != expected.annotation_row_count:
        raise ValueError("PIG completeness: declared annotation count mismatch")
    if observed_piece_ids != declared_piece_ids:
        raise ValueError("PIG completeness: piece/file coverage mismatch")
    if dict(observed_by_piece) != declared_by_piece:
        raise ValueError("PIG completeness: per-piece fingering count mismatch")
    archive = dataset.parent / "PianoFingeringDataset_v1.02.zip"
    archive_sha256 = None
    if archive.is_file():
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        if archive_sha256 != PIG_V102_ZIP_SHA256:
            raise ValueError("PIG completeness: source archive checksum mismatch")
    return PigInventory(
        root=dataset,
        piece_count=len(rows),
        declared_file_count=declared_files,
        observed_file_count=len(files),
        declared_row_count=declared_rows,
        observed_row_count=observed_rows,
        archive_sha256=archive_sha256,
    )
```

- [ ] **Step 4: Harden discovery and implement exact token parsing**

Change `_complete_pig_root()` to return a candidate only after
`inspect_pig_dataset(candidate)` succeeds. Catch `ValueError` while searching,
but include the validation failures in the final `PigUnavailableError`.

Replace `_pig_finger_token()` with:

```python
@dataclass(frozen=True)
class PigFingerToken:
    raw: str
    finger: int
    finger_sign: int
    components: tuple[int, ...]
    compound: bool
    anomaly: str | None


def parse_pig_finger_token(raw: str) -> PigFingerToken | None:
    value = str(raw).strip()
    anomaly = None
    pieces = value.split("_")
    if pieces and pieces[-1] == "":
        if len(pieces) != 2 or not pieces[0]:
            return None
        pieces = pieces[:-1]
        anomaly = "trailing_component_missing"
    signed_components: list[int] = []
    for piece in pieces:
        try:
            signed = int(piece)
        except ValueError:
            return None
        if not 1 <= abs(signed) <= 5:
            return None
        signed_components.append(signed)
    if not signed_components:
        return None
    components = tuple(abs(value) for value in signed_components)
    return PigFingerToken(
        raw=value,
        finger=components[0],
        finger_sign=-1 if signed_components[0] < 0 else 1,
        components=components,
        compound=len(components) > 1,
        anomaly=anomaly,
    )
```

Update `load_pig_canonical()` to serialize these exact fields:

```python
"finger_token": parsed.raw,
"finger_token_raw": parsed.raw,
"finger": parsed.finger,
"finger_sign": parsed.finger_sign,
"finger_components": parsed.components,
"compound_fingering": parsed.compound,
"finger_token_anomaly": parsed.anomaly,
```

- [ ] **Step 5: Update fixture expectations and run tests**

Change `test_ensure_pig_discovers_configured_complete_dataset` into a
fail-closed partial-fixture test. Tests that exercise parsing or physical
policy may continue to call `load_pig_canonical(FIXTURES / "PIG")` directly;
only production discovery requires official counts.

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_evidence.py \
  tests/fingering_audit/test_canonical.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  fingering_audit/pig_inventory.py \
  fingering_audit/acquire.py \
  fingering_audit/canonical.py \
  tests/fingering_audit/test_evidence.py \
  tests/fingering_audit/test_canonical.py
git commit -m "feat: validate complete PIG inventory"
```

---

### Task 2: Complete-PIG Physical Policy and Original-Offset Invariants

**Files:**
- Modify: `fingering_audit/features/audit_flags.py`
- Modify: `fingering_audit/physical_policy.py`
- Modify: `fingering_audit/evidence.py`
- Modify: `tests/fingering_audit/test_audit_flags.py`
- Modify: `tests/fingering_audit/test_evidence.py`

**Interfaces:**
- Consumes: canonical PIG fields from Task 1.
- Produces: `derive_physical_policy(pig_notes: pd.DataFrame, pig_root: Path) -> PhysicalPolicy`.
- Guarantees: all ten finger-pair boundaries exist and equal `max(PRACTICAL_ABS[pair], observed_maximum_or_zero)`.
- Guarantees: `enabled_rules` contains only zero-PIG-violation physical rules.
- Produces later: `physical_policy.yaml` and rule-level validation rows.

- [ ] **Step 1: Add physical-boundary and timing tests**

Add these cases:

```python
def test_physical_flags_ignore_sustain_frame_offset():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 0.4,
                "frame_offset_sec": 2.0,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.5,
                "offset_sec": 0.8,
                "frame_offset_sec": 2.0,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
        ]
    )
    assert not compute_audit_flags(frame).physical_candidate.any()


def test_policy_materializes_published_boundary_without_pig_pair_coverage():
    pig = load_pig_canonical(FIXTURES / "PIG")
    policy = derive_physical_policy(pig, FIXTURES / "PIG")
    assert set(policy.span_boundaries) == set(PRACTICAL_ABS)
    assert policy.span_boundaries["1-5"] == PRACTICAL_ABS["1-5"]
```

Retain the existing tests for both endpoints, the strict 1 ms boundary,
same-pitch exclusion, non-adjacent active notes, and compound exclusion.
Replace the old `test_pair_without_pig_simultaneous_coverage_has_no_boundary`
expectation with the published practical boundary required by this design.

- [ ] **Step 2: Run focused physical tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_audit_flags.py \
  tests/fingering_audit/test_evidence.py -q
```

Expected: the all-pair boundary assertion fails because the current policy
omits pairs without fixture coverage.

- [ ] **Step 3: Materialize conservative boundaries for every pair**

Change the boundary construction to:

```python
boundaries = {
    pair: max(practical, maxima.get(pair, 0))
    for pair, practical in PRACTICAL_ABS.items()
}
```

Keep `_simultaneous_pair_maxima()` restricted to simple finger tokens and
original `offset_sec`. Do not reference `frame_offset_sec`.

Keep `compute_audit_flags()` strict:

```python
float(valid.at[earlier, "_offset"])
> current_onset + timing_epsilon_sec
```

Keep physical span triggering strict:

```python
distance > boundary
```

- [ ] **Step 4: Make physical violations pair-auditable**

Extend `physical_validations_from_flags()` output generation so the policy
writer can report each violating endpoint and reason without losing note IDs.
Do not enable a failing rule:

```python
enabled_rules = frozenset(
    rule_id
    for rule_id, validation in validations.items()
    if validation.status == "pass"
)
```

The recommendation gate must still test every physical validation, including
disabled failing rules.

- [ ] **Step 5: Run unit tests and a read-only complete-PIG smoke scan**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_audit_flags.py \
  tests/fingering_audit/test_evidence.py -q
```

Then run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -c \
'from pathlib import Path; from fingering_audit.canonical import load_pig_canonical; from fingering_audit.physical_policy import derive_physical_policy; p=Path("PIG"); n=load_pig_canonical(p); q=derive_physical_policy(n,p); print(len(n), sorted(q.enabled_rules), {k:v.violation_count for k,v in q.validations.items()})'
```

Expected: unit tests PASS; PIG row count is 100044. Any nonzero complete-PIG
violation remains diagnostic and disabled; do not weaken the rule during this
task.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  fingering_audit/features/audit_flags.py \
  fingering_audit/physical_policy.py \
  fingering_audit/evidence.py \
  tests/fingering_audit/test_audit_flags.py \
  tests/fingering_audit/test_evidence.py
git commit -m "feat: enforce PIG-safe simultaneous physical policy"
```

---

### Task 3: Missingness Pattern Feature Unit

**Files:**
- Create: `fingering_audit/features/pattern_missingness.py`
- Create: `tests/fingering_audit/test_pattern_missingness.py`
- Modify: `fingering_audit/features/__init__.py`

**Interfaces:**
- Produces: `missingness_features(notes: pd.DataFrame) -> pd.DataFrame`.
- Produces: `missingness_candidate_masks(notes: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.Series]`.
- Mask IDs encode exact fixed thresholds and are stable report keys.
- Consumes later: Task 6 combines these masks without reading correctness labels.

- [ ] **Step 1: Write boundary tests for all missingness definitions**

Create a two-recording fixture and assert exact behavior:

```python
def test_missingness_features_do_not_cross_recordings():
    notes = missingness_fixture()
    features = missingness_features(notes)
    assert features.loc[3, "nearest_noinfo_note_distance"] == 1
    assert features.loc[4, "nearest_noinfo_note_distance"] == float("inf")


def test_majority_density_anchor_uses_centered_nine_note_window():
    notes = majority_fixture()
    features = missingness_features(notes)
    masks = missingness_candidate_masks(notes, features)
    assert masks["noinfo_density_notes_w9_p50"].tolist() == [
        False, False, False, False, True, False, False, False, False
    ]


def test_isolated_assigned_note_requires_noinfo_on_both_sides():
    notes = isolated_fixture()
    features = missingness_features(notes)
    assert features["isolated_between_noinfo"].tolist() == [
        False, False, True, False, False
    ]
```

Also parameterize:

- note windows `{5, 9, 17}`;
- time half-windows `{0.25, 0.50, 1.00}`;
- proportions `{0.25, 0.50, 0.75}`;
- nearest note distances `{1, 2, 4}`;
- nearest time distances `{0.10, 0.25, 0.50, 1.00}`; and
- NoInfo run grid `k in {2,3,5}`, `radius in {1,2,4}`.

Assert every mask is false on a `NoInfo` center note.
For every run variant, also materialize a separate
`noinfo_hand_run_k{run}_r{radius}` mask using
`sequence="available_hand"`; it is a comparator and is never substituted for
recording order without the clear-improvement rule.

- [ ] **Step 2: Run the new test module and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_pattern_missingness.py -q
```

Expected: import failure because `pattern_missingness.py` does not exist.

- [ ] **Step 3: Implement recording-local feature computation**

Define constants:

```python
NOINFO_RUN_LENGTHS = (2, 3, 5)
NOINFO_CONTEXT_RADII = (1, 2, 4)
NOINFO_NOTE_WINDOWS = (5, 9, 17)
NOINFO_TIME_HALF_WINDOWS_SEC = (0.25, 0.50, 1.00)
NOINFO_PROPORTIONS = (0.25, 0.50, 0.75)
NOINFO_NOTE_DISTANCES = (1, 2, 4)
NOINFO_TIME_DISTANCES_SEC = (0.10, 0.25, 0.50, 1.00)
```

Use `_assigned_fingering()` to define missingness. For each recording sorted
by `(onset_sec, note_idx)`:

- centered note-window fractions use clipped positional slices;
- centered time-window fractions use `searchsorted`;
- nearest distances scan left-to-right and right-to-left;
- `isolated_between_noinfo` tests adjacent sorted positions; and
- result rows are restored to the input index.

The returned columns must be:

```python
[
    "noinfo_fraction_notes_w5",
    "noinfo_fraction_notes_w9",
    "noinfo_fraction_notes_w17",
    "noinfo_fraction_time_h025",
    "noinfo_fraction_time_h050",
    "noinfo_fraction_time_h100",
    "nearest_noinfo_note_distance",
    "nearest_noinfo_time_distance_sec",
    "isolated_between_noinfo",
]
```

- [ ] **Step 4: Implement fixed mask construction**

Create IDs deterministically:

```python
result[f"noinfo_run_k{run}_r{radius}"] = noinfo_context_mask(
    notes,
    min_run=run,
    radius=radius,
)
result[f"noinfo_density_notes_w{width}_p{int(proportion * 100):02d}"] = (
    assigned & features[f"noinfo_fraction_notes_w{width}"].ge(proportion)
)
result[f"noinfo_density_time_h{int(half_window * 100):03d}_p{int(proportion * 100):02d}"] = (
    assigned
    & features[f"noinfo_fraction_time_h{int(half_window * 100):03d}"].ge(proportion)
)
result[f"noinfo_nearest_notes_d{distance}"] = (
    assigned & features["nearest_noinfo_note_distance"].le(distance)
)
result[f"noinfo_nearest_time_d{int(distance * 100):03d}"] = (
    assigned & features["nearest_noinfo_time_distance_sec"].le(distance)
)
result["noinfo_isolated"] = assigned & features["isolated_between_noinfo"]
```

Set the defensive IDs as metadata in Task 6:
`noinfo_run_k3_r1`, `noinfo_density_notes_w9_p50`, and
`noinfo_nearest_notes_d1`.

- [ ] **Step 5: Run tests and commit Task 3**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_pattern_missingness.py \
  tests/fingering_audit/test_audit_flags.py -q
```

Expected: PASS.

Commit:

```bash
git add \
  fingering_audit/features/pattern_missingness.py \
  fingering_audit/features/__init__.py \
  tests/fingering_audit/test_pattern_missingness.py
git commit -m "feat: add defensive missingness pattern grid"
```

---

### Task 4: Hand-Region, Repeated-Label, Density, and Candidate Catalog Units

**Files:**
- Create: `fingering_audit/features/pattern_context.py`
- Create: `fingering_audit/pattern_catalog.py`
- Create: `tests/fingering_audit/test_pattern_context.py`
- Modify: `fingering_audit/evidence/thresholds.yaml`
- Modify: `fingering_audit/evidence/sources.bib`

**Interfaces:**
- Produces: `context_pattern_features(notes: pd.DataFrame) -> pd.DataFrame`.
- Produces: `fixed_context_masks(notes: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.Series]`.
- Produces: `fit_density_thresholds(features: pd.DataFrame, train_recordings: Collection[str]) -> dict[str, float]`.
- Produces: `density_masks(notes: pd.DataFrame, features: pd.DataFrame, thresholds: Mapping[str, float]) -> dict[str, pd.Series]`.
- Produces: `pattern_catalog() -> pd.DataFrame` with one row per candidate and columns `candidate_id`, `family`, `threshold_summary`, `anchor_status`, `evidence_grade`.
- Consumes later: Task 5 uses `family` and `anchor_status`; Task 7 writes `filter_methods.csv`.

- [ ] **Step 1: Write exact context-pattern tests**

Cover exact-onset inversion, overlap tolerance, interleaving, recording
boundaries, repeated-pitch instability, and training-only density fitting:

```python
def test_exact_onset_strict_hand_inversion_anchor():
    notes = hand_fixture(left_pitch=67, right_pitch=64, delta_sec=0.0)
    features = context_pattern_features(notes)
    masks = fixed_context_masks(notes, features)
    assert masks["hand_inversion_w000_t0"].tolist() == [True, True]


def test_hand_window_does_not_include_note_just_outside_boundary():
    notes = hand_window_fixture(delta_sec=0.100001)
    features = context_pattern_features(notes)
    masks = fixed_context_masks(notes, features)
    assert not masks["hand_overlap_w100_t0"].any()


def test_repeated_pitch_finger_change_is_risk_not_physical():
    notes = repeated_pitch_fixture(ioi_sec=0.25, fingers=(2, 3))
    features = context_pattern_features(notes)
    masks = fixed_context_masks(notes, features)
    assert masks["repeat_finger_change_d025"].tolist() == [True, True]
    assert not compute_audit_flags(notes).physical_candidate.any()


def test_density_thresholds_fit_training_recordings_only():
    features = density_fixture()
    thresholds = fit_density_thresholds(features, {"train"})
    assert thresholds["local_note_count_1s_q995"] == pytest.approx(
        features.loc[features["recording_id"].eq("train"), "local_note_count_1s"].quantile(0.995)
    )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_pattern_context.py -q
```

Expected: imports fail because the new modules do not exist.

- [ ] **Step 3: Implement context features**

Use only:

```python
HAND_HALF_WINDOWS_SEC = (0.0, 0.10, 0.20, 0.50)
HAND_TOLERANCES = (0, 2)
REPEAT_IOI_LIMITS_SEC = (0.10, 0.25, 0.50, 1.00)
DENSITY_QUANTILES = (0.975, 0.990, 0.995)
```

For every recording and every center onset window:

```python
overlap = max(left_pitches) >= min(right_pitches) - tolerance
inversion = max(left_pitches) > min(right_pitches)
interleaving = (
    any(min(right_pitches) <= p <= max(right_pitches) for p in left_pitches)
    and any(min(left_pitches) <= p <= max(left_pitches) for p in right_pitches)
)
```

Mark every assigned note in the qualifying window. Repeated-pitch comparisons
sort by `(onset_sec, note_idx)` within recording and mark both endpoints when
the same MIDI pitch changes hand or changes finger within the same hand at or
below the specified IOI.

Reuse vectorized `local_note_count_1s` and original-offset active-note counts;
do not import `ergonomic.py`.

- [ ] **Step 4: Implement candidate metadata and evidence entries**

`pattern_catalog()` must name every fixed candidate, the three density
quantiles for both density features, and these anchors:

```python
ANCHORS = {
    "noinfo_run": "noinfo_run_k3_r1",
    "noinfo_density": "noinfo_density_notes_w9_p50",
    "noinfo_nearest": "noinfo_nearest_notes_d1",
    "noinfo_isolation": "noinfo_isolated",
    "hand_position": "hand_inversion_w000_t0",
    "repeat_instability": None,
    "density": None,
}
```

Mark repeated instability and density candidates `exploratory` with disabled
anchors. Mark fixed NoInfo and strict hand-inversion candidates
`defensive_pattern`. Update the evidence ledger to say pattern rules indicate
review risk, have no PIG invalidity gate, and cannot be called errors on PIG.
Use existing primary sources only where they directly support the feature;
otherwise describe the threshold as an explicit conservative operational
definition rather than inventing a citation.

- [ ] **Step 5: Run tests and evidence validation**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_pattern_context.py \
  tests/fingering_audit/test_evidence.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  fingering_audit/features/pattern_context.py \
  fingering_audit/pattern_catalog.py \
  fingering_audit/evidence/thresholds.yaml \
  fingering_audit/evidence/sources.bib \
  tests/fingering_audit/test_pattern_context.py
git commit -m "feat: add observable fingering-risk pattern catalog"
```

---

### Task 5: Paired Held-Out Selection, Defensive Fallback, and Pareto Logic

**Files:**
- Create: `fingering_audit/evaluation/selection.py`
- Modify: `fingering_audit/evaluation/bootstrap.py`
- Modify: `fingering_audit/evaluation/__init__.py`
- Create: `tests/fingering_audit/test_selection.py`
- Modify: `tests/fingering_audit/test_metrics.py`

**Interfaces:**
- Produces: `paired_cluster_difference(candidate: pd.Series, anchor: pd.Series, labels: pd.DataFrame, candidate_full: pd.Series, anchor_full: pd.Series, notes: pd.DataFrame, seed: int, replicates: int = 2000) -> dict[str, float]`.
- Produces: `pareto_mask(frame: pd.DataFrame, recall_col: str, workload_col: str) -> pd.Series`.
- Produces: `clear_improvement(candidate_row: Mapping[str, Any], anchor_row: Mapping[str, Any], intervals: Mapping[str, float], superior_folds: int) -> bool`.
- Produces: `select_family_candidate(training_metrics: pd.DataFrame, anchor_id: str | None) -> str | None`.
- Produces: `FoldSelection(held_out_recording: str, family_choices: Mapping[str, str | None], combination_families: tuple[str, ...])`.
- Consumes later: Task 6 stores fold decisions and OOF masks.

- [ ] **Step 1: Write selection-policy tests**

Tests must prove that recall and workload remain separate and that uncertainty
falls back to the anchor:

```python
def test_pareto_keeps_recall_workload_tradeoffs():
    frame = pd.DataFrame(
        {
            "id": ["small", "large", "dominated"],
            "recall": [0.20, 0.40, 0.10],
            "workload": [0.05, 0.10, 0.08],
        }
    )
    assert pareto_mask(frame, "recall", "workload").tolist() == [
        True, True, False
    ]


def test_clear_improvement_requires_both_noninferiority_intervals():
    candidate = {"assigned_error_recall": 0.30, "hard_note_percentage": 0.05}
    anchor = {"assigned_error_recall": 0.20, "hard_note_percentage": 0.05}
    intervals = {
        "recall_difference_ci_low": 0.01,
        "recall_difference_ci_high": 0.19,
        "workload_difference_ci_low": -0.01,
        "workload_difference_ci_high": 0.01,
    }
    assert not clear_improvement(candidate, anchor, intervals, superior_folds=9)


def test_clear_improvement_accepts_supported_recall_gain_without_workload_harm():
    candidate = {"assigned_error_recall": 0.30, "hard_note_percentage": 0.04}
    anchor = {"assigned_error_recall": 0.20, "hard_note_percentage": 0.05}
    intervals = {
        "recall_difference_ci_low": 0.01,
        "recall_difference_ci_high": 0.19,
        "workload_difference_ci_low": -0.02,
        "workload_difference_ci_high": -0.001,
    }
    assert clear_improvement(candidate, anchor, intervals, superior_folds=8)
```

Add a deterministic paired-bootstrap test that calls the function twice with
seed `20260723` and receives identical values.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_selection.py \
  tests/fingering_audit/test_metrics.py -q
```

Expected: import failures for the new selection interfaces.

- [ ] **Step 3: Implement paired recording-cluster differences**

Aggregate per-recording tuples containing assigned-error totals, candidate and
anchor caught errors, assigned-note totals, and candidate and anchor selected
assigned notes. Draw one `(replicates, cluster_count)` matrix with NumPy and
use it for both masks.

Return:

```python
{
    "recall_difference_ci_low": float(np.nanquantile(recall_diff, 0.025)),
    "recall_difference_ci_high": float(np.nanquantile(recall_diff, 0.975)),
    "workload_difference_ci_low": float(np.nanquantile(workload_diff, 0.025)),
    "workload_difference_ci_high": float(np.nanquantile(workload_diff, 0.975)),
    "cluster_count": int(len(cluster_rows)),
    "replicates": int(replicates),
}
```

Candidate-minus-anchor workload differences are beneficial when negative.

- [ ] **Step 4: Implement exact Pareto and fallback rules**

`pareto_mask()` marks row `i` nondominated when no row `j` satisfies:

```python
(recall_j >= recall_i)
and (workload_j <= workload_i)
and ((recall_j > recall_i) or (workload_j < workload_i))
```

`clear_improvement()` returns true only when:

```python
point_noninferior = (
    candidate_recall >= anchor_recall
    and candidate_workload <= anchor_workload
)
point_strict = (
    candidate_recall > anchor_recall
    or candidate_workload < anchor_workload
)
interval_noninferior = (
    intervals["recall_difference_ci_low"] >= 0
    and intervals["workload_difference_ci_high"] <= 0
)
interval_strict = (
    intervals["recall_difference_ci_low"] > 0
    or intervals["workload_difference_ci_high"] < 0
)
stable = superior_folds >= 8
```

The function returns the conjunction of all five values. A disabled anchor is
represented by an all-false mask, not by a missing denominator.

- [ ] **Step 5: Implement deterministic fold selection**

For one training fold, calculate recall and workload for all candidates in one
family. Start from the anchor. A candidate may replace it only when it is
point-Pareto-superior on that training fold. Break multiple qualifying choices
by:

1. highest assigned-error recall;
2. lowest hard-note percentage; and
3. lexicographically smallest candidate ID.

Enumerate every family subset using sorted family names and `itertools.combinations`.
Store the selected family representative and subset in `FoldSelection`.

- [ ] **Step 6: Run tests and commit Task 5**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_selection.py \
  tests/fingering_audit/test_metrics.py -q
```

Expected: PASS.

Commit:

```bash
git add \
  fingering_audit/evaluation/selection.py \
  fingering_audit/evaluation/bootstrap.py \
  fingering_audit/evaluation/__init__.py \
  tests/fingering_audit/test_selection.py \
  tests/fingering_audit/test_metrics.py
git commit -m "feat: add defensive held-out filter selection"
```

---

### Task 6: Physical-Pattern Study Integration and Authoritative Denominators

**Files:**
- Modify: `fingering_audit/contracts.py`
- Modify: `fingering_audit/config.py`
- Modify: `fingering_audit/config/research.yaml`
- Modify: `tests/fingering_audit/fixtures/research-minimal.yaml`
- Rewrite active construction in: `fingering_audit/study.py`
- Modify: `tests/fingering_audit/test_config.py`
- Modify: `tests/fingering_audit/test_filters.py`
- Create: `tests/fingering_audit/test_physical_pattern_study.py`

**Interfaces:**
- Adds config: `expected_recordings`, `expected_assigned_errors`, `expected_noinfo_gt`.
- Produces: `StudyData` with `candidate_masks_full`, `candidate_masks_gt`, `selections_full`, `selections_gt`, `set_metadata`, `fold_decisions`, `features`, `queue_masks_full`, and `queue_masks_gt`.
- Produces: `deployment_masks_full` and `deployment_final_set_id`, kept separate from held-out research masks.
- Produces set IDs: `physical_only`, `pattern_anchor`, `pattern_selected_oof`, `physical_or_pattern_anchor`, `physical_or_pattern_selected_oof`, family candidates, and nested family-subset OOF combinations.
- Keeps integrity masks separate from assigned-note selections.
- Consumes: Tasks 2–5.
- Produces later: Task 7 summaries and Task 8 queue export.

- [ ] **Step 1: Write config and study-contract tests**

Add exact production expectations:

```yaml
expected_recordings: 11
expected_assigned_errors: 117
expected_noinfo_gt: 275
```

Set the fixture expectations:

```yaml
expected_recordings: 1
expected_assigned_errors: 2
expected_noinfo_gt: 1
```

Test that unknown or missing expected values fail config loading and that a
label table with altered denominators fails before selection.

Add a study test:

```python
def test_active_study_contains_only_physical_and_pattern_sets(fixture_config):
    study = build_study(fixture_config)
    forbidden = ("hmm", "crossing", "span_practical", "fast_jump", "rate_q")
    assert not any(
        token in set_id
        for set_id in study.selections_full
        for token in forbidden
    )
    assert {
        "physical_only",
        "pattern_anchor",
        "physical_or_pattern_anchor",
    } <= set(study.selections_full)
```

- [ ] **Step 2: Run focused integration tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_config.py \
  tests/fingering_audit/test_filters.py \
  tests/fingering_audit/test_physical_pattern_study.py -q
```

Expected: failures because the new config fields and active set IDs do not
exist.

- [ ] **Step 3: Add authoritative denominator fields**

Add integer fields to `AuditConfig` and parse them as required positive
integers. At the start of `build_study()`, validate:

```python
recording_count = int(notes["recording_id"].nunique())
assigned_gt = _valid_assignment(labels)
assigned_errors = int((assigned_gt & labels["exact_error"]).sum())
noinfo_gt = int((~assigned_gt).sum())
observed = (
    recording_count,
    assigned_errors,
    noinfo_gt,
)
expected = (
    config.expected_recordings,
    config.expected_assigned_errors,
    config.expected_noinfo_gt,
)
if observed != expected:
    raise ValueError(
        f"authoritative denominator mismatch: expected={expected}, observed={observed}"
    )
```

- [ ] **Step 4: Replace active feature construction**

Remove HMM and `ergonomic_features()` calls from `build_study()`. Do not delete
their modules; historical documentation still references them.

Construct:

```python
missingness = missingness_features(notes)
context = context_pattern_features(notes)
features = pd.concat(
    [missingness.reset_index(drop=True), context.reset_index(drop=True)],
    axis=1,
)
fixed_masks = {
    **missingness_candidate_masks(notes, missingness),
    **fixed_context_masks(notes, context),
}
```

For each held-out recording:

1. fit density thresholds on the other ten recordings;
2. construct density masks;
3. select one representative per family on training rows;
4. enumerate family subsets on training rows;
5. freeze choices; and
6. write only held-out rows into OOF masks.

Project every full mask to GT by `note_id`. Assert no GT-only selection exists.

- [ ] **Step 5: Build anchor, selected, physical, and union masks**

The anchor pattern mask is the OR of:

```python
[
    "noinfo_run_k3_r1",
    "noinfo_density_notes_w9_p50",
    "noinfo_nearest_notes_d1",
    "noinfo_isolated",
    "hand_inversion_w000_t0",
]
```

Repeated instability and density contribute an all-false anchor.

All selected audit masks must satisfy:

```python
selection = selection & assigned & ~integrity
```

Create:

```python
selections_full["physical_only"] = physical
selections_full["pattern_anchor"] = pattern_anchor
selections_full["pattern_selected_oof"] = pattern_selected_oof
selections_full["physical_or_pattern_anchor"] = physical | pattern_anchor
selections_full["physical_or_pattern_selected_oof"] = (
    physical | pattern_selected_oof
)
```

Retain all individual candidates and nested family-subset masks for tables.
Do not add `NoInfo` or integrity rows to assigned selections.

- [ ] **Step 6: Refit only the approved deployment choice**

After all OOF predictions and paired clear-improvement decisions are frozen,
choose one stable threshold per family. Use the learned threshold only if it
passes `clear_improvement()` and is Pareto-superior in at least 8 training
folds; otherwise use the defensive or disabled anchor.

Fit any approved percentile threshold once on all 11 recordings, construct a
single deployment pattern mask over the full corpus, and store:

```python
study.deployment_masks_full = {
    "physical": physical,
    "pattern": deployment_pattern,
    "union": physical | deployment_pattern,
}
study.deployment_final_set_id = (
    "physical_or_pattern_selected"
    if learned_choice_approved
    else "physical_or_pattern_anchor"
)
```

Never use this refitted deployment mask to replace an OOF mask or OOF metric
in research tables.

- [ ] **Step 7: Record method metadata and superseded methods**

`set_metadata` must contain exact family, candidate IDs, thresholds, anchor
status, selection status, and `active=true`. Add historical rows for HMM and
sequential ergonomic methods with `active=false` and
`status="superseded_by_physical_pattern_design"`. Historical rows have no
active mask.

- [ ] **Step 8: Run study tests and full Python regression**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_config.py \
  tests/fingering_audit/test_filters.py \
  tests/fingering_audit/test_physical_pattern_study.py -q
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 6**

```bash
git add \
  fingering_audit/contracts.py \
  fingering_audit/config.py \
  fingering_audit/config/research.yaml \
  tests/fingering_audit/fixtures/research-minimal.yaml \
  fingering_audit/study.py \
  tests/fingering_audit/test_config.py \
  tests/fingering_audit/test_filters.py \
  tests/fingering_audit/test_physical_pattern_study.py
git commit -m "feat: integrate held-out physical-pattern audit study"
```

---

### Task 7: Co-Primary Metrics, Per-Finger Views, Ablation, and Result Tables

**Files:**
- Modify: `fingering_audit/evaluation/metrics.py`
- Modify: `fingering_audit/study.py`
- Rewrite table contract in: `fingering_audit/report.py`
- Modify: `tests/fingering_audit/test_metrics.py`
- Modify: `tests/fingering_audit/test_report.py`

**Interfaces:**
- Produces: `assigned_audit_metrics(selection_gt: pd.Series, labels: pd.DataFrame, selection_full: pd.Series, notes: pd.DataFrame) -> dict[str, float | int]`.
- Produces: `predicted_finger_metrics(selection, labels, set_id) -> pd.DataFrame`.
- Produces: `ground_truth_finger_metrics(selection, labels, set_id) -> pd.DataFrame`.
- Produces: `finger_error_confusion(selection, labels, set_id) -> pd.DataFrame`.
- Produces: `physical_pattern_ablation(*, physical_full: pd.Series, pattern_full: pd.Series, physical_gt: pd.Series, pattern_gt: pd.Series, notes: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame`.
- Produces required result tables from the approved specification.
- Consumes: Task 6 study masks and Task 5 Pareto/bootstrap outputs.

- [ ] **Step 1: Write denominator, finger, confusion, and ablation tests**

Add a fixture with assigned errors and one `NoInfo` prediction:

```python
def test_assigned_recall_excludes_noinfo_ground_truth_rows():
    labels = assigned_metric_fixture()
    selected = pd.Series([True, False, True, True])
    values = assigned_audit_metrics(
        selected,
        labels,
        selected,
        labels,
    )
    assert values["assigned_error_count"] == 2
    assert values["caught_assigned_errors"] == 1
    assert values["assigned_error_recall"] == pytest.approx(0.5)


def test_predicted_and_true_finger_tables_reconcile():
    labels = assigned_metric_fixture()
    selected = pd.Series([True, False, True, False])
    predicted = predicted_finger_metrics(selected, labels, "fixture")
    truth = ground_truth_finger_metrics(selected, labels, "fixture")
    assert predicted["caught_errors"].sum() == 1
    assert truth["caught_errors"].sum() == 1


def test_physical_pattern_ablation_counts_unique_errors():
    labels = ablation_fixture()
    result = physical_pattern_ablation(
        physical_full=pd.Series([True, True, False, False]),
        pattern_full=pd.Series([False, True, True, False]),
        physical_gt=pd.Series([True, True, False, False]),
        pattern_gt=pd.Series([False, True, True, False]),
        labels=labels,
        notes=labels,
    ).set_index("component")
    assert result.loc["physical_unique", "caught_errors"] == 1
    assert result.loc["pattern_unique", "caught_errors"] == 1
```

- [ ] **Step 2: Run metrics/report tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_metrics.py \
  tests/fingering_audit/test_report.py -q
```

Expected: failures for missing interfaces and old 189-set report expectations.

- [ ] **Step 3: Implement assigned and per-finger metrics**

For every set, report:

```python
assigned_gt = _valid_assignment(labels)
assigned_errors = assigned_gt & labels["exact_error"].fillna(False)
assigned_full = _valid_assignment(notes)
error_count = int(assigned_errors.sum())
caught = int((selection_gt & assigned_errors).sum())
assigned_hard_count = int((selection_full & assigned_full).sum())
selected_assigned_gt_count = int((selection_gt & assigned_gt).sum())
assigned_error_recall = _ratio(caught, error_count)
hard_note_percentage = _ratio(
    assigned_hard_count,
    int(assigned_full.sum()),
)
{
    "assigned_error_count": error_count,
    "caught_assigned_errors": caught,
    "assigned_error_recall": assigned_error_recall,
    "assigned_hard_count": assigned_hard_count,
    "hard_note_percentage": hard_note_percentage,
    "assigned_gt_precision": _ratio(caught, selected_assigned_gt_count),
    "recall_minus_hard_percentage_pp": (
        assigned_error_recall - hard_note_percentage
    ) * 100.0,
}
```

Predicted-finger rows group errors by the observed `pred_hand` and
`pred_finger`; ground-truth rows group by `gt_finger_id`. Always materialize
L1–L5 and R1–R5 and retain `NaN` recall for zero-error strata.

The confusion table groups only assigned errors by:

```python
["set_id", "pred_finger_id", "gt_finger_id", "selected"]
```

- [ ] **Step 4: Implement ablation and Pareto tables**

For physical and pattern masks, report four components:

```python
("physical_only", physical)
("pattern_only", pattern)
("union", physical | pattern)
("physical_unique", physical & ~pattern)
("pattern_unique", pattern & ~physical)
```

Each row contains assigned workload, hard-note percentage, caught assigned
errors, assigned-error recall, and incremental values relative to the other
family.

Apply `pareto_mask()` to `assigned_error_recall` and
`hard_note_percentage`. Keep all nondominated rows; do not select by the
difference score.

- [ ] **Step 5: Replace the required result-table contract**

Set `REQUIRED_RESULTS` to:

```python
(
    "filter_methods.csv",
    "threshold_sensitivity.csv",
    "filter_sets.csv",
    "pareto_frontier.csv",
    "per_finger.csv",
    "finger_confusion.csv",
    "per_recording.csv",
    "error_pattern_statistics.csv",
    "pig_validation.csv",
    "pig_anomalies.csv",
    "physical_pattern_ablation.csv",
    "deployment_summary.csv",
    "all_results.parquet",
)
```

Update report tests to assert these exact files and remove the old 189-queue
universe contract. Add reconciliation assertions:

```python
assert per_finger.query("view == 'predicted'")["caught_errors"].sum() == pooled_caught
assert per_finger.query("view == 'ground_truth'")["caught_errors"].sum() == pooled_caught
assert ablation.query("component == 'union'")["caught_errors"].iloc[0] == final_caught
```

Build `error_pattern_statistics.csv` by grouping every predeclared candidate
mask over assigned GT rows into `assigned_error` and `assigned_correct`
strata. Each row contains candidate ID, family, stratum denominator, selected
count, selected percentage, risk ratio, and Fisher-exact inputs. Do not infer
causality from these descriptive associations.

Build `deployment_summary.csv` from `deployment_masks_full`. Mark every row
`evaluation_scope="deployment_refit"` and keep it out of the held-out Pareto
table.

- [ ] **Step 6: Run tests and commit Task 7**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_metrics.py \
  tests/fingering_audit/test_report.py -q
```

Expected: PASS.

Commit:

```bash
git add \
  fingering_audit/evaluation/metrics.py \
  fingering_audit/study.py \
  fingering_audit/report.py \
  tests/fingering_audit/test_metrics.py \
  tests/fingering_audit/test_report.py
git commit -m "feat: report recall-workload and finger-stratified audit metrics"
```

---

### Task 8: Fail-Closed Pipeline, PIG Tables, Final Queue, and Vite Categories

**Files:**
- Modify: `fingering_audit/pipeline.py`
- Modify: `fingering_audit/report.py`
- Modify: `fingering_audit/manifest.py`
- Modify: `annotate/prepare_review_data.py`
- Modify: `annotate/src/auditCategories.js`
- Modify: `annotate/test/auditCategories.test.js`
- Modify: `tests/fingering_audit/test_report.py`
- Modify: `tests/fingering_audit/test_manifest.py`

**Interfaces:**
- Produces: `results/audit_queue.tsv` with one row per selected assigned note.
- Produces queue fields: `physical_must_alert`, `physical_reasons`, `pattern_risk_alert`, `pattern_reasons`, `audit_reasons`, `selected_set_id`.
- Produces: `pig_validation.csv` and `pig_anomalies.csv`.
- Guarantees: a production-success marker exists only when PIG, denominators, fold isolation, tables, and queue reconcile.
- Vite consumes the queue category fields without recomputing research rules.

- [ ] **Step 1: Write fail-closed pipeline and Vite category tests**

Add Python tests for:

- incomplete PIG cannot write `SUCCESS.json`;
- unknown PIG anomaly cannot write `SUCCESS.json`;
- a physical PIG violation emits tables but closes the gate;
- final queue contains assigned notes only;
- queue row count equals the deployment union `assigned_hard_count`;
- every queue reason is present in the corresponding mask; and
- all required files are manifest-hashed.

Add Node test:

```javascript
test('pattern category is explicit and below physical priority', () => {
  const pattern = explicitAuditPriority({
    pattern_risk_alert: true,
    pattern_reasons: ['noinfo_run_k3_r1'],
  });
  const physical = explicitAuditPriority({
    physical_must_alert: true,
    physical_reasons: ['same_finger_simultaneous_keys'],
  });
  assert.equal(auditCategoryForNote({
    pattern_risk_alert: true,
    pattern_reasons: ['noinfo_run_k3_r1'],
  }).id, 'pattern');
  assert.ok(physical.score > pattern.score);
});
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_report.py \
  tests/fingering_audit/test_manifest.py -q
npm --prefix annotate run test:audit-categories
```

Expected: Python failures for new artifacts and Node failure for the missing
pattern category.

- [ ] **Step 3: Persist PIG inventory, validation, and anomalies**

After complete PIG loading, write:

```python
pig_validation = pd.DataFrame(
    [
        {
            "rule_id": rule_id,
            "status": validation.status,
            "violation_count": validation.violation_count,
            "violating_ids": "|".join(validation.violating_ids),
        }
        for rule_id, validation in physical_policy.validations.items()
    ]
)
pig_anomalies = pig_notes.loc[
    pig_notes["finger_token_anomaly"].notna(),
    [
        "pig_note_id",
        "source_line",
        "finger_token_raw",
        "finger_token_anomaly",
    ],
]
pig_anomalies["source_relpath"] = pig_notes.loc[
    pig_notes["finger_token_anomaly"].notna(), "source_path"
].map(lambda value: Path(value).relative_to(pig_root).as_posix())
```

Reconcile anomaly count exactly to one and require its token and location to
match the approved `4_` policy. An additional anomaly closes the gate.
Never write an absolute local filesystem path into committed result tables.

- [ ] **Step 4: Export the final assigned-note queue**

Use `study.deployment_masks_full["union"]`. Its threshold identities were
chosen from frozen OOF clear-improvement decisions in Task 6, then refit once
on all recordings. Do not export the fold-varying OOF mask as the deployment
queue.

Construct rows from canonical notes where the final assigned mask is true:

```python
queue = study.notes.loc[final_mask].copy()
queue["physical_must_alert"] = physical.loc[final_mask].to_numpy()
queue["physical_reasons"] = physical_reasons.loc[final_mask].map(
    lambda value: ",".join(value)
).to_numpy()
queue["pattern_risk_alert"] = pattern.loc[final_mask].to_numpy()
queue["pattern_reasons"] = pattern_reasons.loc[final_mask].map(
    lambda value: ",".join(value)
).to_numpy()
queue["audit_reasons"] = queue.apply(
    lambda row: ",".join(
        value
        for value in (row["physical_reasons"], row["pattern_reasons"])
        if value
    ),
    axis=1,
)
queue["selected_set_id"] = final_set_id
queue.to_csv(results_dir / "audit_queue.tsv", sep="\t", index=False)
```

No integrity or `NoInfo` row may enter this TSV.
Append `audit_queue.tsv` to `REQUIRED_RESULTS` in this task and verify its hash
through the manifest.

- [ ] **Step 5: Update Vite serialization and category priority**

Add pattern fields to `audit_fields()`:

```python
"pattern_risk_alert": bool(getattr(row, "pattern_risk_alert", False)),
"pattern_reasons": list(getattr(row, "pattern_reasons", ()) or ()),
```

Add the category after integrity and before legacy fallback:

```javascript
{
  id: 'pattern',
  flag: 'pattern_risk_alert',
  reasons: 'pattern_reasons',
  label: 'Pattern-risk audit',
  reasonPrefix: 'pattern risk',
  priority: 98,
},
```

Keep physical priority 110 and integrity priority 105. Remove the specialized
`noinfo_context` category only after pattern reasons fully subsume it; retain
backward-compatible reading of old note bundles as a lower-priority legacy
category.

- [ ] **Step 6: Replace pipeline reconciliations**

Remove the old exact 189-set reconciliation. Require:

```python
{
    "pig_inventory_complete": inventory_counts_match,
    "pig_anomaly_policy_exact": anomaly_policy_exact,
    "physical_pig_zero_violations": physical_gate_open,
    "authoritative_denominators": denominators_match,
    "folds_cover_each_recording_once": fold_coverage_exact,
    "full_gt_selection_parity": reconcile_full_gt_selection_parity(study),
    "queue_count_reconciles": len(queue) == deployment_union_hard_count,
    "queue_is_assigned_only": bool(queue["pred_finger_id"].notna().all()),
    "per_finger_reconciles": finger_totals_match,
    "ablation_reconciles": ablation_union_matches,
    "required_results_exist": required_results_exist,
}
```

The manifest may finalize success only when every value is true and the
physical gate is open.

- [ ] **Step 7: Run Python, Node, and Vite verification**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit/test_report.py \
  tests/fingering_audit/test_manifest.py -q
npm --prefix annotate run test:audit-categories
npm --prefix annotate run build
```

Expected: PASS.

- [ ] **Step 8: Commit Task 8**

```bash
git add \
  fingering_audit/pipeline.py \
  fingering_audit/report.py \
  fingering_audit/manifest.py \
  annotate/prepare_review_data.py \
  annotate/src/auditCategories.js \
  annotate/test/auditCategories.test.js \
  tests/fingering_audit/test_report.py \
  tests/fingering_audit/test_manifest.py
git commit -m "feat: export fail-closed physical-pattern audit queue"
```

---

### Task 9: Authoritative Run, Consolidated Documentation, Verification, and Push

**Files:**
- Modify: `docs/fingering-audit-complete.md`
- Modify: `docs/fingering-audit-lab-meeting-summary.md`
- Replace aggregate tables under: `docs/fingering-audit-results/`
- Create from run output: `docs/fingering-audit-results/filter_methods.csv`
- Create from run output: `docs/fingering-audit-results/threshold_sensitivity.csv`
- Create from run output: `docs/fingering-audit-results/filter_sets.csv`
- Create from run output: `docs/fingering-audit-results/pareto_frontier.csv`
- Create from run output: `docs/fingering-audit-results/per_finger.csv`
- Create from run output: `docs/fingering-audit-results/finger_confusion.csv`
- Create from run output: `docs/fingering-audit-results/per_recording.csv`
- Create from run output: `docs/fingering-audit-results/error_pattern_statistics.csv`
- Create from run output: `docs/fingering-audit-results/pig_validation.csv`
- Create from run output: `docs/fingering-audit-results/pig_anomalies.csv`
- Create from run output: `docs/fingering-audit-results/physical_pattern_ablation.csv`
- Create from run output: `docs/fingering-audit-results/deployment_summary.csv`

**Interfaces:**
- Consumes: the complete implementation and local PIG dataset.
- Produces: one verified authoritative run directory, copied aggregate tables, updated consolidated documentation, updated lab summary, and the final GitHub branch.
- Does not commit: PIG source files, the full raw queue if publication policy forbids note-level predictions, audio, video, MIDI, or local artifacts.

- [ ] **Step 1: Run the complete automated test suite**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit -q
npm --prefix annotate run test:audit-categories
npm --prefix annotate run build
```

Expected: all Python tests PASS, all Node tests PASS, and Vite build exits 0.

- [ ] **Step 2: Run preflight with complete local PIG**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m fingering_audit preflight \
  --config fingering_audit/config/research.yaml
```

Expected JSON includes 11 recordings and a discoverable complete PIG root.
Stop if PIG counts, timing sources, or ground truth are incomplete.

- [ ] **Step 3: Execute the authoritative research**

Run:

```bash
PIANOVAM_AUDIT_PYTHON=/home/junhyungp/autofinger/.venv/bin/python \
  ./run_fingering_audit.sh \
  --run-label physical-pattern-pig-v102
```

Capture the printed run directory as `RUN_DIR`. Do not choose a result by
proximity to 30,000 notes; use the defensive fallback decision recorded by the
pipeline.

- [ ] **Step 4: Verify the exact run**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m fingering_audit report \
  --config fingering_audit/config/research.yaml \
  --run-dir "$RUN_DIR" \
  --verify-only
```

Expected:

```json
{
  "verification_status": "PASS"
}
```

The result may contain additional fields, but `verification_status` must be
`PASS`.

- [ ] **Step 5: Independently reconcile key numbers**

Run a read-only script that asserts:

```python
assert assigned_error_count == 117
assert noinfo_gt_count == 275
assert final_queue_rows == deployment_union_assigned_hard_count
assert predicted_finger_caught_sum == final_caught_errors
assert ground_truth_finger_caught_sum == final_caught_errors
assert ablation_union_caught == final_caught_errors
assert pig_validation["violation_count"].sum() == 0
assert len(pig_anomalies) == 1
assert pig_anomalies.iloc[0]["finger_token_raw"] == "4_"
```

If any assertion fails, return to the task that owns the mismatch; do not edit
CSV totals manually.

- [ ] **Step 6: Copy aggregate result tables into documentation**

Copy only the approved aggregate CSVs from `$RUN_DIR/results/` to
`docs/fingering-audit-results/`. Do not copy PIG canonical data or the complete
note-level queue into Git.

Use a deterministic copy list:

```bash
for name in \
  filter_methods.csv \
  threshold_sensitivity.csv \
  filter_sets.csv \
  pareto_frontier.csv \
  per_finger.csv \
  finger_confusion.csv \
  per_recording.csv \
  error_pattern_statistics.csv \
  pig_validation.csv \
  pig_anomalies.csv \
  physical_pattern_ablation.csv \
  deployment_summary.csv
do
  cp "$RUN_DIR/results/$name" "docs/fingering-audit-results/$name"
done
```

- [ ] **Step 7: Rewrite the consolidated and lab-meeting documents from verified tables**

Update `docs/fingering-audit-complete.md` with:

- exact PIG provenance, completeness, and anomaly policy;
- physical and pattern method definitions;
- defensive anchors and every sensitivity grid;
- co-primary recall/workload table and Pareto frontier;
- selected final set and the reason a learned threshold did or did not replace
  its anchor;
- per-predicted-finger and per-ground-truth-finger statistics;
- physical-only, pattern-only, union, and unique-error ablation;
- final queue count and percentage;
- a methods-to-filter-set mapping; and
- a clearly labeled historical section for superseded HMM and sequential
  ergonomic trials.

Update `docs/fingering-audit-lab-meeting-summary.md` with the main verified
numbers only. State that physical availability is a weak filter only if the
verified incremental physical recall/workload table supports that statement.

- [ ] **Step 8: Run documentation and repository checks**

Run:

```bash
rg -n "T""BD|TO""DO|PLACE""HOLDER" \
  docs/fingering-audit-complete.md \
  docs/fingering-audit-lab-meeting-summary.md \
  docs/fingering-audit-results
git diff --check
git status --short
```

Expected: no placeholder matches, no whitespace errors, and no PIG path staged.

- [ ] **Step 9: Re-run final verification before claiming completion**

Run:

```bash
/home/junhyungp/autofinger/.venv/bin/python -m pytest \
  tests/fingering_audit -q
npm --prefix annotate run test:audit-categories
npm --prefix annotate run build
/home/junhyungp/autofinger/.venv/bin/python -m fingering_audit report \
  --config fingering_audit/config/research.yaml \
  --run-dir "$RUN_DIR" \
  --verify-only
```

Expected: every command passes and the report verifier returns
`verification_status=PASS`.

- [ ] **Step 10: Commit verified code, documentation, and aggregate results**

```bash
git add \
  docs/fingering-audit-complete.md \
  docs/fingering-audit-lab-meeting-summary.md \
  docs/fingering-audit-results
git commit -m "docs: publish physical-pattern audit results"
```

- [ ] **Step 11: Confirm no local PIG data is tracked**

Run:

```bash
git ls-files PIG
git diff --cached --name-only
```

Expected: `git ls-files PIG` prints nothing, and no archive, PIG annotation,
score PDF, audio, video, MIDI, local cache, or artifact directory is staged.

- [ ] **Step 12: Push the verified branch**

Run:

```bash
git push origin 260724-audit
```

Expected: GitHub branch `260724-audit` advances to the verified documentation
commit.

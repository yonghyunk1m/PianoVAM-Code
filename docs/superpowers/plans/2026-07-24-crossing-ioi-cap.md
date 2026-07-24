# Crossing IOI Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent non-thumb and step-crossing rules from selecting transitions whose preceding same-hand inter-onset interval exceeds 1000 ms.

**Architecture:** Preserve `non_thumb_crossing` as the raw ergonomic relation. Apply the inclusive `prev_ioi_ms <= 1000` policy only when constructing audit masks, so long-gap examples remain inspectable but cannot select notes. Rerun the complete authoritative-offset study and regenerate all affected result documentation.

**Tech Stack:** Python, pandas, pytest, Parquet/CSV audit artifacts, Markdown.

## Global Constraints

- The 1000 ms boundary is inclusive.
- Missing IOI values and IOIs above 1000 ms do not enter crossing masks.
- Step crossing inherits the capped crossing mask.
- Every hybrid strategy using crossing inherits the same capped mask.
- The 1000 ms cutoff is a user-specified conservative policy, not a published physical invariant.
- Original native `key_offset` timing and the PIG recommendation gate remain unchanged.

---

### Task 1: Test and implement the crossing eligibility boundary

**Files:**
- Modify: `tests/fingering_audit/test_filters.py`
- Modify: `fingering_audit/study.py`
- Modify: `fingering_audit/evidence/thresholds.yaml`

**Interfaces:**
- Consumes: `features["non_thumb_crossing"]` and `features["prev_ioi_ms"]`.
- Produces: `_crossing_mask(features, max_ioi_ms=1000.0) -> pd.Series`.

- [ ] Add a failing unit test asserting that 0 ms and 1000 ms crossings pass, while 1000.001 ms and missing-IOI crossings fail.
- [ ] Run the focused test and confirm that `_crossing_mask` is absent.
- [ ] Implement `_crossing_mask` and use it in `_rule_masks`.
- [ ] Update evidence metadata to record the inclusive, user-specified exploratory IOI cap.
- [ ] Run the focused test and the complete Python suite.

### Task 2: Rerun and verify the authoritative study

**Files:**
- Generate: `artifacts/fingering_audit/<new-run-id>/`

**Interfaces:**
- Consumes: the pinned official timing cache and 105 source fingering TSVs.
- Produces: a new complete result directory with filter, queue, per-finger, and verification tables.

- [ ] Run `./run_fingering_audit.sh --run-label crossing-ioi-cap`.
- [ ] Verify the new run with `python -m fingering_audit report --run-dir <new-run-dir> --verify-only`.
- [ ] Confirm that `bl_crossing` and `bl_step_crossing` retain their seven and three assigned GT errors while excluding every `prev_ioi_ms > 1000` crossing.
- [ ] Confirm timing provenance remains 105/105 files, 508,621/508,621 rows, zero missing offsets, and zero synthetic offsets.

### Task 3: Refresh presentation documentation and publish

**Files:**
- Modify: `docs/fingering-audit-complete.md`
- Modify: `docs/fingering-audit-lab-meeting-summary.md`
- Replace generated snapshots: `docs/fingering-audit-results/*.csv`

**Interfaces:**
- Consumes: verified tables from Task 2.
- Produces: synchronized documentation and GitHub-viewable CSV snapshots.

- [ ] Replace result snapshots with the verified new-run CSVs.
- [ ] Update crossing definitions, counts, precision, and every affected combined-strategy result.
- [ ] Add the 17 authoritative GT crossing-transition examples and explicitly distinguish the raw relation from the capped audit mask.
- [ ] Check Markdown and staged diffs, commit only scoped files, push `260724-audit`, and verify the remote branch commit.

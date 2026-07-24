# Task 9 implementation report

## RED

The focused preflight/manifest/report suite was run before updating its timing
fixtures. It produced 34 passes and 4 failures: pipeline tests attempted the
new mandatory authoritative timing acquisition without a test timing source.
This confirms the old pipeline path cannot silently run with missing offsets.

## GREEN

Implemented in commit `2e6498e`:

- authoritative timing acquisition and exact native-offset enrichment happen
  before study construction;
- `timing_provenance.csv` and manifest timing provenance are persisted;
- exact timing coverage, row, missing-offset, identity, synthetic-offset, and
  full/GT integrity parity reconciliations are required;
- `build_study` accepts fully enriched canonical notes;
- OOF upper-tail GT masks are intersected with the GT mapping of full
  eligibility, preventing excluded full notes from entering GT queues.

Static verification: `py_compile` and `git diff --check` pass. Migrated
pipeline tests use verified local timing fixtures; no synthetic `onset + 0.5`
fallback was introduced.

## Follow-up

The parity invariant is one-way: a GT-domain mask may be narrower than its
full-corpus projection (for example assigned-only calibrated rules), but it
must never select a note excluded by the corresponding full mask. Report
verification now rejects missing provenance files, byte-count mismatches, and
hash mismatches, and requires every source path to exist.

## Test migration (fixture GREEN)

The four pipeline tests now inject the checked-in native timing fixture through
the supported `ensure_authoritative_timing(..., downloader=...)` interface.
The fixture uses the pinned repository/revision contract and exact native
onset/key/frame offsets; it does not use a synthetic offset fallback or network
access. Test wrappers accept the production `notes=` argument and keep the
artifact directory distinct from the timing cache.

Focused pipeline tests: 4 passed.
Full Python suite: **124 passed** (49.48 s; 27 expected pandas warnings).

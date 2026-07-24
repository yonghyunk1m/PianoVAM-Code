from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fingering_audit.acquire import (
    PINNED_TIMING_REPOSITORY,
    PINNED_TIMING_REVISION,
    ensure_authoritative_timing,
)
from fingering_audit.config import load_config
from fingering_audit.timing import attach_authoritative_offsets


FIXTURES = Path(__file__).parent / "fixtures"
HEADER = "# onset\tkey_offset\tframe_offset\tnote\tvelocity\n"


def _config(tmp_path: Path):
    return replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        timing_cache_dir=tmp_path / "timing-cache",
    )


def _downloader(files: dict[str, str], calls: list[str]):
    def download(url: str, destination: Path) -> None:
        calls.append(url)
        recording_id = Path(url).stem
        if recording_id not in files:
            raise OSError(f"unavailable: {recording_id}")
        destination.write_text(files[recording_id], encoding="utf-8")

    return download


def _source(tmp_path: Path, native: str, recording_id: str = "take"):
    calls: list[str] = []
    source = ensure_authoritative_timing(
        _config(tmp_path),
        [recording_id],
        downloader=_downloader({recording_id: native}, calls),
    )
    return source, calls


def _notes(rows: list[dict], recording_id: str = "take") -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "recording_id": recording_id,
                "note_idx": index,
                "note_id": f"{recording_id}#{index}",
                "onset_sec": row["onset"],
                "offset_sec": float("nan"),
                "pitch": row["note"],
                "velocity": row["velocity"],
                "source_path": f"/immutable/{recording_id}.tsv",
                "source_sha256": "f" * 64,
            }
            for index, row in enumerate(rows)
        ]
    )


def test_acquisition_uses_exact_pinned_repository_revision_and_recording(tmp_path):
    calls: list[str] = []
    source = ensure_authoritative_timing(
        _config(tmp_path),
        ["take"],
        downloader=_downloader(
            {"take": HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n"},
            calls,
        ),
    )

    assert len(calls) == 1
    assert (
        calls[0]
        == "https://huggingface.co/datasets/"
        f"{PINNED_TIMING_REPOSITORY}/resolve/"
        f"{PINNED_TIMING_REVISION}/TSV/take.tsv"
    )
    assert source.recording_ids == ("take",)
    assert source.complete
    assert source.provenance["validation_status"].tolist() == [
        "acquisition_valid"
    ]


def test_acquisition_rejects_non_pinned_revision(tmp_path):
    config = replace(_config(tmp_path), timing_revision="a" * 40)

    with pytest.raises(ValueError, match="pinned revision"):
        ensure_authoritative_timing(config, ["take"], downloader=lambda *_: None)


def test_acquisition_rejects_non_40_hex_revision(tmp_path):
    config = replace(_config(tmp_path), timing_revision="main")

    with pytest.raises(ValueError, match="40-character hexadecimal"):
        ensure_authoritative_timing(config, ["take"], downloader=lambda *_: None)


def test_acquisition_rejects_duplicate_recording_requests(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        ensure_authoritative_timing(
            _config(tmp_path), ["take", "take"], downloader=lambda *_: None
        )


def test_acquisition_fails_closed_on_partial_download(tmp_path):
    calls: list[str] = []
    downloader = _downloader(
        {"one": HEADER + "0.000000\t0.100000\t0.200000\t60\t80\n"},
        calls,
    )

    with pytest.raises(RuntimeError, match="two"):
        ensure_authoritative_timing(
            _config(tmp_path), ["one", "two"], downloader=downloader
        )

    assert any(url.endswith("/TSV/two.tsv") for url in calls)


def test_acquisition_reuses_valid_cache_after_rehashing(tmp_path):
    native = HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n"
    source, calls = _source(tmp_path, native)
    first_hash = source.provenance.loc[0, "sha256"]

    def network_forbidden(*_args):
        raise AssertionError("valid cache must not use the network")

    reused = ensure_authoritative_timing(
        _config(tmp_path), ["take"], downloader=network_forbidden
    )

    assert len(calls) == 1
    assert reused.provenance.loc[0, "sha256"] == first_hash
    assert reused.provenance.loc[0, "byte_count"] == len(native.encode())


def test_acquisition_rejects_changed_cached_bytes(tmp_path):
    native = HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n"
    source, _ = _source(tmp_path, native)
    cached = source.cache_dir / "TSV" / "take.tsv"
    cached.write_text(
        HEADER + "1.000000\t1.250000\t1.300000\t60\t80\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        ensure_authoritative_timing(
            _config(tmp_path),
            ["take"],
            downloader=lambda *_: pytest.fail("must not redownload silently"),
        )


def test_exact_join_uses_native_key_offset_without_half_second_fallback(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n",
    )
    notes = _notes([{"onset": 1.0, "note": 60, "velocity": 80}])

    joined = attach_authoritative_offsets(notes, source)

    assert joined.complete
    assert joined.notes["offset_sec"].tolist() == [1.125]
    assert joined.notes.loc[0, "offset_sec"] != pytest.approx(1.5)
    assert joined.provenance.loc[0, "validation_status"] == "exact_join_valid"


def test_exact_join_rejects_row_count_mismatch(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER
        + "1.000000\t1.125000\t1.200000\t60\t80\n"
        + "2.000000\t2.125000\t2.200000\t62\t81\n",
    )
    notes = _notes([{"onset": 1.0, "note": 60, "velocity": 80}])

    with pytest.raises(ValueError, match="row count"):
        attach_authoritative_offsets(notes, source)


def test_exact_join_rejects_duplicate_fingering_identity(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER
        + "1.000000\t1.125000\t1.200000\t60\t80\n"
        + "2.000000\t2.125000\t2.200000\t62\t81\n",
    )
    notes = _notes(
        [
            {"onset": 1.0, "note": 60, "velocity": 80},
            {"onset": 1.0, "note": 60, "velocity": 80},
        ]
    )

    with pytest.raises(ValueError, match="duplicate.*fingering"):
        attach_authoritative_offsets(notes, source)


def test_exact_join_rejects_duplicate_native_identity(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER
        + "1.000000\t1.125000\t1.200000\t60\t80\n"
        + "1.000000\t1.225000\t1.300000\t60\t80\n",
    )
    notes = _notes(
        [
            {"onset": 1.0, "note": 60, "velocity": 80},
            {"onset": 2.0, "note": 62, "velocity": 81},
        ]
    )

    with pytest.raises(ValueError, match="duplicate.*native"):
        attach_authoritative_offsets(notes, source)


@pytest.mark.parametrize(
    ("native_row", "match"),
    [
        ("1.000001\t1.125000\t1.200000\t60\t80", "identity"),
        ("1.000000\t1.125000\t1.200000\t61\t80", "identity"),
        ("1.000000\t1.125000\t1.200000\t60\t81", "velocity"),
        ("nan\t1.125000\t1.200000\t60\t80", "nonfinite"),
        ("1.000000\tinf\t1.200000\t60\t80", "nonfinite"),
        ("1.000000\t0.999999\t1.200000\t60\t80", "before onset"),
    ],
)
def test_exact_join_rejects_timing_mismatches(tmp_path, native_row, match):
    source, _ = _source(tmp_path, HEADER + native_row + "\n")
    notes = _notes([{"onset": 1.0, "note": 60, "velocity": 80}])

    with pytest.raises(ValueError, match=match):
        attach_authoritative_offsets(notes, source)


def test_exact_join_rejects_recording_coverage_mismatch(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n",
    )
    notes = _notes(
        [{"onset": 1.0, "note": 60, "velocity": 80}],
        recording_id="different",
    )

    with pytest.raises(ValueError, match="recording coverage"):
        attach_authoritative_offsets(notes, source)


@pytest.mark.parametrize(
    ("source_change", "provenance_column", "provenance_value"),
    [
        (
            {"repository_id": "Caller/Relabeled"},
            "repository_id",
            "Caller/Relabeled",
        ),
        (
            {"revision": "a" * 40},
            "revision",
            "a" * 40,
        ),
    ],
)
def test_exact_join_independently_rejects_relabeled_source_identity(
    tmp_path, source_change, provenance_column, provenance_value
):
    source, _ = _source(
        tmp_path,
        HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n",
    )
    provenance = source.provenance.copy()
    provenance[provenance_column] = provenance_value
    forged = replace(source, provenance=provenance, **source_change)
    notes = _notes([{"onset": 1.0, "note": 60, "velocity": 80}])

    with pytest.raises(ValueError, match="official pinned"):
        attach_authoritative_offsets(notes, forged)


def test_exact_join_rejects_absolute_or_misplaced_source_paths(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER + "1.000000\t1.125000\t1.200000\t60\t80\n",
    )
    notes = _notes([{"onset": 1.0, "note": 60, "velocity": 80}])
    original = source.cache_dir / "TSV" / "take.tsv"
    misplaced = source.cache_dir / "other" / "take.tsv"
    misplaced.parent.mkdir()
    misplaced.write_bytes(original.read_bytes())

    for replacement_path in (str(original), "other/take.tsv"):
        provenance = source.provenance.copy()
        provenance["relative_source_path"] = replacement_path
        forged = replace(source, provenance=provenance)
        with pytest.raises(ValueError, match="exact relative source path"):
            attach_authoritative_offsets(notes, forged)


def test_exact_join_preserves_arbitrary_dataframe_index(tmp_path):
    source, _ = _source(
        tmp_path,
        HEADER
        + "1.000000\t1.125000\t1.200000\t60\t80\n"
        + "2.000000\t2.250000\t2.300000\t62\t81\n",
    )
    notes = _notes(
        [
            {"onset": 1.0, "note": 60, "velocity": 80},
            {"onset": 2.0, "note": 62, "velocity": 81},
        ]
    )
    notes.index = pd.Index(["first-note", "second-note"], name="source_label")

    joined = attach_authoritative_offsets(notes, source)

    assert joined.notes.index.tolist() == ["first-note", "second-note"]
    assert joined.notes["offset_sec"].tolist() == [1.125, 2.25]

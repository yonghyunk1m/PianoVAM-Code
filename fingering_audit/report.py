from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from .contracts import AuditConfig
from .study import NOINFO_CALIBRATED_VARIANTS, NOINFO_VARIANTS


REQUIRED_RESULTS = (
    "filter_sets.csv",
    "individual_filters.csv",
    "per_finger.csv",
    "workload_per_finger.csv",
    "per_recording.csv",
    "error_types.csv",
    "overlap_matrix.csv",
    "threshold_sensitivity.csv",
    "excluded_rules.csv",
    "pareto_tiers.csv",
    "all_results.parquet",
    "noinfo_sensitivity.csv",
    "queue_summary.csv",
    "queue_workload_per_finger.csv",
)

APPROVED_BASE_RISK_IDS = (
    "mandatory_missing",
    "legacy_current_default",
    "bl_span_practical",
    "bl_span_comfortable",
    "bl_span_relative",
    "bl_crossing",
    "bl_step_crossing",
    "bl_rate_q995",
    "bl_rate_q990",
    "bl_rate_q975",
    "bl_hmm_disagreement",
    "bl_practical_or_rate995",
    "bl_practical_or_crossing",
    "bl_two_signal_strict",
    "wl_model_agreement",
    "wl_strict_obvious",
    "hy_direct_plus_corroborated",
    "hy_two_of_three_families",
    "hy_hierarchical",
)
EXPECTED_COMBINED_QUEUE_IDS = frozenset(
    f"{base_id}__{variant}"
    for base_id in APPROVED_BASE_RISK_IDS
    for variant in NOINFO_VARIANTS
)
EXPECTED_STANDALONE_QUEUE_IDS = frozenset(
    (*NOINFO_VARIANTS, *NOINFO_CALIBRATED_VARIANTS)
)
EXPECTED_QUEUE_SELECTION_IDS = (
    EXPECTED_COMBINED_QUEUE_IDS | EXPECTED_STANDALONE_QUEUE_IDS
)

QUEUE_REPORT_COLUMNS = [
    "base_risk_method",
    "physical_policy_status",
    "noinfo_min_run",
    "noinfo_context_radius",
    "hard_count",
    "hard_percentage_all_notes",
    "gt_error_recall",
    "assigned_gt_error_recall",
    "gt_precision",
    "error_enrichment",
    "incremental_count_beyond_physical",
    "incremental_errors_beyond_physical",
]


def _stable_variant_metadata(sensitivity: pd.DataFrame) -> pd.DataFrame:
    renamed = sensitivity.rename(
        columns={
            "variant": "noinfo_variant",
            "calibration": "noinfo_calibration",
            "min_run": "noinfo_min_run",
            "radius": "noinfo_context_radius",
            "window": "noinfo_window",
            "quantile": "noinfo_quantile",
            "incremental_count_beyond_physical": (
                "standalone_incremental_count_beyond_physical"
            ),
            "incremental_errors_beyond_physical": (
                "standalone_incremental_errors_beyond_physical"
            ),
        }
    )
    stable_columns = [
        "noinfo_calibration",
        "noinfo_min_run",
        "noinfo_context_radius",
        "noinfo_window",
        "noinfo_quantile",
        "evidence_grade",
        "method_identity",
        "threshold_definition",
        "standalone_incremental_count_beyond_physical",
        "standalone_incremental_errors_beyond_physical",
    ]
    stable_columns = [
        column for column in stable_columns if column in renamed
    ]
    for required in ("noinfo_window", "noinfo_quantile"):
        if required not in stable_columns:
            renamed[required] = pd.NA
            stable_columns.append(required)

    records = []
    for variant, group in renamed.groupby(
        "noinfo_variant", sort=False, dropna=False
    ):
        record = {"noinfo_variant": variant}
        for column in stable_columns:
            values = group[column]
            nonmissing = values.loc[values.notna()].drop_duplicates()
            mixes_missing_and_value = values.isna().any() and len(nonmissing)
            if len(nonmissing) > 1 or mixes_missing_and_value:
                raise ValueError(
                    f"{variant}: inconsistent stable field {column}"
                )
            record[column] = (
                nonmissing.iloc[0] if len(nonmissing) else pd.NA
            )
        records.append(record)
    metadata = pd.DataFrame.from_records(records)
    if not metadata["noinfo_variant"].is_unique:
        raise ValueError("noinfo variant metadata must be unique")
    return metadata


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].copy()
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in shown.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _queue_tables(
    tables: Mapping[str, pd.DataFrame],
    physical_policy_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filter_sets = tables["filter_sets"]
    sensitivity = tables["noinfo_sensitivity"].copy()
    variants = set(sensitivity["variant"].dropna().astype(str))
    variant_metadata = _stable_variant_metadata(sensitivity)

    def variant_for(set_id: str) -> str | None:
        candidate = set_id.split("__", 1)[1] if "__" in set_id else set_id
        return candidate if candidate in variants else None

    queue = filter_sets.copy()
    queue["noinfo_variant"] = queue["set_id"].astype(str).map(variant_for)
    queue = queue.loc[queue["noinfo_variant"].notna()].copy()
    queue["base_risk_method"] = queue["set_id"].astype(str).map(
        lambda set_id: set_id.split("__", 1)[0] if "__" in set_id else pd.NA
    )
    queue["physical_policy_status"] = physical_policy_status

    sensitivity_columns = [
        "noinfo_variant",
        "noinfo_calibration",
        "noinfo_min_run",
        "noinfo_context_radius",
        "noinfo_window",
        "noinfo_quantile",
        "standalone_incremental_count_beyond_physical",
        "standalone_incremental_errors_beyond_physical",
    ]
    queue = queue.merge(
        variant_metadata[sensitivity_columns],
        on="noinfo_variant",
        how="left",
        validate="many_to_one",
        suffixes=("", "_standalone"),
    )

    standalone = filter_sets.loc[
        filter_sets["set_id"].isin(variants),
        ["set_id", "hard_count", "gt_selected_errors"],
    ].rename(
        columns={
            "set_id": "noinfo_variant",
            "hard_count": "standalone_hard_count",
            "gt_selected_errors": "standalone_selected_errors",
        }
    )
    queue = queue.merge(
        standalone,
        on="noinfo_variant",
        how="left",
        validate="many_to_one",
    )
    physical_count = (
        queue["standalone_hard_count"]
        - queue["standalone_incremental_count_beyond_physical"]
    )
    physical_errors = (
        queue["standalone_selected_errors"]
        - queue["standalone_incremental_errors_beyond_physical"]
    )
    queue["incremental_count_beyond_physical"] = (
        queue["hard_count"] - physical_count
    ).astype(int)
    queue["incremental_errors_beyond_physical"] = (
        queue["gt_selected_errors"] - physical_errors
    ).astype(int)
    queue["error_enrichment"] = queue["gt_enrichment"]

    identity_columns = [
        "set_id",
        "noinfo_variant",
        "noinfo_calibration",
        "noinfo_window",
        "noinfo_quantile",
        "gt_hard_count",
    ]
    queue = queue[identity_columns + QUEUE_REPORT_COLUMNS]
    queue_ids = set(queue["set_id"])
    workload = tables["workload_per_finger"]
    workload = workload.loc[workload["set_id"].isin(queue_ids)].copy()
    return queue.reset_index(drop=True), workload.reset_index(drop=True)


def write_reports(
    run_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    *,
    corpus_notes: int,
    assigned_notes: int,
    missing_notes: int,
    pig_status: str,
) -> tuple[Path, ...]:
    results_dir = run_dir / "results"
    report_dir = run_dir / "report"
    figures_dir = report_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    filter_sets = tables["filter_sets"].copy()
    for name, frame in tables.items():
        if name == "overlap_matrix":
            frame.to_csv(results_dir / f"{name}.csv")
        else:
            frame.to_csv(results_dir / f"{name}.csv", index=False)
    queue_summary, queue_workload = _queue_tables(tables, pig_status)
    queue_summary.to_csv(results_dir / "queue_summary.csv", index=False)
    queue_workload.to_csv(
        results_dir / "queue_workload_per_finger.csv", index=False
    )
    individual = filter_sets[
        filter_sets["set_id"].isin(
            [
                "mandatory_missing",
                "bl_span_practical",
                "bl_span_comfortable",
                "bl_span_relative",
                "bl_crossing",
                "bl_step_crossing",
                "bl_rate_q995",
                "bl_rate_q990",
                "bl_rate_q975",
                "bl_hmm_disagreement",
            ]
        )
    ]
    individual.to_csv(results_dir / "individual_filters.csv", index=False)
    excluded = pd.DataFrame(
        [
            {
                "rule_family": "PIG validity/support",
                "status": "unavailable",
                "reason": "authoritative PIG v1.02 annotations not present locally",
            },
            {
                "rule_family": "HaMeR note trajectories/candidate margins",
                "status": "unavailable",
                "reason": "only calibration/pixel-point files found; no note-level candidates",
            },
            {
                "rule_family": "MediaPipe temporal confidence",
                "status": "unavailable",
                "reason": "source TSV has labels but no per-frame confidence table",
            },
        ]
    )
    excluded.to_csv(results_dir / "excluded_rules.csv", index=False)
    pareto = filter_sets[
        [
            "strategy",
            "set_id",
            "hard_count",
            "hard_percentage_all_notes",
            "gt_error_recall",
            "gt_precision",
            "pig_status",
            "recommendable",
        ]
    ].copy()
    pareto["pareto_status"] = "not_assigned_by_pipeline"
    pareto.to_csv(results_dir / "pareto_tiers.csv", index=False)
    filter_sets.to_parquet(results_dir / "all_results.parquet", index=False)

    plot = filter_sets[filter_sets["strategy"].ne("integrity")]
    fig, axis = plt.subplots(figsize=(9, 6))
    for strategy, group in plot.groupby("strategy"):
        axis.scatter(
            group["hard_count"],
            group["gt_error_recall"],
            label=strategy,
            alpha=0.8,
        )
    axis.set_xlabel("Selected hard notes")
    axis.set_ylabel("GT exact-error recall (all 1,800 labels)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "workload_vs_recall.png", dpi=160)
    plt.close(fig)

    key_columns = [
        "strategy",
        "set_id",
        "hard_count",
        "hard_percentage_all_notes",
        "gt_error_recall",
        "gt_precision",
        "assigned_gt_error_recall",
        "macro_finger_recall",
        "worst_finger",
        "pig_status",
    ]
    queue_columns = ["set_id", "noinfo_variant", *QUEUE_REPORT_COLUMNS]
    report = f"""# PianoVAM Fingering-Audit Research Report

## Status

Physical-policy status: **{pig_status}**. Candidate sets are compared below,
but this report does not select a recommendation or tune a threshold to a
target workload.

## Audit universes

- Full PianoVAM corpus: {corpus_notes:,} notes.
- Assigned hand/finger labels: {assigned_notes:,} notes.
- Missing hand/finger labels: {missing_notes:,} notes ({missing_notes / corpus_notes:.2%}).

Missing labels are a separate mandatory-repair queue. They are not mixed into
the approximately 30,000-note “hard assigned fingering” workload.

## Filter-set results

{_markdown_table(filter_sets, key_columns)}

## Assigned-audit queue results

{_markdown_table(queue_summary, queue_columns)}

`queue_summary.csv` contains the same rows and exact overall masks used above.
Predicted-finger workload and true-finger recall remain separately auditable
in `queue_workload_per_finger.csv` and `per_finger.csv`, both keyed by
`set_id`. Incremental columns measure the workload and GT errors beyond the
PIG-authorized `physical_must_alert` queue. `data/queue_masks.parquet`
preserves the physical diagnostic, physical must-alert, integrity, and every
Noinfo mask even when the recommendation gate is closed.

## Interpretation safeguards

- `gt_error_recall` uses all 1,800 authoritative labels, including missing
  predictions; `assigned_gt_error_recall` conditions on an existing predicted
  hand/finger.
- Hard-note percentages are computed before adding any display context.
- The LOPO rate thresholds are fitted without the held-out recording.
- Whitelist rows report the complement of notes satisfying all required safe
  conditions; unavailable evidence is never treated as safe.
- The pipeline does not select a recommendation from these candidate rows.
- Threshold decisions and legacy-rule dispositions are documented separately
  in `docs/fingering-audit-threshold-rationale.md`.
"""
    markdown = report_dir / "research_report.md"
    markdown.write_text(report, encoding="utf-8")
    html = report_dir / "research_report.html"
    html.write_text(
        "<html><body><pre>" + report.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>",
        encoding="utf-8",
    )
    return tuple(
        sorted(
            [
                *(results_dir / name for name in REQUIRED_RESULTS),
                markdown,
                html,
                *(figures_dir.glob("*.png")),
            ]
        )
    )


def verify_report(
    config: AuditConfig,
    *,
    run_dir: Path | None,
    latest_success: bool,
) -> dict:
    if run_dir is None:
        candidates = sorted(config.artifact_root.glob("*"))
        if latest_success:
            candidates = [path for path in candidates if (path / "SUCCESS.json").is_file()]
        if not candidates:
            return {"verification_status": "FAIL", "reason": "no matching run"}
        run_dir = candidates[-1]
    missing = [
        name for name in REQUIRED_RESULTS if not (run_dir / "results" / name).is_file()
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "verification_status": "PASS" if not missing else "FAIL",
        "run_dir": str(run_dir.resolve()),
        "manifest_status": manifest["status"],
        "missing_files": missing,
    }

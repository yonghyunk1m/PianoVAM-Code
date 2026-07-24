from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from .contracts import AuditConfig


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
)


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
    pareto["pareto_status"] = "not_assigned_until_pig_gate"
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
    axis.axvline(30_000, color="black", linestyle="--", linewidth=1, label="30k soft target")
    axis.set_xlabel("Selected hard notes")
    axis.set_ylabel("GT exact-error recall (all 1,800 labels)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "workload_vs_recall.png", dpi=160)
    plt.close(fig)

    nearest = plot.iloc[(plot["hard_count"] - 30_000).abs().argsort()[:1]]
    nearest_id = str(nearest.iloc[0]["set_id"])
    finger = tables["per_finger"]
    finger = finger[(finger["set_id"] == nearest_id) & (finger["scope"] == "all_gt")]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(finger["finger_id"], finger["error_recall"].fillna(0))
    axis.set_ylim(0, 1)
    axis.set_ylabel("GT exact-error recall")
    axis.set_title(f"Per-finger recall: {nearest_id}")
    fig.tight_layout()
    fig.savefig(figures_dir / "per_finger_recall_nearest_30k.png", dpi=160)
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
    report = f"""# PianoVAM Fingering-Audit Research Report

## Status

The computational study completed, but the recommendation gate is closed:
**{pig_status}**. Candidate sets are compared below; none is labeled a final
publication recommendation until the authoritative PIG annotations validate
the applicable fingering rules.

## Audit universes

- Full PianoVAM corpus: {corpus_notes:,} notes.
- Assigned hand/finger labels: {assigned_notes:,} notes.
- Missing hand/finger labels: {missing_notes:,} notes ({missing_notes / corpus_notes:.2%}).

Missing labels are a separate mandatory-repair queue. They are not mixed into
the approximately 30,000-note “hard assigned fingering” workload.

## Filter-set results

{_markdown_table(filter_sets, key_columns)}

## Interpretation safeguards

- `gt_error_recall` uses all 1,800 authoritative labels, including missing
  predictions; `assigned_gt_error_recall` conditions on an existing predicted
  hand/finger.
- Hard-note percentages are computed before adding any display context.
- The LOPO rate thresholds are fitted without the held-out recording.
- Whitelist rows report the complement of notes satisfying all required safe
  conditions; unavailable evidence is never treated as safe.
- Every candidate is `recommendable=false` while the PIG gate is unavailable.
- Threshold decisions and legacy-rule dispositions are documented separately
  in `docs/fingering-audit-threshold-rationale.md`.

## Closest workload to 30,000

The mechanically closest evaluated assigned-note set is `{nearest_id}` with
{int(nearest.iloc[0]["hard_count"]):,} notes. This proximity does not override
its evidence grade, held-out GT behavior, per-finger behavior, or PIG status.
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

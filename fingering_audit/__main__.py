from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .preflight import preflight_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PianoVAM fingering audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("preflight", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--config",
            type=Path,
            default=Path("fingering_audit/config/research.yaml"),
        )
        if command == "run":
            sub.add_argument("--limit-recordings", type=int)
            sub.add_argument("--run-label")

    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", type=Path)
    report.add_argument("--latest-success", action="store_true")
    report.add_argument("--verify-only", action="store_true")
    report.add_argument(
        "--config",
        type=Path,
        default=Path("fingering_audit/config/research.yaml"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        print(json.dumps(preflight_summary(config), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        from .pipeline import run_research

        run_dir = run_research(
            config,
            limit_recordings=args.limit_recordings,
            run_label=args.run_label,
        )
        print(run_dir)
        return 0
    from .report import verify_report

    result = verify_report(
        config,
        run_dir=args.run_dir,
        latest_success=args.latest_success,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verification_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

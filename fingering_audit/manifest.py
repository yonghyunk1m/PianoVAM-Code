from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import AuditConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_key(
    name: str,
    inputs: Mapping[str, str],
    config: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        {"name": name, "inputs": inputs, "config": config},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass
class RunManifest:
    run_dir: Path
    payload: dict[str, Any]

    @classmethod
    def start(cls, config: AuditConfig, run_id: str) -> "RunManifest":
        run_dir = config.artifact_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "random_seed": config.random_seed,
            "dependencies": {"python": platform.python_version()},
            "stages": {},
            "reconciliations": {},
        }
        manifest = cls(run_dir=run_dir, payload=payload)
        manifest._write()
        return manifest

    def _write(self) -> None:
        _atomic_json(self.run_dir / "manifest.json", self.payload)

    def complete_stage(
        self,
        name: str,
        key: str,
        artifact: Path,
    ) -> None:
        self.payload["stages"][name] = {"key": key, "artifact": str(artifact)}
        self._write()

    def finalize(self, reconciliations: Mapping[str, bool]) -> None:
        failed = sorted(name for name, passed in reconciliations.items() if not passed)
        if failed:
            raise ValueError(f"failed reconciliation checks: {failed}")
        self.payload["status"] = "success"
        self.payload["reconciliations"] = dict(reconciliations)
        self._write()
        _atomic_json(
            self.run_dir / "SUCCESS.json",
            {"status": "success", "run_id": self.payload["run_id"]},
        )

    def close_recommendation_gate(
        self, reason: str, reconciliations: Mapping[str, bool]
    ) -> None:
        failed = sorted(name for name, passed in reconciliations.items() if not passed)
        if failed:
            raise ValueError(f"failed reconciliation checks: {failed}")
        self.payload["status"] = "complete_with_recommendation_gate_closed"
        self.payload["recommendation_blocker"] = reason
        self.payload["reconciliations"] = dict(reconciliations)
        self._write()
        _atomic_json(
            self.run_dir / "RECOMMENDATION_GATE_CLOSED.json",
            {
                "status": "complete_with_recommendation_gate_closed",
                "run_id": self.payload["run_id"],
                "reason": reason,
            },
        )

    def fail(self, stage: str, error: str) -> None:
        self.payload["status"] = "failed"
        self.payload["failed_stage"] = stage
        self.payload["error"] = error
        self._write()

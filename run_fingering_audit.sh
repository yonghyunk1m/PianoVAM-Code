#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

audit_python="${PIANOVAM_AUDIT_PYTHON:-}"
if [[ -z "$audit_python" ]]; then
  if python3 -c "import numpy, pandas, yaml, pyarrow" >/dev/null 2>&1; then
    audit_python="python3"
  elif [[ -x "/home/junhyungp/autofinger/.venv/bin/python" ]]; then
    audit_python="/home/junhyungp/autofinger/.venv/bin/python"
  else
    echo "No Python environment with the audit dependencies was found." >&2
    exit 2
  fi
fi

exec "$audit_python" -m fingering_audit run \
  --config fingering_audit/config/research.yaml "$@"

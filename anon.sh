#!/bin/bash
# Annotate review app launcher
# Run from project root: ./anon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ANNOTATE_DIR="$SCRIPT_DIR/annotate"
PREPARE_SCRIPT="$ANNOTATE_DIR/prepare_review_data.py"

print_help() {
    cat <<'EOF'
Usage:
  ./anon.sh
  ./anon.sh --sample
  ./anon.sh --midi /path/to/file.mid --video /path/to/file.mp4 --audio /path/to/file.wav [--tsv /path/to/file.tsv ...]

Behavior:
  - No arguments or --sample:
      prepares annotate/public/data/notes.json from testvideo/sample.*
      and, if present, also uses testvideo/fingering_pp.zip for seed labels
      and launches the annotate dev server on port 3333.
  - With custom arguments:
      passes them through to annotate/prepare_review_data.py, then launches
      the annotate dev server on port 3333.

Examples:
  ./anon.sh
  ./anon.sh --sample
  ./anon.sh \
    --midi /path/to/recording.mid \
    --video /path/to/recording.mp4 \
    --audio /path/to/recording.wav \
    --tsv /path/to/recording.tsv \
    --piece "My Piece" \
    --difficulty hard
EOF
}

ensure_npm_deps() {
    if ! command -v npm >/dev/null 2>&1; then
        echo "Error: npm is not installed. Please install Node.js and npm first."
        exit 1
    fi

    if [[ ! -d "$ANNOTATE_DIR/node_modules" ]]; then
        echo "[annotate] Installing npm dependencies..."
        if [[ -f "$ANNOTATE_DIR/package-lock.json" ]]; then
            (cd "$ANNOTATE_DIR" && npm ci)
        else
            (cd "$ANNOTATE_DIR" && npm install)
        fi
        return
    fi

    if [[ ! -x "$ANNOTATE_DIR/node_modules/.bin/vite" ]]; then
        echo "[annotate] vite is missing, reinstalling npm dependencies..."
        if [[ -f "$ANNOTATE_DIR/package-lock.json" ]]; then
            (cd "$ANNOTATE_DIR" && npm ci)
        else
            (cd "$ANNOTATE_DIR" && npm install)
        fi
    fi
}

prepare_sample() {
    local args=(
        --midi "$SCRIPT_DIR/testvideo/sample.mid"
        --video "$SCRIPT_DIR/testvideo/sample.mp4"
        --audio "$SCRIPT_DIR/testvideo/sample.wav"
    )

    if [[ -f "$SCRIPT_DIR/testvideo/fingering_pp.zip" ]]; then
        args+=(
            --fingering-zip "$SCRIPT_DIR/testvideo/fingering_pp.zip"
            --trial "2024-02-14_19-44-26"
            --piece "2024-02-14_19-44-26"
        )
    else
        args+=(
            --trial sample
            --piece sample
        )
    fi

    python "$PREPARE_SCRIPT" "${args[@]}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

if [[ $# -eq 0 || "${1:-}" == "--sample" ]]; then
    echo "[annotate] Preparing sample review bundle..."
    prepare_sample
else
    echo "[annotate] Preparing review bundle from custom inputs..."
    python "$PREPARE_SCRIPT" "$@"
fi

ensure_npm_deps

export CHOKIDAR_USEPOLLING=1
export CHOKIDAR_INTERVAL=1000

echo "[annotate] Launching dev server at http://localhost:3333"
cd "$ANNOTATE_DIR"
exec npm run dev -- --host 0.0.0.0 --port 3333 --strictPort

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT_DIR/docs/submission/technical_synopsis.md"
OUT="$ROOT_DIR/docs/submission/technical_synopsis.pdf"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "[error] pandoc is not installed."
  echo "Install: sudo apt install pandoc"
  echo "Then rerun this script."
  exit 1
fi

pandoc "$SRC" -o "$OUT" --from gfm --pdf-engine=xelatex

echo "[ok] Exported PDF: $OUT"

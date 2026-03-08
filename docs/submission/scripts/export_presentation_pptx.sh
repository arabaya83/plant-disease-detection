#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT_DIR/docs/submission/presentation/management_presentation_content.md"
OUT="$ROOT_DIR/docs/submission/presentation/management_presentation.pptx"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "[error] pandoc is not installed."
  echo "Install: sudo apt install pandoc"
  echo "Then rerun this script."
  exit 1
fi

pandoc "$SRC" -o "$OUT" --from gfm --to pptx

echo "[ok] Exported PPTX: $OUT"
echo "[next] Open and apply final visual design tweaks before submission."

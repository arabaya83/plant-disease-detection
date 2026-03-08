#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BOOK_DIR="$ROOT_DIR/docs/submission/jupyter_book"

if ! command -v jupyter-book >/dev/null 2>&1; then
  echo "[info] jupyter-book not found. Installing into current Python environment..."
  python -m pip install jupyter-book
fi

echo "[run] jupyter-book build $BOOK_DIR"
jupyter-book build "$BOOK_DIR"

echo "[ok] Built site at: $BOOK_DIR/_build/html/index.html"
echo "[next] Publish _build/html via GitHub Pages or your preferred hosting."

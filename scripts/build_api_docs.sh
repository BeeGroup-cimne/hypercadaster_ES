#!/usr/bin/env bash
#
# Build the HTML API reference from the package docstrings with pdoc.
# Output lands in docs/api/ (git-ignored). Install the tool first:
#
#     pip install pdoc
#
# then run from the repository root:
#
#     ./scripts/build_api_docs.sh
#     # open docs/api/index.html
#
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="docs/api"

# pdoc imports the package; one dependency probes the CPU via a subprocess,
# so allow it explicitly (otherwise pdoc aborts the import).
export PDOC_ALLOW_EXEC=1

# Clear stale output first — pdoc doesn't prune pages for modules that are no
# longer documented.
rm -rf "$OUT"

# Docstrings are NumPy-style; the submodule list/order comes from __all__ in
# the package __init__.py (downloaders, energy, interoperability, mergers,
# utils, weather).
python -m pdoc \
    --docformat numpy \
    hypercadaster_ES \
    -o "$OUT"

echo "API docs written to $OUT/index.html"

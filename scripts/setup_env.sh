#!/usr/bin/env bash
# Documentation-only. Read and run the commands below manually.
#
# Assumes:
#   - conda is available
#   - verl is checked in as a subdirectory of this repo at ./verl
#     (cloned from https://github.com/verl-project/verl.git)
#   - Or: export VERL_DIR=/abs/path/to/verl before running.
#
# Creates a `spec_diag` conda env with verl + spec_diag installed editable.
set -euo pipefail

# Resolve verl dir
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPEC_DIAG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_DIR="${VERL_DIR:-$(cd "$SPEC_DIAG_DIR/verl" 2>/dev/null && pwd || echo "")}"

if [ -z "$VERL_DIR" ] || [ ! -d "$VERL_DIR" ]; then
  echo "ERROR: verl directory not found."
  echo "       Expected at $SPEC_DIAG_DIR/verl,"
  echo "       or: export VERL_DIR=/abs/path/to/verl"
  exit 1
fi
echo "spec_diag dir: $SPEC_DIAG_DIR"
echo "verl dir:      $VERL_DIR"

# 1. Create env
conda create -n spec_diag python=3.10 -y
conda activate spec_diag

# 2. Install verl (editable)
pip install -e "$VERL_DIR"

# 3. Install spec_diag (editable, with dev extras)
pip install -e "$SPEC_DIAG_DIR[dev]"

# 4. Sanity check
python -c "import verl, spec_diag; print('verl:', verl.__file__); print('spec_diag:', spec_diag.__file__)"

#!/usr/bin/env bash
# Smoke test launcher: point at verl's stock GRPO example scripts.
#
# Does NOT use spec_diag code. This is a pure verl baseline to confirm the
# install + GPU env work before we start writing experiment code.
#
# Resolves verl via $VERL_DIR, falling back to nested ./verl.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_DIR="${VERL_DIR:-$(cd "$SCRIPT_DIR/../verl" 2>/dev/null && pwd || echo "")}"

if [ -z "$VERL_DIR" ] || [ ! -d "$VERL_DIR" ]; then
  echo "ERROR: verl directory not found."
  echo "       export VERL_DIR=/abs/path/to/verl and re-run."
  exit 1
fi

EXAMPLE_DIR="$VERL_DIR/examples/grpo_trainer"
if [ ! -d "$EXAMPLE_DIR" ]; then
  echo "ERROR: $EXAMPLE_DIR not found."
  echo "       Check your verl commit still contains examples/grpo_trainer/."
  exit 1
fi

echo "verl dir:         $VERL_DIR"
echo "GRPO examples in: $EXAMPLE_DIR"
echo
ls "$EXAMPLE_DIR"
echo
echo "Pick the smallest example script from above and run it manually, e.g.:"
echo "  cd $VERL_DIR && bash examples/grpo_trainer/<chosen_script>.sh"
echo
echo "(Intentionally not auto-executing — GRPO training downloads models"
echo " and burns GPU; you should inspect and adjust paths first.)"

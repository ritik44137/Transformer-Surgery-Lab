#!/usr/bin/env bash
set -euo pipefail

# Execute the command passed to the container, or drop into bash.
if [[ $# -eq 0 ]]; then
  exec bash
else
  exec "$@"
fi

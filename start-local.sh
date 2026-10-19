#!/usr/bin/env bash
# Run as seeed in the Jetson desktop session; sudo is used only for backend services.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$project_dir/tools/start_local.py" "$@"

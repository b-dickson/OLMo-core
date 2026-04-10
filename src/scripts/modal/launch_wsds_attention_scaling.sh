#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

export UV_CACHE_DIR
exec uv run python -m modal run "${SCRIPT_DIR}/attention_ladder_modal.py" -- "$@"

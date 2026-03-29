#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python3 app/sz_ai_mac_app.py

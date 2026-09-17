#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
if command -v uv >/dev/null 2>&1; then
  uv_executable=uv
else
  uv_executable="$project_root/.tools/uv"
fi
"$uv_executable" venv --python 3.11 --seed --allow-existing .venv
"$uv_executable" pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$uv_executable" pip install --python .venv/bin/python -r requirements-venv.txt
.venv/bin/python -m pip check

#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
if [ ! -x .venv-cu130/bin/python ]; then
  .tools/uv venv --python .venv/bin/python --seed .venv-cu130
fi
.tools/uv pip install --python .venv-cu130/bin/python -r requirements-cu130.txt --index-strategy unsafe-best-match
.venv-cu130/bin/python -m pip check
.venv-cu130/bin/python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.version.cuda == "13.0"'

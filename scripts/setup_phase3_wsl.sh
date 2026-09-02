#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment="${TENSORFORGE_VENV:-${HOME}/.venvs/tensorforge-py311}"
uv_binary="${HOME}/.local/bin/uv"

if [[ ! -x "${uv_binary}" ]]; then
  python3 -m pip install --user 'uv>=0.8,<0.9'
fi

"${uv_binary}" python install 3.11
"${uv_binary}" venv --python 3.11 "${environment}"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
"${uv_binary}" pip install \
  --python "${environment}/bin/python" \
  'torch==2.5.1' \
  --index-url https://download.pytorch.org/whl/cu121
"${uv_binary}" pip install \
  --python "${environment}/bin/python" \
  -e "${repository}[dev,kernels]"

"${environment}/bin/python" -c \
  'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available())'

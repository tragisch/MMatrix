#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"

"${WORKSPACE_DIR}/tools/doxygen/generate_tensor_api_docs.sh"
"${WORKSPACE_DIR}/tools/doxygen/generate_matrix_api_docs.sh"

echo "Done. Tensor + Matrix API docs refreshed."

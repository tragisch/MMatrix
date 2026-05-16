#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"
INCLUDE_DIR="${WORKSPACE_DIR}/app/tensor/include"
OUT_DIR="${WORKSPACE_DIR}/doc/api/tensor"
PARSER="${WORKSPACE_DIR}/tools/doxygen/xml_to_markdown.py"

if ! command -v doxygen >/dev/null 2>&1; then
  echo "Error: doxygen is not installed or not in PATH." >&2
  echo "Install it (e.g. macOS: brew install doxygen) and retry." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required." >&2
  exit 1
fi

if [[ ! -d "${INCLUDE_DIR}" ]]; then
  echo "Error: include directory not found: ${INCLUDE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PARSER}" ]]; then
  echo "Error: parser not found: ${PARSER}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mmatrix-doxygen.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

DOXYFILE="${TMP_DIR}/Doxyfile"
cat >"${DOXYFILE}" <<EOF
PROJECT_NAME           = "MMatrix Tensor API"
OUTPUT_DIRECTORY       = ${TMP_DIR}/out
INPUT                  = ${INCLUDE_DIR}
FILE_PATTERNS          = st.h st_batchnorm.h st_conv.h st_pool.h st_shape_ops.h
RECURSIVE              = NO
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES
EXTRACT_LOCAL_CLASSES  = YES
JAVADOC_AUTOBRIEF      = YES
MULTILINE_CPP_IS_BRIEF = YES
QUIET                  = YES
WARN_IF_UNDOCUMENTED   = NO
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
GENERATE_XML           = YES
XML_OUTPUT             = xml
FULL_PATH_NAMES        = NO
EOF

echo "[1/2] Running doxygen XML generation..."
doxygen "${DOXYFILE}"

echo "[2/2] Rendering Markdown API docs..."
python3 "${PARSER}" \
  --xml-dir "${TMP_DIR}/out/xml" \
  --output-dir "${OUT_DIR}"

echo "Done. Updated files in ${OUT_DIR}"

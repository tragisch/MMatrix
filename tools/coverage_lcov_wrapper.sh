#!/usr/bin/env bash
set -euo pipefail

real_generator=""
runfiles=(
  "bazel_tools/tools/test/coverage_report_generator"
  "remote_coverage_tools/coverage_report_generator"
  "remote_coverage_tools/Main"
  "bazel_tools+remote_coverage_tools_extension+remote_coverage_tools/Main"
)

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  for runfile in "${runfiles[@]}"; do
    if [[ -f "${RUNFILES_DIR}/${runfile}" ]]; then
      real_generator="${RUNFILES_DIR}/${runfile}"
      break
    fi
  done
fi

if [[ -z "${real_generator}" && -n "${RUNFILES_MANIFEST_FILE:-}" && -f "${RUNFILES_MANIFEST_FILE}" ]]; then
  for runfile in "${runfiles[@]}"; do
    real_generator="$(rg -n "^${runfile} " "${RUNFILES_MANIFEST_FILE}" | head -n 1 | awk '{print $2}')"
    if [[ -n "${real_generator}" ]]; then
      break
    fi
  done
fi

if [[ -z "${real_generator}" ]]; then
  for runfile in "${runfiles[@]}"; do
    if [[ -f "${runfile}" ]]; then
      real_generator="${runfile}"
      break
    fi
  done
fi

output_file=""
coverage_dir=""
for arg in "$@"; do
  if [[ "${arg}" == --output_file=* ]]; then
    output_file="${arg#--output_file=}"
  elif [[ "${arg}" == --coverage_dir=* ]]; then
    coverage_dir="${arg#--coverage_dir=}"
  fi
done

if [[ -z "${real_generator}" && -n "${coverage_dir}" ]]; then
  execroot="${coverage_dir%%/bazel-out/*}"
  for candidate in "${execroot}"/bazel-out/*/bin/external/*coverage_tools*/Main; do
    if [[ -x "${candidate}" ]]; then
      real_generator="${candidate}"
      break
    fi
  done
  if [[ -z "${real_generator}" && -d "${execroot}/bazel-out" ]]; then
    real_generator="$(find "${execroot}/bazel-out" -type f -name Main -path "*coverage_tools*" -print -quit)"
  fi
fi

if [[ -z "${real_generator}" ]]; then
  for candidate in bazel-out/*/bin/external/*coverage_tools*/Main; do
    if [[ -x "${candidate}" ]]; then
      real_generator="${candidate}"
      break
    fi
  done
  if [[ -z "${real_generator}" ]]; then
    real_generator="$(find . -type f -name Main -path "*coverage_tools*" -print -quit)"
  fi
fi

if [[ -n "${real_generator}" && -x "${real_generator}" ]]; then
  "${real_generator}" "$@"
else
  if [[ -z "${coverage_dir}" || -z "${output_file}" ]]; then
    echo "coverage wrapper: missing coverage_dir or output_file" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${output_file}")"
  coverage_inputs="$(find "${coverage_dir}" -type f -name "_cc_coverage.dat" -print)"
  if [[ -z "${coverage_inputs}" ]]; then
    echo "coverage wrapper: no _cc_coverage.dat found in ${coverage_dir}" >&2
    exit 1
  fi
  cat ${coverage_inputs} > "${output_file}"
fi

if [[ -n "${output_file}" && -f "${output_file}" ]]; then
  sed -E 's#^SF:.*/execroot/_main/#SF:#' "${output_file}" > "${output_file}.tmp"
  mv "${output_file}.tmp" "${output_file}"
fi

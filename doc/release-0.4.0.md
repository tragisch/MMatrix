# Release preparation: 0.4.0

Candidate based on `feat/interpreter-kernels`, initially `9955884`; last version tag: `v0.3.1`.
The worktree was clean before this preparation. Validation commits are collected on `codex/release-0.4.0-validation`; no release tag or publication has been created.

## Publish in the source release

- Public source, headers, module tests and benchmark sources, BUILD files and helper rules.
- `.bazelversion` (9.3.0), `.bazelignore`, `MODULE.bazel` (0.4.0), and the resolved `MODULE.bazel.lock`.
- README, changelog with API migration notes, documentation sources and generated public API pages.
- `doc/requirements.in` alongside the existing pinned requirements file.
- `share/simple_benchmark/simple_bench.h`: required by the already-published MPS benchmark but previously ignored.
- The MIT project license and upstream license text embedded in vendored `log.c`.

A source release is the current deliverable. The local installer also collects third-party static libraries;
a redistributable binary SDK requires separate platform/runtime validation and complete third-party license notices.

## Remove / keep local

Removed obsolete Bzlmod replacement wrappers for Unity, libomp, pcg, zlib, matio, and SuiteSparse,
the unused local HDF5 repository declaration, stale SuiteSparse installer dependency,
and legacy benchmark targets whose sources were not versioned. Vendored `log.c` is built directly as a workspace target, without a local-repository path assumption. Removed the personal Conda activation `.envrc`.
The shared `.clangd` now contains only public include paths.

Keep local and ignored: `third_party/hdf5` and other vendor experiments, legacy benchmark sources/results,
Bazel links, caches, `compile_commands.json`, profiles, build output, editor state, and machine-specific `.envrc`.
`.bazelignore` also excludes local BUILD trees: Git ignore rules alone do not stop Bazel's `//...` discovery.
No ignored vendor or experiment directory was deleted from disk.

## Review findings and repairs

| Finding | Repair |
| --- | --- |
| Gather/permute raw byte APIs can wrap shape/byte products before copying. Reproduced by a failing overflow regression test. | Check products before multiplication and reject dimensions outside supported offset/index ranges. |
| Thread-comparison tests for binary/reduce/matmul never crossed the actual OpenMP thresholds. | Adjust test shapes to exercise the parallel branches. |
| Tensor permute reads `shape[perm[i]]` before validating the permutation. | Validate bounds and uniqueness before shape access. |
| Raw permute uses typed loads/stores for generic byte buffers. | Use byte copies that preserve payloads and support unaligned buffers. |
| Installer references removed SuiteSparse; unconditionally requires OpenBLAS; omits private/transitive archives and tensors. | Collect archives from the C/C++ link closure, deduplicate files, add platform-specific tensor/MPS artifacts. |
| Installer uses GNU `install -T`, unsupported by macOS. | Use portable explicit file destinations. |
| A fresh source release lacks sources named by legacy benchmark targets. | Keep only the maintained MPS target and publish its helper header. |
| Recursive discovery loads ignored HDF5/experimental BUILD files. | Version an explicit `.bazelignore`; remove unused local dependency wiring. |
| CI points at nonexistent `//tests/...`. | Build/test `//...`, pin Bazel and configure its setup. |
| README incorrectly requires SuiteSparse and implies Linux tensor support; API docs omit recent headers. | Correct dependencies/platforms, regenerate API pages, add migrations. |

Review focused on the changes since `v0.3.1`, especially the interpreter boundary, removed dependencies,
source-release completeness and installation. This is not exhaustive verification of every existing numerical routine.

## Validation

- Full local `bazel test //...`: 32 test targets pass on macOS ARM64; 138 top-level targets analyzed and built, including documentation and installer.
- Overflow regression failed before the fix and passed after it.
- Standard installation executed successfully in a temporary directory, including transitive libraries and public tensor headers.
- Tensor interpreter tests pass with AddressSanitizer and UndefinedBehaviorSanitizer (`halt_on_error=1`).
- Standard and bundle installations both build and run a standalone consumer exercising dense/sparse matrices, SGD, tensor conversion/broadcast, and f16 conversion.
- A separate source-only copy without ignored files or Git metadata builds all targets and passes all 32 test targets using the shared build/test cache. This checks source completeness, not an empty-cache build.

The workflow uses the documented upstream [checkout](https://github.com/actions/checkout) and [Bazel setup](https://github.com/bazel-contrib/setup-bazel) action versions.

## Before tagging

- Complete the GitHub Actions matrix on macOS and Ubuntu x86-64, including standard/bundle installation checks. Linux uses system OpenBLAS and LAPACKE.
- Inspect any failed jobs and fix release blockers before tagging.
- Existing optimized builds emit compiler warnings (including fast-math/NaN assumptions). Numerical tests pass locally, but cross-toolchain numerical guarantees and GPU performance have not been newly established.
- Tag the reviewed release commit as `v0.4.0` and publish the source release with the changelog.

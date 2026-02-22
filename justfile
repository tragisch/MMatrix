# Standard-Build
default:
    bazel build //app/matrix:matrix

# Build mit OpenBLAS
openblas:
    bazel build //app/matrix:matrix --define=USE_OPENBLAS=1

# Build mit Accelerate
accelerate:
    bazel build //app/matrix:matrix --define:USE_ACCELERATE=1

# Build mit Accelerate + MPS
mps:
    bazel build //app/matrix:matrix --define=USE_ACCELERATE_MPS=1

# Optional: Clean
clean:
    bazel clean

#!/usr/bin/env bash
# Builds the C++ extension inside the active conda environment.
#
# Usage:
#   conda activate ytenv
#   ./setup.sh           # serial build
#   ./setup.sh --mpi     # MPI-enabled build
#
# Machine-specific paths (MPI_HOME, MPIRUN, BUILD_TYPE, NPROC) can be set in
# build.local (copy build.local.example and edit it — it is gitignored).
# Environment variables take precedence over build.local.
#
# To set up a fresh conda environment first:
#   conda env create -f environment.yml && conda activate ytenv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/src/shockfindCore_cpp"

# Source machine-local overrides if present.
if [ -f "$SCRIPT_DIR/build.local" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/build.local"
fi

# Defaults (can be overridden via build.local or environment variables).
BUILD_TYPE="${BUILD_TYPE:-Release}"
NPROC="${NPROC:-$(nproc)}"

USE_MPI=0
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --mpi)   USE_MPI=1 ;;
        --clean) CLEAN=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ "$CLEAN" -eq 1 ]; then
    echo "==> Cleaning build directory"
    rm -rf "$CPP_DIR/build"
    [ "$#" -eq 1 ] && echo "==> Done." && exit 0
fi

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import pybind11" &>/dev/null; then
    echo "ERROR: pybind11 not found in $PYTHON. Set PYTHON in build.local." >&2
    exit 1
fi
PYBIND_DIR=$("$PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")

if [ "$USE_MPI" -eq 1 ]; then
    # Resolve MPI_HOME: build.local / env var → mpicxx on PATH → error.
    if [ -z "${MPI_HOME:-}" ]; then
        if ! command -v mpicxx &>/dev/null; then
            echo "ERROR: --mpi requested but mpicxx not found." >&2
            echo "Set MPI_HOME in build.local or as an environment variable." >&2
            exit 1
        fi
        MPI_HOME="$(dirname "$(dirname "$(command -v mpicxx)")")"
    fi
    MPIRUN="${MPIRUN:-$MPI_HOME/bin/mpirun}"
    echo "==> Building with MPI  (MPI_HOME=$MPI_HOME)"
    cmake -B "$CPP_DIR/build" -S "$CPP_DIR" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -Dpybind11_DIR="$PYBIND_DIR" \
          -DUSE_MPI=ON \
          -DMPI_HOME="$MPI_HOME" \
          -DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx" \
          -Wno-dev
else
    echo "==> Building serial  (BUILD_TYPE=$BUILD_TYPE)"
    cmake -B "$CPP_DIR/build" -S "$CPP_DIR" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -Dpybind11_DIR="$PYBIND_DIR" \
          -DUSE_MPI=OFF \
          -Wno-dev
fi

cmake --build "$CPP_DIR/build" -j"$NPROC"
cmake --install "$CPP_DIR/build"
echo "==> Done."
if [ "$USE_MPI" -eq 1 ]; then
    echo "    Run with MPI: ${MPIRUN:-mpirun} -n N python main_example.py"
fi

#!/usr/bin/env bash
# Builds the ShockFind C++ extensions inside the active conda environment.
#
# Usage:
#   conda activate ytenv
#   ./setup.sh                          # serial build (shockfindCore_cpp only)
#   ./setup.sh --mpi                    # MPI-enabled build
#   ./setup.sh --octave                 # + Octave AMR octree backend (auto-detect path)
#   ./setup.sh --octave /path/to/Octave/src   # + Octave with explicit src path
#   ./setup.sh --clean                  # wipe all build dirs
#
# Machine-specific paths (MPI_HOME, MPIRUN, BUILD_TYPE, NPROC, OCTAVE_SRC_DIR)
# can be set in build.local (copy build.local.example and edit — it is gitignored).
# Environment variables take precedence over build.local.
#
# To set up a fresh conda environment first:
#   conda env create -f environment.yml && conda activate ytenv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/src/shockfindCore_cpp"
OCT_DIR="$SCRIPT_DIR/src/shockfindCore_octave"

# Source machine-local overrides if present.
if [ -f "$SCRIPT_DIR/build.local" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/build.local"
fi

# Defaults (can be overridden via build.local or environment variables).
BUILD_TYPE="${BUILD_TYPE:-Release}"
NPROC="${NPROC:-$(nproc)}"

USE_MPI=0
USE_OCTAVE=0
OCTAVE_SRC="${OCTAVE_SRC_DIR:-}"   # can be pre-set in build.local or env
CLEAN=0

# ── argument parsing ──────────────────────────────────────────────────────────
i=1
while [ "$i" -le "$#" ]; do
    arg="${!i}"
    case "$arg" in
        --mpi)    USE_MPI=1 ;;
        --clean)  CLEAN=1 ;;
        --octave)
            USE_OCTAVE=1
            # Accept an optional next argument as the Octave src path
            next_i=$((i + 1))
            if [ "$next_i" -le "$#" ]; then
                next="${!next_i}"
                if [[ "$next" != --* ]]; then
                    OCTAVE_SRC="$next"
                    i=$next_i
                fi
            fi
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
    i=$((i + 1))
done

# ── clean ─────────────────────────────────────────────────────────────────────
if [ "$CLEAN" -eq 1 ]; then
    echo "==> Cleaning build directories"
    rm -rf "$CPP_DIR/build"
    rm -rf "$OCT_DIR/build"
    [ "$#" -eq 1 ] && echo "==> Done." && exit 0
fi

# ── resolve Python / pybind11 ─────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import pybind11" &>/dev/null; then
    echo "ERROR: pybind11 not found in $PYTHON. Set PYTHON in build.local." >&2
    exit 1
fi
PYBIND_DIR=$("$PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")

# ── build shockfindCore_cpp ───────────────────────────────────────────────────
if [ "$USE_MPI" -eq 1 ]; then
    if [ -z "${MPI_HOME:-}" ]; then
        if ! command -v mpicxx &>/dev/null; then
            echo "ERROR: --mpi requested but mpicxx not found." >&2
            echo "Set MPI_HOME in build.local or as an environment variable." >&2
            exit 1
        fi
        MPI_HOME="$(dirname "$(dirname "$(command -v mpicxx)")")"
    fi
    MPIRUN="${MPIRUN:-$MPI_HOME/bin/mpirun}"
    echo "==> Building shockfindCore_cpp with MPI  (MPI_HOME=$MPI_HOME)"
    cmake -B "$CPP_DIR/build" -S "$CPP_DIR" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -Dpybind11_DIR="$PYBIND_DIR" \
          -DUSE_MPI=ON \
          -DMPI_HOME="$MPI_HOME" \
          -DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx" \
          -Wno-dev
else
    echo "==> Building shockfindCore_cpp (serial)"
    cmake -B "$CPP_DIR/build" -S "$CPP_DIR" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -Dpybind11_DIR="$PYBIND_DIR" \
          -DUSE_MPI=OFF \
          -Wno-dev
fi
cmake --build "$CPP_DIR/build" -j"$NPROC"
cmake --install "$CPP_DIR/build"
echo "==> shockfindCore_cpp done."

# ── build shockfindCore_octave (optional) ─────────────────────────────────────
if [ "$USE_OCTAVE" -eq 1 ]; then
    # Auto-detect Octave src if not set
    if [ -z "$OCTAVE_SRC" ]; then
        # Common relative location: ShockFind/../../../Octave/src
        CANDIDATE="$(cd "$SCRIPT_DIR" && cd ../../../Octave/src 2>/dev/null && pwd)" || true
        if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE/tree.hpp" ]; then
            OCTAVE_SRC="$CANDIDATE"
            echo "    Auto-detected Octave src: $OCTAVE_SRC"
        fi
    fi
    if [ -z "$OCTAVE_SRC" ] || [ ! -f "$OCTAVE_SRC/tree.hpp" ]; then
        echo "ERROR: --octave requires a valid Octave src directory." >&2
        echo "  Pass it directly:  ./setup.sh --octave /path/to/Octave/src" >&2
        echo "  Or set OCTAVE_SRC_DIR in build.local." >&2
        exit 1
    fi
    echo "==> Building shockfindCore_octave  (Octave src: $OCTAVE_SRC)"
    cmake -B "$OCT_DIR/build" -S "$OCT_DIR" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -Dpybind11_DIR="$PYBIND_DIR" \
          -DOCTAVE_SRC_DIR="$OCTAVE_SRC" \
          -Wno-dev
    cmake --build "$OCT_DIR/build" -j"$NPROC"
    cmake --install "$OCT_DIR/build"
    echo "==> shockfindCore_octave done."
fi

echo ""
echo "==> All done."
if [ "$USE_MPI" -eq 1 ]; then
    echo "    Uniform grid + MPI: ${MPIRUN:-mpirun} -n N python main_example.py"
fi
if [ "$USE_OCTAVE" -eq 1 ]; then
    echo "    Octave AMR pipeline: python src/octree_shockfind_pipeline.py <data_path> --output <results_dir>"
fi

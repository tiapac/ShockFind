cmake_minimum_required(VERSION 3.18)
project(shockfindCore_octave CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ──────────────────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Octave source path ────────────────────────────────────────────────────────
# Path to Octave's src/ directory containing tree.hpp, octree_params.hpp, etc.
# Override at configure time with: cmake -DOCTAVE_SRC_DIR=/path/to/Octave/src
if(NOT DEFINED OCTAVE_SRC_DIR)
    set(OCTAVE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Octave/src"
        CACHE PATH "Path to Octave's src/ directory")
endif()

if(NOT EXISTS "${OCTAVE_SRC_DIR}/tree.hpp")
    message(FATAL_ERROR
        "Cannot find tree.hpp in OCTAVE_SRC_DIR='${OCTAVE_SRC_DIR}'.\n"
        "Set OCTAVE_SRC_DIR to the Octave src/ directory, e.g.:\n"
        "  cmake -B build -DOCTAVE_SRC_DIR=/path/to/Octave/src")
endif()
message(STATUS "Using Octave headers from: ${OCTAVE_SRC_DIR}")

# ── ShockFind core static lib (already built by parent) ───────────────────────
# We need access to core.hpp, types.hpp (already templated), and the
# flux_capacitor / shock_normal_average / transverse_* implementations.
# Rather than re-linking shockcore (which may not be built yet as a sub-project),
# we compile the needed non-template source directly.
# NOTE: core.cpp now contains only non-template functions, so it's safe to compile twice.
set(SHOCKCORE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shockfindCore_cpp")

add_library(shockcore_oct STATIC
    ${SHOCKCORE_SRC_DIR}/core.cpp
    ${SHOCKCORE_SRC_DIR}/mpi_dispatch.cpp
)
target_include_directories(shockcore_oct PUBLIC
    ${SHOCKCORE_SRC_DIR}        # types.hpp, core.hpp
    ${OCTAVE_SRC_DIR}           # tree.hpp, octree_params.hpp, etc.
)
set_target_properties(shockcore_oct PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(shockcore_oct PRIVATE -O3 -Wall -Wextra -Wno-trigraphs)

# ── Python extension ──────────────────────────────────────────────────────────
pybind11_add_module(shockfindCore_octave bindings_octave.cpp)

target_include_directories(shockfindCore_octave PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # octave_accessor.hpp
    ${SHOCKCORE_SRC_DIR}         # types.hpp, core.hpp
    ${OCTAVE_SRC_DIR}            # tree.hpp, octree_params.hpp, etc.
)
target_link_libraries(shockfindCore_octave PRIVATE shockcore_oct)
target_compile_options(shockfindCore_octave PRIVATE -O3)

# Install next to the other .so modules for easy import
install(TARGETS shockfindCore_octave
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)

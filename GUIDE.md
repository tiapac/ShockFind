# ShockFind(Oct) — Code Index & Quick Guide

An updated, C++-accelerated version of **SHOCKFIND** (Lehmann, Federrath & Wardle 2016)
for identifying MHD shock waves in simulation data. It supports two backends:

- **Uniform-grid** path — the classic algorithm on a regular covering grid (`shockfindCore_cpp`).
- **AMR octree** path — the same physics run directly on RAMSES AMR cells via an octree
  (`shockfindCore_octave`, backed by the sibling `Octave` tree library). No covering grid needed.

Both share one templated C++ core; only the *grid accessor* differs.

---

## 1. Repository map (the index)

### Top level
| Path | Role |
|------|------|
| `__init__.py` | Package entry — re-exports `shock_finder` (`from ShockFind import shock_finder`). |
| `main_example.py` | Reference driver for the **uniform-grid** path: yt load → cube → `shock_finder`. |
| `src/octree_shockfind_pipeline.py` | CLI + library entry for the **AMR octree** path (RAMSES → octree → shocks). |
| `setup.sh` | Builds the C++ extensions. `--octave` adds the AMR backend, `--mpi` the MPI build, `--clean` wipes builds. |
| `build.local` | Machine-local build config (gitignored). Sets `PYTHON=…/venvs/shockfind/bin/python` here. |
| `environment.yml` | Conda env spec (`ytenv`): python 3.12, numpy, yt, pybind11, mpi4py, h5py. |
| `README.md` | Upstream attribution / license (Apache 2.0). |
| `*_octree.h5` | Saved octree (HDF5) — large; produced by `--save-octree`, consumed by `--load-octree`. |

### `src/` — the engine
| Path | Role |
|------|------|
| `src/shockfind_interface.py` | **The public `shock_finder` class** — load data, thresholds, candidates, analyse, save/plot. |
| `src/shockfindCore/shockfind.py` | Pure-**Python** reference implementation of the algorithm (`core` base class). Slow but readable; the C++ mirrors it. |
| `src/shockfindCore/churchkey.py` | Small parameter-preset helper (`churchkey(name, params)`). |
| `src/shockfindCoreJ/` | Experimental/alternate ("J") copy of the Python core — not the active path. |

### `src/shockfindCore_cpp/` — uniform-grid C++ core
| Path | Role |
|------|------|
| `types.hpp` | Core structs: `Vec3`, `CellIndex`, `GridAccessor`, `FieldDataT<G>`, `ShockParams`, `LineProfile`, **`ShockResult`**. |
| `core.hpp` | Templated physics: `shock_normal`, `cylinder_average`, `characterise_shock<G>` (works for any accessor `G`). |
| `core.cpp` | Non-templated helpers: `flux_capacitor` (classification), `transverse_*`, `shock_normal_average`, serial `characterise_shocks`. |
| `bindings.cpp` | pybind11 module `shockfindCore_cpp` — `characterise_shocks(...)` over NumPy field arrays. |
| `mpi_dispatch.{hpp,cpp}` | Optional MPI batch dispatch for the uniform-grid path. |
| `CMakeLists.txt` | Build for the uniform-grid module (+ optional MPI). |

### `src/shockfindCore_octave/` — AMR octree C++ core
| Path | Role |
|------|------|
| `octave_accessor.hpp` | **`OctaveGridAccessor`** — the drop-in `GridAccessor` that walks the `MainTree<3>` octree instead of a regular grid (`line_step`, `cyl_offset`, `flat` → Hilbert rank). |
| `bindings_octave.cpp` | pybind11 module `shockfindCore_octave` — `build_octree`, `find_candidates`, `characterise_shocks_octree`, `compute_gradient_of`, `save_octree`/`load_octree`. |
| `CMakeLists.txt` | Build; needs `OCTAVE_SRC_DIR` (the sibling `Octave/src`), optional OpenMP + HDF5. |

### Support
| Path | Role |
|------|------|
| `utils/utils.py`, `utils/logger.py` | Helpers + logging (`setup_logger`, `loglevels`). |
| `visu/` | 3D/interactive plotting (`MakePlot.py`, `MakeRotPlot.py`, `plot_hists.py`). |
| `SB_plots/`, `plot_SB_shocks_rot_back.py` | Superbubble-specific analysis plots. |
| `legacy/`, `dev/` | Old / work-in-progress code — not on the active path. |

---

## 2. Architecture — one core, two grids

The key design: **all physics is templated on a grid accessor `G`** (`core.hpp`). An accessor only
has to answer four questions — `in_bounds`, `flat` (→ field index), `line_step`, `cyl_offset`:

```
                 characterise_shock<G>()   ← templated physics (core.hpp)
                 /                     \
   GridAccessor (uniform grid)     OctaveGridAccessor (AMR octree)
   flat = i*ny*nz + j*nz + k       flat = Hilbert rank of the leaf
   line_step: integer cell steps   line_step: physical steps of local cell size h=1/2^level
        │                                     │
   shockfindCore_cpp.so                shockfindCore_octave.so
        │                                     │
   shock_finder (interface.py)        octree_shockfind_pipeline.py
```

So adding the AMR backend required **no change to the physics** — only a new accessor and a new
pybind11 module. `FieldDataT<G>` carries the field pointers + the accessor together.

---

## 3. Algorithm / data flow

Both paths follow the same five stages (uniform-grid names in `shock_finder`, octree names in the pipeline):

1. **Load fields** — ρ, P, v(x,y,z), B(x,y,z). Uniform grid: a yt covering grid. Octree: RAMSES leaf cells → `build_octree`.
2. **Derived fields** — velocity divergence `div(v)` and density gradient `∇ρ` (both precomputed once).
3. **Find candidates** — cells that are converging *and* have a strong density gradient:
   - `div(v) < -conv_threshold`   (convergence)
   - `|∇ρ| > nablaRho_threshold`   (compression)
   - plus optional filters: Mach ≥ `mach_min`, `∇T·∇ρ > 0`, `|v| > vmag_min`.
   - Physical thresholds come from `vshock_min`, `rhomean` (see `set_thresholds` / `_find_candidates_full`).
4. **Characterise each candidate** (`characterise_shock<G>`):
   - shock **normal** `n = -∇ρ/|∇ρ|`; build a line along `n`; transverse frame from B;
   - **cylinder-average** fields along the line; `flux_capacitor` classifies the jump.
5. **Output** — a fixed list of per-shock arrays + a header of column names.

**Classification codes** (`ShockResult.family`): `12` = fast, `34` = slow, `0` = unclassified.
**Quality flag** (`ShockResult.flag`): `0` ok · `1` density threshold · `2` edge · `3` B/Mach inconsistent · `4` no convergence.

### Output column schema
Uniform-grid path → **17 columns**:
```
0:x 1:y 2:z 3:nx 4:ny 5:nz 6:Family 7:vs 8:vA 9:MachAlf
10:Mach 11:r 12:rho0 13:B0 14:pmag_ratio 15:peak 16:FLAG
```
AMR octree path → **19 columns** (same 17 **plus** two AMR-only trailing columns):
```
17:level    AMR refinement level of the candidate's leaf cell
18:volume   cell volume in normalised [0,1]^3 units = (1/2^level)^3
```
`level`/`volume` are `-1`/`0` on the uniform-grid path (they only make sense for AMR).
Downstream code indexes columns positionally (e.g. `shocks_data()` uses `ss[6]`, `ss[16]`), so the
two extra columns are additive and don't disturb existing tools.

---

## 4. Quick start

### Environment
The active env on this machine is the venv **`venvs/shockfind`** (python 3.12) — its `bin/python`
is set as `PYTHON` in `build.local`. `cmake` comes either from that venv or the `CMake/3.29.3` module.

```bash
source /mnt/beegfs/projects/hpcc250512a2/mpacicco/venvs/shockfind/bin/activate
module load CMake/3.29.3        # only if `cmake` isn't already on PATH
```

### Build
```bash
cd ShockFindOct
./setup.sh                      # uniform-grid module only
./setup.sh --octave             # + AMR octree module (auto-detects ../../../../Octave/src)
./setup.sh --octave /path/to/Octave/src   # explicit Octave path
./setup.sh --clean              # wipe build dirs
```
To rebuild just the AMR module in place after a source edit:
```bash
cd src/shockfindCore_octave/build
cmake --build . -j$(nproc) && cmake --install .   # installs the .so next to the others in src/
```

### Run — AMR octree path (recommended for RAMSES)
```bash
python src/octree_shockfind_pipeline.py /path/to/output_00041/ \
    --output results/ --name my_run
# reuse a prebuilt tree instead of re-reading RAMSES:
python src/octree_shockfind_pipeline.py --load-octree run/my_run_octree.h5
# build once and save it:
python src/octree_shockfind_pipeline.py /path/to/output_00041/ --save-octree run/
```
See `--help` for the full threshold / characterisation options.

### Run — uniform-grid path
```bash
python main_example.py -lvl 8 -out 41 -dp /path/to/sim/
```

### Use as a library
```python
from ShockFind import shock_finder                       # uniform grid
from src.octree_shockfind_pipeline import run_octree_pipeline   # AMR
handle, sf = run_octree_pipeline("/path/to/output_00041")
sf.save_results(path="results/", name="run01")
```

---

## 5. Operational notes / gotchas

- **Do NOT launch `octree_shockfind_pipeline.py` with `mpirun -np N` (N>1).** The script is *not*
  MPI-decomposed — every rank re-loads the full dataset and builds a full octree independently.
  On a big snapshot this multiplies memory N× and causes `std::bad_alloc` in `build_octree`.
  Run it as a **single process**; intra-node parallelism already comes from OpenMP in
  `characterise_shocks_octree`.
- **The octree build is single-threaded** and is the slow phase for large (10⁸-cell) snapshots — the
  extra threads you see are idle OpenMP/BLAS pools until characterisation. A minutes-long build at
  100% of one core is normal, not a hang. Use `--save-octree` / `--load-octree` to pay it only once.
- **Memory budget** ≈ positions+attrs (N × 11 × 8 bytes) + the tree (~N leaves + interior nodes) +
  yt's own overhead. ~93M cells ≈ 70–75 GB in one process.
- **`save_octree`/`load_octree` need the HDF5 build** (`USE_HDF5=ON`, default when HDF5 is found).
- **Two Python ABIs**: prebuilt `.so`s exist for cpython-310 and cpython-312 — make sure the one you
  import matches your interpreter.

---

*Generated as a navigation aid; keep it next to the code and update the column schema section if the
`ShockResult` layout changes.*

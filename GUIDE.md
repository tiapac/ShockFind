# ShockFind(Oct) — Code Index & Guide

An updated, C++-accelerated version of **SHOCKFIND** (Lehmann, Federrath & Wardle 2016)
for identifying MHD shock waves in simulation data. Two backends:

- **Uniform-grid** path: the classic algorithm on a regular covering grid (`shockfindCore_cpp`).
- **AMR octree** path: the same physics run directly on RAMSES AMR cells via an octree
  (`shockfindCore_octave`, built against the sibling `Octave` tree library). No covering grid.

Both share one templated C++ core; only the *grid accessor* differs.

Checked against the code on 2026-10-08 (branch `shockfind_interface_with_octave`). Update §0 when the
cluster software changes and §5 when the `ShockResult` layout changes.

---

## 0. Environment on this cluster (2026-10)

The cluster software lives in `/mnt/beegfs/appsx` (hierarchical Lmod). The GCC 13.3 / Python 3.12
stack under `/mnt/beegfs/apps` is gone, and so is the old venv `venvs/shockfind` (its python was a
link into it). Use:

```bash
module purge
module load GCC/14.3.0 OpenMPI/5.0.8 HDF5/1.14.6 Python/3.13.5
source /mnt/beegfs/projects/hpcc250512a2/mpacicco/venvs/shockfind313/bin/activate   # = loadShockFind in ~/.bashrc
```

`venvs/shockfind313` (Python 3.13.5): numpy 2.5, scipy, h5py, pybind11 3.1, cmake, mpi4py 4.1
(built against OpenMPI 5.0.8), matplotlib, pyvista, astropy, pyxsim, soxs, and **yt 4.5.dev0 from the
patched checkout `/mnt/beegfs/users/mpacicco/yt`** (`site-packages/yt-dev.pth`; reads ramses_h5
aggregates). Its Cython extensions are built in place for cp313 next to the cp310 ones of
`venvs/yt-h5`; rebuild after editing them: `cd ~/yt && CC=gcc CXX=g++ $VENV/bin/python setup.py
build_clib build_ext --inplace -j8` (GCC/14.3.0 loaded). Back to stock yt: `pip install yt==4.4.2`
and delete `yt-dev.pth`. `build.local` sets
`PYTHON=` to its python. `environment.yml` (conda `ytenv`) is historical: that env does not exist.

**SLURM.** Submit with `sbatch --export=NONE`. The job then starts in a bare shell without
`/usr/bin` on PATH (no `srun`, `ls`) and without the `module` command, so a job script must begin with

```bash
export PATH=/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/mnt/beegfs/appsx/modules/all/Core
module purge
module load GCC/14.3.0 OpenMPI/5.0.8 HDF5/1.14.6 Python/3.13.5
source /mnt/beegfs/projects/hpcc250512a2/mpacicco/venvs/shockfind313/bin/activate
```

and run the step with `srun --export=ALL` (otherwise it inherits the bare environment and finds no
`python`). `run_shockfind.sbatch` (any snapshot), `run_neigh_hl.sbatch` and `run_xthin.sbatch` are working examples (partition `medium`; `short`'s nodes
were all down on 2026-10-08).

**PYTHONPATH.** `~/.bashrc` appends `/mnt/beegfs/users/mpacicco`, where **another clone**,
`~/ShockFind` (branch `shockfindCcp`), is importable as `ShockFind`. The octree pipeline does not use
it: `_import_shock_finder()` loads *this* checkout as the `ShockFind` package, whatever the directory
is called. But a plain `from ShockFind import shock_finder` in your own script resolves through
`sys.path` and gets that other clone. `main_example.py` also needs `AnalysisPlayground`
(`$HOME/AnalysisPlayground`, same PYTHONPATH line).

---

## 1. Repository map

### Top level
| Path | Role |
|------|------|
| `__init__.py` | Package entry: `from .src.shockfind_interface import shock_finder`. |
| `main_example.py` | Reference driver for the **uniform-grid** path (yt cube → `shock_finder`). LB-specific, see §4. |
| `run_shockfind.py`, `run_shockfind.sbatch` | One snapshot (directory or aggregated `.h5`) → `<name>_result.pk`; wraps the pipeline below. |
| `src/octree_shockfind_pipeline.py` | CLI + library entry for the **AMR octree** path (RAMSES → octree → shocks). |
| `src/arepo_octree_pipeline.py` | Arepo snapshot → Octave octree HDF5 (no shock finding), §7. |
| `setup.sh` | Builds the C++ extensions: `--octave [OCTAVE_SRC]`, `--mpi`, `--clean`. |
| `build.local` | Machine-local build config (gitignored): `PYTHON`, optional `MPI_HOME`, `BUILD_TYPE`, `NPROC`, `OCTAVE_SRC_DIR`. |
| `run_*.sbatch` | SLURM jobs for the octree pipeline on the LB snapshots. |
| `*_octree.h5` | Saved octree (HDF5), from `--save-octree`, read by `--load-octree`. |

### `src/` — the engine
| Path | Role |
|------|------|
| `src/shockfind_interface.py` | **The `shock_finder` class**: load data, thresholds, candidates, analyse, save/plot. |
| `src/shockfindCore/shockfind.py` | Pure-Python reference implementation (`core`). Slow; the C++ mirrors it. |
| `src/shockfindCore/churchkey.py` | Parameter-preset helper. |
| `src/shockfindCoreJ/` | Experimental copy of the Python core, not on the active path. |

### `src/shockfindCore_cpp/` — uniform-grid C++ core
| Path | Role |
|------|------|
| `types.hpp` | `Vec3`, `CellIndex`, `GridAccessor`, `FieldDataT<G>`, `ShockParams`, `LineProfile`, **`ShockResult`**. |
| `core.hpp` | Templated physics: `shock_normal`, `cylinder_average`, `characterise_shock<G>`. |
| `core.cpp` | `flux_capacitor` (classification), `transverse_*`, `shock_normal_average`, serial `characterise_shocks`. |
| `bindings.cpp` | pybind11 module `shockfindCore_cpp`: `characterise_shocks(...)` over NumPy arrays. |
| `mpi_dispatch.{hpp,cpp}` | Optional MPI batch dispatch for the uniform-grid path. |
| `__init__.py` | Loads `build*/shockfindCore_cpp<ABI>.so` matching the running interpreter; if none, `shock_finder` falls back to the Python core (or raises under MPI). |

### `src/shockfindCore_octave/` — AMR octree C++ core
| Path | Role |
|------|------|
| `octave_accessor.hpp` | **`OctaveGridAccessor`**: the `GridAccessor` that walks `MainTree<3>` (`line_step`, `cyl_offset`, `flat` → Hilbert rank via `rank_by_id`). |
| `bindings_octave.cpp` | pybind11 module `shockfindCore_octave`, API in §6. |
| `CMakeLists.txt` | Needs `OCTAVE_SRC_DIR`; HDF5 (C++ API) and OpenMP optional. |

### Support
| Path | Role |
|------|------|
| `utils/utils.py`, `utils/logger.py` | Helpers + logging. |
| `visu/MakePlot.py`, `visu/Maplot2.py` | pyvista 3D view of a `*_result.pk` (`python visu/MakePlot.py RESULT.pk`); headless without `DISPLAY`, writes `<name>_3Dplot.png`. MakePlot keeps `vs > 0`, Maplot2 `vs > 1e5` cm/s. |
| `visu/MakeRotPlot.py`, `visu/plot_hists.py`, `SB_plots/`, `plot_SB_shocks_rot_back.py` | Other plots. |
| `legacy/`, `dev/` | Old / work in progress. |

---

## 2. Architecture — one core, two grids

All physics is templated on a grid accessor `G` (`core.hpp`). An accessor answers four questions:
`in_bounds`, `flat` (→ field index), `line_step`, `cyl_offset`:

```
                 characterise_shock<G>()   ← templated physics (core.hpp)
                 /                     \
   GridAccessor (uniform grid)     OctaveGridAccessor (AMR octree)
   flat = i*ny*nz + j*nz + k       flat = Hilbert rank of the leaf (rank_by_id)
   line_step: integer cell steps   line_step: physical steps of local cell size h=1/2^level
        │                                     │
   shockfindCore_cpp.so                shockfindCore_octave.so
        │                                     │
   shock_finder (interface.py)        octree_shockfind_pipeline.py
```

The octree side depends on Octave's `MainTree` internals; the symbols used are listed in
`codes/Octave/docs/shockfind_interface.md`. After editing Octave headers, rebuild here
(`./setup.sh --octave`).

---

## 3. Algorithm / data flow (octree path)

1. **Load** (`load_ramses_for_shockfind(info_path, box=None, box_units="code")`): yt
   `ds.all_data()`, gas cell centres normalised to `[0,1]^3` of the domain, and the fields
   `SHOCKFIND_FIELDS = [density, pressure, vx, vy, vz, bx, by, bz]` from yt `("gas", …)` in yt's CGS
   units, then each entry of `_RAMSES_OPTIONAL_FIELDS` present in the dataset (now
   `hydro_scalar_00` ← `("ramses","hydro_scalar_00")`, the passive scalar). Returns
   **`(positions (N,3), attrs (N,K), field_names)`**, K = 8 or 9. `box` selects a sub-region but is
   not on the CLI.
2. **Build** (`build_octree`): one leaf per cell (`max_members=1`), `max_depth=14` (dx = 1/16384),
   every column registered as a Mean field; div v and ∇ρ precomputed on the leaves.
3. **Candidates** (`_find_candidates_full`, all conditions ANDed):
   - convergence. Physical mode (`vshock_min` given): `div v < -vshock_min (compress-1) /
     (compress N_cells dx)`, dx = `handle.cell_size` (normalised units). Raw mode: `div v < div_threshold`.
   - density gradient. Physical mode needs `rhomean`; otherwise `|∇ρ|/ρ > grad_threshold` (0.1 per box
     length, almost no cut).
   - `∇T·∇ρ > 0` with T ∝ P/ρ (**on** by default; `--no-gradTRho` turns it off).
   - Mach: `|v| / (0.7 min(c_s, v_A)) ≥ mach_min` (**1.5** by default).
   - optional `|v| ≥ vmag_min`.
4. **Characterise** (`characterise_shocks_octree`, OpenMP): normal from ∇ρ, line along it, B-based
   transverse frame, cylinder averages, `flux_capacitor` classifies the jump.
5. **Output**: `shock_finder` with `shocks` (19 columns, §5), saved as `<name>_result.pk`.

**Defaults differ between the CLI and the library.** CLI: `--vshock-min 1e5` (physical convergence
always on), no `--rhomean` (so the 0.1 gradient fallback), **periodic** unless `--no-periodic`.
`run_octree_pipeline(...)`: `vshock_min=None` (raw `div_threshold=-0.1`), **non-periodic**.

---

## 4. Build and run

### Build (both extensions)
```bash
# environment of §0 loaded
./setup.sh --clean --octave          # wipes build dirs, builds shockfindCore_cpp + shockfindCore_octave
./setup.sh --octave /path/to/Octave/src   # explicit Octave src (default: ../Octave/src = codes/Octave/src)
```
- The extensions are built for `$PYTHON` (build.local) and installed into `src/`:
  `shockfindCore_octave.cpython-313-…so` (imported from `src/`) and `shockfindCore_cpp…so` (the
  package loader reads the copy in `src/shockfindCore_cpp/build*/`).
- **HDF5**: CMake looks for the C++ API (`libhdf5_cpp`, `H5Cpp.h`) in `$EBROOTHDF5` (set by the HDF5
  module), then `CMAKE_PREFIX_PATH`/system paths. Without it the build only warns ("HDF5 libraries not
  found") and `save_octree`/`load_octree` throw at run time: check for "HDF5 found" in the output.
- A build dir configured for another path or toolchain fails to reconfigure: `./setup.sh --clean`.
- After editing `bindings_octave.cpp` or any Octave header, rebuild with `./setup.sh --octave`
  (CMake does not track the Octave headers reliably).
- Older `.so`s in `src/` (cpython-310/312, from the GCC 13.3 stack) cannot load in the current
  environment; Python only picks the one matching its ABI.

### Run — one snapshot, the easy way (`run_shockfind.py`)
```bash
python run_shockfind.py /path/to/output_00801.h5      # aggregated snapshot (ramses_h5)
python run_shockfind.py /path/to/output_00801         # classic directory
sbatch --export=NONE run_shockfind.sbatch /path/to/output_00801.h5 [pipeline options]
```
Results go to `shockfind_results/` next to the snapshot, name `octree_output_00801`; every other
option goes to the pipeline below unchanged. An `.h5` is read directly when yt has the ramses_h5
reader (`venvs/shockfind313` does, §0); otherwise, or with `--reconstruct`, the directory is rebuilt
byte for byte into `--scratch` (default `$TMPDIR`) with `ramses_h5_aggregate.py` (found via
`--ramses-h5`, `$RAMSES_H5_DIR` or `../ramses_h5`), and deleted afterwards (`--keep-reconstructed`).
Checked on CONDUCTION/output_00012: directory, direct `.h5` and reconstructed `.h5` give identical
results (direct `.h5`: 21 s against 33 s for the directory, 8 cores).

### Run — AMR octree path
```bash
python src/octree_shockfind_pipeline.py /path/to/output_00801
#   results: shockfind_results/ next to output_00801/, name octree_output_00801 → octree_output_00801_result.pk
python src/octree_shockfind_pipeline.py /path/to/output_00801 --save-octree run/    # also writes run/<name>_octree.h5
python src/octree_shockfind_pipeline.py --load-octree run/octree_output_00801_octree.h5   # skip RAMSES + build
python src/octree_shockfind_pipeline.py /path/to/output_00801 -load                 # reload <name>_result.pk only
```
- `--save-octree PATH`: a directory only if it already exists (then `<name>_octree.h5` inside), else the file name.
- With `--load-octree`, results go to `shockfind_results/` next to the directory holding the h5, and
  the name is `octree_<h5 stem>`.
- Other flags (see `--help`): tree (`--max-depth 14 --min-depth 3 --max-members 1 --max-nodes 1e8`),
  candidates (`--vshock-min --rhomean --dx --N-cells 3 --compress 1.1`, raw `--div-threshold
  --grad-threshold`, `--mach-min 1.5 --vmag-min --no-gradTRho`), characterisation (`--gamma
  --no-periodic --method-norm point_gradient --method-plane point_field --Rgrad 3 --Rcylinder 3
  --line-range 10 --shock-ratio 1.1`), `--rescale k` (halve the candidate list k times), `--plot`
  (saves `<name>_3Dplot.png`), `--quiet`. `field_ref` and `box` are library-only.
- The end-of-run Fast/Slow counts include every FLAG.
- Timing (CONDUCTION/output_00012, 1.59M cells, 8 cores): load 20–40 s, build+candidates+characterise
  ~25 s, from a saved octree 5 s.

### Run — uniform-grid path
```bash
python main_example.py -lvl 8 -out 41 -dp /path/to/sim/      # trailing slash required (string concatenation)
```
Hard-coded for the Local Bubble: centre 800 pc, side 400 pc, `vshock_min=1e6`, `rhomean=3e-26`,
non-periodic, name prefix `LB_double`; `-dlvl` sets the level; caches in `DATA_PATH/__shockfind_cache/`.

### Library
```python
import sys; sys.path.insert(0, "/mnt/beegfs/projects/hpcc250512a2/mpacicco/codes/ShockFindOct/src")
import octree_shockfind_pipeline as P
handle, sf = P.run_octree_pipeline("/path/to/output_00801", name="run01",
                                   vshock_min=1e5, periodic=[True]*3,      # = CLI defaults
                                   octree_save_path="run01_octree.h5")
sf.save_results(path="results/", name="run01")
shock_finder = P._import_shock_finder()        # this checkout's class, not a clone on PYTHONPATH
```

### Gotchas
- **No `mpirun -np N>1`** for the octree pipeline: it is not MPI-decomposed; each rank loads
  everything (N× memory, `std::bad_alloc`). One process; OpenMP threads do the characterisation.
- **The tree build is single-threaded** and is the slow phase for 10⁸-cell snapshots (minutes at
  100% of one core is normal). Use `--save-octree`/`--load-octree` to pay it once.
- **Memory** ≈ N × (3 + K) × 8 bytes for positions + attrs, plus the tree (~N leaves + interior
  nodes), plus yt. ~93M cells ≈ 70–75 GB in one process.
- Octrees saved **before 2026-10-08** load with the right fields, but before Octave commit a5219b3
  a loaded tree had `num_leaves == 1` and every neighbour read as 0 (div v, ∇ρ, candidates and
  characterisation on it were wrong). Results from `--load-octree` runs before that date should be
  redone. Fresh-build runs were not affected.

---

## 5. Results

`sf.save_results(path, name)` pickles `[shocks, header]` to `<path>/<name>_result.pk`:
- `shocks`: list of 19 1-D arrays, **column-major** (`shocks[i]` is column i, one entry per candidate);
- `header = (None, [column names])`.

| col | name | meaning, units (RAMSES via yt: CGS) |
|---|---|---|
| 0–2 | x, y, z | octree: **integer** lower-corner coordinate of the leaf at the finest level (`coord << (max_depth-level)`); × 1/2^max_depth = box fraction. Uniform grid: cell index. |
| 3–5 | nx, ny, nz | unit shock normal |
| 6 | Family | 12 fast, 34 slow, 0 uncategorised |
| 7 | vs | shock speed, cm/s (≥ 0) |
| 8 | vA | pre-shock Alfvén speed, cm/s |
| 9 | MachAlf | vs / vA |
| 10 | Mach | vs / pre-shock c_s |
| 11 | r | ρ_post / ρ_pre |
| 12 | rho0 | pre-shock density, g/cm³ |
| 13 | B0 | pre-shock field, gauss |
| 14 | pmag_ratio | post/pre magnetic pressure |
| 15 | peak | always 1 (no peak search: `shock_peak = centre_init`, core.cpp:133) |
| 16 | FLAG | see below |
| 17 | level | AMR level of the leaf (octree only) |
| 18 | volume | (1/2^level)³, box units (octree only) |

The uniform-grid path writes columns 0–16 only.

**FLAG** (`core.cpp`, `core.hpp`): `0` good; `1` no density jump above `shock_ratio`, or no
pre/post window (r, vA, rho0, B0, pmag_ratio still set; vs = 0, Family = 0); `2` centre at the line
edge, or no in-bounds line cells; `3` Family inconsistent with MachAlf (fast with MA ≤ 1, slow with
MA > 1; Family kept); `4` no positive convergence on the line, zero normal, or invalid node.
**Good shocks**: `FLAG == 0`, `Family ∈ {12, 34}`, `vs > 0` (CONDUCTION/00012: 6,740 of 14,383
candidates, median vs 17.8 km/s).

`sf.shocks_data(dx)` (run by the pipeline with dx = cell size) fills `sf.computed_shocks =
[Fshocks, Sshocks, Bohshocks, Fpointers, Spointers]`: positions × dx and the normals of the fast
and slow shocks (the two normal lists were swapped before 2026-10-08). The `.pk` keeps the integers.

**Octree HDF5** (`save_octree`): `/leaves/fields/<name>` for every column given to `build_octree`
(8 MHD fields + `hydro_scalar_00` since ab65327; older files have the 8 only), `/leaves/AMR/*`
(id, coords, level, parent, rank, Hilbert code), `/leaves/adjacency/*`, `/nodes/*`. Full layout:
`codes/Octave/docs/hdf5_format.md`. div v and ∇ρ are not saved; `load_octree` recomputes them.

---

## 6. Python API of `shockfindCore_octave`

| function | notes |
|---|---|
| `build_octree(positions, attrs, field_names, max_depth=14, min_depth=3, max_members=1, max_nodes=100_000_000)` | positions (N,3) in [0,1]; attrs (N,K); all fields Mean-reduced; precomputes div v, ∇ρ |
| `save_octree(handle, filename)` | every non-empty leaf field + AMR + adjacency; needs HDF5 build |
| `load_octree(filename)` | resolves fields by aliases, recomputes div v and ∇ρ; identical to the built tree |
| `compute_gradient_of(handle, field_arr)` | field_arr: length `num_leaves`, Hilbert order → `(gx, gy, gz)` per box length |
| `find_candidates(handle, div_threshold=-0.1, grad_threshold=0.1)` | simple raw-mode cut; the pipeline uses `_find_candidates_full` instead |
| `characterise_shocks_octree(handle, candidates, extra={}, quiet=False)` | candidates: leaf node ids; `extra` keys: method_norm, method_plane, periodic, Rgrad, Rcylinder, line_range, field_ref, gamma, shock_ratio → `(data, header)` |

Field aliases (`build_octree`, `load_octree`): density/rho/Density; pressure/pres/Pressure/p;
vx/velocity_x/Vx (same for y, z); bx/Bx/magnetic_x/Bfield_x (same for y, z). Other names (e.g.
`hydro_scalar_00`) are stored and saved but nothing reads them.

`OctreeHandle` properties: `num_leaves`, `num_nodes`, `max_depth`, `cell_size` (1/2^max_depth),
`rho_arr pres_arr vx_arr vy_arr vz_arr bx_arr by_arr bz_arr`, `div_v_arr`,
`grad_rho_x/y/z`, `leaf_node_indices`, `leaf_levels` (all Hilbert-ordered, length `num_leaves`).

---

## 7. Arepo octree builder (`src/arepo_octree_pipeline.py`, BOND, ab65327)

Builds an Octave `Octree3D` from an Arepo snapshot and writes it to HDF5. It does **not** find
shocks. It uses Octave's own Python module `octree3d` (not `shockfindCore_octave`), which exists
only as `codes/Octave/octree3d.cpython-312-…so` for the conda `souffle` env; put `codes/Octave` on
`PYTHONPATH` (the script's default search path `~/codes/Octave` does not exist here).

- Particles are deposited with their smoothing length: the stored `SmoothingLength`, else
  `(3m/4πρ)^(1/3)`, × `--hsml-factor` (`--hsml-geom sphere|cube`, `--split-mass`).
- Fields: Masses, Density, InternalEnergy, Pressure, Velocities (required); ne, nh, mol_h_frac,
  metallicity, sfr, cooling_rate, virial, cool_shutoff, potential, MagneticField, passive scalars,
  metal fractions when present; derived `temperature` (K, from InternalEnergy in CGS).
  Mass-weighted means (mass summed).
- Other fields in **Arepo code units**, not CGS; positions and HSML in box units; a sub-box (`--box x0 x1 y0 y1 z0 z1` or `--center … --box-side …`)
  is renormalised to [0,1]³.
- CLI: `snap_path`, `--save-octree` (default `<snap>_octree.h5`), `--part-type`, `--max-depth 12`,
  `--min-depth 2`, `--max-members 8`, `--max-nodes 2e8`.
- Not tested with `shockfindCore_octave.load_octree`: field names would resolve, but units are code
  units and B is absent without MagneticField.

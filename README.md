# ShockFind-

This is an updated version of Shockfind, by [A. Lehmann [1]]([#1](https://doi.org/10.1093/mnras/stw2015)) and copyrighed as follow:

**Start with [GUIDE.md](GUIDE.md)**: environment and build on this cluster (§0, §4), how candidates
are selected (§3), the result format with units and FLAG/Family codes (§5), the C++ module API (§6).

- AMR octree path (RAMSES, recommended): `python src/octree_shockfind_pipeline.py /path/to/output_NNNNN`
- Uniform-grid path: `main_example.py` (`shock_finder` class in `src/shockfind_interface.py`)
- Arepo → octree HDF5 (no shock finding): `src/arepo_octree_pipeline.py`
- Build: `./setup.sh --octave` (needs the sibling `Octave` repository, `codes/Octave`)


"""
Copyright 2016 Andrew Lehmann

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


## References
<a id="1">[1]</a> 
Lehmann, A., Federrath, C. and Wardle, M., 2016

SHOCKFIND - an algorithm to identify magnetohydrodynamic shock waves in turbulent clouds

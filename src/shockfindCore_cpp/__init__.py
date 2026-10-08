"""
shockfindCore_cpp — C++ backend for ShockFind.

Loads the compiled pybind11 extension from build/ and exposes it as
the attribute `shockfindCore_cpp` so that

    from .shockfindCore_cpp import shockfindCore_cpp as _cpp

in shockfind_interface.py resolves correctly.  Falls back silently if the
extension has not been compiled yet.
"""
import os as _os
import glob as _glob
import importlib.util as _ilu
import importlib.machinery as _ilm

# build*/ directories (one per toolchain); only an extension for this interpreter loads.
_here = _os.path.dirname(__file__)
_so_files = sorted(f for f in _glob.glob(_os.path.join(_here, "build*", "shockfindCore_cpp*.so"))
                   if f.endswith(tuple(_ilm.EXTENSION_SUFFIXES[:1])))

if _so_files:
    try:
        # spec_from_file_location resolves PyInit_shockfindCore_cpp in the .so.
        _spec   = _ilu.spec_from_file_location("shockfindCore_cpp", _so_files[0])
        _native = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_native)
        shockfindCore_cpp = _native   # exported as package attribute
    except Exception:
        pass  # not built or broken; caller falls back to Python backend

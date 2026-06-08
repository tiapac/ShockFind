"""RAMSES interface package — loads RamsesConfig.oct.py via importlib."""
import importlib.util, os

_spec = importlib.util.spec_from_file_location(
    "RamsesConfig_oct",
    os.path.join(os.path.dirname(__file__), "RamsesConfig.oct.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RamsesInterface = _mod.RamsesInterface
__all__ = ["RamsesInterface"]

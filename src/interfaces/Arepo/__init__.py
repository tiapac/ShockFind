"""Arepo interface package — loads ArepoConfig.oct.py via importlib."""
import importlib.util, os

_spec = importlib.util.spec_from_file_location(
    "ArepoConfig_oct",
    os.path.join(os.path.dirname(__file__), "ArepoConfig.oct.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ArepoInterface = _mod.ArepoInterface
__all__ = ["ArepoInterface"]

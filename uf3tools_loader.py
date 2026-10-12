"""
Import the fitting part of UF3Tools without its optional extras.

``import UF3Tools`` runs ``UF3Tools/__init__.py``, which imports *every*
module of the package, so it needs phonopy, phono3py, phonolammps, mp-api,
dask, dask-jobqueue, optuna, pyswarms, pymongo, ... although none of them is
used to *fit* a potential. This loader:

  1. finds the UF3Tools source folder (the folder containing
     ``uf3_handler.py``),
  2. registers it as the ``UF3Tools`` package *without* running its
     ``__init__.py`` (so automation/ and the HPC tooling are not imported),
  3. while ``UF3Tools.uf3_handler`` is being imported, replaces the few
     validation-only packages it imports at module level (phonopy,
     phonolammps, phono3py, mp_api) by placeholders *if they are not
     installed*. Calling anything from a placeholder raises ImportError.

    from uf3tools_loader import load_uf3_handler
    uf3_handler = load_uf3_handler(r"C:\\path\\to\\UF3Tools")
    uf3_handler.Featurizer, uf3_handler.ModelTrainer, ...
"""
import datetime as _datetime
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

# Packages that UF3Tools.uf3_handler imports (through its validation
# modules) at import time, but that are not needed to fit a potential.
VALIDATION_ONLY_PACKAGES = ("phonopy", "phono3py", "phonolammps", "mp_api")

# uf3 modules imported by uf3_handler. They are imported *before* any
# placeholder exists, so uf3's own "is phonopy installed?" checks are honest.
UF3_MODULES = (
    "uf3.util.json_io", "uf3.util.parallel", "uf3.util.plotting",
    "uf3.util.plotting3d", "uf3.util.subsample", "uf3.data.io",
    "uf3.data.geometry", "uf3.data.composition", "uf3.data.analyze",
    "uf3.representation.bspline", "uf3.representation.process",
    "uf3.regression.least_squares", "uf3.forcefield.calculator",
    "uf3.forcefield.lammps",
)

UF3_FORK_INSTALL = ('pip install '
                    '"git+https://github.com/uf-chenlab/uf3.git@develop"')


class _Missing:
    """Placeholder for anything imported from a missing optional package."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"'{self._name}' is not available because "
            f"'{self._name.split('.')[0]}' is not installed. It is only used "
            "by UF3Tools validation tools, not for fitting.")

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self._name}.{attr}")

    def __mro_entries__(self, bases):
        return (object,)


class _PlaceholderModule(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self.__name__}.{attr}")


class _PlaceholderLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []  # package-like, so submodules can be imported
        module.__uf3tools_placeholder__ = True
        return module

    def exec_module(self, module):
        pass


class _PlaceholderFinder(importlib.abc.MetaPathFinder):
    """Put at the *end* of sys.meta_path, and only for packages that the
    normal import system cannot find."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            return importlib.machinery.ModuleSpec(
                fullname, _PlaceholderLoader(), is_package=True)
        return None


def find_uf3tools_package(search_root):
    """Return the folder that contains UF3Tools' ``uf3_handler.py``.

    ``search_root`` may be that folder itself, the cloned/unzipped repository
    (``UF3Tools`` or ``UF3Tools-main``) or any folder above it (searched up to
    4 levels deep).
    """
    search_root = os.path.abspath(os.path.expanduser(search_root))
    if os.path.isfile(os.path.join(search_root, "uf3_handler.py")):
        return search_root
    base_depth = search_root.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(search_root):
        if root.count(os.sep) - base_depth >= 4:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                         and d not in ("__pycache__", "site-packages"))
        if "uf3_handler.py" in files and os.path.basename(root) == "UF3Tools":
            return root
    raise FileNotFoundError(
        f"Could not find UF3Tools/uf3_handler.py under '{search_root}'. "
        "Point --uf3tools to the folder where you cloned/unzipped "
        "https://github.com/uf-chenlab/UF3Tools")


def _import_uf3():
    try:
        for name in UF3_MODULES:
            importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == "uf3":
            raise ImportError("The 'uf3' package is not installed. UF3Tools "
                              "needs the uf-chenlab fork:\n  "
                              + UF3_FORK_INSTALL) from exc
        raise ImportError(f"uf3 needs the package '{exc.name}', which is "
                          f"not installed:\n  pip install {exc.name}") from exc
    uf3_io = sys.modules["uf3.data.io"]
    if not hasattr(uf3_io, "parse_with_subsampling"):
        raise ImportError(
            "The installed 'uf3' is the original PyPI version. UF3Tools needs "
            "the uf-chenlab fork (develop branch):\n  pip uninstall -y uf3\n  "
            + UF3_FORK_INSTALL)


def load_uf3_handler(uf3tools_path=None, verbose=True):
    """Import and return the ``UF3Tools.uf3_handler`` module.

    uf3tools_path: folder of the UF3Tools code (see find_uf3tools_package).
        If None, an installed UF3Tools (``pip install -e``) is used.
    """
    if "UF3Tools.uf3_handler" in sys.modules:
        return sys.modules["UF3Tools.uf3_handler"]
    _import_uf3()

    if uf3tools_path is None:
        spec = importlib.util.find_spec("UF3Tools")
        if spec is None or not spec.submodule_search_locations:
            raise FileNotFoundError(
                "UF3Tools not found. Pass --uf3tools <folder> or "
                "'pip install -e' the UF3Tools repository.")
        pkg_dir = list(spec.submodule_search_locations)[0]
    else:
        pkg_dir = find_uf3tools_package(uf3tools_path)

    # Register UF3Tools as a package without executing its __init__.py
    pkg = types.ModuleType("UF3Tools")
    pkg.__path__ = [pkg_dir]
    pkg.__file__ = os.path.join(pkg_dir, "__init__.py")
    pkg.__package__ = "UF3Tools"
    sys.modules["UF3Tools"] = pkg

    missing = [p for p in VALIDATION_ONLY_PACKAGES
               if importlib.util.find_spec(p) is None]
    finder = _PlaceholderFinder(missing)
    sys.meta_path.append(finder)
    try:
        handler = importlib.import_module("UF3Tools.uf3_handler")
    finally:
        # Placeholders were only needed while importing UF3Tools.
        sys.meta_path.remove(finder)
        for name in [n for n, m in sys.modules.items()
                     if getattr(m, "__uf3tools_placeholder__", False)]:
            del sys.modules[name]

    # uf3_handler.py does "from UF3Tools.common.UF3Utility import *", which
    # binds the name `datetime` to the datetime *module*, but its LAMMPS
    # writer calls datetime.now(). Point it at the class so export works.
    if not hasattr(getattr(handler, "datetime", None), "now"):
        handler.datetime = _datetime.datetime

    if verbose:
        print(f"Loaded UF3Tools from: {pkg_dir}")
        if missing:
            print("  (not installed, only needed for UF3Tools validation: "
                  + ", ".join(missing) + ")")
    return handler

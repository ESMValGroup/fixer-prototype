"""A package for making xarray datasets analysis ready."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from importlib_metadata import entry_points

if TYPE_CHECKING:
    import xarray as xr

    from fixer.protocol import FixerPlugin, FixFunction


FIXERS: list[FixerPlugin] = [
    plugin.load() for plugin in entry_points(group="fixer.plugins")
]


def _load_function(name: str) -> FixFunction:
    """Load a Python function.

    Parameters
    ----------
    name:
        The fully qualified name of the function.

    Returns
    -------
    :
        The function.

    """
    module_name, function_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def fix(ds: xr.Dataset, name: str) -> xr.Dataset:
    """Fix a dataset.

    Parameters
    ----------
    ds:
        The dataset to fix.
    name:
        The name of the dataset.

    Returns
    -------
    :
        The fixed dataset.

    """
    for fixer in sorted(FIXERS, key=lambda fixer: fixer.priority):
        for fix in fixer.get_fixes(name):
            fix_function = _load_function(fix["function"])
            kwargs = {k: v for k, v in fix.items() if k != "function"}
            ds = fix_function(ds, **kwargs)
    return ds

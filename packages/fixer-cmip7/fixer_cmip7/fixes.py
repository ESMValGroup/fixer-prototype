"""Functions to make CMIP7 xarray datasets analysis ready."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Self

from fixer.fixes import Coordinate, CoordinateBounds, Variable

if TYPE_CHECKING:
    import xarray as xr


class CMIP7Coordinate(Coordinate):
    """Definition of a CMIP7 coordinate variable."""

    @classmethod
    def from_cmor_table(cls, path: Path, entry: str) -> Self:
        """Load a definition from the CMIP7 CMOR tables."""
        spec = json.loads(
            (path / "CMIP7_coordinate.json").read_text(encoding="utf-8"),
        )["axis_entry"][entry]
        dtype = "float64" if spec.get("dtype", "") == "double" else None
        attrs: tuple[str, ...] = (
            "standard_name",
            "long_name",
            "axis",
            "positive",
        )
        if spec["standard_name"] != "time":
            # The time units are incomplete in the CMOR tables.
            attrs = (*attrs[:2], "units", *attrs[2:])
        dims = (
            (spec["out_name"],) if spec["axis"] and not spec["value"] else None
        )
        if spec["must_have_bounds"] == "yes":
            bounds = CoordinateBounds(
                name=f"{spec['out_name']}_bnds",
                dims=(
                    *dims,
                    "bnds",
                )
                if dims
                else ("bnds",),
            )
        else:
            bounds = None
        return cls(
            name=spec["out_name"],
            dtype=dtype,
            dims=dims,
            attrs={k: spec[k] for k in attrs if spec[k]},
            bounds=bounds,
        )


class CMIP7Variable(Variable):
    """Definition of a CMIP7 variable."""

    @classmethod
    def from_cmor_table(cls, path: Path, table_id: str, entry: str) -> Self:
        """Load a definition from the CMIP7 CMOR tables."""
        spec = json.loads(
            (path / f"CMIP7_{table_id}.json").read_text(encoding="utf-8"),
        )["variable_entry"][entry]

        attrs = (
            "standard_name",
            "long_name",
            "units",
            "cell_methods",
            "cell_measures",
            "positive",
        )
        coords = tuple(
            CMIP7Coordinate.from_cmor_table(path, d)
            for d in spec["dimensions"][::-1]
        )
        dims = tuple(d.name for d in coords if d.dims) if coords else None
        return cls(
            name=spec["out_name"],
            dtype="float32",
            dims=dims,
            coords=coords,
            attrs={k: spec[k] for k in attrs if spec[k]},
        )


def reformat(
    ds: xr.Dataset,
    realm: str,
    branded_variable: str,
    dim_map: dict[str, str] | None = None,
    variable_map: dict[str, str] | None = None,
) -> xr.Dataset:
    """Reformat a dataset using the definition from the CMIP7 CMOR tables.

    Parameters
    ----------
    ds:
        The dataset to reformat.
    realm:
        The CMIP7 realm.
    branded_variable:
        The branded variable name.
    dim_map:
        A mapping of dimension names to rename.
    variable_map:
        A mapping of variable names to rename.

    Returns
    -------
    :     The reformatted dataset.
    """
    table_dir = Path(__file__).parent / "cmip7-cmor-tables" / "tables"
    return CMIP7Variable.from_cmor_table(
        path=table_dir,
        table_id=realm,
        entry=branded_variable,
    ).to_dataset(ds, dim_map=dim_map, variable_map=variable_map)

"""Functions to make CMIP7 xarray datasets analysis ready."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Self

from fixer.fixes import Coordinate, CoordinateBounds, Variable

if TYPE_CHECKING:
    import xarray as xr

TABLE_DIR = Path(__file__).parent / "cmip7-cmor-tables" / "tables"


class CMIP7Coordinate(Coordinate):
    """Definition of a CMIP7 coordinate variable."""

    @classmethod
    def from_cmor_table(cls, entry: str, path: Path | None = None) -> Self:
        """Load a definition from the CMIP7 CMOR tables.

        Parameters
        ----------
        entry:
            The name of the coordinate entry in the CMIP7 CMOR tables.
        path:
            The path to the CMIP7 CMOR tables. If None, the default path is
            used.

        Returns
        -------
        :
            The coordinate definition.

        """
        if path is None:
            path = TABLE_DIR

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
            (spec["out_name"],) if spec["axis"] and not spec["value"] else ()
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
        requirements: tuple[str, ...] = (
            "formula",
            "requested",
            "requested_bounds",
            "stored_direction",
            "valid_min",
            "valid_max",
            "value",
            "z_bounds_factors",
            "z_factors",
        )
        return cls(
            name=spec["out_name"],
            dtype=dtype,
            dims=dims,
            attrs={k: spec[k] for k in attrs if spec[k]},
            bounds=bounds,
            requirements={k: spec[k] for k in requirements if spec[k]},
        )


class CMIP7Variable(Variable):
    """Definition of a CMIP7 variable."""

    @classmethod
    def from_cmor_table(
        cls,
        table_id: str,
        entry: str,
        path: Path | None = None,
    ) -> Self:
        """Load a definition from the CMIP7 CMOR tables.

        Parameters
        ----------
        table_id:
            The ID of the CMIP7 table.
        entry:
            The name of the variable entry in the CMIP7 CMOR tables.
        path:
            The path to the CMIP7 CMOR tables. If None, the default path is
            used.

        Returns
        -------
        :
            The variable definition.
        """
        if path is None:
            path = TABLE_DIR
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
            CMIP7Coordinate.from_cmor_table(d, path)
            for d in spec["dimensions"][::-1]
        )
        dims = tuple(d.name for d in coords if d.dims) if coords else ()
        return cls(
            name=spec["out_name"],
            dtype="float32",
            dims=dims,
            coords=coords,
            attrs={k: spec[k] for k in attrs if spec[k]},
        )


def reformat(  # noqa: PLR0913
    ds: xr.Dataset,
    realm: str,
    branded_variable: str,
    dim_map: dict[str, str] | None = None,
    variable_map: dict[str, str] | None = None,
    *,
    keep_global_attrs: bool = False,
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
    keep_global_attrs:
        Whether to keep the global attributes.

    Returns
    -------
    :
        The reformatted dataset.
    """
    return CMIP7Variable.from_cmor_table(
        table_id=realm,
        entry=branded_variable,
    ).to_dataset(
        ds,
        dim_map=dim_map,
        variable_map=variable_map,
        keep_global_attrs=keep_global_attrs,
    )

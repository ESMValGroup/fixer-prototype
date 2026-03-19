"""Demonstrate how to build a fixer configuration automatically."""

# ruff: noqa: T201

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import fixer_cmip7.fixes
import yaml
from fixer_cmip7.tests.test_fixes import create_test_dataset

if TYPE_CHECKING:
    import fixer
    import xarray as xr


def find_coord(
    ds: xr.Dataset,
    coordinate: fixer.fixes.Coordinate,
) -> xr.DataArray | None:
    """Find a coordinate in the dataset that matches the definition.

    Parameters
    ----------
    ds:
        The dataset to search.
    coordinate:
        The coordinate definition to match.

    Returns
    -------
    :
        The matching coordinate, or None if no match is found.
    """
    if (axis := coordinate.attrs.get("axis")) and axis in ds.cf.axes:
        return ds[ds.cf.axes[axis]]
    if (
        standard_name := coordinate.attrs.get("standard_name")
    ) and standard_name in ds.cf.coordinates:
        coord_names = ds.cf.coordinates[standard_name]
        if len(coord_names) == 1:
            return ds[coord_names[0]]
    return None


def find_dim_coord(
    ds: xr.Dataset,
    coordinate: fixer.fixes.Coordinate,
) -> xr.DataArray | None:
    """Find a 1D coordinate in the dataset that matches the definition.

    Parameters
    ----------
    ds:
        The dataset to search.
    coordinate:
        The coordinate definition to match.

    Returns
    -------
    :
        The matching coordinate, or None if no match is found.
    """
    if not coordinate.dims or len(coordinate.dims) != 1:
        return None
    coord_name = None
    if (axis := coordinate.attrs.get("axis")) and axis in ds.cf.axes:
        coord_name = ds.cf.axes[axis]
    elif (
        standard_name := coordinate.attrs.get("standard_name")
    ) and standard_name in ds.cf.coordinates:
        coord_names = [
            coord_name
            for coord_name in ds.cf.coordinates[standard_name]
            if len(ds[coord_name].sizes) == 1
        ]
        if len(coord_names) == 1:
            coord_name = coord_names[0]
    if coord_name is not None:
        return ds[coord_name]
    return None


def build_mapping(  # noqa: C901
    ds: xr.Dataset,
    variable: fixer.fixes.Variable,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return a mapping from variable definition to dataset variables.

    Parameters
    ----------
    ds:
        The dataset to scan.
    variable:
        The variable definition.

    Returns
    -------
    :
        A fixer configuration for the reformat fix.
    """
    dim_map = {}
    variable_map = {}
    for coord_def in variable.coords:
        # Populate the dim_map.
        if (coord := find_dim_coord(ds, coord_def)) is not None:
            dims = list(coord.sizes)
            if len(dims) == 1:
                dim_name = dims[0]
                if dim_name != coord_def.dims[0]:
                    dim_map[coord_def.dims[0]] = dim_name
        # Populate the variable_map.
        if (coord := find_coord(ds, coord_def)) is not None:
            if coord.name != coord_def.name:
                variable_map[coord_def.name] = coord.name
            if (
                coord_def.bounds
                and (standard_name := coord.attrs.get("standard_name"))
                and standard_name in ds.cf.bounds
            ):
                bound_names = ds.cf.bounds[standard_name]
                if len(bound_names) == 1:
                    if bound_names[0] != coord_def.bounds.name:
                        variable_map[coord_def.bounds.name] = bound_names[0]
                    bounds_dim_name = ds.cf.get_bounds_dim_name(standard_name)
                    def_bounds_dim_name = next(
                        iter(set(coord_def.bounds.dims) - set(coord_def.dims)),
                    )
                    if bounds_dim_name != def_bounds_dim_name:
                        dim_map[def_bounds_dim_name] = bounds_dim_name

    # Add the main variable to the variable_map.
    bound_names = {name for names in ds.cf.bounds.values() for name in names}
    variables = [v for v in ds.data_vars if v not in bound_names]
    if len(variables) == 1:
        var_name = variables[0]
        if var_name != variable.name:
            variable_map[variable.name] = var_name

    return dim_map, variable_map


def scan_flip_coords(
    ds: xr.Dataset,
    variable: fixer.fixes.Variable,
) -> list[dict[str, Any]]:
    """Scan a dataset and return a fixer configuration for the flip_coords fix.

    Parameters
    ----------
    ds:
        The dataset to scan.
    variable:
        The variable definition.

    Returns
    -------
    :
        A list of fixer configurations for the flip_coords fix.
    """
    result = []
    for coord_def in variable.coords:
        if (
            coord_def.dims
            and len(coord_def.dims) == 1
            and "stored_direction" in coord_def.requirements
        ):
            # Try to find the coordinate in the dataset.
            coord_name = None
            if (axis := coord_def.attrs.get("axis")) and axis in ds.cf.axes:
                coord_name = ds.cf.axes[axis]
            elif (
                standard_name := coord_def.attrs.get("standard_name")
            ) and standard_name in ds.cf.coordinates:
                coord_names = [
                    coord_name
                    for coord_name in ds.cf.coordinates[standard_name]
                    if len(ds[coord_name].sizes) == 1
                ]
                if len(coord_names) == 1:
                    coord_name = coord_names[0]
            if coord_name is None:
                continue
            # Check the direction of the coordinate.
            coord = ds[coord_name]
            coord_direction = (
                "increasing"
                if coord[0].values < coord[-1].values
                else "decreasing"
            )
            expected_direction = coord_def.requirements["stored_direction"]
            if coord_direction != expected_direction:
                result.append(
                    {
                        "function": "fixer.fixes.flip_coords",
                        "coord": coord_name,
                    },
                )

    return result


def scan_cmip7(
    ds: xr.Dataset,
    realm: str,
    branded_variable: str,
) -> list[dict[str, Any]]:
    """Scan a dataset and return a list of fixes.

    Parameters
    ----------
    ds:
        The dataset to scan.
    realm:
        The CMIP7 realm.
    branded_variable:
        The branded variable name.

    Returns
    -------
    :
        A configurations for the fixes that should be applied to the dataset.
    """
    variable = fixer_cmip7.fixes.CMIP7Variable.from_cmor_table(
        table_id=realm,
        entry=branded_variable,
    )
    dim_map, variable_map = build_mapping(ds, variable)
    return [
        {
            "function": "fixer_cmip7.fixes.reformat",
            "realm": realm,
            "branded_variable": branded_variable,
            "dim_map": dim_map,
            "variable_map": variable_map,
        },
        *scan_flip_coords(ds, variable),
    ]


def main() -> None:
    """Scan a dataset for problems."""
    ds = create_test_dataset()
    print("Original:\n", ds)
    print()
    print("Scanning..")
    fixer_configuration = scan_cmip7(ds, "atmos", "tas_tavg-h2m-hxy-u")
    print("Found fixes:")
    print(yaml.safe_dump(fixer_configuration, sort_keys=False))


if __name__ == "__main__":
    main()

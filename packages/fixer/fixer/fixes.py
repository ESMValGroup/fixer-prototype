"""Functions to make xarray datasets analysis ready."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cf_xarray.units  # noqa: F401 # Needed to support cf-units with pint
import numpy as np
import pandas as pd
import pint
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Mapping

    import dask.array as da


def _convert_units(
    data: da.Array,
    src_units: str | None,
    tgt_units: str | None,
) -> da.Array:
    """Convert units.

    Parameters
    ----------
    data:
        The data to convert.
    src_units:
        The source units.
    tgt_units:
        The target units. If `None`, no conversion is performed.

    Returns
    -------
    :
        The converted data.

    """
    if tgt_units is None:
        return data
    if src_units is None:
        msg = f"Unable to convert unknown units to {tgt_units}."
        raise ValueError(msg)
    return pint.Quantity(data, src_units).to(tgt_units).magnitude


@dataclass
class CoordinateBounds:
    """Definition of a coordinate bounds variable."""

    name: str
    """Name of the coordinate bounds variable."""

    dims: tuple[str, ...]
    """Dimensions of the coordinate bounds variable."""


@dataclass
class Coordinate:
    """Definition of a coordinate variable."""

    name: str
    """Name of the coordinate variable."""

    dtype: str | None = None
    """Data type of the coordinate variable."""

    dims: tuple[str, ...] = ()
    """Dimensions of the coordinate variable."""

    attrs: dict[str, str] = field(default_factory=dict)
    """Attributes of the coordinate variable."""

    bounds: CoordinateBounds | None = None
    """Bounds of the coordinate variable."""

    requirements: dict[str, str] = field(default_factory=dict)
    """Additional requirements of the coordinate variable."""

    def _copy_time_encoding(
        self,
        src: xr.DataArray,
        tgt: xr.DataArray,
    ) -> None:
        """Copy the time encoding, if it exists.

        Parameters
        ----------
        src:
            The source data array to copy from.
        tgt:
            The target data array to copy to.
        """
        time_encoding_keys = ("units", "calendar", "dtype")
        if set(time_encoding_keys).issubset(src.encoding):
            for key in time_encoding_keys:
                tgt.encoding[key] = src.encoding[key]
                tgt.attrs.pop(key, None)

    def to_dataarray(
        self,
        ds: xr.Dataset,
        dim_map: dict[str, str] | None = None,
        variable_map: dict[str, str] | None = None,
    ) -> xr.DataArray:
        """Create a standardized coordinate.

        Parameters
        ----------
        ds:
            The original dataset.
        dim_map:
            An optional mapping from the dimension names in the definition to
            the dimension names in the resulting dataset.
        variable_map:
            An optional mapping from the variable names in the definition to
            the variable names in the resulting dataset.

        Returns
        -------
        :
            The standardized coordinate variable as a data array.
        """
        dim_map = dim_map or {}
        variable_map = variable_map or {}
        attrs = dict(self.attrs or {})
        if self.bounds:
            attrs["bounds"] = self.bounds.name
        original_coord = ds[variable_map.get(self.name, self.name)]
        if self.dims and dim_map:
            order = [dim_map.get(d, d) for d in self.dims]
            data = original_coord.transpose(*order).data
        else:
            data = original_coord.data
        data = _convert_units(
            data,
            original_coord.attrs.get("units"),
            attrs.get("units"),
        )
        if self.dtype is not None:
            data = data.astype(np.dtype(self.dtype))
        coord = xr.DataArray(
            data=data,
            name=self.name,
            dims=self.dims,
            attrs=attrs,
        )
        self._copy_time_encoding(original_coord, coord)
        return coord

    def to_bounds_dataarray(
        self,
        ds: xr.Dataset,
        dim_map: dict[str, str] | None = None,
        variable_map: dict[str, str] | None = None,
    ) -> xr.DataArray | None:
        """Create standardized coordinate bounds.

        Parameters
        ----------
        ds:
            The original dataset.
        dim_map:
            An optional mapping from the dimension names in the definition to
            the dimension names in the resulting dataset.
        variable_map:
            An optional mapping from the variable names in the definition to
            the variable names in the resulting dataset.

        Returns
        -------
        :
            The standardized coordinate bounds as a data array.
        """
        if not self.bounds:
            return None
        dim_map = dim_map or {}
        variable_map = variable_map or {}
        original_coord = ds[variable_map.get(self.name, self.name)]
        original_bounds = ds[
            variable_map.get(self.bounds.name, self.bounds.name)
        ]
        order = [dim_map.get(d, d) for d in self.bounds.dims]
        data = original_bounds.transpose(*order).data
        data = _convert_units(
            data,
            original_coord.attrs.get("units"),
            (self.attrs or {}).get("units"),
        )
        if self.dtype is not None:
            data = data.astype(np.dtype(self.dtype))
        bounds = xr.DataArray(
            data=data,
            name=self.bounds.name,
            dims=self.bounds.dims,
        )
        self._copy_time_encoding(original_coord, bounds)
        return bounds


@dataclass
class Variable:
    """Definition of a physical quantity."""

    name: str
    """Name of the variable."""

    dtype: str | None = None
    """Data type of the variable."""

    dims: tuple[str, ...] = ()
    """Dimensions of the variable."""

    coords: tuple[Coordinate, ...] = ()
    """Coordinates associated with the variable."""

    attrs: dict[str, str] = field(default_factory=dict)
    """Attributes of the variable."""

    requirements: dict[str, str] = field(default_factory=dict)
    """Additional requirements of the coordinate variable."""

    def to_dataset(
        self,
        ds: xr.Dataset,
        dim_map: dict[str, str] | None = None,
        variable_map: dict[str, str] | None = None,
        *,
        keep_global_attrs: bool = False,
    ) -> xr.Dataset:
        """Create a standardized dataset.

        Parameters
        ----------
        ds:
            The original dataset.
        dim_map:
            An optional mapping from the dimension names in the definition to
            the dimension names in the resulting dataset.
        variable_map:
            An optional mapping from the variable names in the definition to
            the variable names in the resulting dataset.
        keep_global_attrs:
            Whether to keep the global attributes.

        Returns
        -------
        :
            The standardized dataset.

        """
        dim_map = dim_map or {}
        variable_map = variable_map or {}
        attrs = self.attrs or {}
        original_var = ds[variable_map.get(self.name, self.name)]
        if self.dims and dim_map:
            order = [dim_map.get(d, d) for d in self.dims]
            data = original_var.transpose(*order).data
        else:
            data = original_var.data
        data = _convert_units(
            data,
            original_var.attrs.get("units"),
            attrs.get("units"),
        )
        if self.dtype is not None:
            data = data.astype(np.dtype(self.dtype))
        coords = {
            c.name: c.to_dataarray(ds, dim_map, variable_map)
            for c in self.coords
        }
        var = xr.DataArray(
            data=data,
            coords=coords,
            dims=self.dims,
            name=self.name,
            attrs=attrs,
        )
        bounds = {
            c.bounds.name: c.to_bounds_dataarray(ds, dim_map, variable_map)
            for c in self.coords
            if c.bounds is not None
        }
        result = xr.Dataset({self.name: var} | bounds, coords=coords)
        if keep_global_attrs:
            result.attrs = dict(ds.attrs)
        return result


def set_global_attrs(
    ds: xr.Dataset,
    attrs: Mapping[str, str | int | float | list[str | int | float]],
    *,
    keep_existing: Iterable[str] | bool = False,
) -> xr.Dataset:
    """Set global attributes on a dataset.

    Parameters
    ----------
    ds:
        The original dataset.
    attrs:
        The attributes to set.
    keep_existing:
        An optional list of attribute keys to keep from the original dataset.
        If `False`, none of the existing attributes are kept, if `True`, all
        existing attributes are kept.

    Returns
    -------
    :
        A copy of the original dataset, with the requested attributes set.
    """
    ds = ds.copy()
    attrs = dict(attrs)
    if keep_existing is True:
        attrs = ds.attrs | attrs
    elif isinstance(keep_existing, Iterable):
        attrs = {
            k: v for k, v in ds.attrs.items() if k in keep_existing
        } | attrs
    ds.attrs = attrs
    return ds


def set_units(
    ds: xr.Dataset,
    name: str,
    units: str,
    *,
    existing_units_are_invalid: bool = False,
) -> xr.Dataset:
    """Set the units of a variable.

    Parameters
    ----------
    ds:
        The original dataset.
    name:
        The name of the variable to set the units for.
    units:
        The new units to set.
    existing_units_are_invalid:
        If `True`, any existing units on the variable are considered invalid
        and will be overwritten. If `False`, existing units will be converted
        to the new units.

    Raises
    ------
    pint.errors.DimensionalityError:
        If the existing units cannot be converted to the new units.

    Returns
    -------
    :
        A copy of the original dataset, with the requested units set on the
        specified variable.
    """
    ds = ds.copy()
    var = ds[name]
    if "units" in var.attrs and not existing_units_are_invalid:
        var.data = _convert_units(
            var.data,
            var.attrs["units"],
            units,
        )
        var.attrs["units"] = units
    elif "units" in var.encoding and not existing_units_are_invalid:
        # TODO: what about the calendar for time units? Does pint handle it?
        # How do we pass it in?
        var.data = _convert_units(
            var.data,
            var.encoding["units"],
            units,
        )
        var.encoding["units"] = units
    else:
        var.attrs["units"] = units
    return ds


def flip_coordinate(ds: xr.Dataset, name: str) -> xr.Dataset:
    """Flip a coordinate variable.

    Parameters
    ----------
    ds:
        The original dataset.
    name:
        The name of the coordinate variable to flip.

    Returns
    -------
    :
        A copy of the original dataset, with the specified coordinate variable
        flipped.
    """
    coord = ds[name]
    dims = coord.dims
    if len(dims) != 1:
        msg = (
            f"Unable to flip coordinate '{name}' with {len(dims)} dimensions."
        )
        raise NotImplementedError(msg)
    dim = dims[0]
    reverse = slice(None, None, -1)
    ds = ds.isel({dim: reverse})
    if "bounds" in coord.attrs and coord.attrs["bounds"] in ds:
        bounds = ds[coord.attrs["bounds"]]
        bounds_dims = set(bounds.dims) - {dim}
        if len(bounds_dims) != 1:
            msg = (
                f"Coordinate {name} should have exactly 1 bounds dimension, "
                f"but dimensions {', '.join(sorted(bounds_dims))} were found."
            )
            raise ValueError(msg)
        bounds_dim = bounds_dims.pop()
        ds[coord.attrs["bounds"]] = bounds.isel({bounds_dim: reverse})

    return ds


def guess_bounds(
    ds: xr.Dataset,
    coordinates: Iterable[str],
    dim: str | Iterable[str] | None = None,
    bounds_dim: str = "bounds",
) -> xr.Dataset:
    """Guess the bounds of coordinate variables.

    Parameters
    ----------
    ds:
        The original dataset.
    coordinates:
        The names of the coordinate variables to guess the bounds for.
    dim:
        Core dimension(s) along which to estimate bounds. For 2D bounds, it can
        be a list of 2 dimension names.
    bounds_dim:
        The name of the bounds dimension to add.

    Returns
    -------
    :
        A copy of the original dataset, with the guessed bounds added.
    """
    return ds.cf.add_bounds(coordinates, dim=dim, output_dim=bounds_dim)


def add_time_bounds(
    ds: xr.Dataset,
    freq: str,
    dim: str = "time",
    bounds_dim: str = "bounds",
) -> xr.Dataset:
    """Add time bounds to a dataset.

    Parameters
    ----------
    ds:
        The original dataset.
    freq:
        The frequency string to use for generating the time bounds.
    dim:
        The name of the time dimension.
    bounds_dim:
        The name of the bounds dimension to add.

    Returns
    -------
    :
        A copy of the original dataset, with the time bounds added.
    """
    ds = ds.copy()
    coord = ds[dim]
    intervals = pd.interval_range(
        coord.data[0],
        freq=freq,
        periods=len(coord.data),
    )
    if len(intervals) != len(coord.data):
        msg = (
            f"Unable to add time bounds with frequency '{freq}' to dimension "
            f"'{dim}' because the number of generated bounds "
            f"({len(intervals)}) does not match the number of existing time "
            f"points ({len(coord.data)})."
        )
        raise ValueError(msg)
    inside = (coord.data >= intervals.left) & (coord.data <= intervals.right)
    if not inside.all():
        msg = (
            f"Unable to add time bounds with frequency '{freq}' to dimension "
            f"'{dim}' because the existing time points are not inside the "
            f"generated bounds at {coord.data[np.where(~inside)]}."
        )
        raise ValueError(msg)

    name = f"{dim}_bounds"
    bounds = xr.DataArray(
        data=[intervals.left, intervals.right],
        dims=(bounds_dim, dim),
        name=name,
    ).transpose(dim, bounds_dim)
    ds[name] = bounds
    coord.attrs["bounds"] = name
    return ds


def insert_missing_time_points(
    ds: xr.Dataset,
    freq: str,
    dim: str = "time",
) -> xr.Dataset:
    """Insert missing data along a dimension.

    Parameters
    ----------
    ds:
        The original dataset.
    freq:
        The frequency string to use for generating the complete set of time
        points. For example, "D" for daily data, "MS" for monthly data at the
        start of the month, etc. See :func:`pandas.date_range` for available
        options.
    dim:
        The name of the dimension along which to insert missing data.

    Returns
    -------
    :
        A copy of the original dataset, with missing data inserted along the
        specified dimension.
    """
    return ds.reindex(
        {
            dim: pd.date_range(
                ds[dim].data[0],
                ds[dim].data[-1],
                freq=freq,
            ),
        },
    )


def merge_dims(
    ds: xr.Dataset,
    dims: Iterable[str],
) -> xr.Dataset:
    """Merge two or more dimensions in a dataset.

    Parameters
    ----------
    ds:
        The original dataset.
    dims:
        The names of the dimensions to merge. The first dimension in the list
        will be kept.

    Returns
    -------
    :
        A copy of the original dataset, with the specified dimensions merged.
    """
    ds = ds.copy()
    dims = tuple(dims)
    sizes = [ds.sizes[dim] for dim in dims]
    if any(size != sizes[0] for size in sizes):
        msg = (
            f"Unable to merge dimensions {', '.join(dims)} with different "
            f"sizes: {', '.join(f'{dim}={ds.sizes[dim]}' for dim in dims)}."
        )
        raise ValueError(msg)

    target_dim = dims[0]
    for dim in dims[1:]:
        for var in ds.data_vars:
            if dim in ds[var].dims:
                ds[var] = ds[var].rename({dim: target_dim})
        for coord in ds.coords:
            if dim in ds[coord].dims:
                ds[coord] = ds[coord].rename({dim: target_dim})

    return ds

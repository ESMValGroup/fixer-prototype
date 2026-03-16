"""Test the `fixer.fixes` module."""

from __future__ import annotations

import cftime
import numpy as np
import pint.errors
import pytest
import xarray as xr

from fixer.fixes import (
    Coordinate,
    CoordinateBounds,
    Variable,
    flip_coordinate,
    set_global_attrs,
    set_units,
)
from fixer.protocol import FixFunction


def test_reformat() -> None:
    """Test the reformat function on a small synthetic dataset."""
    assert isinstance(set_units, FixFunction)
    ds = xr.Dataset.from_dict(
        {
            "coords": {
                "time": {
                    "dims": ("time",),
                    "attrs": {
                        "bounds": "time_bounds",
                    },
                    "data": [
                        cftime.DatetimeNoLeap(
                            1850,
                            1,
                            16,
                            12,
                            0,
                            0,
                            0,
                            has_year_zero=True,
                        ),
                    ],
                    "encoding": {
                        "units": "days since 1850-01-01 00:00:00",
                        "calendar": "noleap",
                        "dtype": "float64",
                    },
                },
                "lat": {
                    "dims": ("y",),
                    "attrs": {
                        "bounds": "lat_bounds",
                        "units": "degrees_north",
                    },
                    "data": [-87.86379883923263, -85.09652698831736],
                },
                "lon": {
                    "dims": ("x",),
                    "attrs": {
                        "bounds": "lon_bounds",
                        "units": "degrees_east",
                    },
                    "data": [0.0, 2.8125, 5.625],
                },
                "height2m": {
                    "dims": (),
                    "attrs": {
                        "units": "m",
                    },
                    "data": 2.0,
                },
            },
            "attrs": {},
            "dims": {"time": 1, "bnds": 2, "y": 2, "x": 3},
            "data_vars": {
                "time_bounds": {
                    "dims": ("time", "bounds"),
                    "attrs": {},
                    "data": [
                        [
                            cftime.DatetimeNoLeap(
                                1850,
                                1,
                                1,
                                0,
                                0,
                                0,
                                0,
                                has_year_zero=True,
                            ),
                            cftime.DatetimeNoLeap(
                                1850,
                                2,
                                1,
                                0,
                                0,
                                0,
                                0,
                                has_year_zero=True,
                            ),
                        ],
                    ],
                },
                "lat_bounds": {
                    "dims": ("y", "bounds"),
                    "attrs": {},
                    "data": [
                        [-90.0, -86.48016291377499],
                        [-86.48016291377499, -83.70471996810181],
                    ],
                },
                "lon_bounds": {
                    "dims": ("x", "bounds"),
                    "attrs": {},
                    "data": [
                        [-1.40625, 1.40625],
                        [1.40625, 4.21875],
                        [4.21875, 7.03125],
                    ],
                },
                "tas": {
                    "dims": ("time", "y", "x"),
                    "attrs": {
                        "units": "degreeC",
                    },
                    "data": [
                        [
                            [
                                24.4741973876953,
                                24.24557495117188,
                                24.0136260986328,
                            ],
                            [
                                24.81895446777344,
                                24.20892333984375,
                                24.6052703857422,
                            ],
                        ],
                    ],
                },
            },
        },
    )
    print("Original:\n", ds)
    print("Converting..")
    definition = Variable(
        name="tas",
        dtype="float32",
        dims=("time", "lat", "lon"),
        coords=(
            Coordinate(
                name="height",
                dtype=None,
                dims=None,
                attrs={
                    "standard_name": "height",
                    "long_name": "height",
                    "units": "m",
                    "axis": "Z",
                    "positive": "up",
                },
                bounds=None,
            ),
            Coordinate(
                name="time",
                dtype=None,
                dims=("time",),
                attrs={
                    "standard_name": "time",
                    "long_name": "Time Intervals",
                    "axis": "T",
                },
                bounds=CoordinateBounds(
                    name="time_bnds",
                    dims=("time", "bnds"),
                ),
            ),
            Coordinate(
                name="lat",
                dtype=None,
                dims=("lat",),
                attrs={
                    "standard_name": "latitude",
                    "long_name": "Latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                },
                bounds=CoordinateBounds(name="lat_bnds", dims=("lat", "bnds")),
            ),
            Coordinate(
                name="lon",
                dtype=None,
                dims=("lon",),
                attrs={
                    "standard_name": "longitude",
                    "long_name": "Longitude",
                    "units": "degrees_east",
                    "axis": "X",
                },
                bounds=CoordinateBounds(name="lon_bnds", dims=("lon", "bnds")),
            ),
        ),
        attrs={
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
            "cell_methods": "area: time: mean",
        },
    )
    result = definition.to_dataset(
        ds,
        dim_map={
            "lat": "y",
            "lon": "x",
            "bnds": "bounds",
        },
        variable_map={
            "height": "height2m",
            "lat_bnds": "lat_bounds",
            "lon_bnds": "lon_bounds",
            "time_bnds": "time_bounds",
        },
    )
    assert result is not ds
    print("Result:\n", result)

    # Perform some checks on the results
    assert result.sizes == {"time": 1, "lat": 2, "lon": 3, "bnds": 2}
    assert set(result.coords) == {"height", "time", "lat", "lon"}
    assert set(result.data_vars) == {
        "tas",
        "time_bnds",
        "lat_bnds",
        "lon_bnds",
    }
    assert result.tas.attrs["standard_name"] == "air_temperature"
    assert result.tas.units == "K"
    assert result.tas.dtype == np.dtype("float32")
    assert result.lat.dtype == np.dtype("float64")
    assert result.lat_bnds.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    ("keep_existing", "expected"),
    [
        (False, {"b": 20, "d": 4}),
        (True, {"a": 1, "b": 20, "c": 3, "d": 4}),
        (["a"], {"a": 1, "b": 20, "d": 4}),
    ],
)
def test_set_global_attrs(
    *,
    keep_existing: list[str] | bool,
    expected: dict[str, int],
) -> None:
    """Test the set_global_attrs function."""
    assert isinstance(set_units, FixFunction)
    ds = xr.Dataset(attrs={"a": 1, "b": 2, "c": 3})
    new_attrs = {"b": 20, "d": 4}
    result = set_global_attrs(ds, new_attrs, keep_existing=keep_existing)
    assert result is not ds
    assert result.attrs == expected


@pytest.mark.parametrize(
    (
        "existing_units",
        "existing_units_are_invalid",
        "new_units",
        "expected_value",
    ),
    [
        ("degC", False, "K", 274.15),
        ("degC", True, "K", 1.0),
    ],
)
def test_set_units(
    *,
    existing_units: str,
    existing_units_are_invalid: bool,
    new_units: str,
    expected_value: float,
) -> None:
    """Test the set_units function."""
    assert isinstance(set_units, FixFunction)
    ds = xr.Dataset({"var": ("x", [1.0])})
    ds["var"].attrs["units"] = existing_units
    result = set_units(
        ds,
        "var",
        units=new_units,
        existing_units_are_invalid=existing_units_are_invalid,
    )
    assert result is not ds
    assert result["var"].attrs["units"] == new_units
    assert result["var"].data[0] == expected_value


def test_set_units_cannot_convert() -> None:
    """Test that set_units raises an error when conversion is not possible."""
    ds = xr.Dataset({"var": ("x", [1.0])})
    ds["var"].attrs["units"] = "m"
    with pytest.raises(pint.errors.DimensionalityError):
        set_units(ds, "var", "K")


def test_flip_coordinate() -> None:
    """Test the flip_coordinate function."""
    assert isinstance(flip_coordinate, FixFunction)
    ds = xr.Dataset.from_dict(
        {
            "coords": {
                "lat": {
                    "dims": ("lat",),
                    "attrs": {
                        "bounds": "lat_bounds",
                        "units": "degrees_north",
                    },
                    "data": [
                        -89.0,
                        -87.0,
                    ],
                },
                "lon": {
                    "dims": ("lon",),
                    "attrs": {
                        "bounds": "lon_bounds",
                        "units": "degrees_east",
                    },
                    "data": [
                        0.0,
                        2.0,
                        4.0,
                    ],
                },
            },
            "attrs": {},
            "dims": {"bnds": 2, "lat": 2, "lon": 3},
            "data_vars": {
                "lat_bounds": {
                    "dims": ("lat", "bnds"),
                    "attrs": {},
                    "data": [
                        [-90.0, -88.0],
                        [-88.0, -86.0],
                    ],
                },
                "lon_bounds": {
                    "dims": ("lon", "bnds"),
                    "attrs": {},
                    "data": [
                        [-1.0, 1.0],
                        [1.0, 3.0],
                        [3.0, 5.0],
                    ],
                },
                "tas": {
                    "dims": ("lat", "lon"),
                    "attrs": {
                        "standard_name": "air_temperature",
                        "units": "K",
                    },
                    "data": [
                        [290.0, 291.0, 292.0],
                        [293.0, 294.0, 295.0],
                    ],
                },
            },
        },
    )

    result = flip_coordinate(ds, "lat")
    assert result is not ds
    # Check that the latitudes and their bounds have been flipped correctly.
    np.testing.assert_array_equal(
        result.lat.values,
        [
            -87.0,
            -89.0,
        ],
    )
    np.testing.assert_array_equal(
        result.lat_bounds.values,
        [
            [-86.0, -88.0],
            [-88.0, -90.0],
        ],
    )
    # Check that the longitude coordinate and its bounds are unchanged.
    np.testing.assert_array_equal(
        result.lon.values,
        [
            0.0,
            2.0,
            4.0,
        ],
    )
    np.testing.assert_array_equal(
        result.lon_bounds.values,
        [
            [-1.0, 1.0],
            [1.0, 3.0],
            [3.0, 5.0],
        ],
    )
    np.testing.assert_array_equal(
        result.tas.values,
        [
            [293.0, 294.0, 295.0],
            [290.0, 291.0, 292.0],
        ],
    )

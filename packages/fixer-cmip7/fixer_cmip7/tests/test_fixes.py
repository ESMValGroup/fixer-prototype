"""Test the `cmip7_fixer.fixes` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cftime
import numpy as np
import xarray as xr
from fixer.protocol import FixFunction

from fixer_cmip7.fixes import reformat

if TYPE_CHECKING:
    from pathlib import Path


def test_reformat(tmp_path: Path) -> None:
    """Test the reformat function on a small synthetic dataset."""
    assert isinstance(reformat, FixFunction)
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
    result = reformat(
        ds,
        realm="atmos",
        branded_variable="tas_tavg-h2m-hxy-u",
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
    print("Saving to NetCDF..")
    file = tmp_path / "tas_fixed.nc"
    result.to_netcdf(file)

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
    try:
        import iris  # noqa: PLC0415
        import iris.loading  # noqa: PLC0415
    except ImportError:
        pass
    else:
        cube = iris.load_cube(file)
        print(cube.summary())
        for problem in iris.loading.LOAD_PROBLEMS.problems:
            print(problem)
        assert not iris.loading.LOAD_PROBLEMS.problems

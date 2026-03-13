"""Test some stuff."""

# ruff: noqa: T201
import xarray as xr
from fixer import fix


def main() -> None:
    """CMORize a file as an example."""
    ds = xr.open_dataset(
        "~/climate_data/CMIP6/CMIP/BCC/BCC-ESM1/historical/r1i1p1f1/Amon/tas/gn/v20181214/tas_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc",
        chunks={"time": 100},
    )
    print("Original:\n", ds)
    print("Converting..")
    result = fix(
        ds,
        name="MIP-DRS7.CMIP7.CMIP.CCCma.CanESM6-MR.historical.r2i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.g13s.v20250622",
    )
    print("Result:\n", result)
    print("Saving to NetCDF..")
    result.to_netcdf("tas_fixed.nc")

    try:
        import iris
        import iris.loading
    except ImportError:
        return

    cube = iris.load_cube("tas_fixed.nc")
    print(cube.summary())
    if iris.loading.LOAD_PROBLEMS.problems:
        print("Problems:")
    for problem in iris.loading.LOAD_PROBLEMS.problems:
        print(problem)


if __name__ == "__main__":
    main()

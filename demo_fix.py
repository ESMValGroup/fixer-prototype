"""Demonstrate how to apply fixes to a dataset."""

# ruff: noqa: T201

from __future__ import annotations

from fixer import fix
from fixer_cmip7.tests.test_fixes import create_test_dataset


def main() -> None:
    """CMORize a dataset as an example."""
    ds = create_test_dataset()
    print("Original:\n", ds)
    print()

    print("Fixing..")
    result = fix(
        ds,
        name="MIP-DRS7.CMIP7.CMIP.CCCma.CanESM6-MR.historical.r2i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.g13s.v20250622",
    )
    print("Fixed dataset:\n", result)
    print("Saving to NetCDF..")
    result.to_netcdf("tas_fixed.nc")

    try:
        import iris  # noqa: PLC0415
        import iris.loading  # noqa: PLC0415
    except ImportError:
        return
    print("Loading with Iris..")
    cube = iris.load_cube("tas_fixed.nc")
    print(cube.summary())
    if iris.loading.LOAD_PROBLEMS.problems:
        print("Problems:")
    for problem in iris.loading.LOAD_PROBLEMS.problems:
        print(problem)


if __name__ == "__main__":
    main()

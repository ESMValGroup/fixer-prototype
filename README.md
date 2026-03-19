# fixer-prototype
A prototype demonstrating how to implement configurable fixes

## Layout

For demonstration purposes, this repository contains three Python packages, but the idea would be that these would live in separate repositories later on:

- [packages/fixer](packages/fixer) The package that users will import. It provides a single function `fix` that takes an xarray dataset
  and the dataset name and returns a correctly standardized dataset based on the available configuration. It also provides the infrastructure
  needed by plugins and widely applicable functions for fixing data in [`fixer.fixes`](packages/fixer/fixer/fixes.py).
- [packages/fixer-cmip7](packages/fixer-cmip7) A plugin that can be used to format CMIP7 data according to the standards set out in the CMIP7 data request.
- [packages/fixer-esa-cci](packages/fixer-esa-cci) A plugin that can be used to format [ESA-CCI data](https://xcube.readthedocs.io/en/latest/dataaccess.html#esa-climate-data-centre-esa-cci-cciodp-ccizarr-esa-cci-kc) according to the standards set out in the CMIP7 data request.

## Example usage

There is an example script, [`demo_fix.py`](demo_fix.py), that demonstrates how to use the `fix` function to fix a dataset. The script can be run with the following command:

```bash
uv run --extra dev demo_fix.py
```

Additionally, there is a script, [`demo_scan.py`](demo_scan.py), that demonstrates how to scan a dataset for problems and generate a fixer configuration. The script can be run with the following command:

```bash
uv run --extra dev demo_scan.py
```

Below is an example of using the `fix` function in Ipython:

```python
In [1]: from fixer import fix

In [2]: from fixer_cmip7.tests.test_fixes import create_test_dataset

In [3]: ds = create_test_dataset()

In [4]: ds
Out[4]:
<xarray.Dataset> Size: 200B
Dimensions:      (time: 1, y: 2, x: 3, bounds: 2)
Coordinates:
  * time         (time) object 8B 1850-01-16 12:00:00
    lat          (y) float64 16B -85.1 -87.86
    lon          (x) float64 24B 0.0 2.812 5.625
    height2m     float64 8B 2.0
Dimensions without coordinates: y, x, bounds
Data variables:
    time_bounds  (time, bounds) object 16B 1850-01-01 00:00:00 1850-02-01 00:...
    lat_bounds   (y, bounds) float64 32B -83.7 -86.48 -86.48 -90.0
    lon_bounds   (x, bounds) float64 48B -1.406 1.406 1.406 4.219 4.219 7.031
    temp         (time, y, x) float64 48B 24.47 24.25 24.01 24.82 24.21 24.61

In [5]: fix(ds, name="MIP-DRS7.CMIP7.CMIP.CCCma.CanESM6-MR.historical.r2i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.g13s.v20250622")
Out[5]:
<xarray.Dataset> Size: 176B
Dimensions:    (time: 1, lat: 2, lon: 3, bnds: 2)
Coordinates:
  * time       (time) object 8B 1850-01-16 12:00:00
  * lat        (lat) float64 16B -85.1 -87.86
  * lon        (lon) float64 24B 0.0 2.812 5.625
    height     float64 8B 2.0
Dimensions without coordinates: bnds
Data variables:
    tas        (time, lat, lon) float32 24B 297.6 297.4 297.2 298.0 297.4 297.8
    time_bnds  (time, bnds) object 16B 1850-01-01 00:00:00 1850-02-01 00:00:00
    lat_bnds   (lat, bnds) float64 32B -83.7 -86.48 -86.48 -90.0
    lon_bnds   (lon, bnds) float64 48B -1.406 1.406 1.406 4.219 4.219 7.031

```

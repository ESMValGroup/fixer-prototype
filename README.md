# fixer-prototype
A prototype demonstrating how to implement configurable fixes

## Layout

For demonstration purposes, this repository contains three Python packages, but the idea would be that these would live in separate repositories later on:

- [packages/fixer](packages/fixer) The package that users will import. It provides a single function `fix` that takes an xarray dataset
  and the dataset name and returns a correctly standardized dataset based on the available configuration. It also provides the infrastructure
  needed by plugins and widely applicable functions for fixing data in [`fixer.fixes`](packages/fixer/fixer/fixes.py).
- [packages/fixer-cmip7](packages/fixer-cmip7) A plugin that can be used to format CMIP7 data according to the standards set out in the CMIP7 data request.
- [packages/fixer-esa-cci](packages/fixer-esa-cci) A plugin that can be used to format ESA-CCI data according to the standards set out in the CMIP7 data request.

## Example usage

There is an example script, [`demo.py`](demo.py), that demonstrates how to use the `fix` function to fix a dataset. The script can be run with the following command:

```bash
uv run --extra dev demo.py
```

Below is an example using Ipython:

```python
In [1]: import xarray as xr

In [2]: from fixer import fix

In [3]: ds = xr.open_dataset(
   ...:     "~/climate_data/CMIP6/CMIP/BCC/BCC-ESM1/historical/r1i1p1f1/Amon/tas/gn/v20181214/tas_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc",
   ...:     chunks={"time": 100},
   ...: )
   ...:

In [4]: ds
Out[4]:
<xarray.Dataset> Size: 65MB
Dimensions:    (time: 1980, bnds: 2, lat: 64, lon: 128)
Coordinates:
  * time       (time) object 16kB 1850-01-16 12:00:00 ... 2014-12-16 12:00:00
  * lat        (lat) float64 512B -87.86 -85.1 -82.31 ... 82.31 85.1 87.86
  * lon        (lon) float64 1kB 0.0 2.812 5.625 8.438 ... 351.6 354.4 357.2
    height     float64 8B ...
Dimensions without coordinates: bnds
Data variables:
    time_bnds  (time, bnds) object 32kB dask.array<chunksize=(100, 2), meta=np.ndarray>
    lat_bnds   (lat, bnds) float64 1kB dask.array<chunksize=(64, 2), meta=np.ndarray>
    lon_bnds   (lon, bnds) float64 2kB dask.array<chunksize=(128, 2), meta=np.ndarray>
    tas        (time, lat, lon) float32 65MB dask.array<chunksize=(100, 64, 128), meta=np.ndarray>
Attributes: (12/49)
    Conventions:            CF-1.7 CMIP-6.2
    activity_id:            CMIP
    branch_method:          Standard
    branch_time_in_child:   0.0
    branch_time_in_parent:  2110.0
    comment:                The model integration starts from the piControl e...
    ...                     ...
    title:                  BCC-ESM1 output prepared for CMIP6
    tracking_id:            hdl:21.14100/e6e5eb74-9e45-44db-9dbe-d6236596b068
    variable_id:            tas
    variant_label:          r1i1p1f1
    license:                CMIP6 model data produced by BCC is licensed unde...
    cmor_version:           3.3.2

In [5]: result = fix(
   ...:     ds,
   ...:     name="MIP-DRS7.CMIP7.CMIP.CCCma.CanESM6-MR.historical.r2i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.g13s.v20250622",
   ...: )
   ...:

In [6]: result
Out[6]:
<xarray.Dataset> Size: 65MB
Dimensions:    (time: 1980, lat: 64, lon: 128, bnds: 2)
Coordinates:
  * time       (time) object 16kB 1850-01-16 12:00:00 ... 2014-12-16 12:00:00
  * lat        (lat) float64 512B -87.86 -85.1 -82.31 ... 82.31 85.1 87.86
  * lon        (lon) float64 1kB 0.0 2.812 5.625 8.438 ... 351.6 354.4 357.2
    height     float64 8B 2.0
Dimensions without coordinates: bnds
Data variables:
    tas        (time, lat, lon) float32 65MB dask.array<chunksize=(100, 64, 128), meta=np.ndarray>
    time_bnds  (time, bnds) object 32kB dask.array<chunksize=(100, 2), meta=np.ndarray>
    lat_bnds   (lat, bnds) float64 1kB dask.array<chunksize=(64, 2), meta=np.ndarray>
    lon_bnds   (lon, bnds) float64 2kB dask.array<chunksize=(128, 2), meta=np.ndarray>

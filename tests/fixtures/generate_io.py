"""Writes small netCDF / Zarr files with Python xarray plus the decoded
reference (`io_expected.json`) for the Rust I/O tests.

Needs: xarray numpy pandas scipy netCDF4 zarr
Usage: python tests/fixtures/generate_io.py tests/fixtures
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
from generate import enc  # noqa: E402

out = sys.argv[1]
times = pd.date_range("2000-01-01", periods=5, freq="D").astype("datetime64[ns]")
temp = np.arange(15.0).reshape(5, 3) * 1.5
temp[1, 2] = np.nan
ds = xr.Dataset(
    {
        "temp": (("time", "x"), temp, {"units": "K", "long_name": "air temperature"}),
        "precip": (("x",), np.array([0.5, np.nan, 2.0]), {"units": "mm"}),
    },
    coords={
        "time": times,
        "x": [10, 20, 30],
        "lat": ("x", [1.5, 2.5, 3.5], {"units": "degrees_north"}),
        "station": ("x", ["aa", "b", "ccc"]),
    },
    attrs={"title": "xarray-rs fixture", "version": 3, "scale": 0.5},
)

expected = {}
ds.to_netcdf(f"{out}/sample_nc3.nc", format="NETCDF3_64BIT", engine="scipy")
expected["sample_nc3.nc"] = enc(xr.open_dataset(f"{out}/sample_nc3.nc", engine="scipy").load())
enc4 = {"temp": {"zlib": True, "complevel": 4, "chunksizes": (2, 3)}}
ds.to_netcdf(f"{out}/sample_nc4.nc", format="NETCDF4", engine="netcdf4", encoding=enc4)
expected["sample_nc4.nc"] = enc(xr.open_dataset(f"{out}/sample_nc4.nc", engine="netcdf4").load())

packed = xr.Dataset({"t2m": (("y",), np.array([280.0, 281.5, np.nan, 290.25]))}, coords={"y": [0.0, 1.0, 2.0, 3.0]})
packed.to_netcdf(f"{out}/sample_packed.nc", format="NETCDF4", engine="netcdf4",
                 encoding={"t2m": {"dtype": "int16", "scale_factor": 0.25, "add_offset": 270.0, "_FillValue": -9999}})
expected["sample_packed.nc"] = enc(xr.open_dataset(f"{out}/sample_packed.nc", engine="netcdf4").load())

for fmt in (2, 3):
    path = f"{out}/sample_v{fmt}.zarr"
    shutil.rmtree(path, ignore_errors=True)
    ds.to_zarr(path, zarr_format=fmt, consolidated=False, mode="w")
    expected[f"sample_v{fmt}.zarr"] = enc(xr.open_zarr(path, consolidated=False).load())

with open(f"{out}/io_expected.json", "w") as f:
    json.dump(expected, f, allow_nan=False)
print("ok", list(expected))

# xarray-rs

[![crates.io](https://img.shields.io/crates/v/xarray-rs.svg)](https://crates.io/crates/xarray-rs)
[![docs.rs](https://docs.rs/xarray-rs/badge.svg)](https://docs.rs/xarray-rs)
[![CI](https://github.com/DaniilPak/xarray-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/DaniilPak/xarray-rs/actions/workflows/ci.yml)

Labelled N-dimensional arrays for Rust: **named dimensions, coordinate labels
and attributes** on top of [`ndarray`](https://crates.io/crates/ndarray), with
arithmetic that aligns by label and broadcasts by dimension name, NaN-aware
reductions, `groupby`, `resample`, `rolling`, `concat`/`merge`, interpolation,
an xarray-style text repr, and netCDF / Zarr / JSON I/O.

> **xarray-rs is an independent Rust project inspired by Python's
> [xarray](https://github.com/pydata/xarray) (Apache-2.0). It is not affiliated
> with, endorsed by, or maintained by the xarray developers or the pydata
> organization.** The API follows xarray's names and semantics so that xarray
> users feel at home, and its behaviour is checked against real xarray (see
> [Testing](#testing)). All credit for the data model and its design goes to
> the xarray authors.

```toml
[dependencies]
xarray-rs = "0.1"
# optional file formats:
# xarray-rs = { version = "0.1", features = ["netcdf", "zarr"] }
```

## Example

```rust
use chrono::NaiveDate;
use ndarray::array;
use xarray_rs::{DataArray, Dataset, Join, align};

let days: Vec<NaiveDate> = (1..=4).map(|d| NaiveDate::from_ymd_opt(2000, 1, d).unwrap()).collect();
let temp = DataArray::new(array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0], [7.0, 8.0]], ["time", "site"])?
    .with_coord("time", days)?
    .with_coord("site", vec!["north", "south"])?
    .with_coord("elevation", ("site", vec![120.0, 45.5]))? // non-index coordinate
    .with_name("temp")
    .with_attr("units", "degC");

// label-based and positional selection
let north = temp.sel("site", "north")?;                     // drops "site"
let early = temp.sel("time", "2000-01-01"..="2000-01-02")?;  // inclusive label slice
let first = temp.isel("time", 0)?;
assert_eq!(north.to_vec(), vec![1.0, 3.0, 5.0, 7.0]);
assert_eq!(early.sizes()["time"], 2);
assert_eq!(first.dims(), ["site"]);

// NaN-aware reductions over named dimensions
assert_eq!(temp.mean("time")?.to_vec(), vec![4.0, 16.0 / 3.0]);
assert_eq!(temp.count(..)?.item()?, 7);

// arithmetic aligns on labels (inner join) and broadcasts by name
let bias = DataArray::new(vec![0.5, 1.5, 9.9], "site")?.with_coord("site", vec!["south", "north", "east"])?;
let fixed = &temp - &bias;
assert_eq!(fixed.sel("site", "north")?.to_vec(), vec![-0.5, 1.5, 3.5, 5.5]);
let outer = align(&[&temp, &bias], Join::Outer)?;           // like xr.align(join="outer")
assert_eq!(outer[0].sizes()["site"], 3);

// groupby / resample / rolling
let by_site = temp.groupby("site")?.max()?;
let two_day = temp.resample("time", "2D")?.mean()?;
let smooth = temp.rolling("time", 3)?.center(true).min_periods(1).mean()?;
assert_eq!(two_day.sizes()["time"], 2);
assert_eq!(by_site.dims(), ["time", "site"]);
assert_eq!(smooth.shape(), temp.shape());

// datasets share coordinates between variables
let ds = Dataset::new().with_var("temp", temp.clone())?.with_attr("title", "demo");
println!("{ds}");
# Ok::<(), xarray_rs::Error>(())
```

`println!("{temp}")` prints the same layout as xarray's repr:

```text
<xarray.DataArray 'temp' (time: 4, site: 2)> Size: 64B
array([[ 1.,  2.],
       [ 3., nan],
       [ 5.,  6.],
       [ 7.,  8.]])
Coordinates:
  * time       (time) datetime64[ns] 32B 2000-01-01 2000-01-02 ... 2000-01-04
  * site       (site) <U5 40B 'north' 'south'
    elevation  (site) float64 16B 120.0 45.5
Attributes:
    units:    degC
```

### File formats

```rust,ignore
use xarray_rs::{Dataset, NetcdfFormat};

// netCDF-3 classic / 64-bit offset and NetCDF-4 (HDF5), pure Rust (feature "netcdf")
let ds = Dataset::open_netcdf("era5.nc")?;        // CF decoding: _FillValue, scale/offset, times
ds.to_netcdf("out.nc")?;                          // NetCDF-4
ds.to_netcdf_with("out3.nc", NetcdfFormat::Netcdf3_64Bit)?;

// Zarr v2 / v3 directories (feature "zarr")
let z = Dataset::open_zarr("store.zarr")?;
z.to_zarr("copy.zarr")?;                          // Zarr v3, readable by xr.open_zarr

// xarray's dict format as JSON (feature "json", on by default)
let json = ds.to_json();                          // same layout as ds.to_dict()
let back = Dataset::<f64>::from_json(&json)?;
```

Files written by xarray-rs were opened with Python xarray (netCDF4-C, scipy and
zarr-python backends) and compared with `xr.testing.assert_identical`.

## Python → Rust cheat sheet

| xarray | xarray-rs |
|---|---|
| `xr.DataArray(data, dims=..., coords=..., name=..., attrs=...)` | `DataArray::new(data, dims)?.with_coord(..)?.with_name(..).with_attr(..)` |
| `da.isel(x=0)`, `da.isel(x=slice(1, 3))`, `da.isel(x=[0, 2])` | `da.isel("x", 0)`, `da.isel("x", 1..3)`, `da.isel("x", vec![0, 2])` |
| `da.sel(x=10)`, `da.sel(x=slice(10, 30))`, `da.sel(x=[10, 30])` | `da.sel("x", 10)`, `da.sel("x", 10..=30)`, `da.sel("x", vec![10, 30])` |
| `da.sel(x=12, method="nearest", tolerance=5)` | `da.sel_method("x", 12, Method::Nearest, Some(5.0))` |
| `da.sel(time="2000-01")` | `da.sel("time", "2000-01")` |
| `da.where(da > 0)`, `da.where(cond, -1)` | `da.where_(&da.gt(0.0))`, `da.where_or(&cond, -1.0)` |
| `da.mean("x")`, `da.sum(["x", "y"])`, `da.max()` | `da.mean("x")`, `da.sum(["x", "y"])`, `da.max(..)` |
| `da.std("x", ddof=1)`, `da.sum("x", skipna=False)` | `da.std_ddof("x", 1)`, `da.sum_noskip("x")` |
| `a + b`, `a * 2`, `xr.align(a, b, join="outer")` | `&a + &b` (or `a.add(&b)?`), `&a * 2.0`, `align(&[&a, &b], Join::Outer)` |
| `da.groupby("time.month").mean()` | `da.groupby("time.month")?.mean()` |
| `da.groupby("time.month") - clim` | `da.groupby("time.month")?.sub(&clim)` |
| `da.resample(time="1D").sum()` | `da.resample("time", "1D")?.sum()` |
| `da.rolling(time=3, center=True).mean()` | `da.rolling("time", 3)?.center(true).mean()` |
| `xr.concat([a, b], dim="x")`, `xr.merge([a, b])` | `concat(&[&a, &b], "x")`, `merge(&[&a, &b])` |
| `da.interp(x=[1.5, 2.5])` | `da.interp("x", vec![1.5, 2.5])` |
| `ds["temp"]`, `ds.mean("time")` | `ds.get("temp")?`, `ds.mean("time")` |
| `xr.open_dataset(p)`, `ds.to_netcdf(p)` | `Dataset::open_netcdf(p)`, `ds.to_netcdf(p)` |
| `xr.open_zarr(p)`, `ds.to_zarr(p)` | `Dataset::open_zarr(p)`, `ds.to_zarr(p)` |

Most methods return `Result`; operators (`+ - * / %`) panic on alignment
errors, like `ndarray` does on shape mismatches — use the method forms
(`a.add(&b)?`) to get a `Result`.

## Parity with xarray

✅ supported and tested against xarray · 🟡 partial · ❌ not yet

| Area | Feature | Status | Notes |
|---|---|---|---|
| Data model | `DataArray`: named dims, index + non-index (N-D) coords, scalar coords, attrs, name | ✅ | generic over the element type (`f64`, `f32`, ints, `bool`, `String`) |
| | `Dataset`: variables sharing dims/coords, attrs | ✅ | all variables share one element type `T` |
| | Coordinate types: int, float, string, bool, datetime (`datetime64[ns]`) | ✅ | timedelta and cftime calendars ❌ |
| | `MultiIndex`, custom indexes, dask / lazy arrays | ❌ | everything is in memory |
| Indexing | `isel` (int, slice with negative step, list) | ✅ | |
| | `sel` exact / list / inclusive slice / descending index | ✅ | |
| | `sel(method="nearest" \| "pad" \| "backfill", tolerance=...)` | ✅ | |
| | partial datetime strings (`"2000-01"`) | ✅ | |
| | vectorized (pointwise) indexing with `DataArray` indexers | ❌ | orthogonal indexing only |
| | `where`, `fillna`, `dropna`, `isnull`/`notnull`, `clip`, `reindex`, `sortby`, `shift`, `roll`, `diff` | ✅ | |
| | `ffill`, `bfill`, `interpolate_na` (linear) | ✅ | |
| Reshaping | `transpose` (with `...`), `squeeze`, `expand_dims`, `rename`, `swap_dims`, `broadcast_like` | ✅ | `stack`/`unstack` ❌ |
| Arithmetic | element-wise ops with alignment (inner join) and broadcasting by name | ✅ | attrs merged with `drop_conflicts`, conflicting non-index coords dropped |
| | `xr.align` (`inner`, `outer`, `left`, `right`, `exact`, `override`), `xr.broadcast` | ✅ | |
| | math (`abs`, `sqrt`, `exp`, `log`, trig, `round`, ...), comparisons, boolean masks | ✅ | |
| | `xr.dot`, `apply_ufunc`, `polyfit` | ❌ | use `zip_with` / `reduce` / `map_along` for custom kernels |
| Reductions | `sum mean min max std var median prod count quantile` over one, several or all dims | ✅ | NaN-skipping by default; `ddof`, `min_count`, `skipna=False` variants |
| | `argmin argmax idxmin idxmax`, `any all` | ✅ | |
| | `cumsum`, `cumprod` | ✅ | |
| GroupBy | `groupby(coord)` with all reductions, `map`, group arithmetic | ✅ | |
| | virtual datetime keys (`time.month`, `time.season`, `time.dayofweek`, ...) | ✅ | |
| | `groupby_bins`, multi-key groupby | ❌ | |
| Resample | `resample` fixed (`s min h D`), `W`, `MS ME QS QE YS YE` with multiples | ✅ | pandas bin edges/labels; empty bins give NaN |
| | `resample(...).interpolate()`, custom `origin`/`offset` | ❌ | |
| Rolling | `rolling(dim, window, center, min_periods)` with all reductions, `construct` | ✅ | `rolling_exp`, `coarsen` ❌ |
| Combining | `concat` (existing or new dim), `merge` (`no_conflicts`), `to_dataarray` | ✅ | `combine_by_coords` ❌ |
| Interpolation | `interp` linear / nearest on numeric and datetime indexes, `interp_like` | ✅ | one dimension per call, no extrapolation |
| I/O | netCDF-3 / NetCDF-4 read + write with CF decoding (feature `netcdf`) | ✅ | pure Rust; groups and partial (lazy) reads ❌ |
| | Zarr v2 read, Zarr v3 read + write (feature `zarr`) | ✅ | local filesystem; cloud stores ❌ |
| | `to_dict` / `from_dict` JSON (feature `json`, default) | ✅ | |
| | pandas / Arrow / Polars conversion | ❌ | planned |
| Display | xarray-style `repr` for `DataArray` and `Dataset` | ✅ | byte-identical to xarray for the tested cases; no HTML repr |

## Testing

`tests/fixtures/generate.py` builds inputs with real Python **xarray 2026.9.0**
(numpy 2.5, pandas 3.1) and records the results of **360 operations** —
indexing, alignment joins, every reduction, groupby, resample with 16
frequencies, 146 rolling-window variants, concat/merge, interpolation and
dataset ops — plus 11 reprs. `tests/parity.rs` replays each case in Rust and
compares dims, values (NaN-aware, 1e-9 relative), coordinates and names.
`tests/fixtures/generate_io.py` writes netCDF-3, NetCDF-4 (zlib, chunked,
packed `int16`) and Zarr v2/v3 stores with xarray that the I/O tests read back.

```sh
cargo test --all-features
# regenerate fixtures (needs a venv with xarray numpy pandas scipy netCDF4 zarr bottleneck)
python tests/fixtures/generate.py tests/fixtures/xarray_cases.json
python tests/fixtures/generate_io.py tests/fixtures
```

## Minimum supported Rust version

Rust 1.91 (edition 2024).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

This crate contains no code from xarray; its behaviour is modelled on xarray,
which is © xarray Developers and licensed under the Apache License 2.0.

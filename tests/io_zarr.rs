//! Zarr reading (stores written by Python xarray) and round trips.
#![cfg(all(feature = "zarr", feature = "json"))]

use serde_json::Value;
use xarray_rs::*;

mod common;
use common::*;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn check_full(id: &str, ds: &Dataset<f64>, exp: &Value) {
    check_ds(id, ds, exp);
    let e = Dataset::<f64>::from_dict(exp).unwrap();
    assert_eq!(ds.attrs(), e.attrs(), "{id}: global attrs");
    for (k, c) in e.coords() {
        let a = ds.coord(k).unwrap_or_else(|_| panic!("{id}: missing coord {k}"));
        assert!(labels_equal(a, c), "{id}: coord {k}");
        assert_eq!(a.attrs(), c.attrs(), "{id}: attrs of coord {k}");
        assert_eq!(a.data().dtype(), c.data().dtype(), "{id}: dtype of coord {k}");
    }
    for name in e.data_vars() {
        assert_eq!(ds.get(name).unwrap().attrs(), e.get(name).unwrap().attrs(), "{id}: attrs of {name}");
    }
}

#[test]
fn reads_stores_written_by_xarray() {
    let exp: Value = serde_json::from_str(include_str!("fixtures/io_expected.json")).unwrap();
    for f in ["sample_v2.zarr", "sample_v3.zarr"] {
        let ds = Dataset::open_zarr(fixture(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
        check_full(f, &ds, &exp[f]);
    }
}

#[test]
fn round_trip() {
    let ds = Dataset::open_zarr(fixture("sample_v3.zarr")).unwrap();
    let dir = std::env::temp_dir().join(format!("xarray-rs-zarr-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ds.to_zarr_chunked(&dir, Some(2)).unwrap();
    let back = Dataset::open_zarr(&dir).unwrap();
    check_full("round_trip", &back, &ds.to_dict());
    std::fs::remove_dir_all(&dir).ok();
}

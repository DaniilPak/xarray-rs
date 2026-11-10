//! netCDF reading (files written by Python xarray) and round trips.
#![cfg(all(feature = "netcdf", feature = "json"))]

use serde_json::Value;
use xarray_rs::*;

mod common;
use common::*;

fn expected() -> Value {
    serde_json::from_str(include_str!("fixtures/io_expected.json")).unwrap()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn check_full(id: &str, ds: &Dataset<f64>, exp: &Value) {
    check_ds(id, ds, exp);
    let e = Dataset::<f64>::from_dict(exp).unwrap();
    assert_eq!(ds.attrs(), e.attrs(), "{id}: global attrs");
    for (k, c) in e.coords() {
        let a = ds.coord(k).unwrap_or_else(|_| panic!("{id}: missing coord {k}"));
        assert!(labels_equal(a, c), "{id}: coord {k}: {:?} vs {:?}", a.labels(), c.labels());
        assert_eq!(a.attrs(), c.attrs(), "{id}: attrs of coord {k}");
        assert_eq!(a.data().dtype(), c.data().dtype(), "{id}: dtype of coord {k}");
    }
    for name in e.data_vars() {
        assert_eq!(ds.get(name).unwrap().attrs(), e.get(name).unwrap().attrs(), "{id}: attrs of {name}");
    }
}

#[test]
fn reads_files_written_by_xarray() {
    let exp = expected();
    for f in ["sample_nc3.nc", "sample_nc4.nc", "sample_packed.nc"] {
        let ds = Dataset::open_netcdf(fixture(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
        check_full(f, &ds, &exp[f]);
    }
}

#[test]
fn round_trip_all_formats() {
    let ds = Dataset::open_netcdf(fixture("sample_nc4.nc")).unwrap();
    for fmt in
        [NetcdfFormat::Netcdf4, NetcdfFormat::Netcdf4Classic, NetcdfFormat::Netcdf3_64Bit, NetcdfFormat::Netcdf3Classic]
    {
        let bytes = ds.to_netcdf_bytes(fmt).unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
        let back = Dataset::from_netcdf_bytes(&bytes).unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
        check_full(&format!("{fmt:?}"), &back, &ds.to_dict());
    }
}

#[test]
fn dataarray_round_trip() {
    let dir = std::env::temp_dir().join(format!("xarray-rs-nc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("da.nc");
    let da = DataArray::new(vec![1.0, f64::NAN, 3.0], "t")
        .unwrap()
        .with_coord(
            "t",
            vec![chrono::NaiveDate::from_ymd_opt(2001, 1, 1).unwrap().and_hms_opt(6, 0, 0).unwrap(); 1]
                .into_iter()
                .chain([
                    chrono::NaiveDate::from_ymd_opt(2001, 1, 1).unwrap().and_hms_opt(12, 0, 0).unwrap(),
                    chrono::NaiveDate::from_ymd_opt(2001, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                ])
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .with_attr("units", "m");
    da.to_netcdf(&path).unwrap();
    let back = DataArray::open_netcdf(&path).unwrap();
    assert!(back.identical(&da), "{back}\n{da}");
    std::fs::remove_dir_all(&dir).ok();
}

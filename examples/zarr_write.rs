//! Writes a small dataset as a Zarr v3 store.
//!
//! cargo run --example zarr_write --features zarr -- out.zarr

use xarray_rs::{DataArray, Dataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).ok_or("output path")?;
    let times: Vec<chrono::NaiveDateTime> = (0..4)
        .map(|d| chrono::NaiveDate::from_ymd_opt(2024, 1, 1 + d).unwrap().and_hms_opt(0, 0, 0).unwrap())
        .collect();
    let temp =
        DataArray::new(ndarray::Array2::from_shape_fn((4, 2), |(i, j)| i as f64 + 0.5 * j as f64), ["time", "site"])?
            .with_coord("time", times)?
            .with_coord("site", vec!["north", "south"])?
            .with_attr("units", "degC");
    let ds = Dataset::new().with_var("temp", temp)?.with_attr("source", "xarray-rs");
    ds.to_zarr(&out)?;
    println!("{}", Dataset::open_zarr(&out)?);
    Ok(())
}

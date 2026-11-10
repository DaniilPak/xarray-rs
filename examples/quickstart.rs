//! Tour of the main features: cargo run --example quickstart

use chrono::NaiveDate;
use ndarray::Array2;
use xarray_rs::{DataArray, Dataset, Join, align};

fn main() -> Result<(), xarray_rs::Error> {
    // 60 days x 3 stations of synthetic temperatures
    let days: Vec<NaiveDate> =
        (0..60).map(|d| NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + chrono::Duration::days(d)).collect();
    let data = Array2::from_shape_fn((60, 3), |(t, s)| 10.0 + s as f64 + (t as f64 / 9.5).sin() * 5.0);
    let temp = DataArray::new(data, ["time", "station"])?
        .with_coord("time", days)?
        .with_coord("station", vec!["oslo", "rome", "lima"])?
        .with_coord("elevation", ("station", vec![23.0, 21.0, 154.0]))?
        .with_name("temp")
        .with_attr("units", "degC");
    println!("{temp}\n");

    // label-based selection, including partial datetime strings
    let feb = temp.sel("time", "2024-02")?.sel("station", "rome")?;
    println!("Rome in February: mean {:.2}\n", feb.mean("time")?.item()?);

    // reductions over named dimensions, groupby and resample
    println!("{}\n", temp.mean("time")?);
    println!("{}\n", temp.groupby("time.month")?.max()?);
    println!("{}\n", temp.resample("time", "W")?.mean()?.isel("time", 0..3)?);

    // rolling windows
    let smooth = temp.rolling("time", 7)?.center(true).mean()?;
    println!("7-day centered mean, first valid: {:.3}\n", smooth.isel("time", 3)?.isel("station", 0)?.item()?);

    // arithmetic aligns by labels and broadcasts by dimension name
    let offset = DataArray::new(vec![0.5, -0.5], "station")?.with_coord("station", vec!["rome", "oslo"])?;
    let adjusted = &temp + &offset; // inner join: only rome and oslo remain
    println!("{}\n", adjusted.isel("time", 0)?);
    let outer = align(&[&temp.isel("time", 0)?, &offset], Join::Outer)?;
    println!("outer-aligned stations: {:?}\n", outer[1].to_vec());

    // datasets
    let ds = Dataset::new()
        .with_var("temp", temp.clone())?
        .with_var("anomaly", temp.groupby("time.month")?.map(|g| g.sub(&g.mean("time")?))?)?;
    println!("{}", ds.isel("time", 0..5)?);
    Ok(())
}

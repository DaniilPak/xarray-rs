//! Reads a netCDF file and writes it back in every supported format.
//!
//! cargo run --example netcdf_convert --features netcdf -- input.nc out_dir

use xarray_rs::{Dataset, NetcdfFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (input, out) = (args.get(1).ok_or("input path")?, args.get(2).ok_or("output dir")?);
    let ds = Dataset::open_netcdf(input)?;
    println!("{ds}");
    for (fmt, name) in [
        (NetcdfFormat::Netcdf4, "nc4.nc"),
        (NetcdfFormat::Netcdf4Classic, "nc4classic.nc"),
        (NetcdfFormat::Netcdf3_64Bit, "nc3_64bit.nc"),
        (NetcdfFormat::Netcdf3Classic, "nc3classic.nc"),
    ] {
        ds.to_netcdf_with(format!("{out}/{name}"), fmt)?;
    }
    Ok(())
}

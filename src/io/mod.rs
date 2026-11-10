//! Serialization and file formats (behind cargo features).

#[cfg(feature = "json")]
pub mod json;
#[cfg(feature = "netcdf")]
pub mod netcdf;
#[cfg(feature = "zarr")]
pub mod zarr;

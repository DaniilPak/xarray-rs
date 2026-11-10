#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod align;
mod attrs;
mod combine;
mod coords;
mod dataarray;
mod dataset;
mod element;
mod error;
mod groupby;
mod indexing;
mod interp;
pub mod io;
pub mod nanops;
mod ops;
mod reduce;
mod repr;
mod rolling;
pub mod time;

pub use align::{Join, align, broadcast};
pub use attrs::{AttrValue, Attrs};
pub use combine::{concat, concat_with_index, merge};
pub use coords::{Coord, CoordData, CoordValue, IntoDims, Label, Method};
pub use dataarray::{DataArray, IntoArrayD, IntoCoord};
pub use dataset::Dataset;
pub use element::{Element, FloatElement};
pub use error::{Error, Result};
pub use groupby::GroupBy;
pub use indexing::{IndexSel, Sel};
pub use interp::InterpMethod;
#[cfg(feature = "json")]
pub use io::json::JsonElement;
#[cfg(feature = "netcdf")]
pub use io::netcdf::NetcdfFormat;
pub use ops::NumOps;
pub use rolling::Rolling;

/// Re-export of the `ndarray` version used by this crate.
pub use ndarray;

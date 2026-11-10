//! netCDF I/O (classic CDF-1/2/5 and NetCDF-4/HDF5) using the pure-Rust
//! [`netcdf-reader`](https://crates.io/crates/netcdf-reader) and
//! [`netcdf-writer`](https://crates.io/crates/netcdf-writer) crates.
//!
//! Reading follows xarray's default CF decoding: `_FillValue` /
//! `missing_value` become NaN, `scale_factor` / `add_offset` are applied, and
//! coordinates with `units = "<unit> since <date>"` (standard calendars) are
//! decoded to datetimes. Variables named after their only dimension become
//! index coordinates; names listed in a `coordinates` attribute become
//! non-index coordinates.

use crate::attrs::{AttrValue, Attrs};
use crate::coords::{Coord, CoordData};
use crate::dataarray::DataArray;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::time;
use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use netcdf_reader::{NcAttrValue, NcFile, NcType};
use netcdf_writer::{NcFileBuilder, NcWriteFormat, NcWriteOptions};
use std::path::Path;

/// On-disk format used by [`Dataset::to_netcdf_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NetcdfFormat {
    /// NetCDF-4 (HDF5-based), the default like xarray's.
    #[default]
    Netcdf4,
    /// NetCDF-4 restricted to the classic data model.
    Netcdf4Classic,
    /// NetCDF-3 64-bit offset (CDF-2).
    Netcdf3_64Bit,
    /// NetCDF-3 classic (CDF-1).
    Netcdf3Classic,
}

impl NetcdfFormat {
    fn is_classic_model(self) -> bool {
        !matches!(self, NetcdfFormat::Netcdf4)
    }
}

fn nc_err(e: impl std::fmt::Display) -> Error {
    Error::Io(format!("netCDF: {e}"))
}

const ENCODING_ATTRS: &[&str] =
    &["_FillValue", "missing_value", "scale_factor", "add_offset", "_Encoding", "_Unsigned", "coordinates"];

fn attr_from_nc(v: &NcAttrValue) -> Option<AttrValue> {
    fn num<T: Copy>(v: &[T], f: impl Fn(T) -> AttrValue, list: impl Fn(&[T]) -> AttrValue) -> Option<AttrValue> {
        match v.len() {
            0 => None,
            1 => Some(f(v[0])),
            _ => Some(list(v)),
        }
    }
    let ints = |v: &[i64]| AttrValue::IntList(v.to_vec());
    match v {
        NcAttrValue::Chars(s) => Some(AttrValue::Str(s.trim_end_matches('\0').to_string())),
        NcAttrValue::Strings(s) if s.len() == 1 => Some(AttrValue::Str(s[0].clone())),
        NcAttrValue::Strings(s) => Some(AttrValue::StrList(s.clone())),
        NcAttrValue::Bytes(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::UBytes(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::Shorts(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::UShorts(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::Ints(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::UInts(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::Int64s(v) => num(v, AttrValue::Int, ints),
        NcAttrValue::UInt64s(v) => num(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), AttrValue::Int, ints),
        NcAttrValue::Floats(v) => num(&v.iter().map(|&x| x as f64).collect::<Vec<_>>(), AttrValue::Float, |l| {
            AttrValue::FloatList(l.to_vec())
        }),
        NcAttrValue::Doubles(v) => num(v, AttrValue::Float, |l| AttrValue::FloatList(l.to_vec())),
        NcAttrValue::UserDefined { .. } => None,
    }
}

fn attrs_from_nc(list: &[netcdf_reader::NcAttribute], skip_encoding: bool) -> Attrs {
    let mut out = Attrs::new();
    for a in list {
        if skip_encoding && ENCODING_ATTRS.contains(&a.name.as_str()) {
            continue;
        }
        if a.name.starts_with("_Netcdf")
            || a.name == "DIMENSION_LIST"
            || a.name == "REFERENCE_LIST"
            || a.name == "CLASS"
            || a.name == "NAME"
        {
            continue;
        }
        if let Some(v) = attr_from_nc(&a.value) {
            out.insert(a.name.clone(), v);
        }
    }
    out
}

/// Parses CF time units like `"days since 2000-01-01"`; returns
/// (nanoseconds per unit, reference timestamp in ns).
fn parse_time_units(units: &str) -> Option<(f64, i64)> {
    let (unit, since) = units.split_once(" since ")?;
    let per = match unit.trim().to_ascii_lowercase().as_str() {
        "days" | "day" | "d" => 86_400e9,
        "hours" | "hour" | "h" | "hr" | "hrs" => 3_600e9,
        "minutes" | "minute" | "min" | "mins" => 60e9,
        "seconds" | "second" | "s" | "sec" | "secs" => 1e9,
        "milliseconds" | "millisecond" | "ms" => 1e6,
        "microseconds" | "microsecond" | "us" => 1e3,
        "nanoseconds" | "nanosecond" | "ns" => 1.0,
        _ => return None,
    };
    let s = since.trim();
    // strip a trailing timezone like "+00:00" or " UTC"
    let s = s.trim_end_matches(" UTC").trim_end_matches("+00:00").trim_end_matches('Z');
    let reference = time::parse_datetime(s).ok()?;
    Some((per, time::to_ns(reference)))
}

fn is_std_calendar(attrs: &[netcdf_reader::NcAttribute]) -> bool {
    match attrs.iter().find(|a| a.name == "calendar").and_then(|a| a.value.as_string()) {
        None => true,
        Some(c) => matches!(c.to_ascii_lowercase().as_str(), "standard" | "gregorian" | "proleptic_gregorian"),
    }
}

fn is_integer(t: &NcType) -> bool {
    matches!(
        t,
        NcType::Byte
            | NcType::Short
            | NcType::Int
            | NcType::UByte
            | NcType::UShort
            | NcType::UInt
            | NcType::Int64
            | NcType::UInt64
    )
}

impl Dataset<f64> {
    /// Opens a netCDF file (classic or NetCDF-4), like `xr.open_dataset`.
    /// All data variables are decoded to `f64`.
    pub fn open_netcdf(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_netcdf_bytes(&bytes)
    }

    /// Reads a netCDF file from memory.
    pub fn from_netcdf_bytes(bytes: &[u8]) -> Result<Self> {
        let file = NcFile::from_bytes(bytes).map_err(nc_err)?;
        let vars = file.variables().map_err(nc_err)?;
        let dim_names: Vec<String> = file.dimensions().map_err(nc_err)?.iter().map(|d| d.name.clone()).collect();
        let mut coord_names: Vec<String> = Vec::new();
        for v in vars {
            if let Some(a) = v.attributes.iter().find(|a| a.name == "coordinates").and_then(|a| a.value.as_string()) {
                coord_names.extend(a.split_whitespace().map(String::from));
            }
        }
        let mut ds = Dataset::new();
        let mut coords: IndexMap<String, Coord> = IndexMap::new();
        for v in vars {
            let name = v.name.clone();
            let mut dims: Vec<String> = v.dimensions.iter().map(|d| d.name.clone()).collect();
            let is_index = dims.len() == 1 && dims[0] == name;
            let is_coord = is_index || coord_names.contains(&name);
            let attrs_skip = attrs_from_nc(&v.attributes, true);
            let is_string = matches!(v.dtype, NcType::Char | NcType::String);
            if is_string {
                let strings = file.read_variable_as_strings(&name).map_err(nc_err)?;
                if v.dtype == NcType::Char && !dims.is_empty() {
                    dims.pop();
                }
                let shape: Vec<usize> = v.dimensions.iter().take(dims.len()).map(|d| d.size as usize).collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), strings).map_err(|e| Error::Io(e.to_string()))?;
                let mut c = Coord::new(dims, CoordData::Str(arr))?;
                c.attrs = attrs_skip;
                coords.insert(name, c);
                continue;
            }
            let data = file.read_variable_unpacked_masked(&name).map_err(nc_err)?;
            let units = v.attributes.iter().find(|a| a.name == "units").and_then(|a| a.value.as_string());
            let time_units = units.as_deref().and_then(parse_time_units).filter(|_| is_std_calendar(&v.attributes));
            if is_coord {
                let mut attrs = attrs_skip;
                let cd = if let Some((per, reference)) = time_units {
                    attrs.shift_remove("units");
                    attrs.shift_remove("calendar");
                    CoordData::DateTime(
                        data.mapv(|x| if x.is_nan() { time::NAT } else { reference + (x * per).round() as i64 }),
                    )
                } else if is_integer(&v.dtype)
                    && !data.iter().any(|x| x.is_nan())
                    && !v.attributes.iter().any(|a| a.name == "scale_factor" || a.name == "add_offset")
                {
                    CoordData::Int(data.mapv(|x| x as i64))
                } else {
                    CoordData::Float(data)
                };
                let mut c = Coord::new(dims, cd)?;
                c.attrs = attrs;
                coords.insert(name, c);
            } else {
                let mut attrs = attrs_skip;
                if let Some(true) = attrs.get("dtype").and_then(|d| d.as_str()).map(|d| d == "bool") {
                    attrs.shift_remove("dtype");
                }
                let da = DataArray::new(data, dims)?.with_attrs(attrs);
                ds.insert(name, da)?;
            }
        }
        // order coordinates: index coords in dimension order, then others
        let mut ordered: IndexMap<String, Coord> = IndexMap::new();
        for d in &dim_names {
            if let Some(c) = coords.shift_remove(d) {
                ordered.insert(d.clone(), c);
            }
        }
        ordered.extend(coords);
        for (k, c) in ordered {
            ds.assign_coord(k, c)?;
        }
        ds.attrs = attrs_from_nc(file.global_attributes().map_err(nc_err)?, false);
        Ok(ds)
    }

    /// Writes a NetCDF-4 file (`ds.to_netcdf(path)`).
    pub fn to_netcdf(&self, path: impl AsRef<Path>) -> Result<()> {
        self.to_netcdf_with(path, NetcdfFormat::Netcdf4)
    }

    /// Writes a netCDF file in the given format.
    pub fn to_netcdf_with(&self, path: impl AsRef<Path>, format: NetcdfFormat) -> Result<()> {
        std::fs::write(path, self.to_netcdf_bytes(format)?)?;
        Ok(())
    }

    /// Encodes the dataset as netCDF bytes.
    pub fn to_netcdf_bytes(&self, format: NetcdfFormat) -> Result<Vec<u8>> {
        let classic = format.is_classic_model();
        let mut b = NcFileBuilder::new();
        let mut dim_ids = IndexMap::new();
        for (d, n) in self.dims() {
            dim_ids.insert(d.clone(), b.add_dimension(d.clone(), n as u64).map_err(nc_err)?);
        }
        let dims_of = |dims: &[String],
                       ids: &IndexMap<String, netcdf_writer::DimensionId>|
         -> Vec<netcdf_writer::DimensionId> { dims.iter().map(|d| ids[d]).collect() };
        // coordinates
        for (name, c) in &self.coords {
            let ids = dims_of(&c.dims, &dim_ids);
            let mut extra: Vec<(String, NcAttrValue)> = Vec::new();
            let var = match &c.data {
                CoordData::Float(a) => {
                    let id = b.add_variable::<f64>(name.clone(), &ids).map_err(nc_err)?;
                    if a.iter().any(|v| v.is_nan()) {
                        b.set_variable_fill_value(id, f64::NAN).map_err(nc_err)?;
                    }
                    b.write_variable(id, &a.iter().copied().collect::<Vec<_>>()).map_err(nc_err)?;
                    id
                }
                CoordData::Int(a) => write_ints(&mut b, name, &ids, a.iter().copied().collect(), classic)?,
                CoordData::Bool(a) => {
                    let id = b.add_variable::<i8>(name.clone(), &ids).map_err(nc_err)?;
                    b.write_variable(id, &a.iter().map(|&v| v as i8).collect::<Vec<_>>()).map_err(nc_err)?;
                    extra.push(("dtype".into(), NcAttrValue::Chars("bool".into())));
                    id
                }
                CoordData::DateTime(a) => {
                    let vals: Vec<i64> = a.iter().copied().collect();
                    let (units, encoded) = encode_times(&vals);
                    extra.push(("units".into(), NcAttrValue::Chars(units)));
                    extra.push(("calendar".into(), NcAttrValue::Chars("proleptic_gregorian".into())));
                    if encoded.iter().any(|v| v.is_none()) {
                        let id = b.add_variable::<f64>(name.clone(), &ids).map_err(nc_err)?;
                        b.set_variable_fill_value(id, f64::NAN).map_err(nc_err)?;
                        b.write_variable(
                            id,
                            &encoded.iter().map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN)).collect::<Vec<_>>(),
                        )
                        .map_err(nc_err)?;
                        id
                    } else {
                        write_ints(&mut b, name, &ids, encoded.into_iter().map(Option::unwrap).collect(), classic)?
                    }
                }
                CoordData::Str(a) => {
                    let strings: Vec<String> = a.iter().cloned().collect();
                    if classic {
                        let width = strings.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
                        let sdim = format!("string{width}");
                        let sid = match dim_ids.get(&sdim) {
                            Some(id) => *id,
                            None => {
                                let id = b.add_dimension(sdim.clone(), width as u64).map_err(nc_err)?;
                                dim_ids.insert(sdim.clone(), id);
                                id
                            }
                        };
                        let mut ids = ids.clone();
                        ids.push(sid);
                        let id = b.add_char_variable(name.clone(), &ids).map_err(nc_err)?;
                        b.write_char_variable_strings(id, &strings).map_err(nc_err)?;
                        extra.push(("_Encoding".into(), NcAttrValue::Chars("utf-8".into())));
                        id
                    } else {
                        let id = b.add_string_variable(name.clone(), &ids).map_err(nc_err)?;
                        b.write_string_variable(id, &strings).map_err(nc_err)?;
                        id
                    }
                }
            };
            for (k, v) in attrs_to_nc(&c.attrs, classic)?.into_iter().chain(extra) {
                b.add_variable_attribute(var, k, v).map_err(nc_err)?;
            }
        }
        // data variables
        for (name, v) in &self.vars {
            let ids = dims_of(&v.dims, &dim_ids);
            let id = b.add_variable::<f64>(name.clone(), &ids).map_err(nc_err)?;
            b.set_variable_fill_value(id, f64::NAN).map_err(nc_err)?;
            b.write_variable(id, &v.data.iter().copied().collect::<Vec<_>>()).map_err(nc_err)?;
            for (k, a) in attrs_to_nc(&v.attrs, classic)? {
                b.add_variable_attribute(id, k, a).map_err(nc_err)?;
            }
            let non_index: Vec<&str> = self
                .coords
                .iter()
                .filter(|(k, c)| !(c.dims.len() == 1 && &c.dims[0] == *k) && c.dims.iter().all(|d| v.dims.contains(d)))
                .map(|(k, _)| k.as_str())
                .collect();
            if !non_index.is_empty() {
                b.add_variable_attribute(id, "coordinates", NcAttrValue::Chars(non_index.join(" "))).map_err(nc_err)?;
            }
        }
        for (k, a) in attrs_to_nc(&self.attrs, classic)? {
            b.add_attribute(k, a).map_err(nc_err)?;
        }
        let fmt = match format {
            NetcdfFormat::Netcdf4 => NcWriteFormat::Nc4,
            NetcdfFormat::Netcdf4Classic => NcWriteFormat::Nc4Classic,
            NetcdfFormat::Netcdf3_64Bit => NcWriteFormat::Offset64,
            NetcdfFormat::Netcdf3Classic => NcWriteFormat::Classic,
        };
        let (_, bytes) = b.to_vec(NcWriteOptions { format: fmt }).map_err(nc_err)?;
        Ok(bytes)
    }
}

fn write_ints(
    b: &mut NcFileBuilder,
    name: &str,
    ids: &[netcdf_writer::DimensionId],
    vals: Vec<i64>,
    classic: bool,
) -> Result<netcdf_writer::VariableId> {
    if !classic {
        let id = b.add_variable::<i64>(name.to_string(), ids).map_err(nc_err)?;
        b.write_variable(id, &vals).map_err(nc_err)?;
        return Ok(id);
    }
    if vals.iter().all(|&v| i32::try_from(v).is_ok()) {
        let id = b.add_variable::<i32>(name.to_string(), ids).map_err(nc_err)?;
        b.write_variable(id, &vals.iter().map(|&v| v as i32).collect::<Vec<_>>()).map_err(nc_err)?;
        Ok(id)
    } else {
        let id = b.add_variable::<f64>(name.to_string(), ids).map_err(nc_err)?;
        b.write_variable(id, &vals.iter().map(|&v| v as f64).collect::<Vec<_>>()).map_err(nc_err)?;
        Ok(id)
    }
}

/// Chooses CF units (coarsest exact unit, reference = first valid time) like
/// xarray's encoder.
fn encode_times(vals: &[i64]) -> (String, Vec<Option<i64>>) {
    let valid: Vec<i64> = vals.iter().copied().filter(|&v| v != time::NAT).collect();
    let reference = valid.iter().copied().min().unwrap_or(0);
    let units: [(&str, i64); 5] = [
        ("days", 86_400_000_000_000),
        ("hours", 3_600_000_000_000),
        ("minutes", 60_000_000_000),
        ("seconds", 1_000_000_000),
        ("nanoseconds", 1),
    ];
    let (uname, per) = units.iter().copied().find(|(_, per)| valid.iter().all(|v| (v - reference) % per == 0)).unwrap();
    let ref_str = match time::from_ns(reference) {
        Some(dt) if reference.rem_euclid(86_400_000_000_000) == 0 => dt.format("%Y-%m-%d").to_string(),
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S").to_string(),
        None => "1970-01-01".into(),
    };
    let encoded = vals.iter().map(|&v| (v != time::NAT).then(|| (v - reference) / per)).collect();
    (format!("{uname} since {ref_str}"), encoded)
}

fn attrs_to_nc(attrs: &Attrs, classic: bool) -> Result<Vec<(String, NcAttrValue)>> {
    let mut out = Vec::new();
    for (k, v) in attrs {
        let a = match v {
            AttrValue::Str(s) => NcAttrValue::Chars(s.clone()),
            AttrValue::Int(i) if classic => match i32::try_from(*i) {
                Ok(x) => NcAttrValue::Ints(vec![x]),
                Err(_) => NcAttrValue::Doubles(vec![*i as f64]),
            },
            AttrValue::Int(i) => NcAttrValue::Int64s(vec![*i]),
            AttrValue::Float(f) => NcAttrValue::Doubles(vec![*f]),
            AttrValue::Bool(b) => NcAttrValue::Bytes(vec![*b as i8]),
            AttrValue::IntList(l) if classic => NcAttrValue::Ints(l.iter().map(|&x| x as i32).collect()),
            AttrValue::IntList(l) => NcAttrValue::Int64s(l.clone()),
            AttrValue::FloatList(l) => NcAttrValue::Doubles(l.clone()),
            AttrValue::StrList(l) if classic => {
                return Err(Error::Io(format!("attribute {k:?}: string lists need the NetCDF-4 format")));
            }
            AttrValue::StrList(l) => NcAttrValue::Strings(l.clone()),
        };
        out.push((k.clone(), a));
    }
    Ok(out)
}

impl DataArray<f64> {
    /// Writes a single array as a NetCDF-4 file (named `__xarray_dataarray_variable__`
    /// if unnamed, like xarray).
    pub fn to_netcdf(&self, path: impl AsRef<Path>) -> Result<()> {
        let name = self.name.clone().unwrap_or_else(|| "__xarray_dataarray_variable__".into());
        self.to_dataset(Some(&name))?.to_netcdf(path)
    }
    /// Opens a netCDF file holding exactly one data variable
    /// (`xr.open_dataarray`).
    pub fn open_netcdf(path: impl AsRef<Path>) -> Result<Self> {
        let ds = Dataset::open_netcdf(path)?;
        let names = ds.data_vars();
        if names.len() != 1 {
            return Err(Error::invalid(format!("file has {} data variables; use Dataset::open_netcdf", names.len())));
        }
        let mut da = ds.get(names[0])?;
        if da.name.as_deref() == Some("__xarray_dataarray_variable__") {
            da.name = None;
        }
        Ok(da)
    }
}

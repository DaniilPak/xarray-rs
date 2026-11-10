//! Zarr stores on the local filesystem via [`zarrs`](https://crates.io/crates/zarrs).
//!
//! Reads Zarr v2 and v3 stores written by xarray (`ds.to_zarr`): dimension
//! names come from `dimension_names` (v3) or `_ARRAY_DIMENSIONS` (v2), and the
//! same CF decoding as for netCDF is applied (fill values, scale/offset,
//! `"<unit> since <date>"` times). Writes Zarr v3 stores readable by
//! `xr.open_zarr` (one chunk per array unless a chunk size is given,
//! zstd-compressed).

use crate::attrs::{AttrValue, Attrs};
use crate::coords::{Coord, CoordData};
use crate::dataarray::DataArray;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::time;
use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::Arc;
use zarrs::array::{Array, ArrayBuilder, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::{Group, GroupBuilder};

fn zerr(e: impl std::fmt::Display) -> Error {
    Error::Io(format!("zarr: {e}"))
}

const SKIP_ATTRS: &[&str] =
    &["_ARRAY_DIMENSIONS", "_FillValue", "missing_value", "scale_factor", "add_offset", "coordinates", "dtype"];

fn attr_from_json(v: &Value) -> Option<AttrValue> {
    Some(match v {
        Value::String(s) => AttrValue::Str(s.clone()),
        Value::Bool(b) => AttrValue::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => AttrValue::Int(i),
            None => AttrValue::Float(n.as_f64()?),
        },
        Value::Array(a) if a.iter().all(|x| x.is_i64()) => {
            AttrValue::IntList(a.iter().filter_map(|x| x.as_i64()).collect())
        }
        Value::Array(a) if a.iter().all(|x| x.is_number()) => {
            AttrValue::FloatList(a.iter().filter_map(|x| x.as_f64()).collect())
        }
        Value::Array(a) if a.iter().all(|x| x.is_string()) => {
            AttrValue::StrList(a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        }
        _ => return None,
    })
}

fn attrs_from_json(m: Option<&Map<String, Value>>) -> Attrs {
    let mut out = Attrs::new();
    if let Some(m) = m {
        for (k, v) in m {
            if SKIP_ATTRS.contains(&k.as_str()) {
                continue;
            }
            if let Some(a) = attr_from_json(v) {
                out.insert(k.clone(), a);
            }
        }
    }
    out
}

fn attrs_to_json(a: &Attrs) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in a {
        let j = match v {
            AttrValue::Str(s) => Value::String(s.clone()),
            AttrValue::Int(i) => Value::from(*i),
            AttrValue::Float(f) => {
                serde_json::Number::from_f64(*f).map(Value::Number).unwrap_or(Value::String("NaN".into()))
            }
            AttrValue::Bool(b) => Value::Bool(*b),
            AttrValue::IntList(l) => Value::from(l.clone()),
            AttrValue::FloatList(l) => Value::Array(
                l.iter().map(|f| serde_json::Number::from_f64(*f).map(Value::Number).unwrap_or(Value::Null)).collect(),
            ),
            AttrValue::StrList(l) => Value::from(l.clone()),
        };
        m.insert(k.clone(), j);
    }
    m
}

/// Element kinds we can decode.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Str,
    Utf32,
}

fn kind_of(meta: &Value) -> Result<Kind> {
    let name = match meta.get("zarr_format").and_then(|v| v.as_u64()) {
        Some(2) => {
            let d = meta.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
            let base = d.trim_start_matches(['<', '>', '|', '=']);
            return Ok(match base {
                "f8" => Kind::F64,
                "f4" => Kind::F32,
                "i1" => Kind::I8,
                "i2" => Kind::I16,
                "i4" => Kind::I32,
                "i8" => Kind::I64,
                "u1" => Kind::U8,
                "u2" => Kind::U16,
                "u4" => Kind::U32,
                "u8" => Kind::U64,
                "b1" => Kind::Bool,
                "O" => Kind::Str,
                b if b.starts_with('U') => Kind::Utf32,
                other => return Err(zerr(format!("unsupported v2 dtype {other:?}"))),
            });
        }
        _ => match meta.get("data_type") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Object(o)) => o.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string(),
            _ => String::new(),
        },
    };
    Ok(match name.as_str() {
        "float64" => Kind::F64,
        "float32" => Kind::F32,
        "int8" => Kind::I8,
        "int16" => Kind::I16,
        "int32" => Kind::I32,
        "int64" => Kind::I64,
        "uint8" => Kind::U8,
        "uint16" => Kind::U16,
        "uint32" => Kind::U32,
        "uint64" => Kind::U64,
        "bool" => Kind::Bool,
        "string" => Kind::Str,
        "fixed_length_utf32" => Kind::Utf32,
        other => return Err(zerr(format!("unsupported data type {other:?}"))),
    })
}

enum Raw {
    Num(Vec<f64>, bool),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

fn read_raw(a: &Array<FilesystemStore>, kind: Kind) -> Result<Raw> {
    let all = a.subset_all();
    macro_rules! num {
        ($t:ty, $int:expr) => {{
            let v: Vec<$t> = a.retrieve_array_subset(&all).map_err(zerr)?;
            Raw::Num(v.into_iter().map(|x| x as f64).collect(), $int)
        }};
    }
    Ok(match kind {
        Kind::F64 => num!(f64, false),
        Kind::F32 => num!(f32, false),
        Kind::I8 => num!(i8, true),
        Kind::I16 => num!(i16, true),
        Kind::I32 => num!(i32, true),
        Kind::I64 => num!(i64, true),
        Kind::U8 => num!(u8, true),
        Kind::U16 => num!(u16, true),
        Kind::U32 => num!(u32, true),
        Kind::U64 => num!(u64, true),
        Kind::Bool => Raw::Bool(a.retrieve_array_subset(&all).map_err(zerr)?),
        Kind::Str => Raw::Str(a.retrieve_array_subset(&all).map_err(zerr)?),
        Kind::Utf32 => {
            let v: Vec<Vec<char>> = a.retrieve_array_subset(&all).map_err(zerr)?;
            Raw::Str(
                v.into_iter().map(|c| c.into_iter().collect::<String>().trim_end_matches('\0').to_string()).collect(),
            )
        }
    })
}

/// Decodes a `_FillValue` attribute (number, "NaN", or base64 bytes in v3).
fn fill_value(v: Option<&Value>, kind: Kind) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if s == "NaN" => Some(f64::NAN),
        Value::String(s) => {
            let bytes = base64_decode(s)?;
            Some(match (kind, bytes.len()) {
                (Kind::F64, 8) => f64::from_le_bytes(bytes.try_into().ok()?),
                (Kind::F32, 4) => f32::from_le_bytes(bytes.try_into().ok()?) as f64,
                (Kind::I64, 8) => i64::from_le_bytes(bytes.try_into().ok()?) as f64,
                (Kind::I32, 4) => i32::from_le_bytes(bytes.try_into().ok()?) as f64,
                (Kind::I16, 2) => i16::from_le_bytes(bytes.try_into().ok()?) as f64,
                (Kind::I8, 1) => bytes[0] as i8 as f64,
                (Kind::U8, 1) => bytes[0] as f64,
                _ => return None,
            })
        }
        _ => None,
    }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        let v = T.iter().position(|&t| t == c)? as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn parse_time_units(units: &str) -> Option<(f64, i64)> {
    let (unit, since) = units.split_once(" since ")?;
    let per = match unit.trim() {
        "days" | "day" => 86_400e9,
        "hours" | "hour" => 3_600e9,
        "minutes" | "minute" => 60e9,
        "seconds" | "second" => 1e9,
        "milliseconds" => 1e6,
        "microseconds" => 1e3,
        "nanoseconds" => 1.0,
        _ => return None,
    };
    let reference = time::parse_datetime(since.trim().trim_end_matches('Z')).ok()?;
    Some((per, time::to_ns(reference)))
}

impl Dataset<f64> {
    /// Opens a Zarr v2 or v3 store directory (`xr.open_zarr`).
    pub fn open_zarr(path: impl AsRef<Path>) -> Result<Self> {
        let store = Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        let group = Group::open(store.clone(), "/").map_err(zerr)?;
        let gmeta = serde_json::to_value(group.metadata()).map_err(zerr)?;
        let mut arrays = Vec::new();
        let mut paths = group.child_array_paths().map_err(zerr)?;
        paths.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        for p in paths {
            let name = p.as_str().trim_start_matches('/').to_string();
            let a = open_array(&store, path.as_ref(), &name)?;
            let meta = serde_json::to_value(a.metadata()).map_err(zerr)?;
            arrays.push((name, a, meta));
        }
        let attrs_of = |meta: &Value| -> Option<Map<String, Value>> {
            meta.get("attributes").and_then(|a| a.as_object()).cloned()
        };
        let arrays: Vec<(String, Array<FilesystemStore>, Value)> = arrays
            .into_iter()
            .map(|(n, a, mut m)| {
                if m.get("attributes").is_none() {
                    m["attributes"] = Value::Object(a.attributes().clone());
                }
                (n, a, m)
            })
            .collect();
        let mut coord_names: Vec<String> = Vec::new();
        for (_, _, meta) in &arrays {
            if let Some(c) =
                attrs_of(meta).and_then(|m| m.get("coordinates").and_then(|c| c.as_str()).map(String::from))
            {
                coord_names.extend(c.split_whitespace().map(String::from));
            }
        }
        let mut ds = Dataset::new();
        let mut coords: IndexMap<String, Coord> = IndexMap::new();
        for (name, a, meta) in &arrays {
            let raw_attrs = attrs_of(meta);
            let dims: Vec<String> = match meta.get("dimension_names").and_then(|d| d.as_array()) {
                Some(d) => d.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect(),
                None => raw_attrs
                    .as_ref()
                    .and_then(|m| m.get("_ARRAY_DIMENSIONS"))
                    .and_then(|d| d.as_array())
                    .map(|d| d.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect())
                    .ok_or_else(|| zerr(format!("array {name:?} has no dimension names")))?,
            };
            let shape: Vec<usize> = a.shape().iter().map(|&n| n as usize).collect();
            let kind = kind_of(meta)?;
            let mut attrs = attrs_from_json(raw_attrs.as_ref());
            let get = |k: &str| raw_attrs.as_ref().and_then(|m| m.get(k));
            let is_index = dims.len() == 1 && dims[0] == *name;
            let is_coord = is_index || coord_names.contains(name);
            let raw = read_raw(a, kind)?;
            let arr = |v: Vec<f64>| ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| zerr(e.to_string()));
            match raw {
                Raw::Str(v) => {
                    let c = Coord {
                        dims,
                        data: CoordData::Str(ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(zerr)?),
                        attrs,
                    };
                    coords.insert(name.clone(), c);
                }
                Raw::Bool(v) => {
                    let data = ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(zerr)?;
                    if is_coord {
                        coords.insert(name.clone(), Coord { dims, data: CoordData::Bool(data), attrs });
                    } else {
                        ds.insert(
                            name.clone(),
                            DataArray::new(data.mapv(|b| b as u8 as f64), dims)?.with_attrs(attrs),
                        )?;
                    }
                }
                Raw::Num(mut v, is_int) => {
                    let fill = fill_value(get("_FillValue"), kind).or_else(|| fill_value(get("missing_value"), kind));
                    let scale = get("scale_factor").and_then(|x| x.as_f64());
                    let offset = get("add_offset").and_then(|x| x.as_f64());
                    let mut masked = false;
                    for x in v.iter_mut() {
                        if let Some(f) = fill
                            && (*x == f || (f.is_nan() && x.is_nan()))
                        {
                            *x = f64::NAN;
                            masked = true;
                            continue;
                        }
                        if let Some(s) = scale {
                            *x *= s;
                        }
                        if let Some(o) = offset {
                            *x += o;
                        }
                    }
                    let units = get("units").and_then(|u| u.as_str()).and_then(parse_time_units);
                    let std_cal = get("calendar")
                        .and_then(|c| c.as_str())
                        .is_none_or(|c| matches!(c, "standard" | "gregorian" | "proleptic_gregorian"));
                    if is_coord {
                        let data =
                            if let (Some((per, reference)), true) = (units, std_cal) {
                                attrs.shift_remove("units");
                                attrs.shift_remove("calendar");
                                CoordData::DateTime(arr(v)?.mapv(|x| {
                                    if x.is_nan() { time::NAT } else { reference + (x * per).round() as i64 }
                                }))
                            } else if is_int && !masked && scale.is_none() && offset.is_none() {
                                CoordData::Int(arr(v)?.mapv(|x| x as i64))
                            } else {
                                CoordData::Float(arr(v)?)
                            };
                        coords.insert(name.clone(), Coord { dims, data, attrs });
                    } else {
                        ds.insert(name.clone(), DataArray::new(arr(v)?, dims)?.with_attrs(attrs))?;
                    }
                }
            }
        }
        let dim_order: Vec<String> = ds.dims().keys().cloned().collect();
        let mut ordered = IndexMap::new();
        for d in &dim_order {
            if let Some(c) = coords.shift_remove(d) {
                ordered.insert(d.clone(), c);
            }
        }
        ordered.extend(coords);
        for (k, c) in ordered {
            ds.assign_coord(k, c)?;
        }
        ds.attrs = attrs_from_json(Some(group.attributes()));
        let _ = gmeta;
        Ok(ds)
    }

    /// Writes a Zarr v3 store (`ds.to_zarr(path, zarr_format=3)`), one chunk per
    /// array.
    pub fn to_zarr(&self, path: impl AsRef<Path>) -> Result<()> {
        self.to_zarr_chunked(path, None)
    }

    /// Writes a Zarr v3 store with at most `chunk` elements along every
    /// dimension.
    pub fn to_zarr_chunked(&self, path: impl AsRef<Path>, chunk: Option<u64>) -> Result<()> {
        std::fs::create_dir_all(path.as_ref())?;
        let store = Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        let mut g = GroupBuilder::new().build(store.clone(), "/").map_err(zerr)?;
        *g.attributes_mut() = attrs_to_json(&self.attrs);
        g.store_metadata().map_err(zerr)?;
        let chunks =
            |shape: &[u64]| -> Vec<u64> { shape.iter().map(|&n| chunk.map_or(n, |c| c.min(n)).max(1)).collect() };
        let codecs = || -> Vec<Arc<dyn zarrs::array::BytesToBytesCodecTraits>> {
            vec![Arc::new(zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec::new(0, false))]
        };
        let write = |name: &str,
                     dims: &[String],
                     shape: &[usize],
                     dt: zarrs::array::DataType,
                     fill: zarrs::array::FillValueMetadata,
                     attrs: Map<String, Value>,
                     data: Payload|
         -> Result<()> {
            let shape: Vec<u64> = shape.iter().map(|&n| n as u64).collect();
            let mut b = ArrayBuilder::new(shape.clone(), chunks(&shape), dt, fill);
            b.dimension_names(Some(dims.to_vec())).bytes_to_bytes_codecs(codecs()).attributes(attrs);
            let a = b.build(store.clone(), &format!("/{name}")).map_err(zerr)?;
            a.store_metadata().map_err(zerr)?;
            let all = a.subset_all();
            match data {
                Payload::F64(v) => a.store_array_subset(&all, v.as_slice()),
                Payload::I64(v) => a.store_array_subset(&all, v.as_slice()),
                Payload::Bool(v) => a.store_array_subset(&all, v.as_slice()),
                Payload::Str(v) => a.store_array_subset(&all, v.as_slice()),
            }
            .map_err(zerr)
        };
        use zarrs::array::FillValueMetadata as F;
        for (name, c) in &self.coords {
            let mut attrs = attrs_to_json(&c.attrs);
            let shape = c.shape().to_vec();
            match &c.data {
                CoordData::Float(a) => write(
                    name,
                    &c.dims,
                    &shape,
                    data_type::float64(),
                    F::from(f64::NAN),
                    attrs,
                    Payload::F64(a.iter().copied().collect()),
                )?,
                CoordData::Int(a) => write(
                    name,
                    &c.dims,
                    &shape,
                    data_type::int64(),
                    F::from(0i64),
                    attrs,
                    Payload::I64(a.iter().copied().collect()),
                )?,
                CoordData::Bool(a) => write(
                    name,
                    &c.dims,
                    &shape,
                    data_type::bool(),
                    F::from(false),
                    attrs,
                    Payload::Bool(a.iter().copied().collect()),
                )?,
                CoordData::Str(a) => write(
                    name,
                    &c.dims,
                    &shape,
                    data_type::string(),
                    F::from(""),
                    attrs,
                    Payload::Str(a.iter().cloned().collect()),
                )?,
                CoordData::DateTime(a) => {
                    let vals: Vec<i64> = a.iter().copied().collect();
                    let (units, enc) = encode_times(&vals);
                    attrs.insert("units".into(), Value::String(units));
                    attrs.insert("calendar".into(), Value::String("proleptic_gregorian".into()));
                    if enc.iter().any(Option::is_none) {
                        attrs.insert("_FillValue".into(), Value::String("NaN".into()));
                        write(
                            name,
                            &c.dims,
                            &shape,
                            data_type::float64(),
                            F::from(f64::NAN),
                            attrs,
                            Payload::F64(enc.iter().map(|v| v.map_or(f64::NAN, |x| x as f64)).collect()),
                        )?
                    } else {
                        write(
                            name,
                            &c.dims,
                            &shape,
                            data_type::int64(),
                            F::from(0i64),
                            attrs,
                            Payload::I64(enc.into_iter().map(Option::unwrap).collect()),
                        )?
                    }
                }
            }
        }
        for (name, v) in &self.vars {
            let mut attrs = attrs_to_json(&v.attrs);
            attrs.insert("_FillValue".into(), Value::String("AAAAAAAA+H8=".into()));
            let non_index: Vec<&str> = self
                .coords
                .iter()
                .filter(|(k, c)| !(c.dims.len() == 1 && &c.dims[0] == *k) && c.dims.iter().all(|d| v.dims.contains(d)))
                .map(|(k, _)| k.as_str())
                .collect();
            if !non_index.is_empty() {
                attrs.insert("coordinates".into(), Value::String(non_index.join(" ")));
            }
            write(
                name,
                &v.dims,
                v.shape(),
                data_type::float64(),
                F::from(f64::NAN),
                attrs,
                Payload::F64(v.data.iter().copied().collect()),
            )?;
        }
        Ok(())
    }
}

/// Opens an array, patching Zarr v2 metadata that zarrs rejects (xarray writes
/// `fill_value: null` for string arrays).
fn open_array(store: &Arc<FilesystemStore>, root: &Path, name: &str) -> Result<Array<FilesystemStore>> {
    match Array::open(store.clone(), &format!("/{name}")) {
        Ok(a) => Ok(a),
        Err(e) => {
            let zarray = root.join(name).join(".zarray");
            let Ok(text) = std::fs::read_to_string(&zarray) else { return Err(zerr(e)) };
            let mut meta: Value = serde_json::from_str(&text).map_err(zerr)?;
            if meta.get("fill_value").is_none_or(Value::is_null) {
                let is_str =
                    meta.get("dtype").and_then(|d| d.as_str()).is_some_and(|d| d.contains('U') || d.contains('O'));
                meta["fill_value"] = if is_str { Value::String(String::new()) } else { Value::from(0) };
            }
            let mut m: zarrs::metadata::v2::ArrayMetadataV2 = serde_json::from_value(meta).map_err(zerr)?;
            if let Ok(attrs) = std::fs::read_to_string(root.join(name).join(".zattrs")) {
                m.attributes = serde_json::from_str(&attrs).map_err(zerr)?;
            }
            Array::new_with_metadata(store.clone(), &format!("/{name}"), zarrs::metadata::ArrayMetadata::V2(m))
                .map_err(zerr)
        }
    }
}

enum Payload {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

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
    let ref_str = time::from_ns(reference)
        .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "1970-01-01 00:00:00".into());
    (
        format!("{uname} since {ref_str}"),
        vals.iter().map(|&v| (v != time::NAT).then(|| (v - reference) / per)).collect(),
    )
}

impl DataArray<f64> {
    /// Writes a single array as a Zarr v3 store.
    pub fn to_zarr(&self, path: impl AsRef<Path>) -> Result<()> {
        let name = self.name.clone().unwrap_or_else(|| "__xarray_dataarray_variable__".into());
        self.to_dataset(Some(&name))?.to_zarr(path)
    }
}

//! `to_dict` / `from_dict` in xarray's dict format (`DataArray.to_dict()`,
//! `xr.DataArray.from_dict`), serialized with `serde_json`.
//!
//! An extra `"dtype"` key is written for every variable so the round trip is
//! lossless; it is optional when reading. NaN is written as `null`, datetimes
//! as ISO-8601 strings.

use crate::attrs::{AttrValue, Attrs};
use crate::coords::{Coord, CoordData};
use crate::dataarray::DataArray;
use crate::dataset::Dataset;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::time;
use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use serde_json::{Map, Number, Value};

/// Element types that can be converted to and from JSON values.
pub trait JsonElement: Element {
    /// Converts to JSON.
    fn to_json(&self) -> Value;
    /// Converts from JSON (`null` means missing).
    fn from_json(v: &Value) -> Result<Self>;
}

fn bad(v: &Value, t: &str) -> Error {
    Error::Io(format!("cannot read {v} as {t}"))
}

macro_rules! json_float {
    ($($t:ty),*) => {$(
        impl JsonElement for $t {
            fn to_json(&self) -> Value {
                Number::from_f64(*self as f64).map(Value::Number).unwrap_or(Value::Null)
            }
            fn from_json(v: &Value) -> Result<Self> {
                match v {
                    Value::Null => Ok(<$t>::NAN),
                    Value::Number(n) => n.as_f64().map(|x| x as $t).ok_or_else(|| bad(v, "float")),
                    Value::Bool(b) => Ok(*b as u8 as $t),
                    Value::String(s) => match s.as_str() {
                        "nan" | "NaN" => Ok(<$t>::NAN),
                        "inf" | "Infinity" => Ok(<$t>::INFINITY),
                        "-inf" | "-Infinity" => Ok(<$t>::NEG_INFINITY),
                        _ => s.parse().map_err(|_| bad(v, "float")),
                    },
                    _ => Err(bad(v, "float")),
                }
            }
        }
    )*};
}
json_float!(f32, f64);

macro_rules! json_int {
    ($($t:ty),*) => {$(
        impl JsonElement for $t {
            fn to_json(&self) -> Value { Value::from(*self) }
            fn from_json(v: &Value) -> Result<Self> {
                match v {
                    Value::Number(n) => n
                        .as_i64()
                        .and_then(|i| <$t>::try_from(i).ok())
                        .or_else(|| n.as_u64().and_then(|i| <$t>::try_from(i).ok()))
                        .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as $t))
                        .ok_or_else(|| bad(v, stringify!($t))),
                    Value::Bool(b) => Ok(*b as $t),
                    _ => Err(bad(v, stringify!($t))),
                }
            }
        }
    )*};
}
json_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl JsonElement for bool {
    fn to_json(&self) -> Value {
        Value::Bool(*self)
    }
    fn from_json(v: &Value) -> Result<Self> {
        match v {
            Value::Bool(b) => Ok(*b),
            Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) != 0.0),
            _ => Err(bad(v, "bool")),
        }
    }
}

impl JsonElement for String {
    fn to_json(&self) -> Value {
        Value::String(self.clone())
    }
    fn from_json(v: &Value) -> Result<Self> {
        match v {
            Value::String(s) => Ok(s.clone()),
            Value::Null => Ok(String::new()),
            other => Ok(other.to_string()),
        }
    }
}

// ---- nested lists <-> arrays -------------------------------------------------

fn shape_of(v: &Value) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut cur = v;
    while let Value::Array(a) = cur {
        shape.push(a.len());
        match a.first() {
            Some(x) => cur = x,
            None => break,
        }
    }
    shape
}

fn flatten<'a>(v: &'a Value, depth: usize, out: &mut Vec<&'a Value>) -> Result<()> {
    if depth == 0 {
        out.push(v);
        return Ok(());
    }
    match v {
        Value::Array(a) => {
            for x in a {
                flatten(x, depth - 1, out)?;
            }
            Ok(())
        }
        _ => Err(Error::Io("ragged nested list".into())),
    }
}

fn read_array<T>(v: &Value, f: impl Fn(&Value) -> Result<T>) -> Result<ArrayD<T>> {
    let shape = shape_of(v);
    let mut flat = Vec::new();
    flatten(v, shape.len(), &mut flat)?;
    let n: usize = shape.iter().product();
    if flat.len() != n {
        return Err(Error::Io("ragged nested list".into()));
    }
    let vals: Vec<T> = flat.into_iter().map(f).collect::<Result<_>>()?;
    ArrayD::from_shape_vec(IxDyn(&shape), vals).map_err(|e| Error::Io(e.to_string()))
}

fn write_array<T>(a: &ArrayD<T>, f: &impl Fn(&T) -> Value) -> Value {
    fn rec<T>(v: ndarray::ArrayViewD<'_, T>, f: &impl Fn(&T) -> Value) -> Value {
        if v.ndim() == 0 {
            return f(v.iter().next().unwrap());
        }
        Value::Array(v.outer_iter().map(|s| rec(s, f)).collect())
    }
    rec(a.view(), f)
}

// ---- attrs ---------------------------------------------------------------------

fn attrs_to_json(a: &Attrs) -> Value {
    let mut m = Map::new();
    for (k, v) in a {
        let j = match v {
            AttrValue::Str(s) => Value::String(s.clone()),
            AttrValue::Int(i) => Value::from(*i),
            AttrValue::Float(f) => f.to_json(),
            AttrValue::Bool(b) => Value::Bool(*b),
            AttrValue::IntList(l) => Value::Array(l.iter().map(|x| Value::from(*x)).collect()),
            AttrValue::FloatList(l) => Value::Array(l.iter().map(|x| x.to_json()).collect()),
            AttrValue::StrList(l) => Value::Array(l.iter().map(|x| Value::String(x.clone())).collect()),
        };
        m.insert(k.clone(), j);
    }
    Value::Object(m)
}

fn attrs_from_json(v: Option<&Value>) -> Result<Attrs> {
    let mut out = Attrs::new();
    let Some(Value::Object(m)) = v else { return Ok(out) };
    for (k, v) in m {
        let a = match v {
            Value::String(s) => AttrValue::Str(s.clone()),
            Value::Bool(b) => AttrValue::Bool(*b),
            Value::Number(n) => match n.as_i64() {
                Some(i) => AttrValue::Int(i),
                None => AttrValue::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            Value::Null => AttrValue::Float(f64::NAN),
            Value::Array(items) => {
                if items.iter().all(|x| x.is_i64()) {
                    AttrValue::IntList(items.iter().map(|x| x.as_i64().unwrap()).collect())
                } else if items.iter().all(|x| x.is_number() || x.is_null()) {
                    AttrValue::FloatList(items.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
                } else {
                    AttrValue::StrList(
                        items.iter().map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string())).collect(),
                    )
                }
            }
            Value::Object(_) => AttrValue::Str(v.to_string()),
        };
        out.insert(k.clone(), a);
    }
    Ok(out)
}

// ---- coordinates -------------------------------------------------------------

fn coord_data_to_json(d: &CoordData) -> (Value, &'static str) {
    match d {
        CoordData::Int(a) => (write_array(a, &|v| Value::from(*v)), "int64"),
        CoordData::Float(a) => (write_array(a, &|v| v.to_json()), "float64"),
        CoordData::Str(a) => (write_array(a, &|v| Value::String(v.clone())), "str"),
        CoordData::Bool(a) => (write_array(a, &|v| Value::Bool(*v)), "bool"),
        CoordData::DateTime(a) => (
            write_array(a, &|v| if *v == time::NAT { Value::Null } else { Value::String(time::format_iso(*v)) }),
            "datetime64[ns]",
        ),
    }
}

fn coord_data_from_json(v: &Value, dtype: Option<&str>) -> Result<CoordData> {
    let dtype = match dtype {
        Some(d) => d.to_string(),
        None => {
            let mut flat = Vec::new();
            flatten(v, shape_of(v).len(), &mut flat)?;
            if flat.iter().all(|x| x.is_i64() || x.is_u64()) && !flat.is_empty() {
                "int64".into()
            } else if flat.iter().all(|x| x.is_boolean()) && !flat.is_empty() {
                "bool".into()
            } else if flat.iter().any(|x| x.is_string()) {
                "str".into()
            } else {
                "float64".into()
            }
        }
    };
    Ok(if dtype.starts_with("int") || dtype.starts_with("uint") {
        CoordData::Int(read_array(v, i64::from_json)?)
    } else if dtype.starts_with("float") {
        CoordData::Float(read_array(v, f64::from_json)?)
    } else if dtype == "bool" {
        CoordData::Bool(read_array(v, bool::from_json)?)
    } else if dtype.starts_with("datetime") {
        CoordData::DateTime(read_array(v, |x| match x {
            Value::Null => Ok(time::NAT),
            Value::String(s) if s == "NaT" => Ok(time::NAT),
            Value::String(s) => Ok(time::to_ns(time::parse_datetime(s)?)),
            Value::Number(n) => Ok(n.as_i64().unwrap_or(time::NAT)),
            other => Err(bad(other, "datetime")),
        })?)
    } else {
        CoordData::Str(read_array(v, String::from_json)?)
    })
}

fn dims_from_json(v: Option<&Value>, ndim: usize) -> Result<Vec<String>> {
    match v {
        Some(Value::Array(a)) => {
            Ok(a.iter().map(|d| d.as_str().map(String::from).unwrap_or_else(|| d.to_string())).collect())
        }
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        _ => Ok((0..ndim).map(|i| format!("dim_{i}")).collect()),
    }
}

fn coord_to_json(c: &Coord) -> Value {
    let (data, dtype) = coord_data_to_json(&c.data);
    let mut m = Map::new();
    m.insert("dims".into(), Value::Array(c.dims.iter().map(|d| Value::String(d.clone())).collect()));
    m.insert("attrs".into(), attrs_to_json(&c.attrs));
    m.insert("data".into(), data);
    m.insert("dtype".into(), Value::String(dtype.into()));
    Value::Object(m)
}

fn coord_from_json(name: &str, v: &Value) -> Result<Coord> {
    let data = v.get("data").ok_or_else(|| Error::Io(format!("coordinate {name:?} has no data")))?;
    let cd = coord_data_from_json(data, v.get("dtype").and_then(|d| d.as_str()))?;
    let dims = dims_from_json(v.get("dims"), cd.shape().len())?;
    let mut c = Coord::new(dims, cd)?;
    c.attrs = attrs_from_json(v.get("attrs"))?;
    Ok(c)
}

fn coords_from_json(v: Option<&Value>) -> Result<IndexMap<String, Coord>> {
    let mut out = IndexMap::new();
    if let Some(Value::Object(m)) = v {
        for (k, c) in m {
            out.insert(k.clone(), coord_from_json(k, c)?);
        }
    }
    Ok(out)
}

fn var_to_json<T: JsonElement>(da: &DataArray<T>) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("dims".into(), Value::Array(da.dims.iter().map(|d| Value::String(d.clone())).collect()));
    m.insert("attrs".into(), attrs_to_json(&da.attrs));
    m.insert("data".into(), write_array(&da.data, &|v: &T| v.to_json()));
    m.insert("dtype".into(), Value::String(T::dtype().into()));
    m
}

impl<T: JsonElement> DataArray<T> {
    /// Converts to xarray's dict format (`da.to_dict()`).
    pub fn to_dict(&self) -> Value {
        let mut m = var_to_json(self);
        let mut cm = Map::new();
        for (k, c) in &self.coords {
            cm.insert(k.clone(), coord_to_json(c));
        }
        m.insert("coords".into(), Value::Object(cm));
        m.insert("name".into(), self.name.clone().map(Value::String).unwrap_or(Value::Null));
        Value::Object(m)
    }
    /// Builds from xarray's dict format (`xr.DataArray.from_dict(d)`).
    pub fn from_dict(v: &Value) -> Result<Self> {
        let data = v.get("data").ok_or_else(|| Error::Io("missing \"data\"".into()))?;
        let arr = read_array(data, T::from_json)?;
        let dims = dims_from_json(v.get("dims"), arr.ndim())?;
        let mut da = DataArray::new(arr, dims)?;
        for (k, c) in coords_from_json(v.get("coords"))? {
            da.assign_coord(k, c)?;
        }
        da.attrs = attrs_from_json(v.get("attrs"))?;
        da.name = v.get("name").and_then(|n| n.as_str()).map(String::from);
        Ok(da)
    }
    /// JSON text of [`to_dict`](Self::to_dict).
    pub fn to_json(&self) -> String {
        self.to_dict().to_string()
    }
    /// Parses JSON produced by [`to_json`](Self::to_json) or by
    /// `json.dumps(da.to_dict())` in Python.
    pub fn from_json(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s).map_err(|e| Error::Io(e.to_string()))?;
        Self::from_dict(&v)
    }
}

impl<T: JsonElement> Dataset<T> {
    /// Converts to xarray's dict format (`ds.to_dict()`).
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        let mut cm = Map::new();
        for (k, c) in &self.coords {
            cm.insert(k.clone(), coord_to_json(c));
        }
        m.insert("coords".into(), Value::Object(cm));
        m.insert("attrs".into(), attrs_to_json(&self.attrs));
        let mut dm = Map::new();
        for (d, n) in self.dims() {
            dm.insert(d, Value::from(n));
        }
        m.insert("dims".into(), Value::Object(dm));
        let mut vm = Map::new();
        for (k, v) in &self.vars {
            vm.insert(k.clone(), Value::Object(var_to_json(v)));
        }
        m.insert("data_vars".into(), Value::Object(vm));
        Value::Object(m)
    }
    /// Builds from xarray's dict format (`xr.Dataset.from_dict(d)`).
    pub fn from_dict(v: &Value) -> Result<Self> {
        let mut ds = Dataset::new();
        for (k, c) in coords_from_json(v.get("coords"))? {
            ds.assign_coord(k, c)?;
        }
        if let Some(Value::Object(m)) = v.get("data_vars") {
            for (k, var) in m {
                let data = var.get("data").ok_or_else(|| Error::Io(format!("variable {k:?} has no data")))?;
                let arr = read_array(data, T::from_json)?;
                let dims = dims_from_json(var.get("dims"), arr.ndim())?;
                let da = DataArray::new(arr, dims)?.with_attrs(attrs_from_json(var.get("attrs"))?);
                ds.insert(k.clone(), da)?;
            }
        }
        ds.attrs = attrs_from_json(v.get("attrs"))?;
        Ok(ds)
    }
    /// JSON text of [`to_dict`](Self::to_dict).
    pub fn to_json(&self) -> String {
        self.to_dict().to_string()
    }
    /// Parses JSON in xarray's dict format.
    pub fn from_json(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s).map_err(|e| Error::Io(e.to_string()))?;
        Self::from_dict(&v)
    }
}

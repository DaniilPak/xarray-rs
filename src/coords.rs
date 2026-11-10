//! Coordinates and labels.

use crate::attrs::{AttrValue, Attrs};
use crate::error::{Error, Result};
use crate::time::{self, NAT};
use chrono::{NaiveDate, NaiveDateTime};
use ndarray::{Array1, ArrayD, Axis, IxDyn};
use std::collections::HashMap;
use std::fmt;

/// Typed values of a coordinate variable.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordData {
    /// 64-bit integers.
    Int(ArrayD<i64>),
    /// 64-bit floats.
    Float(ArrayD<f64>),
    /// Strings.
    Str(ArrayD<String>),
    /// Datetimes as nanoseconds since the Unix epoch (`datetime64[ns]`),
    /// [`NAT`](crate::time::NAT) is missing.
    DateTime(ArrayD<i64>),
    /// Booleans.
    Bool(ArrayD<bool>),
}

macro_rules! map_data {
    ($d:expr, $a:ident => $body:expr) => {
        match $d {
            CoordData::Int($a) => CoordData::Int($body),
            CoordData::Float($a) => CoordData::Float($body),
            CoordData::Str($a) => CoordData::Str($body),
            CoordData::DateTime($a) => CoordData::DateTime($body),
            CoordData::Bool($a) => CoordData::Bool($body),
        }
    };
}
macro_rules! with_data {
    ($d:expr, $a:ident => $body:expr) => {
        match $d {
            CoordData::Int($a) => $body,
            CoordData::Float($a) => $body,
            CoordData::Str($a) => $body,
            CoordData::DateTime($a) => $body,
            CoordData::Bool($a) => $body,
        }
    };
}

impl CoordData {
    /// Shape of the values.
    pub fn shape(&self) -> &[usize] {
        with_data!(self, a => a.shape())
    }
    /// numpy dtype name (`int64`, `float64`, `<U`, `datetime64[ns]`, `bool`).
    pub fn dtype(&self) -> String {
        match self {
            CoordData::Int(_) => "int64".into(),
            CoordData::Float(_) => "float64".into(),
            CoordData::Str(a) => format!("<U{}", a.iter().map(|s| s.chars().count()).max().unwrap_or(1).max(1)),
            CoordData::DateTime(_) => "datetime64[ns]".into(),
            CoordData::Bool(_) => "bool".into(),
        }
    }
    /// Bytes used, computed like numpy's `nbytes`.
    pub fn nbytes(&self) -> usize {
        match self {
            CoordData::Str(a) => 4 * a.len() * a.iter().map(|s| s.chars().count()).max().unwrap_or(1).max(1),
            CoordData::Bool(a) => a.len(),
            other => 8 * with_data!(other, a => a.len()),
        }
    }
    /// The values as `f64` (datetimes as nanoseconds). `None` for strings.
    pub fn to_f64(&self) -> Option<ArrayD<f64>> {
        match self {
            CoordData::Int(a) => Some(a.mapv(|v| v as f64)),
            CoordData::Float(a) => Some(a.clone()),
            CoordData::DateTime(a) => Some(a.mapv(|v| if v == NAT { f64::NAN } else { v as f64 })),
            CoordData::Bool(a) => Some(a.mapv(|v| v as u8 as f64)),
            CoordData::Str(_) => None,
        }
    }
    /// Label at a flat (row-major) position.
    pub fn label_at(&self, flat: usize) -> Label {
        match self {
            CoordData::Int(a) => Label::Int(*a.iter().nth(flat).unwrap()),
            CoordData::Float(a) => Label::Float(*a.iter().nth(flat).unwrap()),
            CoordData::Str(a) => Label::Str(a.iter().nth(flat).unwrap().clone()),
            CoordData::DateTime(a) => Label::DateTime(*a.iter().nth(flat).unwrap()),
            CoordData::Bool(a) => Label::Bool(*a.iter().nth(flat).unwrap()),
        }
    }
    pub(crate) fn from_labels(labels: &[Label]) -> Result<CoordData> {
        let n = labels.len();
        let shape = IxDyn(&[n]);
        let first = labels.first();
        macro_rules! collect {
            ($variant:ident, $pat:pat => $v:expr) => {{
                let mut out = Vec::with_capacity(n);
                for l in labels {
                    match l {
                        $pat => out.push($v),
                        other => return Err(Error::Dtype(format!("mixed label types: {other}"))),
                    }
                }
                CoordData::$variant(ArrayD::from_shape_vec(shape, out).unwrap())
            }};
        }
        Ok(match first {
            None | Some(Label::Float(_)) => {
                let mut out = Vec::with_capacity(n);
                for l in labels {
                    out.push(match l {
                        Label::Float(f) => *f,
                        Label::Int(i) => *i as f64,
                        other => return Err(Error::Dtype(format!("mixed label types: {other}"))),
                    });
                }
                CoordData::Float(ArrayD::from_shape_vec(shape, out).unwrap())
            }
            Some(Label::Int(_)) => {
                if labels.iter().any(|l| matches!(l, Label::Float(_))) {
                    let f: Vec<Label> = labels
                        .iter()
                        .map(|l| match l {
                            Label::Int(i) => Label::Float(*i as f64),
                            o => o.clone(),
                        })
                        .collect();
                    return CoordData::from_labels(&f);
                }
                collect!(Int, Label::Int(i) => *i)
            }
            Some(Label::Str(_)) => collect!(Str, Label::Str(s) => s.clone()),
            Some(Label::DateTime(_)) => collect!(DateTime, Label::DateTime(t) => *t),
            Some(Label::Bool(_)) => collect!(Bool, Label::Bool(b) => *b),
        })
    }
}

/// Element types accepted as coordinate values.
pub trait CoordValue: Clone {
    /// Wraps an array of these values in the matching [`CoordData`] variant.
    fn wrap(a: ArrayD<Self>) -> CoordData;
}
macro_rules! coord_value {
    ($($t:ty => |$a:ident| $e:expr),* $(,)?) => {$(
        impl CoordValue for $t { fn wrap($a: ArrayD<Self>) -> CoordData { $e } }
    )*};
}
coord_value!(
    i64 => |a| CoordData::Int(a),
    i32 => |a| CoordData::Int(a.mapv(|v| v as i64)),
    i16 => |a| CoordData::Int(a.mapv(|v| v as i64)),
    u32 => |a| CoordData::Int(a.mapv(|v| v as i64)),
    u16 => |a| CoordData::Int(a.mapv(|v| v as i64)),
    u8 => |a| CoordData::Int(a.mapv(|v| v as i64)),
    usize => |a| CoordData::Int(a.mapv(|v| v as i64)),
    f64 => |a| CoordData::Float(a),
    f32 => |a| CoordData::Float(a.mapv(|v| v as f64)),
    String => |a| CoordData::Str(a),
    bool => |a| CoordData::Bool(a),
    NaiveDateTime => |a| CoordData::DateTime(a.mapv(time::to_ns)),
    NaiveDate => |a| CoordData::DateTime(a.mapv(|d| time::to_ns(d.and_hms_opt(0, 0, 0).unwrap()))),
);
impl CoordValue for &str {
    fn wrap(a: ArrayD<Self>) -> CoordData {
        CoordData::Str(a.mapv(|s| s.to_string()))
    }
}

impl<T: CoordValue> From<Vec<T>> for CoordData {
    fn from(v: Vec<T>) -> Self {
        T::wrap(Array1::from(v).into_dyn())
    }
}
impl<T: CoordValue, const N: usize> From<[T; N]> for CoordData {
    fn from(v: [T; N]) -> Self {
        T::wrap(Array1::from(v.to_vec()).into_dyn())
    }
}
impl<T: CoordValue> From<ArrayD<T>> for CoordData {
    fn from(v: ArrayD<T>) -> Self {
        T::wrap(v)
    }
}
impl<T: CoordValue> From<Array1<T>> for CoordData {
    fn from(v: Array1<T>) -> Self {
        T::wrap(v.into_dyn())
    }
}
impl<T: CoordValue> From<ndarray::Array2<T>> for CoordData {
    fn from(v: ndarray::Array2<T>) -> Self {
        T::wrap(v.into_dyn())
    }
}

/// Things that can name a list of dimensions.
pub trait IntoDims {
    /// Converts into owned dimension names.
    fn into_dims(self) -> Vec<String>;
}
impl IntoDims for &str {
    fn into_dims(self) -> Vec<String> {
        vec![self.to_string()]
    }
}
impl IntoDims for String {
    fn into_dims(self) -> Vec<String> {
        vec![self]
    }
}
impl IntoDims for &String {
    fn into_dims(self) -> Vec<String> {
        vec![self.clone()]
    }
}
impl<const N: usize> IntoDims for [&str; N] {
    fn into_dims(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}
impl IntoDims for &[&str] {
    fn into_dims(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}
impl IntoDims for Vec<&str> {
    fn into_dims(self) -> Vec<String> {
        self.into_iter().map(String::from).collect()
    }
}
impl IntoDims for Vec<String> {
    fn into_dims(self) -> Vec<String> {
        self
    }
}
impl IntoDims for &[String] {
    fn into_dims(self) -> Vec<String> {
        self.to_vec()
    }
}
impl IntoDims for () {
    fn into_dims(self) -> Vec<String> {
        vec![]
    }
}

/// A coordinate variable: labels along one or more dimensions plus attrs.
#[derive(Debug, Clone, PartialEq)]
pub struct Coord {
    pub(crate) dims: Vec<String>,
    pub(crate) data: CoordData,
    pub(crate) attrs: Attrs,
}

impl Coord {
    /// Creates a coordinate along `dims`.
    pub fn new(dims: impl IntoDims, data: impl Into<CoordData>) -> Result<Coord> {
        let dims = dims.into_dims();
        let data = data.into();
        if dims.len() != data.shape().len() {
            return Err(Error::shape(format!(
                "coordinate has {} dims {:?} but data has shape {:?}",
                dims.len(),
                dims,
                data.shape()
            )));
        }
        Ok(Coord { dims, data, attrs: Attrs::new() })
    }
    /// A 0-d (scalar) coordinate.
    pub fn scalar(label: impl Into<Label>) -> Coord {
        let data = CoordData::from_labels(&[label.into()]).unwrap();
        let data = map_data!(data, a => a.index_axis_move(Axis(0), 0));
        Coord { dims: vec![], data, attrs: Attrs::new() }
    }
    /// Adds an attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }
    /// Dimension names.
    pub fn dims(&self) -> &[String] {
        &self.dims
    }
    /// Values.
    pub fn data(&self) -> &CoordData {
        &self.data
    }
    /// Attributes.
    pub fn attrs(&self) -> &Attrs {
        &self.attrs
    }
    /// Mutable attributes.
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        &mut self.attrs
    }
    /// Shape.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    /// Number of values.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }
    /// Whether the coordinate has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// All labels in row-major order.
    pub fn labels(&self) -> Vec<Label> {
        match &self.data {
            CoordData::Int(a) => a.iter().map(|v| Label::Int(*v)).collect(),
            CoordData::Float(a) => a.iter().map(|v| Label::Float(*v)).collect(),
            CoordData::Str(a) => a.iter().map(|v| Label::Str(v.clone())).collect(),
            CoordData::DateTime(a) => a.iter().map(|v| Label::DateTime(*v)).collect(),
            CoordData::Bool(a) => a.iter().map(|v| Label::Bool(*v)).collect(),
        }
    }
    /// Integer values, if this is an integer coordinate.
    pub fn as_i64(&self) -> Option<&ArrayD<i64>> {
        match &self.data {
            CoordData::Int(a) => Some(a),
            _ => None,
        }
    }
    /// Float values, if this is a float coordinate.
    pub fn as_f64(&self) -> Option<&ArrayD<f64>> {
        match &self.data {
            CoordData::Float(a) => Some(a),
            _ => None,
        }
    }
    /// String values, if this is a string coordinate.
    pub fn as_str(&self) -> Option<&ArrayD<String>> {
        match &self.data {
            CoordData::Str(a) => Some(a),
            _ => None,
        }
    }
    /// Datetime values as `chrono` datetimes (`None` entries are NaT).
    pub fn as_datetime(&self) -> Option<Vec<Option<NaiveDateTime>>> {
        match &self.data {
            CoordData::DateTime(a) => Some(a.iter().map(|v| time::from_ns(*v)).collect()),
            _ => None,
        }
    }
    pub(crate) fn axis(&self, dim: &str) -> Option<usize> {
        self.dims.iter().position(|d| d == dim)
    }
    pub(crate) fn map_values(&self, data: CoordData) -> Coord {
        Coord { dims: self.dims.clone(), data, attrs: self.attrs.clone() }
    }

    // ---- structural operations used by DataArray / Dataset -----------------

    pub(crate) fn take(&self, dim: &str, idx: &[usize]) -> Coord {
        match self.axis(dim) {
            None => self.clone(),
            Some(ax) => self.map_values(map_data!(&self.data, a => a.select(Axis(ax), idx))),
        }
    }
    pub(crate) fn take_opt(&self, dim: &str, idx: &[Option<usize>]) -> Coord {
        let Some(ax) = self.axis(dim) else { return self.clone() };
        if idx.iter().all(Option::is_some) {
            let idx: Vec<usize> = idx.iter().map(|i| i.unwrap()).collect();
            return self.take(dim, &idx);
        }
        let data = match &self.data {
            CoordData::Int(a) => CoordData::Float(take_fill(&a.mapv(|v| v as f64), ax, idx, f64::NAN)),
            CoordData::Float(a) => CoordData::Float(take_fill(a, ax, idx, f64::NAN)),
            CoordData::Str(a) => CoordData::Str(take_fill(a, ax, idx, String::new())),
            CoordData::DateTime(a) => CoordData::DateTime(take_fill(a, ax, idx, NAT)),
            CoordData::Bool(a) => CoordData::Float(take_fill(&a.mapv(|v| v as u8 as f64), ax, idx, f64::NAN)),
        };
        self.map_values(data)
    }
    pub(crate) fn index(&self, dim: &str, i: usize) -> Coord {
        match self.axis(dim) {
            None => self.clone(),
            Some(ax) => {
                let mut c = self.map_values(map_data!(&self.data, a => a.index_axis(Axis(ax), i).to_owned()));
                c.dims.remove(ax);
                c
            }
        }
    }
    pub(crate) fn rename_dim(&mut self, old: &str, new: &str) {
        for d in &mut self.dims {
            if d == old {
                *d = new.to_string();
            }
        }
    }
    pub(crate) fn insert_dim(&self, pos: usize, dim: &str, size: usize) -> Coord {
        let mut c = self.map_values(map_data!(&self.data, a => {
            let v = a.clone().insert_axis(Axis(pos));
            let mut shape = v.shape().to_vec();
            shape[pos] = size;
            v.broadcast(IxDyn(&shape)).unwrap().to_owned()
        }));
        c.dims.insert(pos, dim.to_string());
        c
    }
    pub(crate) fn concat(coords: &[&Coord], dim: &str) -> Result<Coord> {
        let first = coords[0];
        let ax = first.axis(dim).ok_or_else(|| Error::invalid("concat dim missing in coord"))?;
        macro_rules! cat {
            ($variant:ident) => {{
                let mut views = Vec::new();
                for c in coords {
                    match &c.data {
                        CoordData::$variant(a) => views.push(a.view()),
                        _ => return Coord::concat_labels(coords, dim),
                    }
                }
                CoordData::$variant(ndarray::concatenate(Axis(ax), &views).map_err(|e| Error::shape(e.to_string()))?)
            }};
        }
        let data = match &first.data {
            CoordData::Int(_) => cat!(Int),
            CoordData::Float(_) => cat!(Float),
            CoordData::Str(_) => cat!(Str),
            CoordData::DateTime(_) => cat!(DateTime),
            CoordData::Bool(_) => cat!(Bool),
        };
        Ok(first.map_values(data))
    }
    fn concat_labels(coords: &[&Coord], dim: &str) -> Result<Coord> {
        // mixed int/float: promote to float
        if coords.iter().all(|c| c.dims.len() == 1) {
            let labels: Vec<Label> = coords.iter().flat_map(|c| c.labels()).collect();
            let data = CoordData::from_labels(&labels)?;
            return Ok(Coord { dims: vec![dim.to_string()], data, attrs: coords[0].attrs.clone() });
        }
        Err(Error::Dtype("cannot concatenate coordinates of different types".into()))
    }

    /// Builds a lookup index for a 1-D coordinate.
    pub(crate) fn to_index(&self) -> Result<Index> {
        if self.dims.len() != 1 {
            return Err(Error::invalid("only 1-D coordinates can be used as an index"));
        }
        let kind = match &self.data {
            CoordData::Int(a) => IndexKind::I(a.iter().copied().collect(), false),
            CoordData::DateTime(a) => IndexKind::I(a.iter().copied().collect(), true),
            CoordData::Bool(a) => IndexKind::I(a.iter().map(|b| *b as i64).collect(), false),
            CoordData::Float(a) => IndexKind::F(a.iter().copied().collect()),
            CoordData::Str(a) => IndexKind::S(a.iter().cloned().collect()),
        };
        Ok(Index { kind, dim: self.dims[0].clone() })
    }
}

pub(crate) fn take_fill<T: Clone>(a: &ArrayD<T>, ax: usize, idx: &[Option<usize>], fill: T) -> ArrayD<T> {
    let mut shape = a.shape().to_vec();
    shape[ax] = idx.len();
    let mut out = ArrayD::from_elem(IxDyn(&shape), fill);
    for (j, i) in idx.iter().enumerate() {
        if let Some(i) = i {
            out.index_axis_mut(Axis(ax), j).assign(&a.index_axis(Axis(ax), *i));
        }
    }
    out
}

/// A coordinate label used for label-based selection.
#[derive(Debug, Clone, PartialEq)]
pub enum Label {
    /// Integer label.
    Int(i64),
    /// Float label.
    Float(f64),
    /// String label (also parsed as a datetime on datetime indexes).
    Str(String),
    /// Datetime as nanoseconds since the Unix epoch.
    DateTime(i64),
    /// Boolean label.
    Bool(bool),
}

impl Label {
    /// Datetime label from a `chrono` value.
    pub fn datetime(dt: NaiveDateTime) -> Label {
        Label::DateTime(time::to_ns(dt))
    }
    /// The label as `f64` (datetimes as nanoseconds).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Label::Int(i) => Some(*i as f64),
            Label::Float(f) => Some(*f),
            Label::DateTime(t) => Some(*t as f64),
            Label::Bool(b) => Some(*b as u8 as f64),
            Label::Str(_) => None,
        }
    }
    pub(crate) fn hash_key(&self) -> HashKey {
        match self {
            Label::Int(i) | Label::DateTime(i) => HashKey::I(*i),
            Label::Bool(b) => HashKey::I(*b as i64),
            Label::Float(f) => {
                if f.fract() == 0.0 && f.abs() < 9.0e15 {
                    HashKey::I(*f as i64)
                } else {
                    HashKey::F(if f.is_nan() { u64::MAX } else { f.to_bits() })
                }
            }
            Label::Str(s) => HashKey::S(s.clone()),
        }
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Label::Int(i) => write!(f, "{i}"),
            Label::Float(x) => write!(f, "{}", crate::repr::fmt_float(*x)),
            Label::Str(s) => write!(f, "'{s}'"),
            Label::DateTime(t) => write!(f, "{}", time::format_ns(*t, false)),
            Label::Bool(b) => write!(f, "{b}"),
        }
    }
}

macro_rules! label_from {
    ($($t:ty => |$v:ident| $e:expr),* $(,)?) => {$(
        impl From<$t> for Label { fn from($v: $t) -> Self { $e } }
    )*};
}
label_from!(
    i64 => |v| Label::Int(v),
    i32 => |v| Label::Int(v as i64),
    u32 => |v| Label::Int(v as i64),
    usize => |v| Label::Int(v as i64),
    f64 => |v| Label::Float(v),
    f32 => |v| Label::Float(v as f64),
    bool => |v| Label::Bool(v),
    &str => |v| Label::Str(v.to_string()),
    String => |v| Label::Str(v),
    &String => |v| Label::Str(v.clone()),
    NaiveDateTime => |v| Label::datetime(v),
    NaiveDate => |v| Label::datetime(v.and_hms_opt(0, 0, 0).unwrap()),
);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum HashKey {
    I(i64),
    F(u64),
    S(String),
}

/// Lookup structure for a 1-D index coordinate.
#[derive(Debug, Clone)]
pub(crate) struct Index {
    pub(crate) kind: IndexKind,
    pub(crate) dim: String,
}

#[derive(Debug, Clone)]
pub(crate) enum IndexKind {
    /// integers / datetimes (flag) / bools
    I(Vec<i64>, bool),
    F(Vec<f64>),
    S(Vec<String>),
}

/// A label converted to the index's native type for comparisons.
#[derive(Debug, Clone)]
pub(crate) enum Probe {
    I(i64),
    F(f64),
    S(String),
}

/// How [`sel`](crate::DataArray::sel) matches labels that are not in the
/// index.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Method {
    /// Exact match only (default).
    #[default]
    Exact,
    /// Nearest label (ties go to the larger label, like pandas).
    Nearest,
    /// Previous label (forward fill, `"pad"` / `"ffill"`).
    Pad,
    /// Next label (`"backfill"` / `"bfill"`).
    Backfill,
}

impl Index {
    pub(crate) fn len(&self) -> usize {
        match &self.kind {
            IndexKind::I(v, _) => v.len(),
            IndexKind::F(v) => v.len(),
            IndexKind::S(v) => v.len(),
        }
    }
    pub(crate) fn is_datetime(&self) -> bool {
        matches!(self.kind, IndexKind::I(_, true))
    }
    pub(crate) fn label(&self, i: usize) -> Label {
        match &self.kind {
            IndexKind::I(v, true) => Label::DateTime(v[i]),
            IndexKind::I(v, false) => Label::Int(v[i]),
            IndexKind::F(v) => Label::Float(v[i]),
            IndexKind::S(v) => Label::Str(v[i].clone()),
        }
    }
    pub(crate) fn hash_keys(&self) -> Vec<HashKey> {
        (0..self.len()).map(|i| self.label(i).hash_key()).collect()
    }
    pub(crate) fn probe(&self, label: &Label) -> Result<Probe> {
        let err = || Error::Dtype(format!("label {label} does not match the type of index {:?}", self.dim));
        Ok(match (&self.kind, label) {
            (IndexKind::I(_, true), Label::Str(s)) => Probe::I(time::to_ns(time::parse_datetime(s)?)),
            (IndexKind::I(_, _), Label::Int(i) | Label::DateTime(i)) => Probe::I(*i),
            (IndexKind::I(_, _), Label::Bool(b)) => Probe::I(*b as i64),
            (IndexKind::I(_, false), Label::Float(f)) => Probe::F(*f),
            (IndexKind::F(_), Label::Float(f)) => Probe::F(*f),
            (IndexKind::F(_), Label::Int(i)) => Probe::F(*i as f64),
            (IndexKind::S(_), Label::Str(s)) => Probe::S(s.clone()),
            _ => return Err(err()),
        })
    }
    /// Compares index value `i` with a probe.
    pub(crate) fn cmp(&self, i: usize, p: &Probe) -> std::cmp::Ordering {
        use std::cmp::Ordering::Equal;
        match (&self.kind, p) {
            (IndexKind::I(v, _), Probe::I(x)) => v[i].cmp(x),
            (IndexKind::I(v, _), Probe::F(x)) => (v[i] as f64).partial_cmp(x).unwrap_or(Equal),
            (IndexKind::F(v), Probe::F(x)) => v[i].partial_cmp(x).unwrap_or(Equal),
            (IndexKind::F(v), Probe::I(x)) => v[i].partial_cmp(&(*x as f64)).unwrap_or(Equal),
            (IndexKind::S(v), Probe::S(x)) => v[i].as_str().cmp(x.as_str()),
            _ => Equal,
        }
    }
    fn dist(&self, i: usize, p: &Probe) -> f64 {
        match (&self.kind, p) {
            (IndexKind::I(v, _), Probe::I(x)) => (v[i] as i128 - *x as i128).unsigned_abs() as f64,
            (IndexKind::I(v, _), Probe::F(x)) => (v[i] as f64 - x).abs(),
            (IndexKind::F(v), Probe::F(x)) => (v[i] - x).abs(),
            (IndexKind::F(v), Probe::I(x)) => (v[i] - *x as f64).abs(),
            _ => f64::INFINITY,
        }
    }
    /// `Some(true)` if monotonic increasing, `Some(false)` if decreasing.
    pub(crate) fn monotonic(&self) -> Option<bool> {
        let n = self.len();
        let le = |a: usize, b: usize| match &self.kind {
            IndexKind::I(v, _) => v[a] <= v[b],
            IndexKind::F(v) => v[a] <= v[b],
            IndexKind::S(v) => v[a] <= v[b],
        };
        if (1..n).all(|i| le(i - 1, i)) {
            Some(true)
        } else if (1..n).all(|i| le(i, i - 1)) {
            Some(false)
        } else {
            None
        }
    }
    pub(crate) fn is_unique(&self) -> bool {
        let keys = self.hash_keys();
        let set: std::collections::HashSet<&HashKey> = keys.iter().collect();
        set.len() == keys.len()
    }

    /// Positions of an exact label (all matches, in order).
    pub(crate) fn locate(&self, label: &Label) -> Result<Vec<usize>> {
        let p = self.probe(label)?;
        Ok((0..self.len()).filter(|&i| self.cmp(i, &p).is_eq()).collect())
    }

    /// For each label, its position (exact, nearest, pad or backfill).
    pub(crate) fn get_indexer(
        &self,
        labels: &[Label],
        method: Method,
        tolerance: Option<f64>,
    ) -> Result<Vec<Option<usize>>> {
        let n = self.len();
        if method == Method::Exact {
            let mut map: HashMap<HashKey, usize> = HashMap::with_capacity(n);
            for (i, k) in self.hash_keys().into_iter().enumerate() {
                map.entry(k).or_insert(i);
            }
            return labels
                .iter()
                .map(|l| {
                    let p = self.probe(l)?;
                    let key = match p {
                        Probe::I(i) => HashKey::I(i),
                        Probe::F(f) => Label::Float(f).hash_key(),
                        Probe::S(s) => HashKey::S(s),
                    };
                    Ok(map.get(&key).copied())
                })
                .collect();
        }
        let inc = self
            .monotonic()
            .ok_or_else(|| Error::invalid(format!("index {:?} must be monotonic for method {method:?}", self.dim)))?;
        // positions in increasing order
        let order: Vec<usize> = if inc { (0..n).collect() } else { (0..n).rev().collect() };
        let mut out = Vec::with_capacity(labels.len());
        for l in labels {
            let p = self.probe(l)?;
            // first k (in sorted order) with value >= p
            let k = order.partition_point(|&i| self.cmp(i, &p).is_lt());
            let exact = k < n && self.cmp(order[k], &p).is_eq();
            let below = if exact { Some(order[k]) } else { k.checked_sub(1).map(|j| order[j]) };
            let above = if k < n { Some(order[k]) } else { None };
            let pick = match method {
                Method::Pad => below,
                Method::Backfill => above,
                Method::Nearest => match (below, above) {
                    (Some(b), Some(a)) => {
                        if self.dist(b, &p) < self.dist(a, &p) {
                            Some(b)
                        } else {
                            Some(a)
                        }
                    }
                    (b, a) => b.or(a),
                },
                Method::Exact => unreachable!(),
            };
            let pick = match (pick, tolerance) {
                (Some(i), Some(tol)) if self.dist(i, &p) > tol => None,
                (pick, _) => pick,
            };
            out.push(pick);
        }
        Ok(out)
    }

    /// Positions `[start, stop)` selected by an inclusive label slice.
    pub(crate) fn slice_locs(&self, start: Option<&Label>, stop: Option<&Label>) -> Result<(usize, usize)> {
        let n = self.len();
        let ps = start.map(|l| self.slice_probe(l, true)).transpose()?;
        let pe = stop.map(|l| self.slice_probe(l, false)).transpose()?;
        match self.monotonic() {
            Some(true) => {
                let a = ps.map(|p| (0..n).position(|i| !self.cmp(i, &p).is_lt()).unwrap_or(n)).unwrap_or(0);
                let b =
                    pe.map(|p| (0..n).rposition(|i| !self.cmp(i, &p).is_gt()).map(|i| i + 1).unwrap_or(0)).unwrap_or(n);
                Ok((a, b.max(a)))
            }
            Some(false) => {
                let a = ps.map(|p| (0..n).position(|i| !self.cmp(i, &p).is_gt()).unwrap_or(n)).unwrap_or(0);
                let b =
                    pe.map(|p| (0..n).rposition(|i| !self.cmp(i, &p).is_lt()).map(|i| i + 1).unwrap_or(0)).unwrap_or(n);
                Ok((a, b.max(a)))
            }
            None => {
                let find = |l: &Label| -> Result<usize> {
                    let pos = self.locate(l)?;
                    if pos.len() != 1 {
                        return Err(Error::LabelNotFound { dim: self.dim.clone(), label: l.to_string() });
                    }
                    Ok(pos[0])
                };
                let a = start.map(find).transpose()?.unwrap_or(0);
                let b = stop.map(find).transpose()?.map(|i| i + 1).unwrap_or(n);
                Ok((a, b.max(a)))
            }
        }
    }

    fn slice_probe(&self, l: &Label, is_start: bool) -> Result<Probe> {
        // partial datetime strings: start uses period start, stop uses period end
        if let (true, Label::Str(s)) = (self.is_datetime(), l) {
            let (a, b, _) = time::parse_partial(s)?;
            return Ok(Probe::I(if is_start { time::to_ns(a) } else { time::to_ns(b) - 1 }));
        }
        self.probe(l)
    }
}

/// `..` means "all dimensions" for reductions and "the remaining dimensions"
/// in [`transpose`](crate::DataArray::transpose).
impl IntoDims for std::ops::RangeFull {
    fn into_dims(self) -> Vec<String> {
        vec!["...".to_string()]
    }
}

//! Attribute values (`attrs` metadata).

use indexmap::IndexMap;
use std::fmt;

/// Ordered attribute dictionary.
pub type Attrs = IndexMap<String, AttrValue>;

/// One attribute value. Mirrors the value types netCDF/Zarr attributes can
/// hold.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// Text.
    Str(String),
    /// Integer.
    Int(i64),
    /// Floating point number.
    Float(f64),
    /// Boolean.
    Bool(bool),
    /// List of integers.
    IntList(Vec<i64>),
    /// List of floats.
    FloatList(Vec<f64>),
    /// List of strings.
    StrList(Vec<String>),
}

impl AttrValue {
    /// The value as text, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttrValue::Str(s) => Some(s),
            _ => None,
        }
    }
    /// The value as a float, if it is numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AttrValue::Int(i) => Some(*i as f64),
            AttrValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

impl fmt::Display for AttrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn list<T: fmt::Display>(f: &mut fmt::Formatter<'_>, v: &[T]) -> fmt::Result {
            write!(f, "[")?;
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{x}")?;
            }
            write!(f, "]")
        }
        match self {
            AttrValue::Str(s) => write!(f, "{s}"),
            AttrValue::Int(i) => write!(f, "{i}"),
            AttrValue::Float(x) => write!(f, "{}", crate::repr::fmt_float(*x)),
            AttrValue::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            AttrValue::IntList(v) => list(f, v),
            AttrValue::FloatList(v) => list(f, v),
            AttrValue::StrList(v) => list(f, v),
        }
    }
}

macro_rules! from_attr {
    ($($t:ty => |$v:ident| $e:expr),* $(,)?) => {$(
        impl From<$t> for AttrValue { fn from($v: $t) -> Self { $e } }
    )*};
}
from_attr!(
    &str => |v| AttrValue::Str(v.to_string()),
    String => |v| AttrValue::Str(v),
    i64 => |v| AttrValue::Int(v),
    i32 => |v| AttrValue::Int(v as i64),
    u32 => |v| AttrValue::Int(v as i64),
    f64 => |v| AttrValue::Float(v),
    f32 => |v| AttrValue::Float(v as f64),
    bool => |v| AttrValue::Bool(v),
    Vec<i64> => |v| AttrValue::IntList(v),
    Vec<f64> => |v| AttrValue::FloatList(v),
    Vec<String> => |v| AttrValue::StrList(v),
    Vec<&str> => |v| AttrValue::StrList(v.into_iter().map(String::from).collect()),
);

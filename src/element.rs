//! Element types that can be stored in a [`DataArray`](crate::DataArray).

use std::fmt::Debug;

/// A type that can be stored in a [`DataArray`](crate::DataArray) or
/// [`Dataset`](crate::Dataset).
///
/// Implemented for all primitive numbers, `bool` and `String`.
pub trait Element: Clone + Debug + PartialEq + Send + Sync + 'static {
    /// numpy-style dtype name used in the text repr (`float64`, `int32`, ...).
    fn dtype() -> &'static str;
    /// The missing value used when reindexing introduces gaps (`NaN` for
    /// floats). Types without one return `None`, and alignment that would need
    /// it returns an error instead.
    fn missing() -> Option<Self> {
        None
    }
    /// Whether this value is missing (`NaN` for floats).
    fn is_missing(&self) -> bool {
        false
    }
    /// Size in bytes of one element, used for the `Size:` line of the repr.
    fn item_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
    /// Short text used by the repr.
    fn repr(&self) -> String;
    /// Lossy conversion to `f64` used by repr helpers and interpolation of
    /// labels. `None` for non-numeric types.
    fn to_f64(&self) -> Option<f64> {
        None
    }
}

macro_rules! impl_int {
    ($($t:ty => $name:literal),*) => {$(
        impl Element for $t {
            fn dtype() -> &'static str { $name }
            fn repr(&self) -> String { self.to_string() }
            fn to_f64(&self) -> Option<f64> { Some(*self as f64) }
        }
    )*};
}
impl_int!(i8 => "int8", i16 => "int16", i32 => "int32", i64 => "int64",
          u8 => "uint8", u16 => "uint16", u32 => "uint32", u64 => "uint64",
          isize => "int64", usize => "uint64");

macro_rules! impl_float {
    ($($t:ty => $name:literal),*) => {$(
        impl Element for $t {
            fn dtype() -> &'static str { $name }
            fn missing() -> Option<Self> { Some(<$t>::NAN) }
            fn is_missing(&self) -> bool { self.is_nan() }
            fn repr(&self) -> String { crate::repr::fmt_float(*self as f64) }
            fn to_f64(&self) -> Option<f64> { Some(*self as f64) }
        }
    )*};
}
impl_float!(f32 => "float32", f64 => "float64");

impl Element for bool {
    fn dtype() -> &'static str {
        "bool"
    }
    fn repr(&self) -> String {
        if *self { "True" } else { "False" }.to_string()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(if *self { 1.0 } else { 0.0 })
    }
}

impl Element for String {
    fn dtype() -> &'static str {
        "<U"
    }
    fn item_size(&self) -> usize {
        4 * self.chars().count().max(1)
    }
    fn repr(&self) -> String {
        format!("'{self}'")
    }
}

/// Floating point element types (`f32`, `f64`): NaN-aware reductions,
/// interpolation and rolling windows are available for these.
pub trait FloatElement: Element + num_traits::Float + num_traits::FromPrimitive + Default {}
impl FloatElement for f32 {}
impl FloatElement for f64 {}

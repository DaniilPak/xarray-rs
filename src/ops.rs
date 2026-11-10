//! Element-wise arithmetic with automatic alignment and broadcasting by
//! dimension name.

use crate::align::{Join, align2};
use crate::attrs::Attrs;
use crate::coords::Coord;
use crate::dataarray::{DataArray, coord_equals};
use crate::element::{Element, FloatElement};
use crate::error::{Error, Result};
use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn, Zip};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Union of dimension names in order of first appearance, with sizes.
pub(crate) fn union_dims(items: &[(&Vec<String>, &[usize])]) -> Result<(Vec<String>, Vec<usize>)> {
    let mut dims: Vec<String> = Vec::new();
    let mut shape: Vec<usize> = Vec::new();
    for (ds, sh) in items {
        for (d, &n) in ds.iter().zip(sh.iter()) {
            match dims.iter().position(|x| x == d) {
                Some(i) if shape[i] != n => {
                    return Err(Error::Align(format!("dimension {d:?} has conflicting sizes {} and {n}", shape[i])));
                }
                Some(_) => {}
                None => {
                    dims.push(d.clone());
                    shape.push(n);
                }
            }
        }
    }
    Ok((dims, shape))
}

/// Broadcasts an array to `dims`/`shape` (dims not in the array are added).
pub(crate) fn broadcast_to<T: Element>(a: &DataArray<T>, dims: &[String], shape: &[usize]) -> Result<DataArray<T>> {
    let data = broadcast_owned(&a.data, &a.dims, dims, shape)?;
    Ok(DataArray::from_parts(data, dims.to_vec(), a.coords.clone(), a.attrs.clone(), a.name.clone()))
}

/// View of `data` with axes permuted into `dims` order and length-1 axes
/// inserted for missing dims (ready for `broadcast`).
fn laid_out<'a, T>(data: &'a ArrayD<T>, own: &[String], dims: &[String]) -> Result<ndarray::ArrayViewD<'a, T>> {
    let mut order: Vec<usize> = (0..own.len()).collect();
    order.sort_by_key(|&i| dims.iter().position(|d| *d == own[i]).unwrap_or(usize::MAX));
    let mut v = data.view().permuted_axes(order.clone());
    let sorted: Vec<&String> = order.iter().map(|&i| &own[i]).collect();
    let mut k = 0;
    for (pos, d) in dims.iter().enumerate() {
        if k < sorted.len() && sorted[k] == d {
            k += 1;
        } else {
            v = v.insert_axis(Axis(pos));
        }
    }
    Ok(v)
}

pub(crate) fn broadcast_owned<T: Clone>(
    data: &ArrayD<T>,
    own: &[String],
    dims: &[String],
    shape: &[usize],
) -> Result<ArrayD<T>> {
    let mut order: Vec<usize> = (0..own.len()).collect();
    order.sort_by_key(|&i| dims.iter().position(|d| *d == own[i]).unwrap_or(usize::MAX));
    let mut v = data.view().permuted_axes(order.clone());
    let sorted: Vec<&String> = order.iter().map(|&i| &own[i]).collect();
    let mut k = 0;
    for (pos, d) in dims.iter().enumerate() {
        if k < sorted.len() && sorted[k] == d {
            k += 1;
        } else {
            v = v.insert_axis(Axis(pos));
        }
    }
    let b = v
        .broadcast(IxDyn(shape))
        .ok_or_else(|| Error::shape(format!("cannot broadcast {:?} to {:?}", data.shape(), shape)))?;
    Ok(b.to_owned())
}

/// Merges attributes, dropping keys whose values conflict
/// (`keep_attrs="drop_conflicts"`).
pub(crate) fn merge_attrs_drop_conflicts(a: &Attrs, b: &Attrs) -> Attrs {
    let mut out = Attrs::new();
    for (k, v) in a {
        match b.get(k) {
            Some(w) if w != v => {}
            _ => {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    for (k, v) in b {
        if !a.contains_key(k) {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// Merges coordinates of two aligned arrays, dropping conflicting non-index
/// coordinates (xarray's arithmetic behaviour).
pub(crate) fn merge_coords(a: &IndexMap<String, Coord>, b: &IndexMap<String, Coord>) -> IndexMap<String, Coord> {
    let mut out = IndexMap::new();
    for (k, c) in a {
        match b.get(k) {
            Some(o) if !coord_equals(c, o) => {}
            _ => {
                out.insert(k.clone(), c.clone());
            }
        }
    }
    for (k, c) in b {
        if !a.contains_key(k) {
            out.insert(k.clone(), c.clone());
        }
    }
    out
}

/// Combines two already-aligned arrays element-wise after broadcasting.
pub(crate) fn zip_with<A: Element, B: Element, U: Element>(
    a: &DataArray<A>,
    b: &DataArray<B>,
    f: impl Fn(&A, &B) -> U,
) -> Result<DataArray<U>> {
    let (dims, shape) = union_dims(&[(&a.dims, a.shape()), (&b.dims, b.shape())])?;
    let va = laid_out(&a.data, &a.dims, &dims)?;
    let vb = laid_out(&b.data, &b.dims, &dims)?;
    let sh = IxDyn(&shape);
    let (ba, bb) = (
        va.broadcast(sh.clone()).ok_or_else(|| Error::shape("broadcast failed"))?,
        vb.broadcast(sh).ok_or_else(|| Error::shape("broadcast failed"))?,
    );
    let data = Zip::from(&ba).and(&bb).map_collect(&f);
    let name = if a.name == b.name { a.name.clone() } else { None };
    let attrs = merge_attrs_drop_conflicts(&a.attrs, &b.attrs);
    Ok(DataArray::from_parts(data.into_dyn(), dims, merge_coords(&a.coords, &b.coords), attrs, name))
}

impl<T: Element> DataArray<T> {
    /// Aligns (inner join) and broadcasts against `other`, then combines
    /// element-wise with `f`. This is what the arithmetic operators use.
    pub fn zip_with<B: Element, U: Element>(
        &self,
        other: &DataArray<B>,
        f: impl Fn(&T, &B) -> U,
    ) -> Result<DataArray<U>> {
        self.zip_with_join(other, Join::Inner, f)
    }
    /// Like [`zip_with`](Self::zip_with) with an explicit join.
    pub fn zip_with_join<B: Element, U: Element>(
        &self,
        other: &DataArray<B>,
        join: Join,
        f: impl Fn(&T, &B) -> U,
    ) -> Result<DataArray<U>> {
        let (a, b) = align2(self, other, join)?;
        zip_with(&a, &b, f)
    }
}

macro_rules! binop_methods {
    ($($method:ident, $trait:ident, $op:tt);*) => {
        impl<T: Element + Copy + $crate::ops::NumOps> DataArray<T> {
            $(
            #[doc = concat!("`self ", stringify!($op), " other` with alignment (inner join) and broadcasting.")]
            pub fn $method(&self, other: &DataArray<T>) -> Result<DataArray<T>> {
                self.zip_with(other, |a, b| *a $op *b)
            }
            )*
        }
        $(
        impl<T: Element + Copy + NumOps> $trait<&DataArray<T>> for &DataArray<T> {
            type Output = DataArray<T>;
            /// Panics if the arrays cannot be aligned; use the method form
            /// for a `Result`.
            fn $method(self, rhs: &DataArray<T>) -> DataArray<T> {
                DataArray::$method(self, rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }
        impl<T: Element + Copy + NumOps> $trait<DataArray<T>> for DataArray<T> {
            type Output = DataArray<T>;
            fn $method(self, rhs: DataArray<T>) -> DataArray<T> {
                DataArray::$method(&self, &rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }
        impl<T: Element + Copy + NumOps> $trait<&DataArray<T>> for DataArray<T> {
            type Output = DataArray<T>;
            fn $method(self, rhs: &DataArray<T>) -> DataArray<T> {
                DataArray::$method(&self, rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }
        impl<T: Element + Copy + NumOps> $trait<T> for &DataArray<T> {
            type Output = DataArray<T>;
            fn $method(self, rhs: T) -> DataArray<T> {
                self.map(|a| *a $op rhs)
            }
        }
        impl<T: Element + Copy + NumOps> $trait<T> for DataArray<T> {
            type Output = DataArray<T>;
            fn $method(self, rhs: T) -> DataArray<T> {
                self.map(|a| *a $op rhs)
            }
        }
        )*
    };
}

/// Numeric element types supporting `+ - * / %`.
pub trait NumOps:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self> + Sized
{
}
impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T>> NumOps for T {}

binop_methods!(add, Add, +; sub, Sub, -; mul, Mul, *; div, Div, /; rem, Rem, %);

macro_rules! scalar_lhs {
    ($($t:ty),*) => {$(
        impl Add<&DataArray<$t>> for $t { type Output = DataArray<$t>; fn add(self, r: &DataArray<$t>) -> DataArray<$t> { r.map(|v| self + *v) } }
        impl Sub<&DataArray<$t>> for $t { type Output = DataArray<$t>; fn sub(self, r: &DataArray<$t>) -> DataArray<$t> { r.map(|v| self - *v) } }
        impl Mul<&DataArray<$t>> for $t { type Output = DataArray<$t>; fn mul(self, r: &DataArray<$t>) -> DataArray<$t> { r.map(|v| self * *v) } }
        impl Div<&DataArray<$t>> for $t { type Output = DataArray<$t>; fn div(self, r: &DataArray<$t>) -> DataArray<$t> { r.map(|v| self / *v) } }
        impl Add<DataArray<$t>> for $t { type Output = DataArray<$t>; fn add(self, r: DataArray<$t>) -> DataArray<$t> { r.map(|v| self + *v) } }
        impl Sub<DataArray<$t>> for $t { type Output = DataArray<$t>; fn sub(self, r: DataArray<$t>) -> DataArray<$t> { r.map(|v| self - *v) } }
        impl Mul<DataArray<$t>> for $t { type Output = DataArray<$t>; fn mul(self, r: DataArray<$t>) -> DataArray<$t> { r.map(|v| self * *v) } }
        impl Div<DataArray<$t>> for $t { type Output = DataArray<$t>; fn div(self, r: DataArray<$t>) -> DataArray<$t> { r.map(|v| self / *v) } }
    )*};
}
scalar_lhs!(f64, f32, i64, i32);

impl<T: Element + Copy + Neg<Output = T>> Neg for &DataArray<T> {
    type Output = DataArray<T>;
    fn neg(self) -> DataArray<T> {
        self.map(|v| -*v)
    }
}
impl<T: Element + Copy + Neg<Output = T>> Neg for DataArray<T> {
    type Output = DataArray<T>;
    fn neg(self) -> DataArray<T> {
        self.map(|v| -*v)
    }
}

// ---- comparisons -----------------------------------------------------------

impl<T: Element + PartialOrd> DataArray<T> {
    fn cmp_scalar(&self, v: &T, f: impl Fn(&T, &T) -> bool) -> DataArray<bool> {
        let mut out = self.map(|x| f(x, v));
        out.attrs.clear();
        out
    }
    /// `self > value`
    pub fn gt(&self, value: T) -> DataArray<bool> {
        self.cmp_scalar(&value, |a, b| a > b)
    }
    /// `self >= value`
    pub fn ge(&self, value: T) -> DataArray<bool> {
        self.cmp_scalar(&value, |a, b| a >= b)
    }
    /// `self < value`
    pub fn lt(&self, value: T) -> DataArray<bool> {
        self.cmp_scalar(&value, |a, b| a < b)
    }
    /// `self <= value`
    pub fn le(&self, value: T) -> DataArray<bool> {
        self.cmp_scalar(&value, |a, b| a <= b)
    }
    /// `self == value`
    pub fn eq_value(&self, value: T) -> DataArray<bool> {
        self.cmp_scalar(&value, |a, b| a == b)
    }
    /// `self != value`
    pub fn ne_value(&self, value: T) -> DataArray<bool> {
        self.cmp_scalar(&value, |a, b| a != b)
    }
    /// Element-wise `self > other` with alignment and broadcasting.
    pub fn gt_array(&self, other: &DataArray<T>) -> Result<DataArray<bool>> {
        self.zip_with(other, |a, b| a > b)
    }
    /// Element-wise `self < other` with alignment and broadcasting.
    pub fn lt_array(&self, other: &DataArray<T>) -> Result<DataArray<bool>> {
        self.zip_with(other, |a, b| a < b)
    }
    /// Element-wise `self == other` with alignment and broadcasting.
    pub fn eq_array(&self, other: &DataArray<T>) -> Result<DataArray<bool>> {
        self.zip_with(other, |a, b| a == b)
    }
}

impl std::ops::BitAnd for &DataArray<bool> {
    type Output = DataArray<bool>;
    fn bitand(self, rhs: Self) -> DataArray<bool> {
        self.zip_with(rhs, |a, b| *a && *b).unwrap_or_else(|e| panic!("{e}"))
    }
}
impl std::ops::BitOr for &DataArray<bool> {
    type Output = DataArray<bool>;
    fn bitor(self, rhs: Self) -> DataArray<bool> {
        self.zip_with(rhs, |a, b| *a || *b).unwrap_or_else(|e| panic!("{e}"))
    }
}
impl std::ops::Not for &DataArray<bool> {
    type Output = DataArray<bool>;
    fn not(self) -> DataArray<bool> {
        self.map(|v| !*v)
    }
}

// ---- float math ------------------------------------------------------------

macro_rules! unary_float {
    ($($name:ident),*) => {
        impl<T: FloatElement> DataArray<T> {
            $(
            #[doc = concat!("Element-wise `", stringify!($name), "`.")]
            pub fn $name(&self) -> DataArray<T> { self.map(|v| v.$name()) }
            )*
        }
    };
}
unary_float!(abs, sqrt, exp, ln, log10, log2, sin, cos, tan, floor, ceil, round);

impl<T: FloatElement> DataArray<T> {
    /// Element-wise power.
    pub fn powf(&self, p: T) -> DataArray<T> {
        self.map(|v| v.powf(p))
    }
    /// Limits values to `[min, max]` (NaN stays NaN).
    pub fn clip(&self, min: T, max: T) -> DataArray<T> {
        self.map(|v| if v.is_nan() { *v } else { v.max(min).min(max) })
    }
}

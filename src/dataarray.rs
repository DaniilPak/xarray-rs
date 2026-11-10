//! [`DataArray`]: an N-dimensional array with named dimensions, coordinates
//! and attributes.

use crate::attrs::{AttrValue, Attrs};
use crate::coords::{Coord, CoordData, Index, IntoDims, Label, Method};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::indexing::{IndexSel, Sel};
use crate::time;
use indexmap::IndexMap;
use ndarray::{Array, ArrayD, Axis, Dimension, IxDyn};

/// Conversion into an `ndarray::ArrayD`.
pub trait IntoArrayD<T> {
    /// Converts into a dynamic-dimensional array.
    fn into_arrayd(self) -> ArrayD<T>;
}
impl<T, D: Dimension> IntoArrayD<T> for Array<T, D> {
    fn into_arrayd(self) -> ArrayD<T> {
        self.into_dyn()
    }
}
impl<T> IntoArrayD<T> for Vec<T> {
    fn into_arrayd(self) -> ArrayD<T> {
        ndarray::Array1::from(self).into_dyn()
    }
}

/// Something that can become a [`Coord`] when attached under a name.
///
/// * plain values (`Vec`, arrays, `ndarray` arrays) become a 1-D coordinate
///   along the dimension with the same name (an *index* coordinate);
/// * `(dims, values)` tuples attach a coordinate along explicit dimensions
///   (a *non-index* coordinate, possibly N-D);
/// * a ready-made [`Coord`].
pub trait IntoCoord {
    /// Converts into a coordinate named `name`.
    fn into_coord(self, name: &str) -> Result<Coord>;
}
impl IntoCoord for Coord {
    fn into_coord(self, _: &str) -> Result<Coord> {
        Ok(self)
    }
}
impl IntoCoord for CoordData {
    fn into_coord(self, name: &str) -> Result<Coord> {
        if self.shape().is_empty() { Coord::new((), self) } else { Coord::new(name, self) }
    }
}
impl<T: crate::coords::CoordValue> IntoCoord for Vec<T> {
    fn into_coord(self, name: &str) -> Result<Coord> {
        Coord::new(name, self)
    }
}
impl<T: crate::coords::CoordValue, const N: usize> IntoCoord for [T; N] {
    fn into_coord(self, name: &str) -> Result<Coord> {
        Coord::new(name, self)
    }
}
impl<T: crate::coords::CoordValue, D: Dimension> IntoCoord for Array<T, D> {
    fn into_coord(self, name: &str) -> Result<Coord> {
        let a = self.into_dyn();
        if a.ndim() != 1 {
            return Err(Error::invalid(format!("coordinate {name:?} is {}-D; pass (dims, values)", a.ndim())));
        }
        Coord::new(name, a)
    }
}
impl<D: IntoDims, V: Into<CoordData>> IntoCoord for (D, V) {
    fn into_coord(self, _: &str) -> Result<Coord> {
        Coord::new(self.0, self.1)
    }
}

/// An N-dimensional array with named dimensions, coordinate labels and
/// attributes — the Rust counterpart of `xarray.DataArray`.
///
/// ```
/// use xarray_rs::DataArray;
/// use ndarray::array;
///
/// let da = DataArray::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["x", "y"])?
///     .with_coord("x", vec![10, 20])?
///     .with_coord("y", vec!["a", "b", "c"])?
///     .with_name("temp")
///     .with_attr("units", "K");
///
/// let row = da.sel("x", 20)?;            // drops "x"
/// assert_eq!(row.dims(), ["y"]);
/// assert_eq!(da.mean("y")?.to_vec(), vec![2.0, 5.0]);
/// # Ok::<(), xarray_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DataArray<T: Element = f64> {
    pub(crate) data: ArrayD<T>,
    pub(crate) dims: Vec<String>,
    pub(crate) coords: IndexMap<String, Coord>,
    pub(crate) attrs: Attrs,
    pub(crate) name: Option<String>,
}

impl<T: Element> DataArray<T> {
    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    /// Wraps `data` with dimension names (one per axis, all distinct).
    pub fn new(data: impl IntoArrayD<T>, dims: impl IntoDims) -> Result<Self> {
        let data = data.into_arrayd();
        let dims = dims.into_dims();
        if dims.len() != data.ndim() {
            return Err(Error::shape(format!(
                "{} dimension names {:?} for data with shape {:?}",
                dims.len(),
                dims,
                data.shape()
            )));
        }
        for (i, d) in dims.iter().enumerate() {
            if dims[..i].contains(d) {
                return Err(Error::invalid(format!("duplicate dimension name {d:?}")));
            }
        }
        Ok(DataArray { data, dims, coords: IndexMap::new(), attrs: Attrs::new(), name: None })
    }

    /// A 0-dimensional array holding one value.
    pub fn scalar(value: T) -> Self {
        DataArray {
            data: ArrayD::from_elem(IxDyn(&[]), value),
            dims: vec![],
            coords: IndexMap::new(),
            attrs: Attrs::new(),
            name: None,
        }
    }

    /// Array of `shape` filled with `value`.
    pub fn full(value: T, dims: impl IntoDims, shape: &[usize]) -> Result<Self> {
        DataArray::new(ArrayD::from_elem(IxDyn(shape), value), dims)
    }

    pub(crate) fn from_parts(
        data: ArrayD<T>,
        dims: Vec<String>,
        coords: IndexMap<String, Coord>,
        attrs: Attrs,
        name: Option<String>,
    ) -> Self {
        DataArray { data, dims, coords, attrs, name }
    }

    /// Attaches a coordinate (see [`IntoCoord`] for accepted values).
    pub fn with_coord(mut self, name: impl Into<String>, coord: impl IntoCoord) -> Result<Self> {
        let name = name.into();
        let c = coord.into_coord(&name)?;
        self.check_coord(&name, &c)?;
        self.coords.insert(name, c);
        Ok(self)
    }
    /// Same as [`with_coord`](Self::with_coord) but in place.
    pub fn assign_coord(&mut self, name: impl Into<String>, coord: impl IntoCoord) -> Result<()> {
        let name = name.into();
        let c = coord.into_coord(&name)?;
        self.check_coord(&name, &c)?;
        self.coords.insert(name, c);
        Ok(())
    }
    fn check_coord(&self, name: &str, c: &Coord) -> Result<()> {
        for (d, &n) in c.dims.iter().zip(c.shape()) {
            match self.dims.iter().position(|x| x == d) {
                None => {
                    return Err(Error::invalid(format!(
                        "coordinate {name:?} has dimension {d:?} which is not a dimension of the array {:?}",
                        self.dims
                    )));
                }
                Some(ax) if self.data.shape()[ax] != n => {
                    return Err(Error::shape(format!(
                        "coordinate {name:?} has length {n} along {d:?} but the array has {}",
                        self.data.shape()[ax]
                    )));
                }
                _ => {}
            }
        }
        if self.dims.iter().any(|d| d == name) && c.dims != [name] {
            return Err(Error::invalid(format!(
                "coordinate {name:?} has the name of a dimension, so it must be 1-D along it"
            )));
        }
        Ok(())
    }
    /// Sets the name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Adds an attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }
    /// Replaces all attributes.
    pub fn with_attrs(mut self, attrs: Attrs) -> Self {
        self.attrs = attrs;
        self
    }

    // ------------------------------------------------------------------
    // accessors
    // ------------------------------------------------------------------

    /// The underlying array.
    pub fn data(&self) -> &ArrayD<T> {
        &self.data
    }
    /// Alias of [`data`](Self::data) (`.values` in xarray).
    pub fn values(&self) -> &ArrayD<T> {
        &self.data
    }
    /// Mutable access to the values (shape must not change).
    pub fn data_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }
    /// Consumes the array, returning the values.
    pub fn into_data(self) -> ArrayD<T> {
        self.data
    }
    /// Values flattened in row-major order.
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
    /// The single value of a 0-d (or size-1) array.
    pub fn item(&self) -> Result<T> {
        if self.data.len() != 1 {
            return Err(Error::shape(format!("item() needs exactly one element, have {}", self.data.len())));
        }
        Ok(self.data.iter().next().unwrap().clone())
    }
    /// Dimension names.
    pub fn dims(&self) -> &[String] {
        &self.dims
    }
    /// Shape.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    /// Mapping from dimension name to length.
    pub fn sizes(&self) -> IndexMap<String, usize> {
        self.dims.iter().cloned().zip(self.data.shape().iter().copied()).collect()
    }
    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }
    /// Name, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// Sets or clears the name in place.
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
    /// Attributes.
    pub fn attrs(&self) -> &Attrs {
        &self.attrs
    }
    /// Mutable attributes.
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        &mut self.attrs
    }
    /// All coordinates.
    pub fn coords(&self) -> &IndexMap<String, Coord> {
        &self.coords
    }
    /// One coordinate by name.
    pub fn coord(&self, name: &str) -> Result<&Coord> {
        self.coords.get(name).ok_or_else(|| Error::NotFound(name.to_string()))
    }
    /// Names of the index coordinates (1-D coordinates named after their
    /// dimension).
    pub fn indexes(&self) -> Vec<&str> {
        self.dims.iter().filter(|d| self.has_index(d)).map(|d| d.as_str()).collect()
    }
    pub(crate) fn has_index(&self, dim: &str) -> bool {
        self.coords.get(dim).is_some_and(|c| c.dims == [dim])
    }
    pub(crate) fn index(&self, dim: &str) -> Option<Index> {
        self.coords.get(dim).filter(|c| c.dims == [dim]).and_then(|c| c.to_index().ok())
    }
    /// Axis number of a dimension.
    pub fn axis(&self, dim: &str) -> Result<usize> {
        self.dims.iter().position(|d| d == dim).ok_or_else(|| Error::DimNotFound(dim.to_string(), self.dims.clone()))
    }
    /// Length of a dimension.
    pub fn dim_len(&self, dim: &str) -> Result<usize> {
        Ok(self.data.shape()[self.axis(dim)?])
    }
    /// Removes coordinates by name.
    pub fn drop_vars(mut self, names: impl IntoDims) -> Result<Self> {
        for n in names.into_dims() {
            self.coords.shift_remove(&n).ok_or(Error::NotFound(n))?;
        }
        Ok(self)
    }
    /// Drops all non-index coordinates (`reset_coords(drop=True)`).
    pub fn reset_coords(mut self) -> Self {
        let dims = self.dims.clone();
        self.coords.retain(|k, c| dims.contains(k) && c.dims == [k.clone()]);
        self
    }

    // ------------------------------------------------------------------
    // structural primitives
    // ------------------------------------------------------------------

    /// Orthogonal take along one dimension (keeps the dimension).
    pub(crate) fn take_dim(&self, dim: &str, idx: &[usize]) -> Result<Self> {
        let ax = self.axis(dim)?;
        Ok(DataArray {
            data: self.data.select(Axis(ax), idx),
            dims: self.dims.clone(),
            coords: self.coords.iter().map(|(k, c)| (k.clone(), c.take(dim, idx))).collect(),
            attrs: self.attrs.clone(),
            name: self.name.clone(),
        })
    }
    /// Take with missing positions filled by `T::missing()`.
    pub(crate) fn take_dim_opt(&self, dim: &str, idx: &[Option<usize>]) -> Result<Self> {
        if idx.iter().all(Option::is_some) {
            let idx: Vec<usize> = idx.iter().map(|i| i.unwrap()).collect();
            return self.take_dim(dim, &idx);
        }
        let fill = T::missing().ok_or_else(|| {
            Error::Dtype(format!(
                "{} data cannot hold missing values introduced by reindexing; convert to float first",
                T::dtype()
            ))
        })?;
        let ax = self.axis(dim)?;
        Ok(DataArray {
            data: crate::coords::take_fill(&self.data, ax, idx, fill),
            dims: self.dims.clone(),
            coords: self.coords.iter().map(|(k, c)| (k.clone(), c.take_opt(dim, idx))).collect(),
            attrs: self.attrs.clone(),
            name: self.name.clone(),
        })
    }
    /// Select one position and drop the dimension.
    pub(crate) fn index_dim(&self, dim: &str, i: usize) -> Result<Self> {
        let ax = self.axis(dim)?;
        let mut dims = self.dims.clone();
        dims.remove(ax);
        Ok(DataArray {
            data: self.data.index_axis(Axis(ax), i).to_owned(),
            dims,
            coords: self.coords.iter().map(|(k, c)| (k.clone(), c.index(dim, i))).collect(),
            attrs: self.attrs.clone(),
            name: self.name.clone(),
        })
    }

    // ------------------------------------------------------------------
    // indexing
    // ------------------------------------------------------------------

    /// Positional selection along one dimension (`da.isel(dim=...)`).
    ///
    /// Accepts an integer (drops the dimension), a range, or a list of
    /// positions. Negative positions count from the end.
    pub fn isel(&self, dim: &str, sel: impl Into<IndexSel>) -> Result<Self> {
        let n = self.dim_len(dim)?;
        let (pos, keep) = sel.into().resolve(dim, n)?;
        if keep { self.take_dim(dim, &pos) } else { self.index_dim(dim, pos[0]) }
    }
    /// Positional selection along several dimensions at once.
    pub fn isel_multi<S: Into<IndexSel>>(&self, sels: impl IntoIterator<Item = (impl AsRef<str>, S)>) -> Result<Self> {
        let mut out = self.clone();
        for (d, s) in sels {
            out = out.isel(d.as_ref(), s)?;
        }
        Ok(out)
    }

    /// Label-based selection along one dimension (`da.sel(dim=...)`).
    ///
    /// Accepts a label (drops the dimension), a list of labels, or an
    /// inclusive label range such as `10..=30` or `"2000-01-01"..="2000-01-05"`.
    pub fn sel(&self, dim: &str, sel: impl Into<Sel>) -> Result<Self> {
        self.sel_method(dim, sel, Method::Exact, None)
    }
    /// Nearest-label selection (`da.sel(dim=label, method="nearest")`).
    pub fn sel_nearest(&self, dim: &str, sel: impl Into<Sel>) -> Result<Self> {
        self.sel_method(dim, sel, Method::Nearest, None)
    }
    /// Label-based selection with an inexact matching method and optional
    /// tolerance (in index units; nanoseconds for datetimes).
    pub fn sel_method(&self, dim: &str, sel: impl Into<Sel>, method: Method, tolerance: Option<f64>) -> Result<Self> {
        let pos = self.sel_positions(dim, &sel.into(), method, tolerance)?;
        match pos {
            SelPos::One(i) => self.index_dim(dim, i),
            SelPos::Many(v) => self.take_dim(dim, &v),
        }
    }
    /// Label-based selection along several dimensions.
    pub fn sel_multi<S: Into<Sel>>(&self, sels: impl IntoIterator<Item = (impl AsRef<str>, S)>) -> Result<Self> {
        let mut out = self.clone();
        for (d, s) in sels {
            out = out.sel(d.as_ref(), s)?;
        }
        Ok(out)
    }

    pub(crate) fn sel_positions(&self, dim: &str, sel: &Sel, method: Method, tolerance: Option<f64>) -> Result<SelPos> {
        self.axis(dim)?;
        let index = self.index(dim).ok_or_else(|| Error::invalid(format!("no index found for dimension {dim:?}")))?;
        sel_positions(&index, sel, method, tolerance)
    }

    /// Keeps values where `cond` is true and fills the rest with
    /// `T::missing()` (NaN). `cond` is broadcast and aligned by name.
    pub fn where_(&self, cond: &DataArray<bool>) -> Result<Self> {
        let fill = T::missing()
            .ok_or_else(|| Error::Dtype("where_ without `other` needs a float array; use where_or".into()))?;
        self.where_or(cond, fill)
    }
    /// Keeps values where `cond` is true, uses `other` elsewhere.
    pub fn where_or(&self, cond: &DataArray<bool>, other: T) -> Result<Self> {
        let (a, c) = crate::align::align2(self, cond, crate::align::Join::Inner)?;
        let mut out = crate::ops::zip_with(&a, &c, |x, &k| if k { x.clone() } else { other.clone() })?;
        out.name = self.name.clone();
        out.attrs = self.attrs.clone();
        Ok(out)
    }
    /// Keeps values where `cond` is true, takes values from `other` elsewhere.
    pub fn where_else(&self, cond: &DataArray<bool>, other: &DataArray<T>) -> Result<Self> {
        let (a, c) = crate::align::align2(self, cond, crate::align::Join::Inner)?;
        let (a, o) = crate::align::align2(&a, other, crate::align::Join::Inner)?;
        let c = crate::align::align2(&c, &o, crate::align::Join::Inner)?.0;
        let o2 = crate::ops::zip_with(&c, &o, |_, y| y.clone())?;
        let out = crate::ops::zip_with(&a, &c, |x, _| x.clone())?;
        let (dims, shape) = crate::ops::union_dims(&[(&out.dims, out.shape()), (&o2.dims, o2.shape())])?;
        let mut out = crate::ops::broadcast_to(&out, &dims, &shape)?;
        let cb = crate::ops::broadcast_owned(&c.data, &c.dims, &dims, &shape)?;
        let ob = crate::ops::broadcast_owned(&o2.data, &o2.dims, &dims, &shape)?;
        ndarray::Zip::from(&mut out.data).and(&cb).and(&ob).for_each(|x, &k, y| {
            if !k {
                *x = y.clone();
            }
        });
        for (k, v) in &o2.coords {
            out.coords.entry(k.clone()).or_insert_with(|| v.clone());
        }
        out.name = self.name.clone();
        out.attrs = self.attrs.clone();
        Ok(out)
    }

    // ------------------------------------------------------------------
    // reshaping
    // ------------------------------------------------------------------

    /// Reorders dimensions. `order` must name every dimension; use `"..."`
    /// once to stand for all dimensions not listed.
    pub fn transpose(&self, order: impl IntoDims) -> Result<Self> {
        let order = expand_ellipsis(&self.dims, order.into_dims())?;
        let perm: Vec<usize> = order.iter().map(|d| self.axis(d)).collect::<Result<_>>()?;
        Ok(DataArray {
            data: self.data.clone().permuted_axes(perm.clone()).as_standard_layout().into_owned(),
            dims: order,
            coords: self.coords.clone(),
            attrs: self.attrs.clone(),
            name: self.name.clone(),
        })
    }
    /// Reverses the order of dimensions (`da.T`).
    pub fn t(&self) -> Self {
        let mut order = self.dims.clone();
        order.reverse();
        self.transpose(order).expect("valid dims")
    }
    /// Returns a copy with a new name (`da.rename("new")`).
    pub fn rename(&self, name: impl Into<String>) -> Self {
        let mut out = self.clone();
        out.name = Some(name.into());
        out
    }
    /// Renames a dimension and/or coordinate (`da.rename({"old": "new"})`):
    /// a dimension named `old` is renamed, and so is a coordinate named `old`.
    pub fn rename_vars(&self, old: &str, new: &str) -> Result<Self> {
        let is_dim = self.dims.iter().any(|d| d == old);
        let is_coord = self.coords.contains_key(old);
        if !is_dim && !is_coord {
            return Err(Error::NotFound(old.to_string()));
        }
        let mut out = self.clone();
        if is_dim {
            for d in &mut out.dims {
                if d == old {
                    *d = new.to_string();
                }
            }
            for c in out.coords.values_mut() {
                c.rename_dim(old, new);
            }
        }
        if is_coord {
            let idx = out.coords.get_index_of(old).unwrap();
            let (_, c) = out.coords.shift_remove_index(idx).unwrap();
            out.coords.shift_insert(idx, new.to_string(), c);
        }
        Ok(out)
    }
    /// Renames a dimension only (`swap`-free `rename_dims`).
    pub fn rename_dims(&self, old: &str, new: &str) -> Result<Self> {
        self.axis(old)?;
        let mut out = self.clone();
        for d in &mut out.dims {
            if d == old {
                *d = new.to_string();
            }
        }
        for c in out.coords.values_mut() {
            c.rename_dim(old, new);
        }
        Ok(out)
    }
    /// Swaps a dimension for a 1-D coordinate along it (`swap_dims`).
    pub fn swap_dims(&self, old: &str, new: &str) -> Result<Self> {
        let c = self.coord(new)?;
        if c.dims != [old] {
            return Err(Error::invalid(format!("coordinate {new:?} must be 1-D along {old:?}")));
        }
        let mut out = self.rename_dims(old, new)?;
        if let Some(prev) = out.coords.shift_remove(old) {
            // the old index becomes a non-index coord along the new dim
            out.coords.insert(old.to_string(), prev);
        }
        Ok(out)
    }
    /// Drops all length-1 dimensions; their coordinates become scalar
    /// coordinates.
    pub fn squeeze(&self) -> Self {
        let mut out = self.clone();
        let ones: Vec<String> =
            self.dims.iter().zip(self.shape()).filter(|(_, n)| **n == 1).map(|(d, _)| d.clone()).collect();
        for d in ones {
            out = out.index_dim(&d, 0).expect("dim exists");
        }
        out
    }
    /// Drops one length-1 dimension.
    pub fn squeeze_dim(&self, dim: &str) -> Result<Self> {
        if self.dim_len(dim)? != 1 {
            return Err(Error::shape(format!("cannot squeeze dimension {dim:?} with length {}", self.dim_len(dim)?)));
        }
        self.index_dim(dim, 0)
    }
    /// Inserts a new length-1 dimension at axis 0 (`expand_dims`). If a
    /// scalar coordinate with that name exists, it becomes the new index.
    pub fn expand_dims(&self, dim: &str) -> Result<Self> {
        self.expand_dims_at(dim, 0)
    }
    /// Inserts a new length-1 dimension at `axis`.
    pub fn expand_dims_at(&self, dim: &str, axis: usize) -> Result<Self> {
        if self.dims.iter().any(|d| d == dim) {
            return Err(Error::invalid(format!("dimension {dim:?} already exists")));
        }
        if axis > self.ndim() {
            return Err(Error::invalid(format!("axis {axis} out of range")));
        }
        let mut out = self.clone();
        out.data = self.data.clone().insert_axis(Axis(axis));
        out.dims.insert(axis, dim.to_string());
        if let Some(c) = out.coords.get_mut(dim)
            && c.dims.is_empty()
        {
            *c = c.insert_dim(0, dim, 1);
        }
        Ok(out)
    }
    /// Inserts a new dimension at axis 0 with the given coordinate labels; the
    /// data is repeated along it.
    pub fn expand_dims_with(&self, dim: &str, labels: impl Into<CoordData>) -> Result<Self> {
        let labels = labels.into();
        let n = labels.shape().iter().product();
        let mut out = self.expand_dims(dim)?;
        let mut shape = out.data.shape().to_vec();
        shape[0] = n;
        out.data = out.data.broadcast(IxDyn(&shape)).unwrap().to_owned();
        out.coords.insert(dim.to_string(), Coord::new(dim, labels)?);
        Ok(out)
    }

    // ------------------------------------------------------------------
    // element-wise helpers
    // ------------------------------------------------------------------

    /// Applies `f` element-wise, keeping dims and coords (attrs dropped like
    /// xarray's default `keep_attrs=False`... except for `map`, which keeps
    /// them).
    pub fn map<U: Element>(&self, f: impl Fn(&T) -> U) -> DataArray<U> {
        DataArray {
            data: self.data.map(f),
            dims: self.dims.clone(),
            coords: self.coords.clone(),
            attrs: self.attrs.clone(),
            name: self.name.clone(),
        }
    }
    /// Replaces the values with an array of the same shape.
    pub fn with_data<U: Element>(&self, data: ArrayD<U>) -> Result<DataArray<U>> {
        if data.shape() != self.shape() {
            return Err(Error::shape(format!("expected shape {:?}, got {:?}", self.shape(), data.shape())));
        }
        Ok(DataArray {
            data,
            dims: self.dims.clone(),
            coords: self.coords.clone(),
            attrs: self.attrs.clone(),
            name: self.name.clone(),
        })
    }
    /// Numeric cast (`astype`).
    pub fn astype<U>(&self) -> DataArray<U>
    where
        T: num_traits::AsPrimitive<U>,
        U: Element + Copy,
    {
        self.map(|v| v.as_())
    }
    /// Boolean mask of missing values.
    pub fn isnull(&self) -> DataArray<bool> {
        let mut m = self.map(|v| v.is_missing());
        m.attrs.clear();
        m
    }
    /// Boolean mask of present values.
    pub fn notnull(&self) -> DataArray<bool> {
        let mut m = self.map(|v| !v.is_missing());
        m.attrs.clear();
        m
    }
    /// Replaces missing values with `value`.
    pub fn fillna(&self, value: T) -> Self {
        self.map(|v| if v.is_missing() { value.clone() } else { v.clone() })
    }
    /// Drops labels along `dim` where any (`all = false`) or all
    /// (`all = true`) values are missing.
    pub fn dropna(&self, dim: &str, all: bool) -> Result<Self> {
        let ax = self.axis(dim)?;
        let keep: Vec<usize> = (0..self.shape()[ax])
            .filter(|&i| {
                let lane = self.data.index_axis(Axis(ax), i);
                if all { !lane.iter().all(|v| v.is_missing()) } else { !lane.iter().any(|v| v.is_missing()) }
            })
            .collect();
        self.take_dim(dim, &keep)
    }
    /// True if dims, coordinates and values are equal (NaN equals NaN).
    pub fn equals(&self, other: &Self) -> bool {
        self.dims == other.dims
            && self.shape() == other.shape()
            && self.data.iter().zip(other.data.iter()).all(|(a, b)| a == b || (a.is_missing() && b.is_missing()))
            && self.coords.len() == other.coords.len()
            && self.coords.iter().all(|(k, c)| other.coords.get(k).is_some_and(|o| coord_equals(c, o)))
    }
    /// Like [`equals`](Self::equals) but also compares name and attrs.
    pub fn identical(&self, other: &Self) -> bool {
        self.equals(other)
            && self.name == other.name
            && self.attrs == other.attrs
            && self.coords.iter().all(|(k, c)| other.coords.get(k).is_some_and(|o| c.attrs == o.attrs))
    }

    /// Conforms to new labels along `dim`, filling gaps with the missing value.
    pub fn reindex(&self, dim: &str, labels: impl Into<CoordData>) -> Result<Self> {
        self.reindex_method(dim, labels, Method::Exact, None)
    }
    /// Reindex with an inexact matching method.
    pub fn reindex_method(
        &self,
        dim: &str,
        labels: impl Into<CoordData>,
        method: Method,
        tolerance: Option<f64>,
    ) -> Result<Self> {
        let new = Coord::new(dim, labels)?;
        let idx = self.index(dim).ok_or_else(|| Error::invalid(format!("no index for dimension {dim:?}")))?;
        let pos = idx.get_indexer(&new.labels(), method, tolerance)?;
        let mut out = self.take_dim_opt(dim, &pos)?;
        let attrs = out.coords.get(dim).map(|c| c.attrs.clone()).unwrap_or_default();
        out.coords.insert(dim.to_string(), Coord { attrs, ..new });
        Ok(out)
    }
    /// Sorts along `dim` by its index (ascending or descending).
    pub fn sortby(&self, dim: &str, ascending: bool) -> Result<Self> {
        let idx = self.index(dim).ok_or_else(|| Error::invalid(format!("no index for dimension {dim:?}")))?;
        let mut order: Vec<usize> = (0..idx.len()).collect();
        let probes: Vec<_> = (0..idx.len()).map(|i| idx.probe(&idx.label(i)).unwrap()).collect();
        order.sort_by(|&a, &b| idx.cmp(a, &probes[b]));
        if !ascending {
            order.reverse();
        }
        self.take_dim(dim, &order)
    }
    /// Shifts values along `dim` by `n` positions, filling with missing values
    /// (coordinates stay put).
    pub fn shift(&self, dim: &str, n: isize) -> Result<Self> {
        let len = self.dim_len(dim)? as isize;
        let pos: Vec<Option<usize>> = (0..len)
            .map(|i| {
                let j = i - n;
                (0..len).contains(&j).then_some(j as usize)
            })
            .collect();
        let mut out = self.take_dim_opt(dim, &pos)?;
        out.coords = self.coords.clone();
        Ok(out)
    }
    /// Rolls values along `dim` by `n` positions (coordinates stay put, like
    /// `roll_coords=False`).
    pub fn roll(&self, dim: &str, n: isize) -> Result<Self> {
        let len = self.dim_len(dim)? as isize;
        if len == 0 {
            return Ok(self.clone());
        }
        let pos: Vec<usize> = (0..len).map(|i| (i - n).rem_euclid(len) as usize).collect();
        let mut out = self.take_dim(dim, &pos)?;
        out.coords = self.coords.clone();
        Ok(out)
    }
    /// Broadcasts against `other`, returning `self` expanded to the union of
    /// both arrays' dimensions.
    pub fn broadcast_like<U: Element>(&self, other: &DataArray<U>) -> Result<Self> {
        let (a, b) = crate::align::align2(self, other, crate::align::Join::Outer)?;
        let (dims, shape) = crate::ops::union_dims(&[(&b.dims, b.shape()), (&a.dims, a.shape())])?;
        let mut out = crate::ops::broadcast_to(&a, &dims, &shape)?;
        for (k, c) in &b.coords {
            if c.dims == [k.clone()] {
                out.coords.entry(k.clone()).or_insert_with(|| c.clone());
            }
        }
        Ok(out)
    }

    /// Converts to a single-variable [`Dataset`](crate::Dataset).
    pub fn to_dataset(&self, name: Option<&str>) -> Result<crate::Dataset<T>> {
        let name = name
            .map(String::from)
            .or_else(|| self.name.clone())
            .ok_or_else(|| Error::invalid("unnamed DataArray needs a name"))?;
        let mut ds = crate::Dataset::new();
        ds.insert(name, self.clone())?;
        Ok(ds)
    }
}

/// Result of resolving a label selector to positions.
pub(crate) enum SelPos {
    One(usize),
    Many(Vec<usize>),
}

pub(crate) fn sel_positions(index: &Index, sel: &Sel, method: Method, tolerance: Option<f64>) -> Result<SelPos> {
    let dim = &index.dim;
    match sel {
        Sel::Label(l) => {
            if let (true, Label::Str(s), Method::Exact) = (index.is_datetime(), l, method) {
                let (a, b, res) = time::parse_partial(s)?;
                let values = match &index.kind {
                    crate::coords::IndexKind::I(v, _) => v,
                    _ => unreachable!(),
                };
                if res < time::index_resolution(values) {
                    let (a, b) = (time::to_ns(a), time::to_ns(b));
                    let pos: Vec<usize> = (0..values.len()).filter(|&i| values[i] >= a && values[i] < b).collect();
                    if pos.is_empty() {
                        return Err(Error::LabelNotFound { dim: dim.clone(), label: l.to_string() });
                    }
                    return Ok(SelPos::Many(pos));
                }
            }
            if method == Method::Exact {
                let pos = index.locate(l)?;
                match pos.len() {
                    0 => Err(Error::LabelNotFound { dim: dim.clone(), label: l.to_string() }),
                    1 => Ok(SelPos::One(pos[0])),
                    _ => Ok(SelPos::Many(pos)),
                }
            } else {
                let pos = index.get_indexer(std::slice::from_ref(l), method, tolerance)?;
                pos[0].map(SelPos::One).ok_or_else(|| Error::LabelNotFound { dim: dim.clone(), label: l.to_string() })
            }
        }
        Sel::Labels(ls) => {
            let pos = index.get_indexer(ls, method, tolerance)?;
            let mut out = Vec::with_capacity(pos.len());
            for (p, l) in pos.into_iter().zip(ls) {
                out.push(p.ok_or_else(|| Error::LabelNotFound { dim: dim.clone(), label: l.to_string() })?);
            }
            Ok(SelPos::Many(out))
        }
        Sel::Slice(a, b) => {
            if method != Method::Exact {
                return Err(Error::invalid("cannot use method with a slice selector"));
            }
            let (s, e) = index.slice_locs(a.as_ref(), b.as_ref())?;
            Ok(SelPos::Many((s..e).collect()))
        }
    }
}

pub(crate) fn coord_equals(a: &Coord, b: &Coord) -> bool {
    a.dims == b.dims
        && match (&a.data, &b.data) {
            (CoordData::Float(x), CoordData::Float(y)) => {
                x.shape() == y.shape() && x.iter().zip(y.iter()).all(|(p, q)| p == q || (p.is_nan() && q.is_nan()))
            }
            (x, y) => x == y,
        }
}

pub(crate) fn expand_ellipsis(all: &[String], order: Vec<String>) -> Result<Vec<String>> {
    let rest: Vec<String> = all.iter().filter(|d| !order.contains(d)).cloned().collect();
    let mut out = Vec::new();
    let mut seen_ellipsis = false;
    for d in order {
        if d == "..." {
            if seen_ellipsis {
                return Err(Error::invalid("only one ellipsis allowed"));
            }
            seen_ellipsis = true;
            out.extend(rest.iter().cloned());
        } else {
            if !all.contains(&d) {
                return Err(Error::DimNotFound(d, all.to_vec()));
            }
            out.push(d);
        }
    }
    if out.len() != all.len() {
        return Err(Error::invalid(format!("transpose order {out:?} must be a permutation of {all:?}")));
    }
    Ok(out)
}

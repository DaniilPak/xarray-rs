//! [`Dataset`]: a dict-like collection of [`DataArray`]s sharing dimensions
//! and coordinates.

use crate::align::Join;
use crate::attrs::{AttrValue, Attrs};
use crate::coords::{Coord, CoordData, Index, IntoDims, Method};
use crate::dataarray::{DataArray, IntoCoord, SelPos, coord_equals, sel_positions};
use crate::element::{Element, FloatElement};
use crate::error::{Error, Result};
use crate::indexing::{IndexSel, Sel};
use indexmap::IndexMap;

/// A collection of named data variables that share dimensions and
/// coordinates — the counterpart of `xarray.Dataset`.
///
/// All data variables have the same element type `T`.
///
/// ```
/// use xarray_rs::{DataArray, Dataset};
/// let t = DataArray::new(vec![280.0, 285.0, 290.0], "x")?.with_coord("x", vec![1, 2, 3])?;
/// let p = DataArray::new(vec![1000.0, 990.0, 980.0], "x")?;
/// let ds = Dataset::new().with_var("temp", t)?.with_var("pres", p)?.with_attr("title", "demo");
/// assert_eq!(ds.get("pres")?.coord("x")?.len(), 3);
/// let m = ds.mean("x")?;
/// assert_eq!(m.get("temp")?.item()?, 285.0);
/// # Ok::<(), xarray_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset<T: Element = f64> {
    /// data variables, stored without coordinates
    pub(crate) vars: IndexMap<String, DataArray<T>>,
    pub(crate) coords: IndexMap<String, Coord>,
    pub(crate) attrs: Attrs,
}

impl<T: Element> Default for Dataset<T> {
    fn default() -> Self {
        Dataset { vars: IndexMap::new(), coords: IndexMap::new(), attrs: Attrs::new() }
    }
}

impl<T: Element> Dataset<T> {
    /// Empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a data variable; its coordinates are merged into the dataset's.
    pub fn with_var(mut self, name: impl Into<String>, da: DataArray<T>) -> Result<Self> {
        self.insert(name, da)?;
        Ok(self)
    }
    /// Adds or replaces a data variable in place.
    pub fn insert(&mut self, name: impl Into<String>, mut da: DataArray<T>) -> Result<()> {
        let name = name.into();
        let sizes = self.dims_excluding(Some(&name));
        for (d, n) in da.sizes() {
            if let Some(&m) = sizes.get(&d)
                && m != n
            {
                return Err(Error::shape(format!("variable {name:?} has length {n} along {d:?}, dataset has {m}")));
            }
        }
        for (k, c) in std::mem::take(&mut da.coords) {
            match self.coords.get(&k) {
                Some(old) if !coord_equals(old, &c) => {
                    return Err(Error::Merge(format!(
                        "coordinate {k:?} of variable {name:?} conflicts with the dataset's; align or merge first"
                    )));
                }
                Some(_) => {}
                None => {
                    self.coords.insert(k, c);
                }
            }
        }
        da.name = Some(name.clone());
        self.vars.insert(name, da);
        Ok(())
    }
    /// Adds a coordinate.
    pub fn with_coord(mut self, name: impl Into<String>, coord: impl IntoCoord) -> Result<Self> {
        self.assign_coord(name, coord)?;
        Ok(self)
    }
    /// Adds a coordinate in place.
    pub fn assign_coord(&mut self, name: impl Into<String>, coord: impl IntoCoord) -> Result<()> {
        let name = name.into();
        let c = coord.into_coord(&name)?;
        let sizes = self.dims();
        for (d, &n) in c.dims.iter().zip(c.shape()) {
            if let Some(&m) = sizes.get(d)
                && m != n
            {
                return Err(Error::shape(format!("coordinate {name:?} has length {n} along {d:?}, dataset has {m}")));
            }
        }
        self.coords.insert(name, c);
        Ok(())
    }
    /// Adds an attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }

    /// Dimension sizes, in order of first appearance.
    pub fn dims(&self) -> IndexMap<String, usize> {
        self.dims_excluding(None)
    }

    fn dims_excluding(&self, skip: Option<&str>) -> IndexMap<String, usize> {
        let mut ordered = IndexMap::new();
        for (k, v) in &self.vars {
            if Some(k.as_str()) == skip {
                continue;
            }
            for (d, n) in v.dims.iter().zip(v.shape()) {
                ordered.entry(d.clone()).or_insert(*n);
            }
        }
        for c in self.coords.values() {
            for (d, n) in c.dims.iter().zip(c.shape()) {
                ordered.entry(d.clone()).or_insert(*n);
            }
        }
        ordered
    }
    /// Names of the data variables.
    pub fn data_vars(&self) -> Vec<&str> {
        self.vars.keys().map(|s| s.as_str()).collect()
    }
    /// Whether a data variable exists.
    pub fn contains(&self, name: &str) -> bool {
        self.vars.contains_key(name)
    }
    /// Number of data variables.
    pub fn len(&self) -> usize {
        self.vars.len()
    }
    /// Whether there are no data variables.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
    /// Coordinates.
    pub fn coords(&self) -> &IndexMap<String, Coord> {
        &self.coords
    }
    /// One coordinate.
    pub fn coord(&self, name: &str) -> Result<&Coord> {
        self.coords.get(name).ok_or_else(|| Error::NotFound(name.to_string()))
    }
    /// Attributes.
    pub fn attrs(&self) -> &Attrs {
        &self.attrs
    }
    /// Mutable attributes.
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        &mut self.attrs
    }
    /// A data variable as a [`DataArray`] with the coordinates that apply to
    /// it (those whose dims are a subset of the variable's dims).
    pub fn get(&self, name: &str) -> Result<DataArray<T>> {
        let v = self.vars.get(name).ok_or_else(|| Error::NotFound(name.to_string()))?;
        let mut da = v.clone();
        for (k, c) in &self.coords {
            if c.dims.iter().all(|d| v.dims.contains(d)) {
                da.coords.insert(k.clone(), c.clone());
            }
        }
        Ok(da)
    }
    /// Iterates over `(name, DataArray)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, DataArray<T>)> + '_ {
        self.vars.keys().map(move |k| (k.as_str(), self.get(k).unwrap()))
    }
    /// Removes data variables or coordinates by name.
    pub fn drop_vars(mut self, names: impl IntoDims) -> Result<Self> {
        for n in names.into_dims() {
            if self.vars.shift_remove(&n).is_none() && self.coords.shift_remove(&n).is_none() {
                return Err(Error::NotFound(n));
            }
        }
        Ok(self)
    }
    /// Keeps only the listed data variables.
    pub fn select_vars(&self, names: impl IntoDims) -> Result<Self> {
        let mut out = Dataset { vars: IndexMap::new(), coords: self.coords.clone(), attrs: self.attrs.clone() };
        for n in names.into_dims() {
            let v = self.vars.get(&n).ok_or_else(|| Error::NotFound(n.clone()))?;
            out.vars.insert(n, v.clone());
        }
        Ok(out)
    }

    pub(crate) fn index(&self, dim: &str) -> Option<Index> {
        self.coords.get(dim).filter(|c| c.dims == [dim]).and_then(|c| c.to_index().ok())
    }
    fn dim_len(&self, dim: &str) -> Result<usize> {
        self.dims()
            .get(dim)
            .copied()
            .ok_or_else(|| Error::DimNotFound(dim.to_string(), self.dims().keys().cloned().collect()))
    }

    // ---- structural primitives ------------------------------------------------

    pub(crate) fn take_dim(&self, dim: &str, idx: &[usize]) -> Result<Self> {
        let mut out = self.clone();
        for v in out.vars.values_mut() {
            if v.dims.iter().any(|d| d == dim) {
                *v = v.take_dim(dim, idx)?;
            }
        }
        for c in out.coords.values_mut() {
            *c = c.take(dim, idx);
        }
        Ok(out)
    }
    pub(crate) fn take_dim_opt(&self, dim: &str, idx: &[Option<usize>]) -> Result<Self> {
        let mut out = self.clone();
        for v in out.vars.values_mut() {
            if v.dims.iter().any(|d| d == dim) {
                *v = v.take_dim_opt(dim, idx)?;
            }
        }
        for c in out.coords.values_mut() {
            *c = c.take_opt(dim, idx);
        }
        Ok(out)
    }
    pub(crate) fn index_dim(&self, dim: &str, i: usize) -> Result<Self> {
        let mut out = self.clone();
        for v in out.vars.values_mut() {
            if v.dims.iter().any(|d| d == dim) {
                *v = v.index_dim(dim, i)?;
            }
        }
        for c in out.coords.values_mut() {
            *c = c.index(dim, i);
        }
        Ok(out)
    }

    /// Positional selection along one dimension for every variable.
    pub fn isel(&self, dim: &str, sel: impl Into<IndexSel>) -> Result<Self> {
        let n = self.dim_len(dim)?;
        let (pos, keep) = sel.into().resolve(dim, n)?;
        if keep { self.take_dim(dim, &pos) } else { self.index_dim(dim, pos[0]) }
    }
    /// Label-based selection along one dimension for every variable.
    pub fn sel(&self, dim: &str, sel: impl Into<Sel>) -> Result<Self> {
        self.sel_method(dim, sel, Method::Exact, None)
    }
    /// Nearest-label selection.
    pub fn sel_nearest(&self, dim: &str, sel: impl Into<Sel>) -> Result<Self> {
        self.sel_method(dim, sel, Method::Nearest, None)
    }
    /// Label-based selection with a matching method.
    pub fn sel_method(&self, dim: &str, sel: impl Into<Sel>, method: Method, tolerance: Option<f64>) -> Result<Self> {
        self.dim_len(dim)?;
        let index = self.index(dim).ok_or_else(|| Error::invalid(format!("no index found for dimension {dim:?}")))?;
        match sel_positions(&index, &sel.into(), method, tolerance)? {
            SelPos::One(i) => self.index_dim(dim, i),
            SelPos::Many(v) => self.take_dim(dim, &v),
        }
    }
    /// Reindexes along `dim` (missing labels become NaN).
    pub fn reindex(&self, dim: &str, labels: impl Into<CoordData>) -> Result<Self> {
        let new = Coord::new(dim, labels)?;
        let idx = self.index(dim).ok_or_else(|| Error::invalid(format!("no index for dimension {dim:?}")))?;
        let pos = idx.get_indexer(&new.labels(), Method::Exact, None)?;
        let mut out = self.take_dim_opt(dim, &pos)?;
        out.coords.insert(dim.to_string(), new);
        Ok(out)
    }

    /// Applies `f` to every data variable (with its coordinates) and builds a
    /// new dataset from the results. Coordinates of the results replace the
    /// dataset's; unrelated coordinates are kept when their dims still exist.
    pub fn map<U: Element>(&self, f: impl Fn(&DataArray<T>) -> Result<DataArray<U>>) -> Result<Dataset<U>> {
        let mut out: Dataset<U> = Dataset { vars: IndexMap::new(), coords: IndexMap::new(), attrs: self.attrs.clone() };
        let mut results = Vec::new();
        for k in self.vars.keys() {
            results.push((k.clone(), f(&self.get(k)?)?));
        }
        let mut dims_after: Vec<String> = Vec::new();
        for (_, r) in &results {
            for d in &r.dims {
                if !dims_after.contains(d) {
                    dims_after.push(d.clone());
                }
            }
        }
        // first take coordinates from the results
        for (_, r) in &results {
            for (k, c) in &r.coords {
                out.coords.entry(k.clone()).or_insert_with(|| c.clone());
            }
        }
        // then keep the dataset's own coordinates whose dims survived
        for (k, c) in &self.coords {
            if !out.coords.contains_key(k) && c.dims.iter().all(|d| dims_after.contains(d)) {
                out.coords.insert(k.clone(), c.clone());
            }
        }
        for (k, mut r) in results {
            r.coords.clear();
            r.name = Some(k.clone());
            out.vars.insert(k, r);
        }
        Ok(out)
    }

    /// Renames a variable, coordinate or dimension.
    pub fn rename(&self, old: &str, new: &str) -> Result<Self> {
        let mut out = self.clone();
        let mut found = false;
        if let Some(i) = out.vars.get_index_of(old) {
            let (_, mut v) = out.vars.shift_remove_index(i).unwrap();
            v.name = Some(new.to_string());
            out.vars.shift_insert(i, new.to_string(), v);
            found = true;
        }
        if let Some(i) = out.coords.get_index_of(old) {
            let (_, c) = out.coords.shift_remove_index(i).unwrap();
            out.coords.shift_insert(i, new.to_string(), c);
            found = true;
        }
        if self.dims().contains_key(old) {
            for v in out.vars.values_mut() {
                *v = v.rename_dims(old, new).unwrap_or_else(|_| v.clone());
            }
            for c in out.coords.values_mut() {
                c.rename_dim(old, new);
            }
            found = true;
        }
        if !found {
            return Err(Error::NotFound(old.to_string()));
        }
        Ok(out)
    }
    /// Transposes every variable that has all the listed dims (`...` allowed).
    pub fn transpose(&self, order: impl IntoDims) -> Result<Self> {
        let order = order.into_dims();
        let mut out = self.clone();
        for v in out.vars.values_mut() {
            let mine: Vec<String> = order.iter().filter(|d| *d == "..." || v.dims.contains(d)).cloned().collect();
            let mine = if mine.iter().any(|d| d == "...") {
                mine
            } else {
                let mut m = mine;
                if m.len() != v.dims.len() {
                    m.push("...".into());
                }
                m
            };
            *v = v.transpose(mine)?;
        }
        Ok(out)
    }
    /// Drops length-1 dimensions.
    pub fn squeeze(&self) -> Result<Self> {
        let mut out = self.clone();
        for (d, n) in self.dims() {
            if n == 1 {
                out = out.index_dim(&d, 0)?;
            }
        }
        Ok(out)
    }
    /// Adds a length-1 dimension to every data variable.
    pub fn expand_dims(&self, dim: &str) -> Result<Self> {
        let mut out = self.clone();
        for v in out.vars.values_mut() {
            *v = v.expand_dims(dim)?;
        }
        if let Some(c) = out.coords.get_mut(dim)
            && c.dims.is_empty()
        {
            *c = c.insert_dim(0, dim, 1);
        }
        Ok(out)
    }

    /// Converts to one [`DataArray`] by stacking the variables along a new
    /// leading dimension `dim` (`to_dataarray`).
    pub fn to_dataarray(&self, dim: &str) -> Result<DataArray<T>> {
        let arrays: Vec<DataArray<T>> = self.vars.keys().map(|k| self.get(k)).collect::<Result<_>>()?;
        let refs: Vec<&DataArray<T>> = arrays.iter().collect();
        let b = crate::align::broadcast(&refs)?;
        let refs: Vec<&DataArray<T>> = b.iter().collect();
        let mut out = crate::combine::concat(&refs, dim)?;
        out.coords.insert(dim.to_string(), Coord::new(dim, self.vars.keys().cloned().collect::<Vec<_>>())?);
        out.attrs = self.attrs.clone();
        out.name = None;
        Ok(out)
    }

    /// True if variables, coordinates and values are equal.
    pub fn equals(&self, other: &Self) -> bool {
        self.vars.len() == other.vars.len()
            && self.vars.iter().all(|(k, v)| other.vars.get(k).is_some_and(|o| v.equals(o)))
            && self.coords.len() == other.coords.len()
            && self.coords.iter().all(|(k, c)| other.coords.get(k).is_some_and(|o| coord_equals(c, o)))
    }

    /// Aligns all variables and coordinates of two datasets.
    pub fn align_with(&self, other: &Self, join: Join) -> Result<(Self, Self)> {
        let va = self.align_view();
        let vb = other.align_view();
        let idx = crate::align::joined_indexes(&[va, vb], join, &[])?;
        Ok((self.reindex_to(&idx)?, other.reindex_to(&idx)?))
    }
    pub(crate) fn align_view(&self) -> crate::align::AlignView<'_> {
        let d = self.dims();
        crate::align::AlignView {
            dims: d.keys().cloned().collect(),
            sizes: d.values().copied().collect(),
            coords: &self.coords,
        }
    }
    pub(crate) fn reindex_to(&self, indexes: &IndexMap<String, Coord>) -> Result<Self> {
        let mut out = self.clone();
        for (dim, new) in indexes {
            match self.coords.get(dim).filter(|c| c.dims == [dim.clone()]) {
                None => {
                    if self.dims().contains_key(dim) {
                        out.coords.insert(dim.clone(), new.clone());
                    }
                }
                Some(old) if coord_equals(old, new) => {}
                Some(old) => {
                    let idx = old.to_index()?;
                    if !idx.is_unique() {
                        return Err(Error::Align(format!("index {dim:?} has duplicate values")));
                    }
                    let pos = idx.get_indexer(&new.labels(), Method::Exact, None)?;
                    out = out.take_dim_opt(dim, &pos)?;
                    out.coords.insert(dim.clone(), Coord { attrs: old.attrs.clone(), ..new.clone() });
                }
            }
        }
        Ok(out)
    }

    /// Element-wise binary op between datasets with matching variable names
    /// (inner join on labels; variables present in only one side are dropped).
    pub fn zip_with(&self, other: &Self, f: impl Fn(&T, &T) -> T + Copy) -> Result<Self> {
        let (a, b) = self.align_with(other, Join::Inner)?;
        let mut out = Dataset {
            vars: IndexMap::new(),
            coords: IndexMap::new(),
            attrs: crate::ops::merge_attrs_drop_conflicts(&a.attrs, &b.attrs),
        };
        let mut arrays = Vec::new();
        for k in a.vars.keys() {
            if b.vars.contains_key(k) {
                arrays.push((k.clone(), a.get(k)?.zip_with(&b.get(k)?, f)?));
            }
        }
        for (k, mut r) in arrays {
            for (ck, c) in std::mem::take(&mut r.coords) {
                out.coords.entry(ck).or_insert(c);
            }
            r.name = Some(k.clone());
            out.vars.insert(k, r);
        }
        for (k, c) in a.coords.iter().chain(b.coords.iter()) {
            out.coords.entry(k.clone()).or_insert_with(|| c.clone());
        }
        Ok(out)
    }
    /// Applies `f` element-wise to every variable.
    pub fn map_values(&self, f: impl Fn(&T) -> T) -> Self {
        let mut out = self.clone();
        for v in out.vars.values_mut() {
            *v = v.map(&f);
        }
        out
    }
}

macro_rules! ds_reduce {
    ($($name:ident => $doc:literal),*) => {
        impl<T: FloatElement> Dataset<T> {
            $(
            #[doc = $doc]
            ///
            /// Variables that lack all of `dims` are kept unchanged.
            pub fn $name(&self, dims: impl IntoDims) -> Result<Self> {
                let dims = dims.into_dims();
                self.reduce_vars(&dims, |da, d| da.$name(d))
            }
            )*
        }
    };
}
ds_reduce!(sum => "Sum over `dims` for each variable.", mean => "Mean over `dims`.", min => "Minimum over `dims`.",
    max => "Maximum over `dims`.", std => "Standard deviation over `dims`.", var => "Variance over `dims`.",
    median => "Median over `dims`.", prod => "Product over `dims`.");

impl<T: FloatElement> Dataset<T> {
    fn reduce_vars(
        &self,
        dims: &[String],
        f: impl Fn(&DataArray<T>, Vec<String>) -> Result<DataArray<T>>,
    ) -> Result<Self> {
        let all = dims.iter().any(|d| d == "...");
        let ds_dims = self.dims();
        for d in dims {
            if d != "..." && !ds_dims.contains_key(d) {
                return Err(Error::DimNotFound(d.clone(), ds_dims.keys().cloned().collect()));
            }
        }
        let reduced: Vec<String> = if all { ds_dims.keys().cloned().collect() } else { dims.to_vec() };
        let mut out = self.map(|da| {
            let mine: Vec<String> = reduced.iter().filter(|d| da.dims.contains(d)).cloned().collect();
            if mine.is_empty() && !da.dims.is_empty() { Ok(da.clone()) } else { f(da, mine) }
        })?;
        out.coords.retain(|_, c| !c.dims.iter().any(|d| reduced.contains(d)));
        Ok(out)
    }
    /// Number of valid values per variable (as floats).
    pub fn count(&self, dims: impl IntoDims) -> Result<Self> {
        let dims = dims.into_dims();
        self.reduce_vars(&dims, |da, d| Ok(da.count(d)?.map(|&c| T::from_i64(c).unwrap())))
    }
    /// Cumulative sum along `dim` for variables that have it.
    pub fn cumsum(&self, dim: &str) -> Result<Self> {
        self.map(|da| if da.dims.iter().any(|d| d == dim) { da.cumsum(dim) } else { Ok(da.clone()) })
    }
}

//! `concat` and `merge`.

use crate::align::{Join, joined_indexes};
use crate::coords::{Coord, CoordData};
use crate::dataarray::{DataArray, coord_equals};
use crate::dataset::Dataset;
use crate::element::Element;
use crate::error::{Error, Result};
use indexmap::IndexMap;
use ndarray::Axis;

/// Concatenates arrays along `dim`, like `xr.concat(arrays, dim)`.
///
/// If `dim` already exists it is extended; otherwise a new leading dimension
/// is created (scalar coordinates named `dim` become its index). Other
/// dimensions are aligned with an outer join first.
pub fn concat<T: Element>(arrays: &[&DataArray<T>], dim: &str) -> Result<DataArray<T>> {
    if arrays.is_empty() {
        return Err(Error::invalid("concat needs at least one array"));
    }
    let exclude = vec![dim.to_string()];
    let views: Vec<_> = arrays.iter().map(|a| a.align_view()).collect();
    let idx = joined_indexes(&views, Join::Outer, &exclude)?;
    let mut aligned: Vec<DataArray<T>> = arrays.iter().map(|a| a.reindex_to(&idx)).collect::<Result<_>>()?;
    let existing = aligned[0].dims.iter().any(|d| d == dim);
    if !existing {
        // promote to a new leading dimension
        let scalar_coords: Vec<String> =
            aligned[0].coords.iter().filter(|(_, c)| c.dims.is_empty()).map(|(k, _)| k.clone()).collect();
        let varying: Vec<String> = scalar_coords
            .into_iter()
            .filter(|k| {
                k == dim
                    || aligned.iter().any(|a| a.coords.get(k).is_none_or(|c| !coord_equals(c, &aligned[0].coords[k])))
            })
            .collect();
        for a in aligned.iter_mut() {
            let mut e = a.expand_dims(dim)?;
            for k in &varying {
                if k != dim
                    && let Some(c) = e.coords.get(k).cloned()
                {
                    e.coords.insert(k.clone(), c.insert_dim(0, dim, 1));
                }
            }
            *a = e;
        }
    }
    let first = &aligned[0];
    let order = first.dims.clone();
    let mut parts = Vec::with_capacity(aligned.len());
    for a in &aligned {
        if a.dims.len() != order.len() || !a.dims.iter().all(|d| order.contains(d)) {
            return Err(Error::shape(format!("cannot concat arrays with dims {:?} and {:?}", order, a.dims)));
        }
        parts.push(a.transpose(order.clone())?);
    }
    let ax = first.axis(dim)?;
    let views: Vec<_> = parts.iter().map(|p| p.data.view()).collect();
    let data = ndarray::concatenate(Axis(ax), &views).map_err(|e| Error::shape(e.to_string()))?;
    let mut coords = IndexMap::new();
    for (k, c) in &first.coords {
        if c.dims.iter().any(|d| d == dim) {
            let all: Option<Vec<&Coord>> = parts.iter().map(|p| p.coords.get(k)).collect();
            if let Some(all) = all {
                coords.insert(k.clone(), Coord::concat(&all, dim)?);
            }
        } else {
            coords.insert(k.clone(), c.clone());
        }
    }
    let name = if parts.iter().all(|p| p.name == first.name) { first.name.clone() } else { None };
    Ok(DataArray::from_parts(data, order, coords, first.attrs.clone(), name))
}

/// Concatenates along a new dimension `dim` whose index is `labels`.
pub fn concat_with_index<T: Element>(
    arrays: &[&DataArray<T>],
    dim: &str,
    labels: impl Into<CoordData>,
) -> Result<DataArray<T>> {
    let mut out = concat(arrays, dim)?;
    let c = Coord::new(dim, labels)?;
    if c.len() != out.dim_len(dim)? {
        return Err(Error::shape("number of labels must match number of arrays"));
    }
    out.coords.insert(dim.to_string(), c);
    Ok(out)
}

impl<T: Element> Dataset<T> {
    /// Concatenates datasets along `dim` (`xr.concat(datasets, dim)`).
    pub fn concat(datasets: &[&Dataset<T>], dim: &str) -> Result<Dataset<T>> {
        if datasets.is_empty() {
            return Err(Error::invalid("concat needs at least one dataset"));
        }
        let first = datasets[0];
        let existing = first.dims().contains_key(dim);
        let mut out = Dataset::new();
        out.attrs = first.attrs.clone();
        for name in first.vars.keys() {
            let arrays: Vec<DataArray<T>> = datasets.iter().map(|d| d.get(name)).collect::<Result<_>>()?;
            let has = arrays[0].dims.iter().any(|d| d == dim);
            let r = if has || !existing {
                let refs: Vec<&DataArray<T>> = arrays.iter().collect();
                concat(&refs, dim)?
            } else {
                arrays[0].clone()
            };
            out.insert(name.clone(), r)?;
        }
        for (k, c) in &first.coords {
            if out.coords.contains_key(k) {
                continue;
            }
            if c.dims.iter().any(|d| d == dim) {
                let all: Option<Vec<&Coord>> = datasets.iter().map(|d| d.coords.get(k)).collect();
                if let Some(all) = all {
                    out.coords.insert(k.clone(), Coord::concat(&all, dim)?);
                }
            } else if !(k == dim && !existing) {
                out.coords.insert(k.clone(), c.clone());
            }
        }
        Ok(out)
    }

    /// Merges datasets (`xr.merge`): indexes are outer-joined and variables
    /// combined. Variables with the same name must agree wherever both have
    /// values (`compat="no_conflicts"`).
    pub fn merge(datasets: &[&Dataset<T>]) -> Result<Dataset<T>> {
        let views: Vec<_> = datasets.iter().map(|d| d.align_view()).collect();
        let idx = joined_indexes(&views, Join::Outer, &[])?;
        let aligned: Vec<Dataset<T>> = datasets.iter().map(|d| d.reindex_to(&idx)).collect::<Result<_>>()?;
        let mut out = Dataset::new();
        for d in &aligned {
            for (k, v) in &d.attrs {
                out.attrs.entry(k.clone()).or_insert_with(|| v.clone());
            }
            for (k, c) in &d.coords {
                match out.coords.get(k) {
                    None => {
                        out.coords.insert(k.clone(), c.clone());
                    }
                    Some(o) if coord_equals(o, c) => {}
                    Some(_) => return Err(Error::Merge(format!("conflicting values for coordinate {k:?}"))),
                }
            }
            for k in d.vars.keys() {
                let da = d.get(k)?;
                match out.vars.get(k) {
                    None => out.insert(k.clone(), da)?,
                    Some(_) => {
                        let prev = out.get(k)?;
                        let conflict =
                            prev.zip_with_join(&da, Join::Exact, |a, b| !a.is_missing() && !b.is_missing() && a != b)?;
                        if conflict.data.iter().any(|c| *c) {
                            return Err(Error::Merge(format!("conflicting values for variable {k:?}")));
                        }
                        let merged =
                            prev.zip_with_join(
                                &da,
                                Join::Exact,
                                |a, b| if a.is_missing() { b.clone() } else { a.clone() },
                            )?;
                        out.vars.insert(k.clone(), merged.reset_coords_all().with_name(k.clone()));
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<T: Element> DataArray<T> {
    pub(crate) fn reset_coords_all(mut self) -> Self {
        self.coords.clear();
        self
    }
}

/// Merges named arrays into a dataset (`xr.merge([da1, da2])`).
pub fn merge<T: Element>(arrays: &[&DataArray<T>]) -> Result<Dataset<T>> {
    let mut dss = Vec::new();
    for a in arrays {
        dss.push(a.to_dataset(None)?);
    }
    let refs: Vec<&Dataset<T>> = dss.iter().collect();
    Dataset::merge(&refs)
}

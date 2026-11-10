//! Alignment by coordinate labels (`xr.align`) and broadcasting by
//! dimension name (`xr.broadcast`).

use crate::coords::{Coord, CoordData, Label, Method};
use crate::dataarray::{DataArray, coord_equals};
use crate::element::Element;
use crate::error::{Error, Result};
use indexmap::IndexMap;
use std::collections::HashSet;

/// How indexes are combined when aligning objects (`join=` in xarray).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Join {
    /// Intersection of labels (default for arithmetic).
    #[default]
    Inner,
    /// Union of labels; gaps are filled with the missing value.
    Outer,
    /// Labels of the first object.
    Left,
    /// Labels of the last object.
    Right,
    /// Indexes must already be equal, otherwise an error.
    Exact,
    /// Ignore labels; sizes must match and the first object's index wins.
    Override,
}

/// Minimal view of an object's dims/sizes/coords for computing the joined
/// indexes.
pub(crate) struct AlignView<'a> {
    pub dims: Vec<String>,
    pub sizes: Vec<usize>,
    pub coords: &'a IndexMap<String, Coord>,
}

impl AlignView<'_> {
    fn index(&self, dim: &str) -> Option<&Coord> {
        self.coords.get(dim).filter(|c| c.dims == [dim])
    }
    fn size(&self, dim: &str) -> Option<usize> {
        self.dims.iter().position(|d| d == dim).map(|i| self.sizes[i])
    }
}

/// Computes the joined index coordinate for every dimension that needs one.
/// Dimensions listed in `exclude` are left alone.
pub(crate) fn joined_indexes(
    views: &[AlignView<'_>],
    join: Join,
    exclude: &[String],
) -> Result<IndexMap<String, Coord>> {
    let mut all_dims: Vec<String> = Vec::new();
    for v in views {
        for d in &v.dims {
            if !all_dims.contains(d) && !exclude.contains(d) {
                all_dims.push(d.clone());
            }
        }
    }
    let mut out = IndexMap::new();
    for dim in all_dims {
        let with_index: Vec<&Coord> = views.iter().filter_map(|v| v.index(&dim)).collect();
        let unlabeled: Vec<usize> =
            views.iter().filter(|v| v.index(&dim).is_none()).filter_map(|v| v.size(&dim)).collect();
        if with_index.is_empty() {
            let sizes: HashSet<usize> = unlabeled.iter().copied().collect();
            if sizes.len() > 1 {
                return Err(Error::Align(format!("conflicting sizes for dimension {dim:?}: {sizes:?}")));
            }
            continue;
        }
        let all_equal = with_index.iter().all(|c| coord_equals(c, with_index[0]));
        let joined: Coord = if all_equal {
            with_index[0].clone()
        } else {
            match join {
                Join::Exact => {
                    return Err(Error::Align(format!("indexes along dimension {dim:?} are not equal (join='exact')")));
                }
                Join::Override => {
                    let n = with_index[0].len();
                    if with_index.iter().any(|c| c.len() != n) {
                        return Err(Error::Align(format!("cannot override index {dim:?}: sizes differ")));
                    }
                    with_index[0].clone()
                }
                Join::Left => with_index[0].clone(),
                Join::Right => with_index[with_index.len() - 1].clone(),
                Join::Inner | Join::Outer => {
                    let mut labels = with_index[0].labels();
                    for c in &with_index[1..] {
                        let other = c.labels();
                        let keys: HashSet<_> = other.iter().map(Label::hash_key).collect();
                        if join == Join::Inner {
                            labels.retain(|l| keys.contains(&l.hash_key()));
                        } else {
                            let have: HashSet<_> = labels.iter().map(Label::hash_key).collect();
                            labels.extend(other.into_iter().filter(|l| !have.contains(&l.hash_key())));
                            sort_labels(&mut labels);
                        }
                    }
                    let data = if labels.is_empty() {
                        empty_like(&with_index[0].data)
                    } else {
                        CoordData::from_labels(&labels)?
                    };
                    Coord { dims: vec![dim.clone()], data, attrs: with_index[0].attrs.clone() }
                }
            }
        };
        let n = joined.len();
        if let Some(bad) = unlabeled.iter().find(|&&s| s != n) {
            return Err(Error::Align(format!(
                "cannot align along dimension {dim:?}: unlabeled size {bad} differs from index size {n}"
            )));
        }
        out.insert(dim, joined);
    }
    Ok(out)
}

fn empty_like(d: &CoordData) -> CoordData {
    use ndarray::{ArrayD, IxDyn};
    match d {
        CoordData::Int(_) => CoordData::Int(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()),
        CoordData::Float(_) => CoordData::Float(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()),
        CoordData::Str(_) => CoordData::Str(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()),
        CoordData::DateTime(_) => CoordData::DateTime(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()),
        CoordData::Bool(_) => CoordData::Bool(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()),
    }
}

pub(crate) fn sort_labels(labels: &mut [Label]) {
    labels.sort_by(|a, b| match (a, b) {
        (Label::Str(x), Label::Str(y)) => x.cmp(y),
        (Label::Int(x), Label::Int(y)) | (Label::DateTime(x), Label::DateTime(y)) => x.cmp(y),
        _ => a.as_f64().unwrap_or(f64::NAN).total_cmp(&b.as_f64().unwrap_or(f64::NAN)),
    });
}

impl<T: Element> DataArray<T> {
    pub(crate) fn align_view(&self) -> AlignView<'_> {
        AlignView { dims: self.dims.clone(), sizes: self.shape().to_vec(), coords: &self.coords }
    }
    /// Reindexes this array onto the given joined indexes.
    pub(crate) fn reindex_to(&self, indexes: &IndexMap<String, Coord>) -> Result<Self> {
        let mut out = self.clone();
        for (dim, new) in indexes {
            if !self.dims.contains(dim) {
                continue;
            }
            match self.coords.get(dim).filter(|c| c.dims == [dim.clone()]) {
                None => {
                    // unlabeled dimension: adopt the index (sizes already checked)
                    out.coords.insert(dim.clone(), new.clone());
                }
                Some(old) if coord_equals(old, new) => {}
                Some(old) => {
                    if old.len() == new.len()
                        && old.data.dtype() != new.data.dtype()
                        && old.data.to_f64() == new.data.to_f64()
                    {
                        continue;
                    }
                    let idx = old.to_index()?;
                    if !idx.is_unique() {
                        return Err(Error::Align(format!(
                            "cannot reindex or align along dimension {dim:?}: index has duplicate values"
                        )));
                    }
                    let pos = idx.get_indexer(&new.labels(), Method::Exact, None)?;
                    out = out.take_dim_opt(dim, &pos)?;
                    let attrs = old.attrs.clone();
                    out.coords.insert(dim.clone(), Coord { attrs, ..new.clone() });
                }
            }
        }
        Ok(out)
    }
}

/// Aligns two arrays (possibly of different element types).
pub fn align2<A: Element, B: Element>(
    a: &DataArray<A>,
    b: &DataArray<B>,
    join: Join,
) -> Result<(DataArray<A>, DataArray<B>)> {
    let idx = joined_indexes(&[a.align_view(), b.align_view()], join, &[])?;
    if join == Join::Override {
        return Ok((override_to(a, &idx)?, override_to(b, &idx)?));
    }
    Ok((a.reindex_to(&idx)?, b.reindex_to(&idx)?))
}

fn override_to<T: Element>(a: &DataArray<T>, idx: &IndexMap<String, Coord>) -> Result<DataArray<T>> {
    let mut out = a.clone();
    for (d, c) in idx {
        if out.dims.contains(d) {
            out.coords.insert(d.clone(), c.clone());
        }
    }
    Ok(out)
}

/// Aligns any number of arrays, like `xr.align(*objects, join=...)`.
///
/// ```
/// use xarray_rs::{DataArray, Join, align};
/// let a = DataArray::new(vec![1.0, 2.0, 3.0], "x")?.with_coord("x", vec![0, 1, 2])?;
/// let b = DataArray::new(vec![10.0, 20.0], "x")?.with_coord("x", vec![1, 5])?;
/// let out = align(&[&a, &b], Join::Outer)?;
/// assert_eq!(out[0].coord("x")?.as_i64().unwrap().iter().copied().collect::<Vec<_>>(), vec![0, 1, 2, 5]);
/// # Ok::<(), xarray_rs::Error>(())
/// ```
pub fn align<T: Element>(arrays: &[&DataArray<T>], join: Join) -> Result<Vec<DataArray<T>>> {
    let views: Vec<_> = arrays.iter().map(|a| a.align_view()).collect();
    let idx = joined_indexes(&views, join, &[])?;
    arrays.iter().map(|a| if join == Join::Override { override_to(a, &idx) } else { a.reindex_to(&idx) }).collect()
}

/// Broadcasts arrays against each other by dimension name, like
/// `xr.broadcast` (aligns with an outer join first).
pub fn broadcast<T: Element>(arrays: &[&DataArray<T>]) -> Result<Vec<DataArray<T>>> {
    let aligned = align(arrays, Join::Outer)?;
    let shapes: Vec<(&Vec<String>, &[usize])> = aligned.iter().map(|a| (&a.dims, a.shape())).collect();
    let (dims, shape) = crate::ops::union_dims(&shapes)?;
    let mut all_coords: IndexMap<String, Coord> = IndexMap::new();
    for a in &aligned {
        for (k, c) in &a.coords {
            if c.dims == [k.clone()] {
                all_coords.entry(k.clone()).or_insert_with(|| c.clone());
            }
        }
    }
    aligned
        .iter()
        .map(|a| {
            let mut b = crate::ops::broadcast_to(a, &dims, &shape)?;
            for (k, c) in &all_coords {
                b.coords.entry(k.clone()).or_insert_with(|| c.clone());
            }
            Ok(b)
        })
        .collect()
}

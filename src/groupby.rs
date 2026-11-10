//! `groupby` over a coordinate and `resample` over a datetime index.

use crate::align::sort_labels;
use crate::coords::{Coord, CoordData, Label};
use crate::dataarray::DataArray;
use crate::element::{Element, FloatElement};
use crate::error::{Error, Result};
use crate::nanops;
use crate::time::{self, NAT};
use chrono::{Datelike, Duration};
use ndarray::{ArrayD, IxDyn};

/// Groups of positions along one dimension, each with a label. Created by
/// [`DataArray::groupby`] or [`DataArray::resample`].
#[derive(Debug, Clone)]
pub struct GroupBy<'a, T: Element> {
    obj: &'a DataArray<T>,
    /// dimension being grouped
    dim: String,
    /// name of the output dimension
    out_dim: String,
    labels: CoordData,
    groups: Vec<Vec<usize>>,
    /// resample keeps empty bins and fills them with NaN
    resample: bool,
}

impl<T: Element> DataArray<T> {
    /// Groups along the dimension of a 1-D coordinate by its unique values
    /// (sorted), like `da.groupby("letters")`.
    ///
    /// Virtual datetime components are supported: `"time.year"`,
    /// `"time.month"`, `"time.day"`, `"time.hour"`, `"time.dayofyear"`,
    /// `"time.dayofweek"`, `"time.quarter"` and `"time.season"`
    /// (`"DJF"`, `"MAM"`, `"JJA"`, `"SON"`).
    pub fn groupby(&self, key: &str) -> Result<GroupBy<'_, T>> {
        let (coord_name, comp) = match key.split_once('.') {
            Some((c, comp)) if self.coords.contains_key(c) => (c, Some(comp)),
            _ => (key, None),
        };
        let c = self.coord(coord_name)?;
        if c.dims.len() != 1 {
            return Err(Error::invalid(format!("groupby key {key:?} must be a 1-D coordinate")));
        }
        let dim = c.dims[0].clone();
        let keys: Vec<Label> = match comp {
            None => c.labels(),
            Some(comp) => {
                let CoordData::DateTime(a) = &c.data else {
                    return Err(Error::Dtype(format!("{coord_name:?} is not a datetime coordinate")));
                };
                a.iter()
                    .map(|&v| {
                        let x = time::component(v, comp)?;
                        Ok(if comp == "season" {
                            Label::Str(["DJF", "MAM", "JJA", "SON"][x as usize].to_string())
                        } else {
                            Label::Int(x)
                        })
                    })
                    .collect::<Result<_>>()?
            }
        };
        let out_dim = comp.unwrap_or(coord_name).to_string();
        let mut uniq: Vec<Label> = Vec::new();
        let mut seen = std::collections::HashMap::new();
        for k in &keys {
            if matches!(k, Label::Float(f) if f.is_nan()) || matches!(k, Label::DateTime(NAT)) {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(k.hash_key()) {
                e.insert(());
                uniq.push(k.clone());
            }
        }
        sort_labels(&mut uniq);
        let pos: std::collections::HashMap<_, usize> =
            uniq.iter().enumerate().map(|(i, l)| (l.hash_key(), i)).collect();
        let mut groups = vec![Vec::new(); uniq.len()];
        for (i, k) in keys.iter().enumerate() {
            if let Some(&g) = pos.get(&k.hash_key()) {
                groups[g].push(i);
            }
        }
        let mut labels = CoordData::from_labels(&uniq)?;
        if let (CoordData::DateTime(_), CoordData::Int(a)) = (&c.data, &labels)
            && comp.is_none()
        {
            labels = CoordData::DateTime(a.clone());
        }
        Ok(GroupBy { obj: self, dim, out_dim, labels, groups, resample: false })
    }

    /// Resamples a datetime dimension to a new frequency, like
    /// `da.resample(time="1D")`.
    ///
    /// Frequencies: `"<n>s"`, `"<n>min"`, `"<n>h"`, `"<n>D"`, `"<n>W"`
    /// (weeks ending Sunday), `"<n>MS"` / `"<n>ME"` (month start/end),
    /// `"<n>QS"` / `"<n>QE"`, `"<n>YS"` / `"<n>YE"`. Bins follow pandas'
    /// defaults (closed/labelled left for start anchors and fixed
    /// frequencies, right for end anchors); empty bins give NaN.
    pub fn resample(&self, dim: &str, freq: &str) -> Result<GroupBy<'_, T>> {
        let c = self.coord(dim)?;
        let CoordData::DateTime(a) = &c.data else {
            return Err(Error::Dtype(format!("resample needs a datetime index on {dim:?}")));
        };
        if c.dims != [dim] {
            return Err(Error::invalid(format!("{dim:?} must be an index coordinate")));
        }
        let values: Vec<i64> = a.iter().copied().collect();
        let (labels, assign) = resample_bins(&values, freq)?;
        let mut groups = vec![Vec::new(); labels.len()];
        for (i, g) in assign.into_iter().enumerate() {
            if let Some(g) = g {
                groups[g].push(i);
            }
        }
        let n = labels.len();
        Ok(GroupBy {
            obj: self,
            dim: dim.to_string(),
            out_dim: dim.to_string(),
            labels: CoordData::DateTime(ArrayD::from_shape_vec(IxDyn(&[n]), labels).unwrap()),
            groups,
            resample: true,
        })
    }
}

impl<'a, T: Element> GroupBy<'a, T> {
    /// Group labels.
    pub fn labels(&self) -> Vec<Label> {
        Coord { dims: vec![self.out_dim.clone()], data: self.labels.clone(), attrs: Default::default() }.labels()
    }
    /// Positions (along the grouped dimension) of each group.
    pub fn group_positions(&self) -> &[Vec<usize>] {
        &self.groups
    }
    /// Each group's label and sub-array.
    pub fn groups(&self) -> Result<Vec<(Label, DataArray<T>)>> {
        self.labels().into_iter().zip(&self.groups).map(|(l, g)| Ok((l, self.obj.take_dim(&self.dim, g)?))).collect()
    }

    fn label_coord(&self) -> Coord {
        Coord { dims: vec![self.out_dim.clone()], data: self.labels.clone(), attrs: Default::default() }
    }

    /// Reduces each group over the grouped dimension with a kernel; the group
    /// dimension takes the place of the grouped one.
    pub fn reduce<U: Element>(&self, f: impl Fn(&[T]) -> U, empty: U) -> Result<DataArray<U>> {
        let obj = self.obj;
        let ax = obj.axis(&self.dim)?;
        let moved = obj.data.view().permuted_axes(crate::reduce::move_last(obj.ndim(), ax));
        let std = moved.as_standard_layout();
        let n = obj.shape()[ax];
        let lanes: Vec<&[T]> = if n == 0 { vec![] } else { std.as_slice().unwrap().chunks_exact(n).collect() };
        let g = self.groups.len();
        let mut out: Vec<U> = Vec::with_capacity(lanes.len() * g);
        let mut buf: Vec<T> = Vec::new();
        for lane in &lanes {
            for grp in &self.groups {
                if grp.is_empty() && self.resample {
                    out.push(empty.clone());
                    continue;
                }
                buf.clear();
                buf.extend(grp.iter().map(|&i| lane[i].clone()));
                out.push(f(&buf));
            }
        }
        let mut shape: Vec<usize> = crate::reduce::move_last(obj.ndim(), ax).iter().map(|&i| obj.shape()[i]).collect();
        *shape.last_mut().unwrap() = g;
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap();
        let data = arr.permuted_axes(crate::reduce::move_back(obj.ndim(), ax)).as_standard_layout().into_owned();
        let mut dims = obj.dims.clone();
        dims[ax] = self.out_dim.clone();
        let mut coords: indexmap::IndexMap<String, Coord> = obj
            .coords
            .iter()
            .filter(|(_, c)| !c.dims.contains(&self.dim))
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect();
        coords.insert(self.out_dim.clone(), self.label_coord());
        Ok(DataArray::from_parts(data, dims, coords, obj.attrs.clone(), obj.name.clone()))
    }

    /// Applies `f` to every group and combines the results. If the results
    /// keep the grouped dimension, they are put back in the original order
    /// (e.g. anomalies `g - g.mean(dim)`); otherwise they are stacked along
    /// the group dimension.
    pub fn map<U: Element>(&self, f: impl Fn(&DataArray<T>) -> Result<DataArray<U>>) -> Result<DataArray<U>> {
        let mut results = Vec::new();
        for g in &self.groups {
            if g.is_empty() {
                continue;
            }
            results.push(f(&self.obj.take_dim(&self.dim, g)?)?);
        }
        if results.is_empty() {
            return Err(Error::invalid("no groups"));
        }
        let refs: Vec<&DataArray<U>> = results.iter().collect();
        if results[0].dims.contains(&self.dim) {
            let cat = crate::combine::concat(&refs, &self.dim)?;
            let order: Vec<usize> = self.groups.iter().flatten().copied().collect();
            let mut inv = vec![0; order.len()];
            for (k, &p) in order.iter().enumerate() {
                inv[p] = k;
            }
            let mut out = cat.take_dim(&self.dim, &inv)?;
            if let Ok(orig) = self
                .obj
                .transpose(out.dims.clone().into_iter().filter(|d| self.obj.dims.contains(d)).collect::<Vec<_>>())
                && orig.dims == out.dims
            {
                for (k, c) in &self.obj.coords {
                    if c.dims.iter().all(|d| out.dims.contains(d)) {
                        out.coords.insert(k.clone(), c.clone());
                    }
                }
            }
            Ok(out)
        } else {
            let non_empty: Vec<Label> =
                self.labels().into_iter().zip(&self.groups).filter(|(_, g)| !g.is_empty()).map(|(l, _)| l).collect();
            let data = CoordData::from_labels(&non_empty)?;
            let mut out = crate::combine::concat(&refs, &self.out_dim)?;
            out.coords.insert(self.out_dim.clone(), Coord::new(self.out_dim.as_str(), data)?);
            Ok(out)
        }
    }

    /// Group-wise binary operation with an array indexed by the group
    /// dimension, e.g. `(da.groupby("time.month") - climatology)`.
    pub fn binary_op<U: Element>(&self, other: &DataArray<T>, f: impl Fn(&T, &T) -> U + Copy) -> Result<DataArray<U>> {
        if !other.dims.contains(&self.out_dim) {
            return Err(Error::invalid(format!("other must have dimension {:?}", self.out_dim)));
        }
        let labels = self.labels();
        let mut results = Vec::new();
        let mut order = Vec::new();
        for (l, g) in labels.iter().zip(&self.groups) {
            if g.is_empty() {
                continue;
            }
            let part = self.obj.take_dim(&self.dim, g)?;
            let o = other.sel(&self.out_dim, l.clone())?.drop_vars_lenient(&self.out_dim);
            let mut r = part.zip_with(&o, f)?;
            r.coords.shift_remove(&self.out_dim);
            results.push(r);
            order.extend(g.iter().copied());
        }
        let refs: Vec<&DataArray<U>> = results.iter().collect();
        let cat = crate::combine::concat(&refs, &self.dim)?;
        let mut inv = vec![0; order.len()];
        for (k, &p) in order.iter().enumerate() {
            inv[p] = k;
        }
        let mut out = cat.take_dim(&self.dim, &inv)?;
        if self.out_dim != self.dim {
            // keep the group key as a coordinate along the grouped dimension
            let n = self.obj.dim_len(&self.dim)?;
            let mut keys: Vec<Label> = vec![Label::Float(f64::NAN); n];
            for (l, g) in labels.iter().zip(&self.groups) {
                for &i in g {
                    keys[i] = l.clone();
                }
            }
            out.coords.insert(self.out_dim.clone(), Coord::new(self.dim.as_str(), CoordData::from_labels(&keys)?)?);
        }
        Ok(out)
    }
}

impl<T: Element> DataArray<T> {
    pub(crate) fn drop_vars_lenient(mut self, name: &str) -> Self {
        self.coords.shift_remove(name);
        self
    }
}

impl<T: FloatElement> GroupBy<'_, T> {
    /// Group sum.
    pub fn sum(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nansum, T::nan())
    }
    /// Group mean.
    pub fn mean(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmean, T::nan())
    }
    /// Group minimum.
    pub fn min(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmin, T::nan())
    }
    /// Group maximum.
    pub fn max(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmax, T::nan())
    }
    /// Group standard deviation (`ddof=0`).
    pub fn std(&self) -> Result<DataArray<T>> {
        self.reduce(|v| nanops::nanstd(v, 0), T::nan())
    }
    /// Group variance (`ddof=0`).
    pub fn var(&self) -> Result<DataArray<T>> {
        self.reduce(|v| nanops::nanvar(v, 0), T::nan())
    }
    /// Group median.
    pub fn median(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmedian, T::nan())
    }
    /// Group product.
    pub fn prod(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanprod, T::nan())
    }
    /// Number of valid values per group (as floats so empty resample bins can
    /// be NaN).
    pub fn count(&self) -> Result<DataArray<T>> {
        self.reduce(|v| T::from_usize(nanops::count(v)).unwrap(), T::nan())
    }
    /// First value per group (NaN skipped).
    pub fn first(&self) -> Result<DataArray<T>> {
        self.reduce(|v| v.iter().copied().find(|x| !x.is_nan()).unwrap_or(T::nan()), T::nan())
    }
    /// Last value per group (NaN skipped).
    pub fn last(&self) -> Result<DataArray<T>> {
        self.reduce(|v| v.iter().rev().copied().find(|x| !x.is_nan()).unwrap_or(T::nan()), T::nan())
    }
    /// `self - other` per group.
    pub fn sub(&self, other: &DataArray<T>) -> Result<DataArray<T>> {
        self.binary_op(other, |a, b| *a - *b)
    }
    /// `self + other` per group.
    pub fn add(&self, other: &DataArray<T>) -> Result<DataArray<T>> {
        self.binary_op(other, |a, b| *a + *b)
    }
    /// `self / other` per group.
    pub fn div(&self, other: &DataArray<T>) -> Result<DataArray<T>> {
        self.binary_op(other, |a, b| *a / *b)
    }
    /// `self * other` per group.
    pub fn mul(&self, other: &DataArray<T>) -> Result<DataArray<T>> {
        self.binary_op(other, |a, b| *a * *b)
    }
}

// ---------------------------------------------------------------------------
// resample bins
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Freq {
    Fixed(i64),
    /// months per step, end-anchored?
    Months(i32, bool, u32),
    Weeks(i64),
}

fn parse_freq(s: &str) -> Result<(i64, Freq)> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: i64 = if digits.is_empty() {
        1
    } else {
        digits.parse().map_err(|_| Error::invalid(format!("bad frequency {s:?}")))?
    };
    if n <= 0 {
        return Err(Error::invalid("frequency must be positive"));
    }
    let unit = &s[digits.len()..];
    const S: i64 = 1_000_000_000;
    let f = match unit {
        "ns" | "N" => Freq::Fixed(1),
        "us" | "U" => Freq::Fixed(1_000),
        "ms" | "L" => Freq::Fixed(1_000_000),
        "s" | "S" => Freq::Fixed(S),
        "min" | "T" => Freq::Fixed(60 * S),
        "h" | "H" => Freq::Fixed(3600 * S),
        "D" | "d" => Freq::Fixed(86_400 * S),
        "W" | "W-SUN" => Freq::Weeks(n),
        "MS" => Freq::Months(n as i32, false, 1),
        "ME" | "M" => Freq::Months(n as i32, true, 1),
        "QS" | "QS-JAN" => Freq::Months(3 * n as i32, false, 3),
        "QE" | "Q" | "QE-DEC" => Freq::Months(3 * n as i32, true, 3),
        "YS" | "AS" | "YS-JAN" => Freq::Months(12 * n as i32, false, 12),
        "YE" | "Y" | "A" | "YE-DEC" => Freq::Months(12 * n as i32, true, 12),
        _ => return Err(Error::invalid(format!("unsupported frequency {s:?}"))),
    };
    let f = match f {
        Freq::Fixed(x) => Freq::Fixed(x * n),
        other => other,
    };
    Ok((n, f))
}

/// Returns bin labels and the bin of every timestamp.
fn resample_bins(values: &[i64], freq: &str) -> Result<(Vec<i64>, Vec<Option<usize>>)> {
    let (_, f) = parse_freq(freq)?;
    let valid: Vec<i64> = values.iter().copied().filter(|&v| v != NAT).collect();
    if valid.is_empty() {
        return Ok((vec![], vec![None; values.len()]));
    }
    let first = *valid.iter().min().unwrap();
    let last = *valid.iter().max().unwrap();
    const DAY: i64 = 86_400_000_000_000;
    match f {
        Freq::Fixed(step) => {
            let origin = first.div_euclid(DAY) * DAY; // start_day
            let start = origin + (first - origin).div_euclid(step) * step;
            let nbins = ((last - start).div_euclid(step) + 1) as usize;
            let labels = (0..nbins).map(|k| start + k as i64 * step).collect();
            let assign = values.iter().map(|&v| (v != NAT).then(|| ((v - start).div_euclid(step)) as usize)).collect();
            Ok((labels, assign))
        }
        Freq::Weeks(n) => {
            // W-SUN: closed right, labelled by the Sunday ending the week
            let to_date = |v: i64| time::from_ns(v).unwrap().date();
            let d0 = to_date(first);
            let label0 = d0 + Duration::days(((6 - d0.weekday().num_days_from_monday() as i64) + 7) % 7);
            let step = 7 * n;
            let gidx = |v: i64| -> usize {
                let diff = (to_date(v) - label0).num_days();
                if diff <= 0 { 0 } else { ((diff + step - 1) / step) as usize }
            };
            let nbins = gidx(last) + 1;
            let labels = (0..nbins)
                .map(|k| time::to_ns((label0 + Duration::days(step * k as i64)).and_hms_opt(0, 0, 0).unwrap()))
                .collect();
            let assign = values.iter().map(|&v| (v != NAT).then(|| gidx(v))).collect();
            Ok((labels, assign))
        }
        Freq::Months(step, end, period) => {
            let mi = |v: i64| {
                let d = time::from_ns(v).unwrap();
                d.year() * 12 + d.month0() as i32
            };
            let p = period as i32;
            let m_first = mi(first);
            let m0 = if end {
                // period end month containing the first timestamp
                m_first - m_first.rem_euclid(p) + (p - 1)
            } else {
                m_first - m_first.rem_euclid(p)
            };
            let gidx = |v: i64| -> usize {
                let m = mi(v);
                if end {
                    let diff = m - m0;
                    if diff <= 0 { 0 } else { ((diff + step - 1) / step) as usize }
                } else {
                    ((m - m0).div_euclid(step)) as usize
                }
            };
            let nbins = gidx(last) + 1;
            let labels = (0..nbins)
                .map(|k| {
                    let m = m0 + step * k as i32;
                    let (y, mo) = (m.div_euclid(12), m.rem_euclid(12) as u32);
                    let dt = if end {
                        time::month_end(y, mo)
                    } else {
                        chrono::NaiveDate::from_ymd_opt(y, mo + 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()
                    };
                    time::to_ns(dt)
                })
                .collect();
            let assign = values.iter().map(|&v| (v != NAT).then(|| gidx(v))).collect();
            Ok((labels, assign))
        }
    }
}

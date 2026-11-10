//! Interpolation along a dimension and gap filling.

use crate::coords::{Coord, CoordData};
use crate::dataarray::DataArray;
use crate::element::FloatElement;
use crate::error::{Error, Result};
use ndarray::{ArrayD, Axis, IxDyn};

/// Interpolation method for [`DataArray::interp_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpMethod {
    /// Piecewise linear (default).
    #[default]
    Linear,
    /// Nearest neighbour.
    Nearest,
}

impl<T: FloatElement> DataArray<T> {
    /// Linear interpolation onto new coordinate values along `dim`
    /// (`da.interp(dim=new)`). Points outside the original range are NaN.
    /// Works for numeric and datetime indexes.
    pub fn interp(&self, dim: &str, new: impl Into<CoordData>) -> Result<Self> {
        self.interp_with(dim, new, InterpMethod::Linear)
    }

    /// Interpolation with a chosen method.
    pub fn interp_with(&self, dim: &str, new: impl Into<CoordData>, method: InterpMethod) -> Result<Self> {
        let ax = self.axis(dim)?;
        let old = self.coord(dim)?;
        if old.dims != [dim] {
            return Err(Error::invalid(format!("interp needs an index coordinate on {dim:?}")));
        }
        let new = Coord::new(dim, new)?;
        let xs = old.data.to_f64().ok_or_else(|| Error::Dtype("cannot interpolate over a string index".into()))?;
        let mut xs: Vec<f64> = xs.iter().copied().collect();
        let xn = new.data.to_f64().ok_or_else(|| Error::Dtype("new coordinates must be numeric or datetime".into()))?;
        let xn: Vec<f64> = xn.iter().copied().collect();
        // datetimes: work relative to the first value to keep precision
        let shift = if matches!(old.data, CoordData::DateTime(_)) { xs.first().copied().unwrap_or(0.0) } else { 0.0 };
        if shift != 0.0 {
            xs.iter_mut().for_each(|v| *v -= shift);
        }
        let xn: Vec<f64> = xn.iter().map(|v| v - shift).collect();
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
        let sx: Vec<f64> = order.iter().map(|&i| xs[i]).collect();
        let n_old = sx.len();
        let weights: Vec<Option<(usize, usize, f64, f64)>> = xn
            .iter()
            .map(|&x| {
                if x.is_nan() || n_old == 0 || x < sx[0] || x > sx[n_old - 1] {
                    return None;
                }
                // scipy.interpolate.interp1d: searchsorted(left), clipped to [1, n-1]
                if n_old == 1 {
                    return Some((order[0], order[0], 0.0, 0.0));
                }
                let k = sx.partition_point(|&v| v < x).clamp(1, n_old - 1);
                let (lo, hi) = (k - 1, k);
                Some(match method {
                    InterpMethod::Linear => (order[lo], order[hi], x - sx[lo], sx[hi] - sx[lo]),
                    InterpMethod::Nearest => {
                        let mid = (sx[lo] + sx[hi]) / 2.0;
                        let j = if x <= mid { order[lo] } else { order[hi] };
                        (j, j, 0.0, 0.0)
                    }
                })
            })
            .collect();
        let mut shape = self.shape().to_vec();
        shape[ax] = xn.len();
        let mut out = ArrayD::from_elem(IxDyn(&shape), T::nan());
        for (j, w) in weights.iter().enumerate() {
            if let Some((a, b, dx, span)) = *w {
                let va = self.data.index_axis(Axis(ax), a);
                let vb = self.data.index_axis(Axis(ax), b);
                let (dx, span) = (T::from_f64(dx).unwrap(), T::from_f64(span).unwrap());
                let mut dst = out.index_axis_mut(Axis(ax), j);
                ndarray::Zip::from(&mut dst).and(&va).and(&vb).for_each(|d, &y0, &y1| {
                    *d = if a == b { y0 } else { (y1 - y0) / span * dx + y0 };
                });
            }
        }
        let mut coords = indexmap::IndexMap::new();
        for (k, c) in &self.coords {
            if k == dim {
                coords.insert(k.clone(), Coord { attrs: c.attrs.clone(), ..new.clone() });
            } else if !c.dims.iter().any(|d| d == dim) {
                coords.insert(k.clone(), c.clone());
            } else if c.dims == [dim] && matches!(c.data, CoordData::Float(_) | CoordData::Int(_)) {
                // numeric non-index coordinates are interpolated too
                let v = c.data.to_f64().unwrap();
                let vals: Vec<f64> = weights
                    .iter()
                    .map(|w| match *w {
                        None => f64::NAN,
                        Some((a, b, dx, span)) => {
                            if a == b {
                                v[[a]]
                            } else {
                                (v[[b]] - v[[a]]) / span * dx + v[[a]]
                            }
                        }
                    })
                    .collect();
                coords.insert(
                    k.clone(),
                    Coord {
                        dims: c.dims.clone(),
                        data: CoordData::Float(ndarray::Array1::from(vals).into_dyn()),
                        attrs: c.attrs.clone(),
                    },
                );
            }
        }
        Ok(DataArray::from_parts(out, self.dims.clone(), coords, self.attrs.clone(), self.name.clone()))
    }

    /// Interpolates onto another array's index along `dim`.
    pub fn interp_like<U: crate::Element>(&self, other: &DataArray<U>, dim: &str) -> Result<Self> {
        self.interp(dim, other.coord(dim)?.data.clone())
    }

    /// Fills NaN by propagating the last valid value forward along `dim`
    /// (`limit` = max consecutive fills).
    pub fn ffill(&self, dim: &str, limit: Option<usize>) -> Result<Self> {
        self.map_along(dim, |lane| fill_forward(lane, limit))
    }
    /// Fills NaN with the next valid value along `dim`.
    pub fn bfill(&self, dim: &str, limit: Option<usize>) -> Result<Self> {
        self.map_along(dim, |lane| {
            let mut rev: Vec<T> = lane.iter().rev().copied().collect();
            rev = fill_forward(&rev, limit);
            rev.reverse();
            rev
        })
    }
    /// Fills interior NaN gaps by linear interpolation over the index of
    /// `dim` (or positions, if it has no index).
    pub fn interpolate_na(&self, dim: &str) -> Result<Self> {
        let xs: Vec<f64> = match self.coords.get(dim).and_then(|c| c.data.to_f64()) {
            Some(a) => a.iter().copied().collect(),
            None => (0..self.dim_len(dim)?).map(|i| i as f64).collect(),
        };
        self.map_along(dim, |lane| {
            let valid: Vec<usize> = (0..lane.len()).filter(|&i| !lane[i].is_nan()).collect();
            let mut out = lane.to_vec();
            for w in valid.windows(2) {
                let (a, b) = (w[0], w[1]);
                for i in a + 1..b {
                    let t = T::from_f64((xs[i] - xs[a]) / (xs[b] - xs[a])).unwrap();
                    out[i] = lane[a] + (lane[b] - lane[a]) * t;
                }
            }
            out
        })
    }
}

fn fill_forward<T: FloatElement>(lane: &[T], limit: Option<usize>) -> Vec<T> {
    let mut last: Option<T> = None;
    let mut run = 0usize;
    lane.iter()
        .map(|&v| {
            if v.is_nan() {
                run += 1;
                match last {
                    Some(l) if limit.is_none_or(|lim| run <= lim) => l,
                    _ => v,
                }
            } else {
                last = Some(v);
                run = 0;
                v
            }
        })
        .collect()
}

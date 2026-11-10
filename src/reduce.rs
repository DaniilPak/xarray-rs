//! Reductions over named dimensions.

use crate::coords::{Coord, IntoDims};
use crate::dataarray::DataArray;
use crate::element::{Element, FloatElement};
use crate::error::{Error, Result};
use crate::nanops;
use ndarray::{ArrayD, IxDyn};

impl<T: Element> DataArray<T> {
    pub(crate) fn resolve_dims(&self, dims: impl IntoDims) -> Result<Vec<String>> {
        let dims = dims.into_dims();
        if dims.iter().any(|d| d == "...") {
            return Ok(self.dims.clone());
        }
        for d in &dims {
            self.axis(d)?;
        }
        Ok(dims)
    }

    /// Reduces over `dims` with a custom kernel that receives the values of
    /// each reduced block (in row-major order of the reduced dims).
    ///
    /// Coordinates that depend on a reduced dimension are dropped; name and
    /// attributes are kept (xarray's current `keep_attrs` default).
    pub fn reduce<U: Element>(&self, dims: impl IntoDims, f: impl Fn(&[T]) -> U) -> Result<DataArray<U>> {
        let red = self.resolve_dims(dims)?;
        let red_ax: Vec<usize> = red.iter().map(|d| self.axis(d)).collect::<Result<_>>()?;
        let keep_ax: Vec<usize> = (0..self.ndim()).filter(|i| !red_ax.contains(i)).collect();
        let mut perm = keep_ax.clone();
        perm.extend(&red_ax);
        let shape = self.shape();
        let keep_shape: Vec<usize> = keep_ax.iter().map(|&i| shape[i]).collect();
        let m: usize = red_ax.iter().map(|&i| shape[i]).product();
        let n_out: usize = keep_shape.iter().product();
        let permuted = self.data.view().permuted_axes(perm);
        let std = permuted.as_standard_layout();
        let flat = std.as_slice().expect("standard layout");
        let out: Vec<U> =
            if m == 0 { (0..n_out).map(|_| f(&[])).collect() } else { flat.chunks_exact(m).map(&f).collect() };
        let data = ArrayD::from_shape_vec(IxDyn(&keep_shape), out).expect("shape");
        let dims: Vec<String> = keep_ax.iter().map(|&i| self.dims[i].clone()).collect();
        let coords = self
            .coords
            .iter()
            .filter(|(_, c)| !c.dims.iter().any(|d| red.contains(d)))
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect();
        Ok(DataArray::from_parts(data, dims, coords, self.attrs.clone(), self.name.clone()))
    }

    /// Applies a 1-D transform along `dim` that keeps its length (used by
    /// cumulative ops).
    pub fn map_along<U: Element>(&self, dim: &str, f: impl Fn(&[T]) -> Vec<U>) -> Result<DataArray<U>> {
        let ax = self.axis(dim)?;
        let n = self.shape()[ax];
        let moved = self.data.view().permuted_axes(move_last(self.ndim(), ax));
        let std = moved.as_standard_layout();
        let flat = std.as_slice().unwrap();
        let mut out: Vec<U> = Vec::with_capacity(flat.len());
        if n > 0 {
            for lane in flat.chunks_exact(n) {
                let r = f(lane);
                if r.len() != n {
                    return Err(Error::shape("map_along must preserve length"));
                }
                out.extend(r);
            }
        }
        let shape: Vec<usize> = move_last(self.ndim(), ax).iter().map(|&i| self.shape()[i]).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap();
        let back = arr.permuted_axes(move_back(self.ndim(), ax)).as_standard_layout().into_owned();
        Ok(DataArray::from_parts(back, self.dims.clone(), self.coords.clone(), self.attrs.clone(), self.name.clone()))
    }
}

pub(crate) fn move_last(ndim: usize, ax: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..ndim).filter(|&i| i != ax).collect();
    p.push(ax);
    p
}
pub(crate) fn move_back(ndim: usize, ax: usize) -> Vec<usize> {
    // inverse of move_last
    let fwd = move_last(ndim, ax);
    let mut inv = vec![0; ndim];
    for (i, &p) in fwd.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// NaN-aware reductions (`skipna=True`, xarray's default for floats).
///
/// `dims` can be one name, several names, or `..` for all dimensions.
impl<T: FloatElement> DataArray<T> {
    /// Sum (NaN skipped; all-NaN gives 0).
    pub fn sum(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::nansum)
    }
    /// Sum that returns NaN when fewer than `min_count` values are valid.
    pub fn sum_min_count(&self, dims: impl IntoDims, min_count: usize) -> Result<Self> {
        self.reduce(dims, |v| nanops::nansum_min_count(v, min_count))
    }
    /// Product (NaN skipped).
    pub fn prod(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::nanprod)
    }
    /// Mean (NaN skipped).
    pub fn mean(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::nanmean)
    }
    /// Minimum (NaN skipped).
    pub fn min(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::nanmin)
    }
    /// Maximum (NaN skipped).
    pub fn max(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::nanmax)
    }
    /// Population standard deviation (`ddof=0`, NaN skipped).
    pub fn std(&self, dims: impl IntoDims) -> Result<Self> {
        self.std_ddof(dims, 0)
    }
    /// Standard deviation with `ddof`.
    pub fn std_ddof(&self, dims: impl IntoDims, ddof: usize) -> Result<Self> {
        self.reduce(dims, |v| nanops::nanstd(v, ddof))
    }
    /// Population variance (`ddof=0`, NaN skipped).
    pub fn var(&self, dims: impl IntoDims) -> Result<Self> {
        self.var_ddof(dims, 0)
    }
    /// Variance with `ddof`.
    pub fn var_ddof(&self, dims: impl IntoDims, ddof: usize) -> Result<Self> {
        self.reduce(dims, |v| nanops::nanvar(v, ddof))
    }
    /// Median (NaN skipped).
    pub fn median(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::nanmedian)
    }
    /// Quantile `q` in `[0, 1]` with linear interpolation (NaN skipped).
    pub fn quantile(&self, q: f64, dims: impl IntoDims) -> Result<Self> {
        if !(0.0..=1.0).contains(&q) {
            return Err(Error::invalid("quantile must be in [0, 1]"));
        }
        let mut out = self.reduce(dims, |v| nanops::nanquantile(v, q))?;
        out.coords.insert("quantile".into(), Coord::scalar(q));
        Ok(out)
    }
    /// Number of non-NaN values.
    pub fn count(&self, dims: impl IntoDims) -> Result<DataArray<i64>> {
        self.reduce(dims, |v| nanops::count(v) as i64)
    }
    /// Sum without skipping NaN (`skipna=False`).
    pub fn sum_noskip(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::sum)
    }
    /// Mean without skipping NaN (`skipna=False`).
    pub fn mean_noskip(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::mean)
    }
    /// Minimum without skipping NaN.
    pub fn min_noskip(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::min)
    }
    /// Maximum without skipping NaN.
    pub fn max_noskip(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, nanops::max)
    }
    /// Position of the minimum along one dimension (NaN skipped). Errors on
    /// all-NaN slices, like xarray.
    pub fn argmin(&self, dim: &str) -> Result<DataArray<i64>> {
        self.arg_reduce(dim, nanops::nanargmin)
    }
    /// Position of the maximum along one dimension (NaN skipped).
    pub fn argmax(&self, dim: &str) -> Result<DataArray<i64>> {
        self.arg_reduce(dim, nanops::nanargmax)
    }
    fn arg_reduce(&self, dim: &str, f: fn(&[T]) -> Option<usize>) -> Result<DataArray<i64>> {
        let r = self.reduce(dim, |v| f(v).map(|i| i as i64).unwrap_or(-1))?;
        if r.data.iter().any(|&i| i < 0) {
            return Err(Error::invalid("All-NaN slice encountered"));
        }
        Ok(r)
    }
    /// Label (index coordinate value) of the minimum along `dim`.
    pub fn idxmin(&self, dim: &str) -> Result<DataArray<f64>> {
        self.idx_reduce(dim, nanops::nanargmin)
    }
    /// Label (index coordinate value) of the maximum along `dim`.
    pub fn idxmax(&self, dim: &str) -> Result<DataArray<f64>> {
        self.idx_reduce(dim, nanops::nanargmax)
    }
    fn idx_reduce(&self, dim: &str, f: fn(&[T]) -> Option<usize>) -> Result<DataArray<f64>> {
        let labels =
            self.coord(dim)?.data.to_f64().ok_or_else(|| Error::Dtype("idxmin/idxmax need a numeric index".into()))?;
        let mut r = self.reduce(dim, |v| f(v).map(|i| labels[[i]]).unwrap_or(f64::NAN))?;
        r.name = Some(dim.to_string());
        Ok(r)
    }
    /// Cumulative sum along `dim` (NaN treated as 0).
    pub fn cumsum(&self, dim: &str) -> Result<Self> {
        self.map_along(dim, nanops::nancumsum)
    }
    /// Cumulative product along `dim` (NaN treated as 1).
    pub fn cumprod(&self, dim: &str) -> Result<Self> {
        self.map_along(dim, nanops::nancumprod)
    }
    /// n-th order forward difference along `dim` (`diff(dim, n)`, label="upper").
    pub fn diff(&self, dim: &str, n: usize) -> Result<Self> {
        let mut out = self.clone();
        for _ in 0..n {
            let len = out.dim_len(dim)?;
            if len == 0 {
                return Ok(out);
            }
            let upper = out.take_dim(dim, &(1..len).collect::<Vec<_>>())?;
            let lower = out.take_dim(dim, &(0..len - 1).collect::<Vec<_>>())?;
            let ax = out.axis(dim)?;
            let data = &upper.data - &lower.data;
            out = DataArray::from_parts(
                data,
                upper.dims.clone(),
                upper.coords.clone(),
                out.attrs.clone(),
                out.name.clone(),
            );
            let _ = ax;
        }
        Ok(out)
    }
}

impl DataArray<bool> {
    /// True if any value along `dims` is true.
    pub fn any(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, |v| v.iter().any(|&b| b))
    }
    /// True if all values along `dims` are true.
    pub fn all(&self, dims: impl IntoDims) -> Result<Self> {
        self.reduce(dims, |v| v.iter().all(|&b| b))
    }
    /// Number of true values.
    pub fn count_true(&self, dims: impl IntoDims) -> Result<DataArray<i64>> {
        self.reduce(dims, |v| v.iter().filter(|&&b| b).count() as i64)
    }
}

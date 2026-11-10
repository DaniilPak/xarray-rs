//! Moving-window operations (`da.rolling(dim=window)`).

use crate::dataarray::DataArray;
use crate::element::FloatElement;
use crate::error::{Error, Result};
use crate::nanops;

/// A moving window along one dimension. Build with
/// [`DataArray::rolling`], configure with [`center`](Rolling::center) and
/// [`min_periods`](Rolling::min_periods), then reduce.
///
/// ```
/// use xarray_rs::DataArray;
/// let da = DataArray::new(vec![1.0_f64, 2.0, 3.0, 4.0], "t")?;
/// let m = da.rolling("t", 2)?.mean()?;
/// assert!(m.to_vec()[0].is_nan());
/// assert_eq!(&m.to_vec()[1..], &[1.5, 2.5, 3.5]);
/// # Ok::<(), xarray_rs::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Rolling<'a, T: FloatElement> {
    obj: &'a DataArray<T>,
    dim: String,
    window: usize,
    center: bool,
    min_periods: Option<usize>,
}

impl<T: FloatElement> DataArray<T> {
    /// Moving window of `window` positions along `dim`.
    pub fn rolling(&self, dim: &str, window: usize) -> Result<Rolling<'_, T>> {
        self.axis(dim)?;
        if window == 0 {
            return Err(Error::invalid("window must be at least 1"));
        }
        Ok(Rolling { obj: self, dim: dim.to_string(), window, center: false, min_periods: None })
    }
}

impl<T: FloatElement> Rolling<'_, T> {
    /// Centers the window on each label (default: trailing window).
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }
    /// Minimum number of valid values for a result (default: window size).
    pub fn min_periods(mut self, n: usize) -> Self {
        self.min_periods = Some(n);
        self
    }

    fn bounds(&self, i: usize, n: usize) -> (usize, usize) {
        let w = self.window as isize;
        let start = if self.center { i as isize - w / 2 } else { i as isize - w + 1 };
        let end = start + w; // exclusive
        (start.max(0) as usize, (end.max(0) as usize).min(n))
    }

    /// Applies `f` to the valid values of each window.
    pub fn reduce(&self, f: impl Fn(&[T]) -> T) -> Result<DataArray<T>> {
        let min_p = self.min_periods.unwrap_or(self.window).max(1);
        self.obj.map_along(&self.dim, |lane| {
            let n = lane.len();
            (0..n)
                .map(|i| {
                    let (a, b) = self.bounds(i, n);
                    let win = &lane[a..b];
                    if nanops::count(win) < min_p { T::nan() } else { f(win) }
                })
                .collect()
        })
    }
    /// Moving sum.
    pub fn sum(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nansum)
    }
    /// Moving mean.
    pub fn mean(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmean)
    }
    /// Moving minimum.
    pub fn min(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmin)
    }
    /// Moving maximum.
    pub fn max(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmax)
    }
    /// Moving standard deviation (`ddof=0`).
    pub fn std(&self) -> Result<DataArray<T>> {
        self.reduce(|v| nanops::nanstd(v, 0))
    }
    /// Moving variance (`ddof=0`).
    pub fn var(&self) -> Result<DataArray<T>> {
        self.reduce(|v| nanops::nanvar(v, 0))
    }
    /// Moving median.
    pub fn median(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanmedian)
    }
    /// Moving product.
    pub fn prod(&self) -> Result<DataArray<T>> {
        self.reduce(nanops::nanprod)
    }
    /// Number of valid values in each window (NaN where below `min_periods`).
    pub fn count(&self) -> Result<DataArray<T>> {
        self.reduce(|v| T::from_usize(nanops::count(v)).unwrap())
    }
    /// Materializes the windows as a new trailing dimension `window_dim`
    /// (positions outside the array are NaN), like `rolling(...).construct()`.
    pub fn construct(&self, window_dim: &str) -> Result<DataArray<T>> {
        let obj = self.obj;
        let ax = obj.axis(&self.dim)?;
        let n = obj.shape()[ax];
        let w = self.window;
        let lane_out: Vec<Vec<isize>> = (0..n)
            .map(|i| {
                let start = if self.center { i as isize - (w as isize) / 2 } else { i as isize - w as isize + 1 };
                (0..w as isize).map(|k| start + k).collect()
            })
            .collect();
        let mut shape = obj.shape().to_vec();
        shape.push(w);
        let mut out = ndarray::ArrayD::from_elem(ndarray::IxDyn(&shape), T::nan());
        for (i, pos) in lane_out.iter().enumerate() {
            for (k, &p) in pos.iter().enumerate() {
                if p >= 0 && (p as usize) < n {
                    let src = obj.data.index_axis(ndarray::Axis(ax), p as usize);
                    let mut dst = out.index_axis_mut(ndarray::Axis(obj.ndim()), k);
                    let mut dst = dst.index_axis_mut(ndarray::Axis(ax), i);
                    dst.assign(&src);
                }
            }
        }
        let mut dims = obj.dims.clone();
        dims.push(window_dim.to_string());
        Ok(DataArray::from_parts(out, dims, obj.coords.clone(), obj.attrs.clone(), obj.name.clone()))
    }
}

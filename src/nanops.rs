//! NaN-skipping kernels on slices (numpy's `nan*` functions).

use num_traits::Float;

/// Sum ignoring NaN (all-NaN or empty gives 0).
pub fn nansum<T: Float>(v: &[T]) -> T {
    v.iter().filter(|x| !x.is_nan()).fold(T::zero(), |a, &b| a + b)
}
/// Sum with NaN propagation.
pub fn sum<T: Float>(v: &[T]) -> T {
    v.iter().fold(T::zero(), |a, &b| a + b)
}
/// Sum ignoring NaN; NaN when fewer than `min_count` valid values.
pub fn nansum_min_count<T: Float>(v: &[T], min_count: usize) -> T {
    if count(v) < min_count { T::nan() } else { nansum(v) }
}
/// Product ignoring NaN.
pub fn nanprod<T: Float>(v: &[T]) -> T {
    v.iter().filter(|x| !x.is_nan()).fold(T::one(), |a, &b| a * b)
}
/// Number of non-NaN values.
pub fn count<T: Float>(v: &[T]) -> usize {
    v.iter().filter(|x| !x.is_nan()).count()
}
/// Mean ignoring NaN (all-NaN gives NaN).
pub fn nanmean<T: Float>(v: &[T]) -> T {
    let n = count(v);
    if n == 0 { T::nan() } else { nansum(v) / T::from(n).unwrap() }
}
/// Mean with NaN propagation.
pub fn mean<T: Float>(v: &[T]) -> T {
    if v.is_empty() { T::nan() } else { sum(v) / T::from(v.len()).unwrap() }
}
/// Variance ignoring NaN with `ddof` delta degrees of freedom.
pub fn nanvar<T: Float>(v: &[T], ddof: usize) -> T {
    let n = count(v);
    if n <= ddof {
        return T::nan();
    }
    let m = nanmean(v);
    let ss = v.iter().filter(|x| !x.is_nan()).fold(T::zero(), |a, &b| a + (b - m) * (b - m));
    ss / T::from(n - ddof).unwrap()
}
/// Variance with NaN propagation.
pub fn var<T: Float>(v: &[T], ddof: usize) -> T {
    if v.iter().any(|x| x.is_nan()) { T::nan() } else { nanvar(v, ddof) }
}
/// Standard deviation ignoring NaN.
pub fn nanstd<T: Float>(v: &[T], ddof: usize) -> T {
    nanvar(v, ddof).sqrt()
}
/// Minimum ignoring NaN (all-NaN gives NaN).
pub fn nanmin<T: Float>(v: &[T]) -> T {
    v.iter().filter(|x| !x.is_nan()).fold(None, |a: Option<T>, &b| Some(a.map_or(b, |a| a.min(b)))).unwrap_or(T::nan())
}
/// Maximum ignoring NaN (all-NaN gives NaN).
pub fn nanmax<T: Float>(v: &[T]) -> T {
    v.iter().filter(|x| !x.is_nan()).fold(None, |a: Option<T>, &b| Some(a.map_or(b, |a| a.max(b)))).unwrap_or(T::nan())
}
/// Minimum with NaN propagation.
pub fn min<T: Float>(v: &[T]) -> T {
    if v.iter().any(|x| x.is_nan()) { T::nan() } else { nanmin(v) }
}
/// Maximum with NaN propagation.
pub fn max<T: Float>(v: &[T]) -> T {
    if v.iter().any(|x| x.is_nan()) { T::nan() } else { nanmax(v) }
}
/// Position of the minimum, ignoring NaN (`None` if all NaN).
pub fn nanargmin<T: Float>(v: &[T]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, x) in v.iter().enumerate() {
        if !x.is_nan() && best.is_none_or(|b| *x < v[b]) {
            best = Some(i);
        }
    }
    best
}
/// Position of the maximum, ignoring NaN (`None` if all NaN).
pub fn nanargmax<T: Float>(v: &[T]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, x) in v.iter().enumerate() {
        if !x.is_nan() && best.is_none_or(|b| *x > v[b]) {
            best = Some(i);
        }
    }
    best
}
/// Quantile ignoring NaN with linear interpolation (numpy's default method).
pub fn nanquantile<T: Float>(v: &[T], q: f64) -> T {
    let mut s: Vec<T> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if s.is_empty() {
        return T::nan();
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = T::from(pos - lo as f64).unwrap();
    if lo == hi { s[lo] } else { s[lo] + (s[hi] - s[lo]) * frac }
}
/// Median ignoring NaN.
pub fn nanmedian<T: Float>(v: &[T]) -> T {
    nanquantile(v, 0.5)
}
/// Cumulative sum treating NaN as 0 (numpy's `nancumsum`).
pub fn nancumsum<T: Float>(v: &[T]) -> Vec<T> {
    let mut acc = T::zero();
    v.iter()
        .map(|&x| {
            if !x.is_nan() {
                acc = acc + x;
            }
            acc
        })
        .collect()
}
/// Cumulative product treating NaN as 1.
pub fn nancumprod<T: Float>(v: &[T]) -> Vec<T> {
    let mut acc = T::one();
    v.iter()
        .map(|&x| {
            if !x.is_nan() {
                acc = acc * x;
            }
            acc
        })
        .collect()
}

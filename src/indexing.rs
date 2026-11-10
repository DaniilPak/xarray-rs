//! Selector types for [`isel`](crate::DataArray::isel) and
//! [`sel`](crate::DataArray::sel).

use crate::coords::Label;
use crate::error::{Error, Result};
use chrono::{NaiveDate, NaiveDateTime};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Positional selector along one dimension (numpy semantics: negative
/// positions count from the end).
#[derive(Debug, Clone, PartialEq)]
pub enum IndexSel {
    /// A single position; the dimension is dropped.
    At(isize),
    /// A Python-style slice `start:stop:step`; the dimension is kept.
    Slice {
        /// First position (default: start of the axis).
        start: Option<isize>,
        /// Position to stop before (default: end of the axis).
        stop: Option<isize>,
        /// Step (non-zero; negative steps walk backwards).
        step: isize,
    },
    /// An explicit list of positions; the dimension is kept.
    Positions(Vec<isize>),
}

impl IndexSel {
    /// Python-style slice `start:stop:step`.
    pub fn slice(start: Option<isize>, stop: Option<isize>, step: isize) -> IndexSel {
        IndexSel::Slice { start, stop, step }
    }

    /// Resolves the selector into explicit positions. Returns `(positions,
    /// keep_dim)`.
    pub(crate) fn resolve(&self, dim: &str, n: usize) -> Result<(Vec<usize>, bool)> {
        let ni = n as isize;
        let norm = |i: isize| -> Result<usize> {
            let j = if i < 0 { i + ni } else { i };
            if j < 0 || j >= ni {
                Err(Error::OutOfBounds { dim: dim.to_string(), index: i, size: n })
            } else {
                Ok(j as usize)
            }
        };
        match self {
            IndexSel::At(i) => Ok((vec![norm(*i)?], false)),
            IndexSel::Positions(v) => Ok((v.iter().map(|&i| norm(i)).collect::<Result<_>>()?, true)),
            IndexSel::Slice { start, stop, step } => {
                let step = *step;
                if step == 0 {
                    return Err(Error::invalid("slice step cannot be zero"));
                }
                let clamp = |i: isize, lo: isize, hi: isize| i.max(lo).min(hi);
                let fix = |i: isize| if i < 0 { i + ni } else { i };
                let mut out = Vec::new();
                if step > 0 {
                    let a = clamp(start.map(fix).unwrap_or(0), 0, ni);
                    let b = clamp(stop.map(fix).unwrap_or(ni), 0, ni);
                    let mut i = a;
                    while i < b {
                        out.push(i as usize);
                        i += step;
                    }
                } else {
                    let a = clamp(start.map(fix).unwrap_or(ni - 1), -1, ni - 1);
                    let b = clamp(stop.map(fix).unwrap_or(-1), -1, ni - 1);
                    let mut i = a;
                    while i > b {
                        out.push(i as usize);
                        i += step;
                    }
                }
                Ok((out, true))
            }
        }
    }
}

macro_rules! isel_int {
    ($($t:ty),*) => {$(
        impl From<$t> for IndexSel { fn from(v: $t) -> Self { IndexSel::At(v as isize) } }
        impl From<Range<$t>> for IndexSel {
            fn from(r: Range<$t>) -> Self { IndexSel::slice(Some(r.start as isize), Some(r.end as isize), 1) }
        }
        impl From<RangeInclusive<$t>> for IndexSel {
            fn from(r: RangeInclusive<$t>) -> Self {
                IndexSel::slice(Some(*r.start() as isize), Some(*r.end() as isize + 1), 1)
            }
        }
        impl From<RangeFrom<$t>> for IndexSel {
            fn from(r: RangeFrom<$t>) -> Self { IndexSel::slice(Some(r.start as isize), None, 1) }
        }
        impl From<RangeTo<$t>> for IndexSel {
            fn from(r: RangeTo<$t>) -> Self { IndexSel::slice(None, Some(r.end as isize), 1) }
        }
        impl From<Vec<$t>> for IndexSel {
            fn from(v: Vec<$t>) -> Self { IndexSel::Positions(v.into_iter().map(|i| i as isize).collect()) }
        }
        impl From<&[$t]> for IndexSel {
            fn from(v: &[$t]) -> Self { IndexSel::Positions(v.iter().map(|&i| i as isize).collect()) }
        }
        impl<const N: usize> From<[$t; N]> for IndexSel {
            fn from(v: [$t; N]) -> Self { IndexSel::Positions(v.iter().map(|&i| i as isize).collect()) }
        }
    )*};
}
isel_int!(i32, i64, isize, usize, u32);

impl From<RangeFull> for IndexSel {
    fn from(_: RangeFull) -> Self {
        IndexSel::slice(None, None, 1)
    }
}

/// Label-based selector along one dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum Sel {
    /// One label. Drops the dimension when the label matches exactly one
    /// position. On datetime indexes a partial string such as `"2000-01"`
    /// selects the whole period and keeps the dimension.
    Label(Label),
    /// A list of labels; keeps the dimension.
    Labels(Vec<Label>),
    /// Inclusive label slice like `slice(start, stop)` in xarray.
    Slice(Option<Label>, Option<Label>),
}

impl Sel {
    /// Inclusive label slice; `None` means open-ended.
    pub fn slice(start: Option<impl Into<Label>>, stop: Option<impl Into<Label>>) -> Sel {
        Sel::Slice(start.map(Into::into), stop.map(Into::into))
    }
    /// List of labels.
    pub fn labels<L: Into<Label>>(labels: impl IntoIterator<Item = L>) -> Sel {
        Sel::Labels(labels.into_iter().map(Into::into).collect())
    }
}

macro_rules! sel_from {
    ($($t:ty),*) => {$(
        impl From<$t> for Sel { fn from(v: $t) -> Self { Sel::Label(v.into()) } }
        impl From<Vec<$t>> for Sel { fn from(v: Vec<$t>) -> Self { Sel::Labels(v.into_iter().map(Into::into).collect()) } }
        impl<const N: usize> From<[$t; N]> for Sel {
            fn from(v: [$t; N]) -> Self { Sel::Labels(v.into_iter().map(Into::into).collect()) }
        }
        impl From<RangeInclusive<$t>> for Sel {
            fn from(r: RangeInclusive<$t>) -> Self {
                let (a, b) = r.into_inner();
                Sel::Slice(Some(a.into()), Some(b.into()))
            }
        }
        impl From<RangeFrom<$t>> for Sel {
            fn from(r: RangeFrom<$t>) -> Self { Sel::Slice(Some(r.start.into()), None) }
        }
        impl From<RangeToInclusive<$t>> for Sel {
            fn from(r: RangeToInclusive<$t>) -> Self { Sel::Slice(None, Some(r.end.into())) }
        }
    )*};
}
sel_from!(i64, i32, u32, usize, f64, f32, bool, &str, String, NaiveDateTime, NaiveDate);

impl From<Label> for Sel {
    fn from(l: Label) -> Self {
        Sel::Label(l)
    }
}
impl From<Vec<Label>> for Sel {
    fn from(v: Vec<Label>) -> Self {
        Sel::Labels(v)
    }
}
impl From<RangeFull> for Sel {
    fn from(_: RangeFull) -> Self {
        Sel::Slice(None, None)
    }
}

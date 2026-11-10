//! Text representation modelled on xarray's `repr` (and numpy's
//! `array_repr` for the values).

use crate::attrs::Attrs;
use crate::coords::{Coord, CoordData};
use crate::dataarray::DataArray;
use crate::dataset::Dataset;
use crate::element::Element;
use crate::time;
use indexmap::IndexMap;
use ndarray::{ArrayD, ArrayViewD, Axis};
use std::fmt;

const DISPLAY_WIDTH: usize = 80;
const MAX_ROWS: usize = 12;

/// Python's `repr(float)`.
pub(crate) fn fmt_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let e = format!("{x:e}"); // shortest digits, e.g. "1.5e-5"
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..16).contains(&exp) {
        let s = format!("{x}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// Python's `format(x, ".{p}")` (general format, at least one decimal in
/// fixed notation).
pub(crate) fn fmt_general(x: f64, p: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let e = format!("{:.*e}", p - 1, x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp >= -4 && exp < (p as i32 - 1).max(1) {
        let dec = (p as i32 - 1 - exp).max(0) as usize;
        let mut s = format!("{x:.dec$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
        }
        if s.ends_with('.') || !s.contains('.') {
            if !s.contains('.') {
                s.push('.');
            }
            s.push('0');
        }
        s
    } else {
        let mut m = mant.to_string();
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// xarray's `render_human_readable_nbytes`.
pub(crate) fn human_bytes(n: usize) -> String {
    let units = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];
    let mut v = n as f64;
    let mut unit = units[0];
    for (i, u) in units.iter().enumerate() {
        unit = u;
        if v < 1000.0 || i == units.len() - 1 {
            break;
        }
        v /= 1000.0;
    }
    format!("{v:.0}{unit}")
}

// ---------------------------------------------------------------------------
// numpy-style array formatting
// ---------------------------------------------------------------------------

/// Positional float digits with at most `prec` fractional digits, trailing
/// zeros trimmed but the dot kept ("1.", "2.25").
fn positional(x: f64, prec: usize) -> String {
    let s = format!("{}", x.abs());
    let s = if s.contains('e') { format!("{:.prec$}", x.abs()) } else { s };
    let s = match s.split_once('.') {
        Some((_, f)) if f.len() > prec => format!("{:.prec$}", x.abs()),
        _ => s,
    };
    let mut s = if s.contains('.') { s } else { format!("{s}.") };
    while s.ends_with('0') && s.contains('.') && !s.ends_with(".") {
        s.pop();
    }
    if x.is_sign_negative() && x != 0.0 || (x == 0.0 && x.is_sign_negative()) { format!("-{s}") } else { s }
}

/// Scientific digits with at most `prec` fractional mantissa digits.
fn scientific(x: f64, prec: usize) -> (String, String, i32) {
    // shortest repr first, then limit precision
    let e = format!("{:e}", x.abs());
    let (m, ex) = e.split_once('e').unwrap();
    let (m, ex) = match m.split_once('.') {
        Some((_, f)) if f.len() > prec => {
            let e2 = format!("{:.prec$e}", x.abs());
            let (m2, ex2) = e2.split_once('e').unwrap();
            (m2.trim_end_matches('0').to_string(), ex2.to_string())
        }
        _ => (m.to_string(), ex.to_string()),
    };
    let ex: i32 = ex.parse().unwrap();
    let (int, frac) = match m.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (m.clone(), String::new()),
    };
    let int = if x.is_sign_negative() { format!("-{int}") } else { int };
    (int, frac, ex)
}

fn float_formatter(vals: &[f64], precision: usize) -> Box<dyn Fn(f64) -> String> {
    let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    let nz: Vec<f64> = finite.iter().filter(|v| **v != 0.0).map(|v| v.abs()).collect();
    let mut exp_format = false;
    if !nz.is_empty() {
        let max = nz.iter().cloned().fold(f64::MIN, f64::max);
        let min = nz.iter().cloned().fold(f64::MAX, f64::min);
        exp_format = max >= 1e8 || min < 1e-4 || max / min > 1000.0;
    }
    let has_nonfinite = finite.len() != vals.len();
    let neginf = vals.iter().any(|v| v.is_infinite() && *v < 0.0);
    if finite.is_empty() {
        let width = vals.iter().map(|v| fmt_nonfinite(*v).len()).max().unwrap_or(0);
        return Box::new(move |v| format!("{:>width$}", fmt_nonfinite(v)));
    }
    if exp_format {
        let parts: Vec<(String, String, i32)> = finite.iter().map(|&v| scientific(v, precision)).collect();
        let prec = parts.iter().map(|p| p.1.len()).max().unwrap_or(0);
        let exp_size = parts.iter().map(|p| p.2.abs().to_string().len()).max().unwrap_or(1).max(2);
        let mut pad_left = parts.iter().map(|p| p.0.len()).max().unwrap_or(1);
        let pad_right = exp_size + 2 + prec;
        if has_nonfinite {
            let offset = pad_right + 1;
            pad_left = pad_left.max(3usize.saturating_sub(offset)).max((3 + neginf as usize).saturating_sub(offset));
        }
        Box::new(move |v: f64| {
            if !v.is_finite() {
                let s = fmt_nonfinite(v);
                let w = pad_left + pad_right + 1;
                return format!("{s:>w$}");
            }
            let e = format!("{:.prec$e}", v.abs());
            let (m, ex) = e.split_once('e').unwrap();
            let ex: i32 = ex.parse().unwrap();
            let (int, frac) = m.split_once('.').unwrap_or((m, ""));
            let int = if v.is_sign_negative() { format!("-{int}") } else { int.to_string() };
            format!("{int:>pad_left$}.{frac}e{}{:0>exp_size$}", if ex < 0 { '-' } else { '+' }, ex.abs())
        })
    } else {
        let strs: Vec<String> = finite.iter().map(|&v| positional(v, precision)).collect();
        let mut pad_left = strs.iter().map(|s| s.split('.').next().unwrap().len()).max().unwrap_or(1);
        let pad_right = strs.iter().map(|s| s.split('.').nth(1).unwrap_or("").len()).max().unwrap_or(0);
        if has_nonfinite {
            let offset = pad_right + 1;
            pad_left = pad_left.max(3usize.saturating_sub(offset)).max((3 + neginf as usize).saturating_sub(offset));
        }
        Box::new(move |v: f64| {
            if !v.is_finite() {
                let s = fmt_nonfinite(v);
                let w = pad_left + pad_right + 1;
                return format!("{s:>w$}");
            }
            let s = positional(v, precision);
            let (i, f) = s.split_once('.').unwrap();
            format!("{i:>pad_left$}.{f:<pad_right$}")
        })
    }
}

fn fmt_nonfinite(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v > 0.0 {
        "inf".into()
    } else {
        "-inf".into()
    }
}

/// numpy's `repr(array)` (linewidth 75, precision 8, threshold 1000,
/// edgeitems 3).
pub(crate) fn numpy_repr<T: Element>(a: &ArrayD<T>) -> String {
    let dtype = T::dtype();
    let summarize = a.len() > 1000;
    let edge = 3;
    // collect the values that will be shown to build the formatter
    let shown = leading_trailing(a.view(), if summarize { Some(edge) } else { None });
    let fmt: Box<dyn Fn(&T) -> String> = if dtype.starts_with("float") {
        let vals: Vec<f64> = shown.iter().map(|v| v.to_f64().unwrap()).collect();
        let f = float_formatter(&vals, 8);
        Box::new(move |v: &T| f(v.to_f64().unwrap()))
    } else if dtype.starts_with("int") || dtype.starts_with("uint") {
        let w = shown.iter().map(|v| v.repr().len()).max().unwrap_or(1);
        Box::new(move |v: &T| format!("{:>w$}", v.repr()))
    } else if dtype == "bool" {
        Box::new(move |v: &T| {
            let r = v.repr();
            if r == "True" { " True".to_string() } else { r }
        })
    } else {
        Box::new(|v: &T| v.repr())
    };
    let suffix = match dtype {
        "float64" | "int64" | "bool" => String::new(),
        "<U" => {
            let n = a.iter().map(|v| v.item_size() / 4).max().unwrap_or(1);
            format!(", dtype='<U{n}'")
        }
        other => format!(", dtype={other}"),
    };
    if a.is_empty() {
        let shape = if a.ndim() > 1 {
            format!(", shape={:?}", a.shape()).replace('[', "(").replace(']', ")")
        } else {
            String::new()
        };
        let d = if dtype == "<U" { "'<U1'".to_string() } else { dtype.to_string() };
        return format!("array([]{shape}, dtype={d})");
    }
    if a.ndim() == 0 {
        let v = a.iter().next().unwrap();
        let s = if dtype == "bool" { v.repr() } else { fmt(v).trim_start().to_string() };
        return format!("array({s}{suffix})");
    }
    let body = format_rec(a.view(), &fmt, "       ", 74, if summarize { Some(edge) } else { None });
    format!("array({body}{suffix})")
}

fn leading_trailing<T: Clone>(a: ArrayViewD<'_, T>, edge: Option<usize>) -> Vec<T> {
    match edge {
        None => a.iter().cloned().collect(),
        Some(e) => {
            if a.ndim() == 0 {
                return a.iter().cloned().collect();
            }
            let n = a.shape()[0];
            let idx: Vec<usize> = if n > 2 * e { (0..e).chain(n - e..n).collect() } else { (0..n).collect() };
            idx.into_iter().flat_map(|i| leading_trailing(a.index_axis(Axis(0), i), Some(e))).collect()
        }
    }
}

fn extend_line(s: &mut String, line: &mut String, word: &str, width: usize, prefix: &str) {
    let needs_wrap = line.len() + word.len() > width && line.len() > prefix.len();
    if needs_wrap {
        s.push_str(line.trim_end());
        s.push('\n');
        *line = prefix.to_string();
    }
    line.push_str(word);
}

fn format_rec<T>(
    a: ArrayViewD<'_, T>,
    fmt: &dyn Fn(&T) -> String,
    hanging: &str,
    width: usize,
    edge: Option<usize>,
) -> String {
    let n = a.shape()[0];
    let next_hanging = format!("{hanging} ");
    let next_width = width - 1;
    let summary = edge.is_some_and(|e| 2 * e < n);
    let (lead, trail) = if summary { (edge.unwrap(), edge.unwrap()) } else { (0, n) };
    let mut s = String::new();
    let sub = |i: usize| -> String {
        let v = a.index_axis(Axis(0), i);
        if v.ndim() == 0 { fmt(v.iter().next().unwrap()) } else { format_rec(v, fmt, &next_hanging, next_width, edge) }
    };
    if a.ndim() == 1 {
        let elem_width = width - 1;
        let mut line = hanging.to_string();
        for i in 0..lead {
            extend_line(&mut s, &mut line, &sub(i), elem_width, hanging);
            line.push_str(", ");
        }
        if summary {
            extend_line(&mut s, &mut line, "...", elem_width, hanging);
            line.push_str(", ");
        }
        for k in (2..=trail).rev() {
            extend_line(&mut s, &mut line, &sub(n - k), elem_width, hanging);
            line.push_str(", ");
        }
        if n > 0 {
            extend_line(&mut s, &mut line, &sub(n - 1), width, hanging);
        }
        s.push_str(&line);
    } else {
        let line_sep = format!(",{}", "\n".repeat(a.ndim() - 1));
        for i in 0..lead {
            s.push_str(hanging);
            s.push_str(&sub(i));
            s.push_str(&line_sep);
        }
        if summary {
            s.push_str(hanging);
            s.push_str("...");
            s.push_str(&line_sep);
        }
        for k in (2..=trail).rev() {
            s.push_str(hanging);
            s.push_str(&sub(n - k));
            s.push_str(&line_sep);
        }
        s.push_str(hanging);
        s.push_str(&sub(n - 1));
    }
    format!("[{}]", &s[hanging.len()..])
}

fn limit_lines(s: &str, limit: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() > limit {
        let half = limit / 2;
        let mut out: Vec<&str> = lines[..half].to_vec();
        out.push("...");
        out.extend(&lines[lines.len() - half..]);
        out.join("\n")
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// inline (one-line) summaries
// ---------------------------------------------------------------------------

fn format_flat(items: &[String], size: usize, max_width: usize, get: &dyn Fn(usize) -> String) -> String {
    let _ = items;
    let max_relevant = size.max(1).min((max_width.div_ceil(2)).max(2));
    let n_front = max_relevant.div_ceil(2).min(size);
    let n_back = (max_relevant / 2).min(size);
    let front: Vec<String> = (0..n_front).map(get).collect();
    let back: Vec<String> = (size - n_back..size).map(get).collect();
    let mut relevant: Vec<String> = Vec::new();
    let mut fi = front.iter();
    let mut bi = back.iter().rev();
    loop {
        let a = fi.next();
        let b = bi.next();
        if a.is_none() && b.is_none() {
            break;
        }
        if let Some(a) = a {
            relevant.push(a.clone());
        }
        if let Some(b) = b {
            relevant.push(b.clone());
        }
    }
    relevant.truncate(max_relevant);
    let mut cum = Vec::new();
    let mut acc = 0usize;
    for s in &relevant {
        acc += s.len() + 1;
        cum.push(acc - 1);
    }
    let (count, padding) = if size > 2 && (max_relevant < size || cum.iter().any(|&c| c > max_width)) {
        let padding = " ... ";
        let first = cum.iter().position(|&c| c + padding.len() - 1 > max_width).unwrap_or(0);
        (size.min(first.max(2)), padding)
    } else {
        (size, if size <= 1 { "" } else { " " })
    };
    let num_front = count.div_ceil(2);
    let num_back = count - num_front;
    let mut s = front[..num_front.min(front.len())].join(" ");
    s.push_str(padding);
    if num_back > 0 {
        s.push_str(&back[back.len() - num_back..].join(" "));
    }
    if s.chars().count() > max_width {
        s = s.chars().take(max_width.saturating_sub(3)).collect::<String>() + "...";
    }
    s
}

fn coord_item(d: &CoordData, i: usize) -> String {
    match d {
        CoordData::Int(a) => a.iter().nth(i).unwrap().to_string(),
        CoordData::Float(a) => fmt_general(*a.iter().nth(i).unwrap(), 4),
        CoordData::Str(a) => format!("'{}'", a.iter().nth(i).unwrap()),
        CoordData::DateTime(a) => {
            let v = *a.iter().nth(i).unwrap();
            let midnight = v != time::NAT && v.rem_euclid(86_400_000_000_000) == 0;
            time::format_ns(v, midnight)
        }
        CoordData::Bool(a) => if *a.iter().nth(i).unwrap() { "True" } else { "False" }.to_string(),
    }
}

fn element_item<T: Element>(v: &T) -> String {
    if T::dtype().starts_with("float") { fmt_general(v.to_f64().unwrap(), 4) } else { v.repr() }
}

fn pretty(s: &str, n: usize) -> String {
    let s = maybe_truncate(s, n);
    format!("{s:<n$}")
}

fn maybe_truncate(s: &str, n: usize) -> String {
    if s.chars().count() > n { s.chars().take(n.saturating_sub(3)).collect::<String>() + "..." } else { s.to_string() }
}

fn summarize_line(
    name: &str,
    dims: &[String],
    dtype: &str,
    nbytes: usize,
    is_index: bool,
    col_width: usize,
    values: &dyn Fn(usize) -> String,
) -> String {
    let marker = if is_index { "*" } else { " " };
    let first = pretty(&format!("  {marker} {name} "), col_width);
    let dims_str = if dims.is_empty() { String::new() } else { format!("({}) ", dims.join(", ")) };
    let front = format!("{first}{dims_str}{dtype} {} ", human_bytes(nbytes));
    let width = DISPLAY_WIDTH.saturating_sub(front.len());
    format!("{front}{}", values(width))
}

fn coord_line(name: &str, c: &Coord, col_width: usize) -> String {
    let n = c.len();
    let is_index = c.dims.len() == 1 && c.dims[0] == name;
    summarize_line(name, &c.dims, &c.data.dtype(), c.data.nbytes(), is_index, col_width, &|w| {
        format_flat(&[], n, w, &|i| coord_item(&c.data, i))
    })
}

fn col_width<'a>(names: impl Iterator<Item = &'a String>) -> usize {
    names.map(|s| s.chars().count()).max().unwrap_or(0).max(7) + 6
}

fn ordered_coords<'a>(coords: &'a IndexMap<String, Coord>, dims: &[String]) -> Vec<(&'a String, &'a Coord)> {
    let mut out = Vec::new();
    for d in dims {
        if let Some(c) = coords.get(d) {
            out.push((coords.get_key_value(d).unwrap().0, c));
        }
        for (k, c) in coords {
            if k != d && c.dims.first() == Some(d) {
                out.push((k, c));
            }
        }
    }
    for (k, c) in coords {
        if !out.iter().any(|(o, _)| *o == k) {
            out.push((k, c));
        }
    }
    out
}

fn section(out: &mut String, title: &str, lines: Vec<String>) {
    out.push_str(&format!("\n{title}:"));
    if lines.len() > MAX_ROWS {
        let half = MAX_ROWS / 2;
        for l in &lines[..half] {
            out.push_str(&format!("\n{l}"));
        }
        out.push_str("\n    ...");
        for l in &lines[lines.len() - half..] {
            out.push_str(&format!("\n{l}"));
        }
    } else {
        for l in lines {
            out.push_str(&format!("\n{l}"));
        }
    }
}

fn attrs_section(out: &mut String, attrs: &Attrs) {
    if attrs.is_empty() {
        return;
    }
    let w = col_width(attrs.keys());
    let lines = attrs
        .iter()
        .map(|(k, v)| {
            let key = pretty(&format!("    {k}:"), w);
            let val = v.to_string().replace('\t', "\\t").replace('\n', "\\n");
            maybe_truncate(&format!("{key} {val}"), DISPLAY_WIDTH)
        })
        .collect();
    section(out, "Attributes", lines);
}

fn data_nbytes<T: Element>(a: &ArrayD<T>) -> usize {
    if T::dtype() == "<U" {
        let w = a.iter().map(|v| v.item_size()).max().unwrap_or(4);
        w * a.len()
    } else {
        a.iter().next().map(|v| v.item_size()).unwrap_or(std::mem::size_of::<T>()) * a.len()
    }
}

fn dtype_of<T: Element>(a: &ArrayD<T>) -> String {
    if T::dtype() == "<U" {
        format!("<U{}", a.iter().map(|v| v.item_size() / 4).max().unwrap_or(1))
    } else {
        T::dtype().to_string()
    }
}

impl<T: Element> fmt::Display for DataArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name.as_ref().map(|n| format!("'{n}' ")).unwrap_or_default();
        let dims: Vec<String> = self.dims.iter().zip(self.shape()).map(|(d, n)| format!("{d}: {n}")).collect();
        let mut out = format!(
            "<xarray.DataArray {name}({})> Size: {}\n{}",
            dims.join(", "),
            human_bytes(data_nbytes(&self.data)),
            limit_lines(&numpy_repr(&self.data), 40)
        );
        if !self.coords.is_empty() {
            let w = col_width(self.coords.keys());
            let lines =
                ordered_coords(&self.coords, &self.dims).into_iter().map(|(k, c)| coord_line(k, c, w)).collect();
            section(&mut out, "Coordinates", lines);
        }
        let unindexed: Vec<&str> = self.dims.iter().filter(|d| !self.has_index(d)).map(|s| s.as_str()).collect();
        if !unindexed.is_empty() {
            out.push_str(&format!("\nDimensions without coordinates: {}", unindexed.join(", ")));
        }
        attrs_section(&mut out, &self.attrs);
        f.write_str(&out)
    }
}

impl<T: Element> fmt::Display for Dataset<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = self.dims();
        let total: usize = self.vars.values().map(|v| data_nbytes(&v.data)).sum::<usize>()
            + self.coords.values().map(|c| c.data.nbytes()).sum::<usize>();
        let w = col_width(self.coords.keys().chain(self.vars.keys()));
        let mut out = format!("<xarray.Dataset> Size: {}", human_bytes(total));
        let dim_s: Vec<String> = dims.iter().map(|(d, n)| format!("{d}: {n}")).collect();
        out.push_str(&format!("\n{}({})", pretty("Dimensions:", w), dim_s.join(", ")));
        let dim_names: Vec<String> = dims.keys().cloned().collect();
        if !self.coords.is_empty() {
            let lines =
                ordered_coords(&self.coords, &dim_names).into_iter().map(|(k, c)| coord_line(k, c, w)).collect();
            section(&mut out, "Coordinates", lines);
        }
        let unindexed: Vec<&str> = dim_names
            .iter()
            .filter(|d| !self.coords.get(*d).is_some_and(|c| c.dims == [d.to_string()]))
            .map(|s| s.as_str())
            .collect();
        if !unindexed.is_empty() {
            out.push_str(&format!("\nDimensions without coordinates: {}", unindexed.join(", ")));
        }
        let lines = self
            .vars
            .iter()
            .map(|(k, v)| {
                let n = v.size();
                let flat: Vec<&T> = v.data.iter().collect();
                summarize_line(k, &v.dims, &dtype_of(&v.data), data_nbytes(&v.data), false, w, &|width| {
                    format_flat(&[], n, width, &|i| element_item(flat[i]))
                })
            })
            .collect();
        section(&mut out, "Data variables", lines);
        attrs_section(&mut out, &self.attrs);
        f.write_str(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats() {
        assert_eq!(fmt_float(1.0), "1.0");
        assert_eq!(fmt_float(1e-5), "1e-05");
        assert_eq!(fmt_float(1e20), "1e+20");
        assert_eq!(fmt_float(0.1), "0.1");
        for (x, s) in [
            (1.0, "1.0"),
            (123.456, "123.5"),
            (1234.5, "1.234e+03"),
            (999.99, "1e+03"),
            (100.0, "100.0"),
            (0.0001234, "0.0001234"),
            (1e-5, "1e-05"),
            (1.23456, "1.235"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(fmt_general(x, 4), s, "{x}");
        }
        assert_eq!(human_bytes(48), "48B");
        assert_eq!(human_bytes(8000), "8kB");
    }

    #[test]
    fn numpy_like() {
        use ndarray::array;
        assert_eq!(
            numpy_repr(&array![[1.0, f64::NAN, 3.0], [4.0, 5.0, 6.0]].into_dyn()),
            "array([[ 1., nan,  3.],\n       [ 4.,  5.,  6.]])"
        );
        assert_eq!(numpy_repr(&array![1.0, 2000.0].into_dyn()), "array([1.e+00, 2.e+03])");
        assert_eq!(numpy_repr(&array![1.5, 2.25, -3.0].into_dyn()), "array([ 1.5 ,  2.25, -3.  ])");
        assert_eq!(numpy_repr(&array![true, false].into_dyn()), "array([ True, False])");
        assert_eq!(numpy_repr(&array![1i32, 2].into_dyn()), "array([1, 2], dtype=int32)");
        assert_eq!(numpy_repr(&ndarray::arr0(5.0).into_dyn()), "array(5.)");
    }
}

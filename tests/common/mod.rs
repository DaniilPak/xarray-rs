//! Shared comparison helpers for the integration tests.
#![allow(dead_code)]

use serde_json::Value;
use xarray_rs::*;

pub fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

pub fn labels_equal(a: &Coord, b: &Coord) -> bool {
    let (la, lb) = (a.labels(), b.labels());
    la.len() == lb.len()
        && la.iter().zip(&lb).all(|(x, y)| match (x, y) {
            (Label::Str(p), Label::Str(q)) => p == q,
            (Label::DateTime(p), Label::DateTime(q)) => p == q,
            _ => match (x.as_f64(), y.as_f64()) {
                (Some(p), Some(q)) => close(p, q),
                _ => false,
            },
        })
}

pub fn check(id: &str, actual: &DataArray<f64>, expected: &Value) {
    let exp = DataArray::<f64>::from_dict(expected).unwrap_or_else(|e| panic!("{id}: bad expected: {e}"));
    assert_eq!(actual.dims(), exp.dims(), "{id}: dims");
    assert_eq!(actual.shape(), exp.shape(), "{id}: shape");
    for (i, (a, b)) in actual.data().iter().zip(exp.data().iter()).enumerate() {
        assert!(
            close(*a, *b),
            "{id}: value #{i}: got {a}, expected {b}\nactual: {:?}\nexpected: {:?}",
            actual.to_vec(),
            exp.to_vec()
        );
    }
    let mut an: Vec<&String> = actual.coords().keys().collect();
    let mut en: Vec<&String> = exp.coords().keys().collect();
    an.sort();
    en.sort();
    assert_eq!(an, en, "{id}: coordinate names");
    for (k, c) in exp.coords() {
        let a = actual.coord(k).unwrap();
        assert_eq!(a.dims(), c.dims(), "{id}: dims of coord {k}");
        assert!(labels_equal(a, c), "{id}: coord {k}: got {:?}, expected {:?}", a.labels(), c.labels());
    }
    assert_eq!(actual.name(), exp.name(), "{id}: name");
}

pub fn check_ds(id: &str, actual: &Dataset<f64>, expected: &Value) {
    let exp = Dataset::<f64>::from_dict(expected).unwrap_or_else(|e| panic!("{id}: bad expected: {e}"));
    let mut a = actual.data_vars();
    let mut e = exp.data_vars();
    a.sort();
    e.sort();
    assert_eq!(a, e, "{id}: data vars");
    for name in exp.data_vars() {
        let mut e = exp.get(name).unwrap().to_dict();
        e["name"] = Value::String(name.to_string());
        check(&format!("{id}/{name}"), &actual.get(name).unwrap(), &e);
    }
}

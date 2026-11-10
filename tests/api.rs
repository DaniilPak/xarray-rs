//! Ergonomics and behaviour tests of the public API.

use chrono::NaiveDate;
use ndarray::array;
use xarray_rs::*;

fn sample() -> DataArray<f64> {
    DataArray::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["x", "y"])
        .unwrap()
        .with_coord("x", vec![10, 20])
        .unwrap()
        .with_coord("y", vec!["a", "b", "c"])
        .unwrap()
        .with_coord("lon", ("y", vec![0.5, 1.5, 2.5]))
        .unwrap()
        .with_name("temp")
        .with_attr("units", "K")
}

#[test]
fn literal_selectors_infer() {
    let da = sample();
    assert_eq!(da.isel("x", 0).unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(da.isel("y", -1).unwrap().to_vec(), vec![3.0, 6.0]);
    assert_eq!(da.isel("y", 1..3).unwrap().shape(), &[2, 2]);
    assert_eq!(da.isel("y", ..).unwrap().shape(), &[2, 3]);
    assert_eq!(
        da.isel("y", vec![2, 0]).unwrap().coord("y").unwrap().labels(),
        vec![Label::from("c"), Label::from("a")]
    );
    assert_eq!(da.sel("x", 20).unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    assert_eq!(da.sel("y", "b"..="c").unwrap().shape(), &[2, 2]);
    assert_eq!(da.sel("x", ..=15).unwrap().shape(), &[1, 3]);
    assert_eq!(da.sel("y", ["c", "a"]).unwrap().isel("x", 0).unwrap().to_vec(), vec![3.0, 1.0]);
    assert_eq!(da.sel_nearest("x", 16).unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    assert!(matches!(da.sel("x", 15), Err(Error::LabelNotFound { .. })));
    assert!(matches!(da.isel("x", 5), Err(Error::OutOfBounds { .. })));
    assert!(matches!(da.isel("nope", 0), Err(Error::DimNotFound(..))));
    // a scalar selection keeps the label as a scalar coordinate
    let s = da.sel("x", 10).unwrap();
    assert_eq!(s.coord("x").unwrap().dims().len(), 0);
}

#[test]
fn operators_align_and_broadcast() {
    let a = sample();
    let b = DataArray::new(vec![100.0, 200.0, 300.0], "x").unwrap().with_coord("x", vec![0, 10, 20]).unwrap();
    let c = &a + &b; // inner join on x, broadcast over y
    assert_eq!(c.dims(), ["x", "y"]);
    assert_eq!(c.to_vec(), vec![201.0, 202.0, 203.0, 304.0, 305.0, 306.0]);
    let d = &a * 2.0;
    assert_eq!(d.to_vec()[5], 12.0);
    assert_eq!(d.attrs()["units"], AttrValue::from("K"));
    let e = 1.0 - &a;
    assert_eq!(e.to_vec()[0], 0.0);
    let n = -&a;
    assert_eq!(n.to_vec()[0], -1.0);
    let mask = a.gt(2.5);
    assert_eq!(a.where_(&mask).unwrap().count(..).unwrap().item().unwrap(), 4);
    assert_eq!(a.where_or(&mask, 0.0).unwrap().sum(..).unwrap().item().unwrap(), 18.0);
    let both = &mask & &a.lt(5.5);
    assert_eq!(both.count_true(..).unwrap().item().unwrap(), 3);
    let outer = a.zip_with_join(&b, Join::Outer, |x, y| x + y).unwrap();
    assert_eq!(outer.shape(), &[3, 3]);
    assert!(outer.to_vec()[0].is_nan());
}

#[test]
fn reductions_and_transforms() {
    let a = sample();
    assert_eq!(a.mean("y").unwrap().to_vec(), vec![2.0, 5.0]);
    assert_eq!(a.sum(["x", "y"]).unwrap().item().unwrap(), 21.0);
    assert_eq!(a.max(..).unwrap().item().unwrap(), 6.0);
    assert_eq!(a.argmax("y").unwrap().to_vec(), vec![2, 2]);
    assert!(a.mean("y").unwrap().coords().contains_key("x"));
    assert!(!a.mean("y").unwrap().coords().contains_key("lon"));
    assert_eq!(a.cumsum("y").unwrap().isel("x", 1).unwrap().to_vec(), vec![4.0, 9.0, 15.0]);
    assert_eq!(a.transpose(["y", "x"]).unwrap().shape(), &[3, 2]);
    assert_eq!(a.t().dims(), ["y", "x"]);
    assert_eq!(a.expand_dims("band").unwrap().dims(), ["band", "x", "y"]);
    assert_eq!(a.expand_dims("band").unwrap().squeeze().dims(), ["x", "y"]);
    assert_eq!(a.rename_vars("x", "row").unwrap().coord("row").unwrap().dims(), ["row"]);
    assert_eq!(a.rename("t2").name(), Some("t2"));
    assert_eq!(a.astype::<i64>().to_vec(), vec![1, 2, 3, 4, 5, 6]);
    assert!(a.equals(&a.clone()));
    assert!(!a.identical(&a.rename("other")));
}

#[test]
fn groupby_resample_rolling() {
    let days: Vec<NaiveDate> = (1..=10).map(|d| NaiveDate::from_ymd_opt(2000, 1, d).unwrap()).collect();
    let ts =
        DataArray::new((1..=10).map(f64::from).collect::<Vec<_>>(), "time").unwrap().with_coord("time", days).unwrap();
    let weekly = ts.resample("time", "5D").unwrap().mean().unwrap();
    assert_eq!(weekly.to_vec(), vec![3.0, 8.0]);
    let by_dow = ts.groupby("time.dayofweek").unwrap().count().unwrap();
    assert_eq!(by_dow.dims(), ["dayofweek"]);
    assert_eq!(by_dow.sum(..).unwrap().item().unwrap(), 10.0);
    let r = ts.rolling("time", 3).unwrap().center(true).mean().unwrap();
    assert_eq!(r.to_vec()[1], 2.0);
    assert!(r.to_vec()[0].is_nan());
    let anomalies = ts.groupby("time.month").unwrap().map(|g| g.sub(&g.mean("time")?)).unwrap();
    assert_eq!(anomalies.sum(..).unwrap().item().unwrap(), 0.0);
    let i =
        ts.interp("time", vec![NaiveDate::from_ymd_opt(2000, 1, 2).unwrap().and_hms_opt(12, 0, 0).unwrap()]).unwrap();
    assert_eq!(i.to_vec(), vec![2.5]);
}

#[test]
fn dataset_ops() {
    let a = sample();
    let b = DataArray::new(vec![7.0, 8.0], "x").unwrap().with_coord("x", vec![10, 20]).unwrap();
    let ds = Dataset::new().with_var("a", a.clone()).unwrap().with_var("b", b).unwrap().with_attr("title", "demo");
    assert_eq!(ds.dims().get("y"), Some(&3));
    assert_eq!(ds.data_vars(), ["a", "b"]);
    let m = ds.mean("y").unwrap();
    assert_eq!(m.get("a").unwrap().to_vec(), vec![2.0, 5.0]);
    assert_eq!(m.get("b").unwrap().to_vec(), vec![7.0, 8.0]);
    let s = ds.sel("x", 20).unwrap();
    assert_eq!(s.get("b").unwrap().item().unwrap(), 8.0);
    let both = Dataset::concat(&[&ds.isel("x", 0..1).unwrap(), &ds.isel("x", 1..2).unwrap()], "x").unwrap();
    assert!(both.equals(&ds));
    let other = DataArray::new(vec![1.0, 2.0], "x").unwrap().with_coord("x", vec![20, 30]).unwrap().with_name("c");
    let merged = Dataset::merge(&[&ds, &other.to_dataset(None).unwrap()]).unwrap();
    assert_eq!(merged.dims()["x"], 3);
    assert!(merged.get("a").unwrap().isel("x", 2).unwrap().to_vec()[0].is_nan());
    let arr = ds.select_vars(vec!["b"]).unwrap().to_dataarray("variable").unwrap();
    assert_eq!(arr.dims(), ["variable", "x"]);
    assert!(ds.to_string().starts_with("<xarray.Dataset>"));
}

#[test]
fn display_matches_xarray_style() {
    let s = sample().to_string();
    let expected = "<xarray.DataArray 'temp' (x: 2, y: 3)> Size: 48B
array([[1., 2., 3.],
       [4., 5., 6.]])
Coordinates:
  * x        (x) int64 16B 10 20
  * y        (y) <U1 12B 'a' 'b' 'c'
    lon      (y) float64 24B 0.5 1.5 2.5
Attributes:
    units:    K";
    assert_eq!(s, expected);
}

#[test]
fn integer_arrays_refuse_missing_values() {
    let a = DataArray::new(vec![1i64, 2, 3], "x").unwrap().with_coord("x", vec![0, 1, 2]).unwrap();
    let b = DataArray::new(vec![1i64, 2], "x").unwrap().with_coord("x", vec![1, 5]).unwrap();
    assert!((&a + &b).to_vec() == vec![3]);
    assert!(matches!(align(&[&a, &b], Join::Outer), Err(Error::Dtype(_))));
    assert!(align(&[&a.astype::<f64>(), &b.astype::<f64>()], Join::Outer).is_ok());
}

#[cfg(feature = "json")]
#[test]
fn json_round_trip() {
    let a = sample();
    let back = DataArray::<f64>::from_json(&a.to_json()).unwrap();
    assert!(back.identical(&a));
    let ds = Dataset::new().with_var("a", a).unwrap();
    assert!(Dataset::<f64>::from_json(&ds.to_json()).unwrap().equals(&ds));
}

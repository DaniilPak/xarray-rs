//! Parity tests against reference results produced by Python xarray
//! (see tests/fixtures/generate.py).
#![cfg(feature = "json")]

use serde_json::Value;
use std::sync::OnceLock;
use xarray_rs::*;

mod common;
use common::*;

fn fixtures() -> &'static Value {
    static F: OnceLock<Value> = OnceLock::new();
    F.get_or_init(|| serde_json::from_str(include_str!("fixtures/xarray_cases.json")).unwrap())
}

fn input(name: &str) -> DataArray<f64> {
    DataArray::from_dict(&fixtures()["inputs"][name]).unwrap_or_else(|e| panic!("input {name}: {e}"))
}

fn label(v: &Value) -> Label {
    match v {
        Value::Number(n) if n.is_i64() => Label::Int(n.as_i64().unwrap()),
        Value::Number(n) => Label::Float(n.as_f64().unwrap()),
        Value::String(s) => Label::Str(s.clone()),
        Value::Bool(b) => Label::Bool(*b),
        other => panic!("bad label {other}"),
    }
}

fn sel_of(v: &Value) -> Sel {
    match v {
        Value::Array(a) => Sel::Labels(a.iter().map(label).collect()),
        other => Sel::Label(label(other)),
    }
}

fn opt_label(v: &Value) -> Option<Label> {
    if v.is_null() { None } else { Some(label(v)) }
}

fn dims_of(v: &Value) -> Vec<String> {
    match v {
        Value::Null => vec!["...".into()],
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().map(|d| d.as_str().unwrap().to_string()).collect(),
        other => panic!("bad dims {other}"),
    }
}

fn to_f64<T: Element + Copy + num_traits_shim::ToF64>(da: &DataArray<T>) -> DataArray<f64> {
    da.map(|v| v.to_f())
}

mod num_traits_shim {
    pub trait ToF64 {
        fn to_f(&self) -> f64;
    }
    impl ToF64 for i64 {
        fn to_f(&self) -> f64 {
            *self as f64
        }
    }
    impl ToF64 for bool {
        fn to_f(&self) -> f64 {
            *self as u8 as f64
        }
    }
    impl ToF64 for f64 {
        fn to_f(&self) -> f64 {
            *self
        }
    }
}

enum Out {
    One(DataArray<f64>),
    Many(Vec<DataArray<f64>>),
    Ds(Dataset<f64>),
}

fn run(case: &Value) -> Result<Out> {
    let op = case["op"].as_str().unwrap();
    let kw = &case["kwargs"];
    let ins: Vec<&str> = case["inputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let a = || input(ins[0]);
    let b = || input(ins[1]);
    let s = |k: &str| kw[k].as_str().unwrap().to_string();
    let dim = || s("dim");
    use Out::*;
    Ok(match op {
        "isel" => {
            let sel: IndexSel = match &kw["index"] {
                Value::Array(v) => IndexSel::Positions(v.iter().map(|i| i.as_i64().unwrap() as isize).collect()),
                v => IndexSel::At(v.as_i64().unwrap() as isize),
            };
            One(a().isel(&dim(), sel)?)
        }
        "isel_slice" => {
            let sl = kw["slice"].as_array().unwrap();
            let g = |i: usize| sl[i].as_i64().map(|v| v as isize);
            One(a().isel(&dim(), IndexSel::slice(g(0), g(1), g(2).unwrap_or(1)))?)
        }
        "sel" => One(a().sel(&dim(), sel_of(&kw["label"]))?),
        "sel_slice" => {
            let mut da = a();
            if kw["pre"] == "sort_desc" {
                da = da.sortby(&dim(), false)?;
            }
            One(da.sel(&dim(), Sel::Slice(opt_label(&kw["start"]), opt_label(&kw["stop"])))?)
        }
        "sel_method" => {
            let m = match kw["method"].as_str().unwrap() {
                "nearest" => Method::Nearest,
                "pad" => Method::Pad,
                "backfill" => Method::Backfill,
                _ => unreachable!(),
            };
            One(a().sel_method(&dim(), sel_of(&kw["label"]), m, kw["tolerance"].as_f64())?)
        }
        "transpose" => One(a().transpose(dims_of(&kw["order"]))?),
        "squeeze" => One(a().isel("x", vec![1])?.squeeze()),
        "expand_dims" => One(a().expand_dims(&dim())?),
        "rename_vars" => One(a().rename_vars(&s("old"), &s("new"))?),
        "rename" => One(a().rename(s("name"))),
        "where_gt" => {
            let da = a();
            let cond = da.gt(kw["threshold"].as_f64().unwrap());
            match kw["other"].as_f64() {
                Some(o) => One(da.where_or(&cond, o)?),
                None => One(da.where_(&cond)?),
            }
        }
        "shift" => One(a().shift(&dim(), kw["n"].as_i64().unwrap() as isize)?),
        "roll" => One(a().roll(&dim(), kw["n"].as_i64().unwrap() as isize)?),
        "diff" => One(a().diff(&dim(), kw["n"].as_u64().unwrap() as usize)?),
        "sortby" => One(a().sortby(&dim(), kw["ascending"].as_bool().unwrap())?),
        "reindex" => {
            let labels: Vec<i64> = kw["labels"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            One(a().reindex(&dim(), labels)?)
        }
        "fillna" => One(a().fillna(kw["value"].as_f64().unwrap())),
        "dropna" => One(a().dropna(&dim(), kw["how"] == "all")?),
        "ffill" => One(a().ffill(&dim(), kw["limit"].as_u64().map(|v| v as usize))?),
        "bfill" => One(a().bfill(&dim(), kw["limit"].as_u64().map(|v| v as usize))?),
        "interpolate_na" => One(a().interpolate_na(&dim())?),
        "isnull" => One(to_f64(&a().isnull())),
        "clip" => One(a().clip(kw["min"].as_f64().unwrap(), kw["max"].as_f64().unwrap())),
        "reduce" => {
            let da = a();
            let d = dims_of(&kw["dims"]);
            let skip = kw["skipna"].as_bool().unwrap_or(true);
            let ddof = kw["ddof"].as_u64().unwrap_or(0) as usize;
            let one = || d[0].clone();
            One(match kw["func"].as_str().unwrap() {
                "sum" if kw["min_count"].is_u64() => da.sum_min_count(d, kw["min_count"].as_u64().unwrap() as usize)?,
                "sum" if !skip => da.sum_noskip(d)?,
                "mean" if !skip => da.mean_noskip(d)?,
                "sum" => da.sum(d)?,
                "mean" => da.mean(d)?,
                "min" => da.min(d)?,
                "max" => da.max(d)?,
                "std" => da.std_ddof(d, ddof)?,
                "var" => da.var_ddof(d, ddof)?,
                "median" => da.median(d)?,
                "prod" => da.prod(d)?,
                "count" => to_f64(&da.count(d)?),
                "argmin" => to_f64(&da.argmin(&one())?),
                "argmax" => to_f64(&da.argmax(&one())?),
                "idxmin" => da.idxmin(&one())?,
                "idxmax" => da.idxmax(&one())?,
                f => panic!("unknown reduction {f}"),
            })
        }
        "quantile" => One(a().quantile(kw["q"].as_f64().unwrap(), dims_of(&kw["dims"]))?),
        "cumsum" => One(a().cumsum(&dim())?),
        "cumprod" => One(a().cumprod(&dim())?),
        "binop" => {
            let (mut x, y) = (a(), b());
            if kw["pre_isel_first"].is_object() {
                x = x.isel("time", 0..10)?;
            }
            One(match kw["func"].as_str().unwrap() {
                "add" => x.add(&y)?,
                "sub" => x.sub(&y)?,
                "mul" => x.mul(&y)?,
                "div" => x.div(&y)?,
                _ => unreachable!(),
            })
        }
        "scalar" => {
            let v = kw["value"].as_f64().unwrap();
            let x = a();
            One(match kw["func"].as_str().unwrap() {
                "add" => &x + v,
                "mul" => &x * v,
                "rsub" => v - &x,
                _ => unreachable!(),
            })
        }
        "align" => {
            let join = match kw["join"].as_str().unwrap() {
                "inner" => Join::Inner,
                "outer" => Join::Outer,
                "left" => Join::Left,
                "right" => Join::Right,
                "exact" => Join::Exact,
                _ => unreachable!(),
            };
            Many(align(&[&a(), &b()], join)?)
        }
        "broadcast" => Many(broadcast(&[&a(), &b()])?),
        "broadcast_like" => One(a().broadcast_like(&b())?),
        "groupby" => {
            let mut da = a();
            if let Some(x) = kw["pre_x"].as_array() {
                da = da.with_coord("x", x.iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>())?;
            }
            let g = da.groupby(&s("key"))?;
            One(match kw["func"].as_str().unwrap() {
                "mean" => g.mean()?,
                "sum" => g.sum()?,
                "max" => g.max()?,
                "min" => g.min()?,
                "std" => g.std()?,
                "count" => g.count()?,
                "median" => g.median()?,
                f => panic!("{f}"),
            })
        }
        "groupby_anomaly" => {
            let da = a();
            One(da.groupby(&s("key"))?.map(|g| g.sub(&g.mean("time")?))?)
        }
        "groupby_sub" => {
            let da = a();
            let clim = da.groupby(&s("key"))?.mean()?;
            One(da.groupby(&s("key"))?.sub(&clim)?)
        }
        "resample" => {
            let da = a();
            let r = da.resample(&dim(), &s("freq"))?;
            One(match kw["func"].as_str().unwrap() {
                "mean" => r.mean()?,
                "sum" => r.sum()?,
                "max" => r.max()?,
                "count" => r.count()?,
                f => panic!("{f}"),
            })
        }
        "rolling" => {
            let da = a();
            let mut r =
                da.rolling(&dim(), kw["window"].as_u64().unwrap() as usize)?.center(kw["center"].as_bool().unwrap());
            if let Some(mp) = kw["min_periods"].as_u64() {
                r = r.min_periods(mp as usize);
            }
            One(match kw["func"].as_str().unwrap() {
                "mean" => r.mean()?,
                "sum" => r.sum()?,
                "max" => r.max()?,
                "min" => r.min()?,
                "std" => r.std()?,
                "var" => r.var()?,
                "count" => r.count()?,
                "median" => r.median()?,
                f => panic!("{f}"),
            })
        }
        "rolling_construct" => One(a().rolling(&dim(), 3)?.construct(&s("window_dim"))?),
        "concat" => {
            let da = a();
            One(concat(&[&da.isel("x", vec![0, 1])?, &da.isel("x", vec![3])?], "x")?)
        }
        "concat_new" => {
            let da = a();
            One(concat(&[&da.isel("x", 0)?, &da.isel("x", 2)?], "x")?)
        }
        "concat_newdim" => {
            let da = a();
            One(concat(&[&da, &(&da * 2.0)], &dim())?)
        }
        "concat2" => {
            let da = a();
            let second = da.isel("x", vec![1])?.reindex("y", vec!["a", "b"])?;
            One(concat(&[&da.isel("x", vec![0])?, &second], "x")?)
        }
        "merge" => Ds(merge(&[&a().rename("a"), &b().rename("c")])?),
        "interp" | "interp_nearest" => {
            let da = a();
            let vals = kw["values"].as_array().unwrap();
            let method = if op == "interp" { InterpMethod::Linear } else { InterpMethod::Nearest };
            if kw["datetime"].as_bool() == Some(true) {
                let t: Vec<chrono::NaiveDateTime> =
                    vals.iter().map(|v| time::parse_datetime(v.as_str().unwrap()).unwrap()).collect();
                One(da.interp_with(&dim(), t, method)?)
            } else {
                let v: Vec<f64> = vals.iter().map(|v| v.as_f64().unwrap()).collect();
                One(da.interp_with(&dim(), v, method)?)
            }
        }
        "ds_reduce" => {
            let ds = Dataset::<f64>::from_dict(&fixtures()["inputs"][ins[0]])?;
            let d = dims_of(&kw["dims"]);
            Ds(match kw["func"].as_str().unwrap() {
                "mean" => ds.mean(d)?,
                "sum" => ds.sum(d)?,
                _ => unreachable!(),
            })
        }
        "ds_isel" => Ds(Dataset::<f64>::from_dict(&fixtures()["inputs"][ins[0]])?
            .isel(&dim(), kw["index"].as_i64().unwrap() as isize)?),
        "ds_sel" => Ds(Dataset::<f64>::from_dict(&fixtures()["inputs"][ins[0]])?.sel(&dim(), sel_of(&kw["label"]))?),
        "ds_to_dataarray" => {
            let ds = Dataset::<f64>::from_dict(&fixtures()["inputs"][ins[0]])?;
            One(ds.select_vars(vec!["a", "b"])?.to_dataarray("variable")?)
        }
        other => panic!("unknown op {other}"),
    })
}

#[test]
fn xarray_parity() {
    let cases = fixtures()["cases"].as_array().unwrap();
    let mut failures = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap().to_string();
        let res = std::panic::catch_unwind(|| {
            let out = run(case);
            if case["error"].as_bool() == Some(true) {
                assert!(out.is_err(), "{id}: expected an error");
                return;
            }
            let out = out.unwrap_or_else(|e| panic!("{id}: unexpected error: {e}"));
            let exp = &case["expected"];
            match out {
                Out::One(da) => check(&id, &da, exp),
                Out::Many(v) => {
                    let e = exp.as_array().unwrap();
                    assert_eq!(v.len(), e.len(), "{id}: count");
                    for (i, (da, e)) in v.iter().zip(e).enumerate() {
                        check(&format!("{id}[{i}]"), da, e);
                    }
                }
                Out::Ds(ds) => check_ds(&id, &ds, exp),
            }
        });
        if let Err(e) = res {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            failures.push(format!("{id}: {msg}"));
        }
    }
    if !failures.is_empty() {
        panic!("{} of {} cases failed:\n{}", failures.len(), cases.len(), failures.join("\n\n"));
    }
    eprintln!("{} xarray parity cases passed", cases.len());
}

#[test]
fn repr_parity() {
    let reprs = fixtures()["reprs"].as_object().unwrap();
    let mut failures = Vec::new();
    for (k, expected) in reprs {
        let actual = if k == "DS" {
            Dataset::<f64>::from_dict(&fixtures()["inputs"]["DS"]).unwrap().to_string()
        } else if let Some(v) = fixtures()["repr_inputs"].get(k) {
            let da = DataArray::<f64>::from_dict(v).unwrap();
            if k == "int" { da.astype::<i64>().to_string() } else { da.to_string() }
        } else {
            input(k).to_string()
        };
        if actual != expected.as_str().unwrap() {
            failures.push(format!("--- {k}: expected\n{}\n--- got\n{actual}", expected.as_str().unwrap()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

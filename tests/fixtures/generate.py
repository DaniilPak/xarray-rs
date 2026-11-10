"""Generates reference results with Python xarray for the Rust parity tests.

Run inside a throwaway venv:  python -m venv v && v/bin/pip install xarray numpy pandas
then:  v/bin/python tests/fixtures/generate.py tests/fixtures/xarray_cases.json
"""
import json
import math
import sys

import numpy as np
import pandas as pd
import xarray as xr


def enc_values(a):
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64):
        out = np.datetime_as_string(a.astype("datetime64[ns]"), unit="auto").tolist()
        return out, "datetime64[ns]"
    if np.issubdtype(a.dtype, np.floating):
        return [None if (isinstance(x, float) and math.isnan(x)) else x for x in a.ravel().tolist()], "float64"
    if np.issubdtype(a.dtype, np.integer):
        return a.ravel().tolist(), "int64"
    if a.dtype == bool:
        return a.ravel().tolist(), "bool"
    return [str(x) for x in a.ravel().tolist()], "str"


def nest(flat, shape):
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = int(np.prod(shape[1:]))
    return [nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def enc_var(v):
    flat, dtype = enc_values(v.values)
    if np.issubdtype(np.asarray(v.values).dtype, np.datetime64):
        flat = list(flat) if isinstance(flat, list) else [flat]
        flat = [None if s == "NaT" else s for s in np.ravel(flat)]
    return {"dims": list(v.dims), "data": nest(flat, v.shape), "dtype": dtype,
            "attrs": {k: (v_.item() if hasattr(v_, "item") else v_) for k, v_ in v.attrs.items()}}


def enc(obj):
    if isinstance(obj, xr.DataArray):
        d = enc_var(obj.variable)
        d["coords"] = {k: enc_var(c.variable) for k, c in obj.coords.items()}
        d["name"] = obj.name
        return d
    if isinstance(obj, xr.Dataset):
        return {"coords": {k: enc_var(c.variable) for k, c in obj.coords.items()},
                "data_vars": {k: enc_var(v.variable) for k, v in obj.data_vars.items()},
                "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in obj.attrs.items()}}
    raise TypeError(type(obj))


def main():
    rng = np.random.default_rng(42)

    A = xr.DataArray(
        np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.5], [-2.0, 8.0, np.nan], [0.5, 0.25, 9.0]]),
        dims=("x", "y"),
        coords={"x": [10, 20, 30, 40], "y": ["a", "b", "c"], "lat": ("x", [1.5, 2.5, 3.5, 4.5])},
        name="temp", attrs={"units": "K"},
    )
    B = xr.DataArray(
        np.arange(6.0).reshape(3, 2) + 1,
        dims=("x", "z"), coords={"x": [20, 30, 50], "z": [1, 2]}, name="temp",
    )
    C = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="y", coords={"y": ["c", "a", "d"]}, name="other")
    D = xr.DataArray(np.array([3.0, 1.0, 2.0, 5.0, 4.0]), dims="x", coords={"x": [5.0, 1.0, 2.5, 4.0, 3.0]})
    G = xr.DataArray(
        rng.normal(size=(6, 2)).round(3), dims=("x", "y"),
        coords={"x": np.arange(6), "y": [0, 1], "letters": ("x", list("abbcac"))},
    )
    times = pd.date_range("2000-01-01", periods=60, freq="D")
    Tv = rng.normal(size=(60, 2)).round(3) * 10
    Tv[5, 0] = np.nan
    Tv[17, 1] = np.nan
    T = xr.DataArray(Tv, dims=("time", "loc"), coords={"time": times, "loc": ["p", "q"]}, name="t2m")
    H = xr.DataArray(np.arange(20.0), dims="time", coords={"time": pd.date_range("2000-01-01 03:00", periods=20, freq="5h")})
    M = xr.DataArray(rng.normal(size=30).round(3), dims="time", coords={"time": pd.date_range("2000-01-15", periods=30, freq="MS") + pd.Timedelta(days=14)})
    R = xr.DataArray(np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, np.nan, 9.0, 10.0]), dims="t", coords={"t": np.arange(10) * 2})
    R2 = xr.DataArray(rng.normal(size=(3, 8)).round(3), dims=("s", "t"), coords={"t": np.arange(8)})
    INPUTS = {"A": A, "B": B, "C": C, "D": D, "G": G, "T": T, "H": H, "M": M, "R": R, "R2": R2}

    cases = []


    def case(id, op, inputs, expected=None, error=False, **kw):
        c = {"id": id, "op": op, "inputs": inputs, "kwargs": kw}
        if error:
            c["error"] = True
        else:
            c["expected"] = enc(expected) if not isinstance(expected, list) else [enc(e) for e in expected]
        cases.append(c)


    def safe(id, op, inputs, fn, **kw):
        try:
            res = fn()
        except Exception:
            case(id, op, inputs, error=True, **kw)
            return
        case(id, op, inputs, res, **kw)


    # ---- indexing -----------------------------------------------------------------
    for i, idx in enumerate([0, -1, [0, 2], [3, 1, 1]]):
        case(f"isel_x_{i}", "isel", ["A"], A.isel(x=idx), dim="x", index=idx)
    for i, sl in enumerate([(1, 3, 1), (None, None, 2), (None, None, -1), (3, 0, -2), (-3, None, 1)]):
        case(f"isel_slice_{i}", "isel_slice", ["A"], A.isel(x=slice(*sl)), dim="x", slice=list(sl))
    case("isel_y_scalar", "isel", ["A"], A.isel(y=1), dim="y", index=1)
    safe("isel_oob", "isel", ["A"], lambda: A.isel(x=7), dim="x", index=7)
    for i, lab in enumerate([20, 40]):
        case(f"sel_label_{i}", "sel", ["A"], A.sel(x=lab), dim="x", label=lab)
    case("sel_str", "sel", ["A"], A.sel(y="b"), dim="y", label="b")
    case("sel_list", "sel", ["A"], A.sel(y=["c", "a"]), dim="y", label=["c", "a"])
    case("sel_slice", "sel_slice", ["A"], A.sel(x=slice(15, 35)), dim="x", start=15, stop=35)
    case("sel_slice_open", "sel_slice", ["A"], A.sel(x=slice(None, 30)), dim="x", start=None, stop=30)
    case("sel_slice_str", "sel_slice", ["A"], A.sel(y=slice("b", None)), dim="y", start="b", stop=None)
    safe("sel_missing", "sel", ["A"], lambda: A.sel(x=25), dim="x", label=25)
    for m in ["nearest", "pad", "backfill"]:
        for lab in [24, 26, 36, 5]:
            safe(f"sel_{m}_{lab}", "sel_method", ["A"], lambda: A.sel(x=lab, method=m), dim="x", label=lab, method=m)
    case("sel_nearest_list", "sel_method", ["A"], A.sel(x=[11, 29, 100], method="nearest"), dim="x", label=[11, 29, 100], method="nearest")
    safe("sel_nearest_tol", "sel_method", ["A"], lambda: A.sel(x=[11, 26], method="nearest", tolerance=2), dim="x", label=[11, 26], method="nearest", tolerance=2)
    case("sel_float_unsorted", "sel", ["D"], D.sel(x=2.5), dim="x", label=2.5)
    case("sel_desc_slice", "sel_slice", ["D"], D.sortby("x", ascending=False).sel(x=slice(4.5, 2.0)), dim="x", start=4.5, stop=2.0, pre="sort_desc")
    case("sel_time_day", "sel", ["T"], T.sel(time="2000-01-10"), dim="time", label="2000-01-10")
    case("sel_time_month", "sel", ["T"], T.sel(time="2000-02"), dim="time", label="2000-02")
    case("sel_time_slice", "sel_slice", ["T"], T.sel(time=slice("2000-01-25", "2000-02-03")), dim="time", start="2000-01-25", stop="2000-02-03")
    case("sel_time_slice_month", "sel_slice", ["T"], T.sel(time=slice("2000-01", "2000-01")), dim="time", start="2000-01", stop="2000-01")
    case("sel_time_nearest", "sel_method", ["H"], H.sel(time="2000-01-02 06:00", method="nearest"), dim="time", label="2000-01-02 06:00", method="nearest")

    # ---- reshaping ----------------------------------------------------------------
    case("transpose", "transpose", ["A"], A.transpose("y", "x"), order=["y", "x"])
    case("transpose_ellipsis", "transpose", ["T"], T.transpose("loc", ...), order=["loc", "..."])
    case("squeeze", "squeeze", ["A"], A.isel(x=[1]).squeeze(), pre_isel={"x": [1]})
    case("expand_dims", "expand_dims", ["A"], A.expand_dims("band"), dim="band")
    case("rename_dim", "rename_vars", ["A"], A.rename({"x": "xx"}), old="x", new="xx")
    case("rename_coord", "rename_vars", ["A"], A.rename({"lat": "latitude"}), old="lat", new="latitude")
    case("rename_name", "rename", ["A"], A.rename("newname"), name="newname")
    case("where_gt", "where_gt", ["A"], A.where(A > 2.0), threshold=2.0)
    case("where_other", "where_gt", ["A"], A.where(A > 2.0, -1.0), threshold=2.0, other=-1.0)
    case("shift", "shift", ["A"], A.shift(x=1), dim="x", n=1)
    case("shift_neg", "shift", ["A"], A.shift(x=-2), dim="x", n=-2)
    case("roll", "roll", ["A"], A.roll(x=1, roll_coords=False), dim="x", n=1)
    case("diff", "diff", ["A"], A.diff("x"), dim="x", n=1)
    case("diff2", "diff", ["D"], D.diff("x", n=2), dim="x", n=2)
    case("sortby", "sortby", ["D"], D.sortby("x"), dim="x", ascending=True)
    case("sortby_desc", "sortby", ["D"], D.sortby("x", ascending=False), dim="x", ascending=False)
    case("reindex", "reindex", ["A"], A.reindex(x=[40, 15, 10]), dim="x", labels=[40, 15, 10])
    case("fillna", "fillna", ["A"], A.fillna(0.0), value=0.0)
    case("dropna_any", "dropna", ["A"], A.dropna("x"), dim="x", how="any")
    case("dropna_all", "dropna", ["R"], R.dropna("t", how="all"), dim="t", how="all")
    case("ffill", "ffill", ["R"], R.ffill("t"), dim="t")
    case("ffill_limit", "ffill", ["R"], R.ffill("t", limit=1), dim="t", limit=1)
    case("bfill", "bfill", ["R"], R.bfill("t"), dim="t")
    case("interpolate_na", "interpolate_na", ["R"], R.interpolate_na("t"), dim="t")
    case("isnull", "isnull", ["A"], A.isnull())
    case("clip", "clip", ["A"], A.clip(0.0, 5.0), min=0.0, max=5.0)

    # ---- reductions ---------------------------------------------------------------
    for red in ["sum", "mean", "min", "max", "std", "var", "median", "count", "prod"]:
        for dims in ["x", "y", ["x", "y"], None]:
            tag = "all" if dims is None else ("_".join(dims) if isinstance(dims, list) else dims)
            res = getattr(A, red)(dim=dims) if dims is not None else getattr(A, red)()
            case(f"reduce_{red}_{tag}", "reduce", ["A"], res, func=red, dims=dims)
        case(f"reduce_{red}_T_time", "reduce", ["T"], getattr(T, red)("time"), func=red, dims="time")
    case("std_ddof1", "reduce", ["A"], A.std("x", ddof=1), func="std", dims="x", ddof=1)
    case("var_ddof1", "reduce", ["T"], T.var("time", ddof=1), func="var", dims="time", ddof=1)
    case("sum_noskip", "reduce", ["A"], A.sum("y", skipna=False), func="sum", dims="y", skipna=False)
    case("mean_noskip", "reduce", ["A"], A.mean("x", skipna=False), func="mean", dims="x", skipna=False)
    case("sum_min_count", "reduce", ["A"], A.sum("y", min_count=3), func="sum", dims="y", min_count=3)
    for f in ["argmin", "argmax", "idxmin", "idxmax"]:
        case(f"{f}_x", "reduce", ["A"], getattr(A, f)("x"), func=f, dims="x")
    for q in [0.0, 0.25, 0.9]:
        case(f"quantile_{q}", "quantile", ["A"], A.quantile(q, dim="x"), q=q, dims="x")
    case("cumsum_y", "cumsum", ["A"], A.cumsum("y"), dim="y")
    case("cumsum_x", "cumsum", ["A"], A.cumsum("x"), dim="x")
    case("cumprod_y", "cumprod", ["A"], A.cumprod("y"), dim="y")

    # ---- arithmetic / alignment ----------------------------------------------------
    for name, fn in [("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b), ("div", lambda a, b: a / b)]:
        case(f"binop_{name}_AB", "binop", ["A", "B"], fn(A, B), func=name)
        case(f"binop_{name}_AC", "binop", ["A", "C"], fn(A, C), func=name)
    case("binop_scalar_add", "scalar", ["A"], A + 2.5, func="add", value=2.5)
    case("binop_scalar_rsub", "scalar", ["A"], 10.0 - A, func="rsub", value=10.0)
    case("binop_scalar_mul", "scalar", ["A"], A * -2.0, func="mul", value=-2.0)
    case("binop_self_T", "binop", ["T", "T"], T.isel(time=slice(0, 10)) * T, func="mul", pre_isel_first={"time": [0, 10]})
    for join in ["inner", "outer", "left", "right"]:
        case(f"align_{join}", "align", ["A", "B"], list(xr.align(A, B, join=join)), join=join)
        case(f"align_{join}_str", "align", ["A", "C"], list(xr.align(A, C, join=join)), join=join)
    safe("align_exact", "align", ["A", "B"], lambda: list(xr.align(A, B, join="exact")), join="exact")
    case("broadcast", "broadcast", ["A", "B"], list(xr.broadcast(A, B)))
    case("broadcast_like", "broadcast_like", ["C", "B"], C.broadcast_like(B))

    # ---- groupby ------------------------------------------------------------------
    for red in ["mean", "sum", "max", "min", "std", "count", "median"]:
        case(f"groupby_letters_{red}", "groupby", ["G"], getattr(G.groupby("letters"), red)(), key="letters", func=red)
    for key in ["time.month", "time.dayofweek", "time.season"]:
        case(f"groupby_{key}_mean", "groupby", ["T"], T.groupby(key).mean(), key=key, func="mean")
    case("groupby_x_dup_sum", "groupby", ["G"], G.assign_coords(x=[0, 1, 1, 2, 0, 2]).groupby("x").sum(), key="x", func="sum", pre_x=[0, 1, 1, 2, 0, 2])
    case("groupby_anomaly", "groupby_anomaly", ["T"], T.groupby("time.month").map(lambda g: g - g.mean("time")), key="time.month")
    clim = T.groupby("time.month").mean()
    case("groupby_sub_clim", "groupby_sub", ["T"], T.groupby("time.month") - clim, key="time.month")

    # ---- resample -----------------------------------------------------------------
    for freq in ["3D", "W", "MS", "ME", "2D", "10D"]:
        for red in ["mean", "sum", "max", "count"]:
            case(f"resample_T_{freq}_{red}", "resample", ["T"], getattr(T.resample(time=freq), red)(), dim="time", freq=freq, func=red)
    for freq in ["12h", "1D", "7h", "2h"]:
        for red in ["mean", "sum"]:
            case(f"resample_H_{freq}_{red}", "resample", ["H"], getattr(H.resample(time=freq), red)(), dim="time", freq=freq, func=red)
    for freq in ["QS", "QE", "YS", "YE", "2MS", "3ME"]:
        case(f"resample_M_{freq}_mean", "resample", ["M"], M.resample(time=freq).mean(), dim="time", freq=freq, func="mean")

    # ---- rolling ------------------------------------------------------------------
    for w in [1, 3, 4]:
        for center in [False, True]:
            for mp in [None, 1, 2]:
                for red in ["mean", "sum", "max", "min", "std", "count", "median", "var"]:
                    kw = dict(t=w, center=center)
                    if mp is not None:
                        kw["min_periods"] = mp
                    safe(f"rolling_{w}_{center}_{mp}_{red}", "rolling", ["R"], lambda: getattr(R.rolling(**kw), red)(), dim="t", window=w, center=center, min_periods=mp, func=red)
    case("rolling_2d", "rolling", ["R2"], R2.rolling(t=3, center=True).mean(), dim="t", window=3, center=True, min_periods=None, func="mean")
    case("rolling_construct", "rolling_construct", ["R"], R.rolling(t=3).construct("w"), dim="t", window=3, center=False, window_dim="w")

    # ---- combine ------------------------------------------------------------------
    case("concat_existing", "concat", ["A", "B"], xr.concat([A.isel(x=[0, 1]), A.isel(x=[3])], dim="x"), dim="x", pre="split_A")
    case("concat_new", "concat_new", ["A"], xr.concat([A.isel(x=0), A.isel(x=2)], dim="x"), dim="x")
    case("concat_newdim", "concat_newdim", ["A"], xr.concat([A, A * 2], dim="run"), dim="run")
    case("merge", "merge", ["A", "C"], xr.merge([A.rename("a"), C.rename("c")], join="outer"))
    case("concat_outer", "concat2", ["A", "C"], xr.concat([A.isel(x=[0]), A.isel(x=[1]).reindex(y=["a", "b"])], dim="x", join="outer"), dim="x")

    # ---- interp ---------------------------------------------------------------------
    case("interp_linear", "interp", ["A"], A.interp(x=[10.0, 15.0, 27.5, 40.0, 41.0, 5.0]), dim="x", values=[10.0, 15.0, 27.5, 40.0, 41.0, 5.0])
    case("interp_unsorted", "interp", ["D"], D.interp(x=[1.5, 2.0, 3.5, 4.75]), dim="x", values=[1.5, 2.0, 3.5, 4.75])
    case("interp_nearest", "interp_nearest", ["D"], D.interp(x=[1.5, 2.0, 3.6, 4.75], method="nearest"), dim="x", values=[1.5, 2.0, 3.6, 4.75])
    tnew = pd.date_range("2000-01-01 06:00", periods=5, freq="18h")
    case("interp_time", "interp", ["H"], H.interp(time=tnew), dim="time", values=[str(t) for t in tnew], datetime=True)

    # ---- dataset ------------------------------------------------------------------
    DS = xr.Dataset({"a": A, "b": ("x", np.array([1.0, 2.0, np.nan, 4.0])), "c": ("y", np.array([7.0, 8.0, 9.0]))}, attrs={"title": "t"})
    INPUTS["DS"] = DS
    case("ds_mean_x", "ds_reduce", ["DS"], DS.mean("x"), func="mean", dims="x")
    case("ds_sum_all", "ds_reduce", ["DS"], DS.sum(), func="sum", dims=None)
    case("ds_isel", "ds_isel", ["DS"], DS.isel(x=1), dim="x", index=1)
    case("ds_sel_list", "ds_sel", ["DS"], DS.sel(y=["c", "a"]), dim="y", label=["c", "a"])
    case("ds_to_dataarray", "ds_to_dataarray", ["DS"], DS[["a", "b"]].to_dataarray("variable"), vars=["a", "b"])

    # ---- repr -------------------------------------------------------------------------
    reprs = {
        "A": repr(A),
        "B": repr(B),
        "C": repr(C),
        "D": repr(D),
        "R": repr(R),
        "R2": repr(R2),
        "H_head": repr(H.isel(time=slice(0, 4)).assign_coords(time=H.time.values[:4].astype("datetime64[ns]"))),
        "DS": repr(DS),
        "scalar": repr(xr.DataArray(5.0)),
        "big": repr(xr.DataArray(np.arange(1000.0).reshape(10, 100), dims=("a", "b"))),
        "int": repr(xr.DataArray(np.arange(5), dims="i", coords={"i": [1, 2, 3, 4, 5]}, attrs={"long_name": "count"})),
    }
    reprs_inputs = {"H_head": enc(H.isel(time=slice(0, 4))), "scalar": enc(xr.DataArray(5.0)),
                    "big": enc(xr.DataArray(np.arange(1000.0).reshape(10, 100), dims=("a", "b"))),
                    "int": enc(xr.DataArray(np.arange(5.0), dims="i", coords={"i": [1, 2, 3, 4, 5]}, attrs={"long_name": "count"}))}

    out = {
        "xarray_version": xr.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "inputs": {k: enc(v) for k, v in INPUTS.items()},
        "cases": cases,
        "reprs": reprs,
        "repr_inputs": reprs_inputs,
    }
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=None, allow_nan=False)
    print(len(cases), "cases")


if __name__ == "__main__":
    main()

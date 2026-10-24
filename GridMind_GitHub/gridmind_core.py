# GridMind core functions (generated from the notebook; do not edit by hand)

FEATURES = []   # set by app.py from the saved model

import sys, subprocess

import os, time, warnings

from datetime import date, datetime, timedelta, timezone

import numpy as np

import pandas as pd

import requests

import lightgbm as lgb

import matplotlib.pyplot as plt

from scipy.optimize import linprog

import joblib

P_MAX_MW   = 50.0

E_MAX_MWH  = 100.0

SOC_MIN, SOC_MAX = 5.0, 95.0

RTE        = 0.88

ETA        = RTE ** 0.5

DEG_COST   = 5.0

DT         = 0.5

E_STEP     = P_MAX_MW * DT

SOC_START  = 50.0

TUNE_MODELS = True

BASE = "https://data.elexon.co.uk/bmrs/api/v1"

S = requests.Session()

def get(url, params=None, retries=3):
    err = None
    for _ in range(retries):
        try:
            r = S.get(url, params=params, timeout=90)
            if r.status_code == 429:
                time.sleep(5); continue
            r.raise_for_status()
            js = r.json()
            return js["data"] if isinstance(js, dict) and "data" in js else js
        except Exception as e:
            err = e; time.sleep(2)
    print(f"   ! request failed: {url} {params or ''} -> {err}")
    return []

def chunks(start, end, days=7):
    d = start
    while d <= end:
        e = min(d + timedelta(days=days - 1), end)
        yield d, e
        d = e + timedelta(days=1)

def iso(d):  # date -> ISO midnight UTC
    return f"{d.isoformat()}T00:00:00Z"

def fetch_prices(start, end):
    # Imbalance (system) price per half-hour -> our TARGET
    rows = []
    d = start
    while d <= end:
        rows += get(f"{BASE}/balancing/settlement/system-prices/{d.isoformat()}")
        print(f"   prices {d}", end="\r"); d += timedelta(days=1)
    print()
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df[["startTime", "systemSellPrice", "netImbalanceVolume"]].rename(
        columns={"systemSellPrice": "price", "netImbalanceVolume": "niv"})
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df.drop_duplicates("startTime", keep="last")

def fetch_fuel(start, end):
    # Generation by fuel type every 5 min -> half-hourly averages
    rows = []
    for a, b in chunks(start, end):
        rows += get(f"{BASE}/datasets/FUELINST/stream",
                    {"publishDateTimeFrom": iso(a), "publishDateTimeTo": iso(b + timedelta(days=1))})
        print(f"   fuel mix to {b}", end="\r")
    print()
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True).dt.floor("30min")
    return (df.pivot_table(index="startTime", columns="fuelType", values="generation", aggfunc="mean")
              .add_prefix("gen_").reset_index())

def fetch_mid(start, end):
    # Market Index Data: wholesale price per half-hour (volume-weighted across providers)
    rows = []
    for a, b in chunks(start, end):
        rows += get(f"{BASE}/datasets/MID", {"from": iso(a), "to": iso(b + timedelta(days=1))})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["pv"] = df["price"] * df["volume"]
    g = df.groupby("startTime")[["pv", "volume", "price"]].agg({"pv": "sum", "volume": "sum", "price": "mean"})
    g["market_price"] = np.where(g["volume"] > 0, g["pv"] / g["volume"].replace(0, np.nan), g["price"])
    return g[["market_price"]].reset_index()

def fetch_forecast(dataset, start, end, value_col):
    # Every published forecast (with publishTime) so we can use only what was known at the time
    rows = []
    for a, b in chunks(start - timedelta(days=2), end):
        rows += get(f"{BASE}/datasets/{dataset}/stream",
                    {"publishDateTimeFrom": iso(a), "publishDateTimeTo": iso(b + timedelta(days=1))})
        print(f"   {dataset} to {b}", end="\r")
    print()
    df = pd.DataFrame(rows)
    if df.empty or value_col not in df.columns: return pd.DataFrame()
    df = df[["publishTime", "startTime", value_col]].rename(columns={value_col: "value"})
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df.dropna()

def fetch_all(start, end):
    print("Fetching imbalance prices..."); prices = fetch_prices(start, end)
    print("Fetching fuel mix...");         fuel   = fetch_fuel(start, end)
    print("Fetching market index...");     mid    = fetch_mid(start, end)
    print("Fetching wind forecasts...");   windf  = fetch_forecast("WINDFOR", start, end, "generation")
    print("Fetching demand forecasts..."); demf   = fetch_forecast("NDF", start, end, "demand")
    for name, d in [("prices", prices), ("fuel", fuel), ("mid", mid), ("wind fc", windf), ("demand fc", demf)]:
        print(f"   {name:10s}: {len(d):>8,} rows" + ("   <-- EMPTY (feature will be skipped)" if d.empty else ""))
    return prices, fuel, mid, windf, demf

FC_LEAD = pd.Timedelta("60min")

DA_LEAD = pd.Timedelta("24h")

def asat(fc, times, lead=FC_LEAD, hourly=False):
    # For each target time, the latest forecast for that time published at least `lead` earlier
    if fc.empty: return pd.Series(np.nan, index=times)
    t = pd.DataFrame({"startTime": times})
    t["key"] = t["startTime"].dt.floor("60min") if hourly else t["startTime"]
    t["cutoff"] = t["startTime"] - lead
    f = fc.rename(columns={"startTime": "key"}).sort_values("publishTime")
    out = pd.merge_asof(t.sort_values("cutoff"), f, left_on="cutoff", right_on="publishTime",
                        by="key", direction="backward")
    return out.set_index("startTime")["value"].reindex(times)

def combine(prices, fuel, mid, windf, demf, extend_periods=0):
    t0 = prices["startTime"].min()
    t1 = prices["startTime"].max() + pd.Timedelta(minutes=30) * extend_periods
    grid = pd.DataFrame({"startTime": pd.date_range(t0, t1, freq="30min", tz="UTC")})
    df = grid.merge(prices, on="startTime", how="left")
    for extra in (fuel, mid):
        if len(extra): df = df.merge(extra, on="startTime", how="left")
    hourly = (not windf.empty) and (windf["startTime"].dt.minute == 0).all()
    df["wind_fc"]   = asat(windf, df["startTime"], hourly=hourly).values
    df["demand_fc"] = asat(demf, df["startTime"]).values
    df["wind_fc_da"]   = asat(windf, df["startTime"], lead=DA_LEAD, hourly=hourly).values
    df["demand_fc_da"] = asat(demf, df["startTime"], lead=DA_LEAD).values
    return df

FILL_LIMIT = 4

def clean_data(raw, verbose=True):
    df = raw.copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    n_dup = df.startTime.duplicated().sum()
    df = df.drop_duplicates("startTime", keep="last").sort_values("startTime")
    grid = pd.date_range(df.startTime.min(), df.startTime.max(), freq="30min", tz="UTC")
    n_gap = len(grid) - len(df)
    df = df.set_index("startTime").reindex(grid).rename_axis("startTime").reset_index()

    # 1. drop columns with no information (e.g. coal after 2024) or mostly missing
    num = df.select_dtypes("number").columns
    drop = [c for c in num if c not in ("price", "niv", "market_price", "wind_fc", "demand_fc", "wind_fc_da", "demand_fc_da")
            and (df[c].nunique(dropna=True) <= 1 or df[c].isna().mean() > 0.5)]
    df = df.drop(columns=drop)

    # 2. physically impossible values
    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    clip_cols = [c for c in gen_cols if not c.startswith("gen_INT") and c != "gen_PS"]   # imports/pumping can be negative
    n_clip = int(sum((df[c] < 0).sum() for c in clip_cols))
    for c in clip_cols: df[c] = df[c].clip(lower=0)
    n_bad_fc = 0
    if "wind_fc" in df:   n_bad_fc += int((df.wind_fc < 0).sum());   df.loc[df.wind_fc < 0, "wind_fc"] = np.nan
    if "wind_fc_da" in df: n_bad_fc += int((df.wind_fc_da < 0).sum()); df.loc[df.wind_fc_da < 0, "wind_fc_da"] = np.nan
    if "demand_fc_da" in df: n_bad_fc += int((df.demand_fc_da <= 0).sum()); df.loc[df.demand_fc_da <= 0, "demand_fc_da"] = np.nan
    if "demand_fc" in df: n_bad_fc += int((df.demand_fc <= 0).sum()); df.loc[df.demand_fc <= 0, "demand_fc"] = np.nan

    # 3. fill short feature gaps from the PAST only (never the target)
    feat_cols = gen_cols + [c for c in ("market_price", "wind_fc", "demand_fc", "wind_fc_da", "demand_fc_da") if c in df]
    before = int(df[feat_cols].isna().sum().sum())
    df[feat_cols] = df[feat_cols].ffill(limit=FILL_LIMIT)
    n_fill = before - int(df[feat_cols].isna().sum().sum())

    if verbose:
        print(f"Removed {n_dup} duplicate rows; added {n_gap} empty rows for missing half-hours (kept as gaps)")
        print(f"Dropped uninformative columns: {drop or 'none'}")
        print(f"Clipped {n_clip} negative generation values to 0; removed {n_bad_fc} impossible forecasts")
        print(f"Forward-filled {n_fill:,} short feature gaps (<= {FILL_LIMIT * 30} min). Prices left untouched.")
        print(f"Half-hours with a usable price: {df.price.notna().sum():,} of {len(df):,}")
    return df

def col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

def build_features(raw):
    df = raw.sort_values("startTime").reset_index(drop=True).copy()
    local = df["startTime"].dt.tz_convert("Europe/London")
    gen = [c for c in df.columns if c.startswith("gen_")]
    df["gas"]     = col(df, "gen_CCGT").fillna(0) + col(df, "gen_OCGT").fillna(0)
    df["imports"] = df[[c for c in gen if c.startswith("gen_INT")]].sum(axis=1, min_count=1) if gen else np.nan
    df["total_gen"] = df[gen].sum(axis=1, min_count=1) if gen else np.nan
    df["wind_act"]  = col(df, "gen_WIND")

    f = pd.DataFrame({"startTime": df["startTime"], "price": col(df, "price")})
    p = df["price"]
    for k in (2, 3, 4, 6, 48, 336):                      # 48 = same time yesterday, 336 = last week
        f[f"price_lag{k}"] = p.shift(k)
    known = p.shift(2)
    f["price_mean_6"]  = known.rolling(6, min_periods=3).mean()
    f["price_std_6"]   = known.rolling(6, min_periods=3).std()
    f["price_max_48"]  = known.rolling(48, min_periods=12).max()
    f["ref_price"]     = known.rolling(336, min_periods=48).median()   # 7-day typical price
    # typical price at this time of day over the last 7 days (all already published)
    f["tod_profile"]   = pd.concat([p.shift(48 * k) for k in range(1, 8)], axis=1).median(axis=1)
    # smoother 28-day daily shape used for planning ahead (still only published prices)
    f["profile28"] = (pd.concat([p.shift(48 * k) for k in range(1, 29)], axis=1).median(axis=1)
                        .rolling(5, center=True, min_periods=1).median())
    f["price_vs_ref"]  = f["price_lag2"] - f["ref_price"]
    f["niv_lag2"]      = col(df, "niv").shift(2)
    f["niv_lag3"]      = col(df, "niv").shift(3)
    f["mkt_lag1"]      = col(df, "market_price").shift(1)
    f["mkt_lag2"]      = col(df, "market_price").shift(2)
    f["mkt_minus_ref"] = f["mkt_lag1"] - f["ref_price"]
    f["wind_act_lag1"] = df["wind_act"].shift(1)
    f["wind_fc"]       = col(df, "wind_fc")
    f["wind_err_lag1"] = df["wind_act"].shift(1) - col(df, "wind_fc").shift(1)   # +: more wind than expected
    f["wind_ramp_fc"]  = col(df, "wind_fc") - df["wind_act"].shift(1)            # expected change in wind
    f["demand_fc"]     = col(df, "demand_fc")
    f["residual_fc"]   = col(df, "demand_fc") - col(df, "wind_fc")                      # demand left for gas etc.
    f["gas_lag1"]      = df["gas"].shift(1)
    f["gas_share_lag1"] = (df["gas"] / df["total_gen"]).shift(1)
    f["nuclear_lag1"]  = col(df, "gen_NUCLEAR").shift(1)
    f["imports_lag1"]  = df["imports"].shift(1)
    f["gas_ramp_lag1"] = df["gas"].shift(1) - df["gas"].shift(3)
    hr = local.dt.hour + local.dt.minute / 60
    f["hour"] = hr
    f["hour_sin"], f["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    f["weekday"] = local.dt.weekday
    f["weekend"] = (local.dt.weekday >= 5).astype(int)
    f["month"]   = local.dt.month
    f["is_peak"] = ((hr >= 16) & (hr < 19.5)).astype(int)
    # latest market price: the anchor for change models. Live, the newest prices may not be published yet,
    # so fall back to the most recent known value (forward-fill uses only the past)
    f["base"] = f["mkt_lag1"].fillna(f["price_lag2"]).ffill()
    # day-ahead forecasts (published >= 24h before each half-hour): inputs to the multi-horizon model
    f["wind_fc_da"]   = col(df, "wind_fc_da")
    f["demand_fc_da"] = col(df, "demand_fc_da")
    f["resid_da"]     = f["demand_fc_da"] - f["wind_fc_da"]
    f["local_date"] = local.dt.date
    return f

def step(soc, action, price):
    # action in [-1, 1]: + = discharge, - = charge (fraction of max power)
    if action > 1e-9:
        out = min(action * E_STEP, (soc - SOC_MIN) * ETA)
        return soc - out / ETA, 0.0, out, out * price - DEG_COST * out
    if action < -1e-9:
        inn = min(-action * E_STEP, (SOC_MAX - soc) / ETA)
        return soc + inn * ETA, inn, 0.0, -inn * price
    return soc, 0.0, 0.0, 0.0

def lp_schedule(soc0, p_buy, p_sell, end_soc=SOC_START, end_equal=False):
    # Best charge/discharge plan over the horizon for given buy/sell price paths (linear program)
    n = len(p_buy)
    c = np.concatenate([p_buy, -p_sell + DEG_COST, np.zeros(n)])
    A = np.zeros((n, 3 * n)); b = np.zeros(n)
    idx = np.arange(n)
    A[idx, idx] = -ETA; A[idx, n + idx] = 1 / ETA; A[idx, 2 * n + idx] = 1
    A[idx[1:], 2 * n + idx[1:] - 1] = -1; b[0] = soc0
    bounds = [(0, E_STEP)] * (2 * n) + [(SOC_MIN, SOC_MAX)] * n
    term = np.zeros((1, 3 * n)); term[0, -1] = 1
    if end_equal:
        res = linprog(c, A_eq=np.vstack([A, term]), b_eq=np.append(b, end_soc), bounds=bounds, method="highs")
    else:
        res = linprog(c, A_eq=A, b_eq=b, A_ub=-term, b_ub=[-min(end_soc, SOC_MAX)], bounds=bounds, method="highs")
    if not res.success: return None, None
    return res.x[:n], res.x[n:2 * n]

HORIZON = 44

def run_mpc(df, p1, kappa=0.0, soc=SOC_START, future=None):
    # Rolling optimisation: every half-hour, plan the next 22h, do only the first step, then re-plan.
    #   step 1 price  = p1 (our forecast for the coming half-hour)
    #   later steps   = typical price for that time of day (28-day median, already published)
    #   kappa         = trading hurdle (GBP/MWh) so the plan ignores spreads smaller than forecast noise
    #   future        = optional (rows x 43) forecast for steps 2..44, replacing the typical shape
    actual, prof = df["price"].values, df["profile28"].values
    log = []
    for i in range(len(df)):
        fut = prof[i + 1:i + HORIZON]
        if len(fut) < HORIZON - 1:
            fut = np.concatenate([fut, np.full(HORIZON - 1 - len(fut), prof[i])])
        if future is not None:          # optional forecast for the later steps (multi-horizon model)
            fz = future[i]
            fut = np.where(np.isfinite(fz), fz, fut)
        path = np.concatenate([[p1[i]], fut])
        ch, dis = lp_schedule(soc, path + kappa, path - kappa)
        a = 0.0 if ch is None else (dis[0] - ch[0]) / E_STEP
        soc, e_in, e_out, pnl = step(soc, a, actual[i])
        log.append((df["startTime"].iloc[i], a, soc, e_in, e_out, pnl))
    return pd.DataFrame(log, columns=["startTime", "action", "soc", "e_in", "e_out", "pnl"])

def blend(df, forecast, alpha):
    # Trust the forecast in proportion to its proven skill: alpha=1 uses it fully, 0 ignores it
    prof = df["profile28"].values
    return prof + alpha * (np.asarray(forecast) - prof)

def run_fixed(df, soc=SOC_START):
    # What many simple sites do: charge overnight, discharge at the evening peak
    log = []
    for t, h, p in zip(df["startTime"], df["hour"], df["price"]):
        a = -1.0 if 1 <= h < 4.5 else 1.0 if 16.5 <= h < 19.5 else 0.0
        soc, e_in, e_out, pnl = step(soc, a, p)
        log.append((t, a, soc, e_in, e_out, pnl))
    return pd.DataFrame(log, columns=["startTime", "action", "soc", "e_in", "e_out", "pnl"])

def perfect_hindsight(df, soc0=SOC_START):
    # Upper bound: the best possible schedule for each day if every price were known in advance
    total, out = 0.0, []
    for _, d in df.groupby("local_date", sort=True):
        p = d["price"].values
        ch, dis = lp_schedule(soc0, p, p, end_soc=soc0, end_equal=True)
        if ch is None: continue
        pnl = p * dis - p * ch - DEG_COST * dis
        total += pnl.sum()
        out.append(pd.DataFrame({"startTime": d["startTime"].values, "pnl": pnl}))
    return total, pd.concat(out) if out else pd.DataFrame(columns=["startTime", "pnl"])

CHUNK, GAP = 672, 2

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def fit_cand(kind, df):
    # the two simple baselines need no training
    if kind in ("persist", "market"): return None
    # LightGBM with "l1" loss = predicts the median, so rare price spikes don't dominate training
    if kind == "lgb_price":                      # learns the price directly
        return lgb.LGBMRegressor(objective="l1", **LGB_PARAMS).fit(df[FEATURES], df.price)
    if kind == "lgb_change":                     # learns the CHANGE from the latest market price (adapts to price level)
        return lgb.LGBMRegressor(objective="l1", **LGB_PARAMS).fit(df[FEATURES], df.price - df.base)
    raise ValueError(kind)

def predict_cand(kind, obj, df):
    if kind == "persist":    return df.price_lag2.values                     # "price stays as last published"
    if kind == "market":     return df.base.values                           # "price = latest market price"
    if kind == "lgb_price":  return obj.predict(df[FEATURES])
    if kind == "lgb_change": return df.base.values + obj.predict(df[FEATURES])
    raise ValueError(kind)

def walk_forward(kind):
    out = np.full(n, np.nan)
    for s in range(WF_START, n, CHUNK):
        e_ = min(s + CHUNK, n)
        obj = fit_cand(kind, data.iloc[:s - GAP])
        out[s:e_] = predict_cand(kind, obj, data.iloc[s:e_])
    return out

def band(q50, window=1440):
    # 80% range from the last 30 days of REAL out-of-sample errors (only errors already known)
    res = pd.Series(data.price.values - q50).shift(GAP)
    lo = res.rolling(window, min_periods=200).quantile(0.10).values
    hi = res.rolling(window, min_periods=200).quantile(0.90).values
    return pd.DataFrame({"q10": q50 + lo, "q50": q50, "q90": q50 + hi}, index=data.index)

HZ = HORIZON - 1

NOW_COLS = ["price_lag2", "price_mean_6", "price_std_6", "price_max_48", "ref_price", "niv_lag2",
            "wind_err_lag1", "wind_act_lag1", "gas_lag1", "mkt_lag1"]

AT_COLS = ["hour", "weekday", "weekend", "month", "is_peak", "profile28", "tod_profile", "price_lag48",
           "price_lag336", "wind_fc_da", "demand_fc_da", "resid_da"]

MH_KS_TRAIN = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 38, 43])

MH_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def arrays(f):
    return {c: pd.to_numeric(f[c], errors="coerce").values.astype(float) for c in f.columns
            if c not in ("startTime", "local_date")}

def mh_frame(FA, js, ks):
    # one row per (decision half-hour j, steps ahead k); target = price at j+k minus its typical value
    N = len(FA["price"])
    J = np.repeat(np.asarray(js), len(ks)); K = np.tile(np.asarray(ks), len(js))
    valid = (J + K) < N
    T = np.minimum(J + K, N - 1)
    nan = np.full(len(J), np.nan)
    X = {f"now_{c}": FA.get(c, nan)[J] if c in FA else nan for c in NOW_COLS}
    X.update({f"at_{c}": FA[c][T] if c in FA else nan for c in AT_COLS})
    X["k"] = K
    X["dev_now"] = FA["base"][J] - FA["profile28"][J]
    X["wind_da_change"] = (FA["wind_fc_da"][T] if "wind_fc_da" in FA else nan) - FA["wind_act_lag1"][J]
    y = FA["price"][T] - FA["profile28"][T]
    return pd.DataFrame(X), y, T, valid

def fit_mh(FA, train_js, cutoff_pos, params=None):
    X, y, T, valid = mh_frame(FA, train_js, MH_KS_TRAIN)
    ok = valid & (T <= cutoff_pos - 3) & np.isfinite(y)       # only targets already published at cutoff
    return lgb.LGBMRegressor(objective="l1", **(params or MH_PARAMS)).fit(X[ok], y[ok])

def predict_mh(model, FA, js):
    ks = np.arange(1, HZ + 1)
    X, _, T, valid = mh_frame(FA, js, ks)
    p = model.predict(X) + FA["profile28"][T]
    p[~valid] = np.nan
    return p.reshape(len(js), HZ)

def future_path(lo, hi, beta):
    # plan prices for steps 2..44: typical shape moved towards the ML forecast by `beta` (0 = old behaviour)
    prof, mh = PROF_FUT[lo:hi], MH[lo:hi]
    return np.where(np.isfinite(mh) & np.isfinite(prof), prof + beta * (mh - prof), prof)

def mae_by_k(pred, lo, hi):
    a = ACT_FUT[lo:hi]; m = np.isfinite(a) & np.isfinite(pred[lo:hi])
    return np.array([np.mean(np.abs(a[:, k] - pred[lo:hi, k])[m[:, k]]) for k in range(HZ)])

NICE = {
    "price_lag2": "Latest published imbalance price", "price_lag3": "Imbalance price 1.5h ago",
    "price_lag4": "Imbalance price 2h ago", "price_lag6": "Imbalance price 3h ago",
    "price_lag48": "Price same time yesterday", "price_lag336": "Price same time last week",
    "price_mean_6": "Average price, last 3 hours", "price_std_6": "Price volatility, last 3 hours",
    "price_max_48": "Highest price, last 24 hours", "ref_price": "Typical price this week",
    "tod_profile": "Typical price at this time of day", "profile28": "Typical price at this time of day",
    "price_vs_ref": "Latest price vs typical", "niv_lag2": "Grid short/long (imbalance volume)",
    "niv_lag3": "Grid short/long 1.5h ago", "mkt_lag1": "Latest market index price",
    "mkt_lag2": "Market index price 1h ago", "mkt_minus_ref": "Market price vs typical",
    "wind_act_lag1": "Wind output now", "wind_fc": "Wind forecast", "wind_err_lag1": "Wind surprise (actual - forecast)",
    "wind_ramp_fc": "Expected change in wind", "demand_fc": "Demand forecast",
    "residual_fc": "Demand left after wind (forecast)", "gas_lag1": "Gas generation now",
    "gas_share_lag1": "Gas share of generation", "nuclear_lag1": "Nuclear output",
    "imports_lag1": "Imports via interconnectors", "gas_ramp_lag1": "Change in gas output (last hour)",
    "hour": "Time of day", "hour_sin": "Time of day", "hour_cos": "Time of day", "weekday": "Day of week",
    "weekend": "Weekend", "month": "Month", "is_peak": "Evening peak", "wind_fc_da": "Day-ahead wind forecast",
    "demand_fc_da": "Day-ahead demand forecast", "resid_da": "Day-ahead demand left after wind",
    "k": "Hours ahead", "dev_now": "Today's price vs typical", "wind_da_change": "Forecast wind change vs now",
}

def nice(c):
    if c.startswith("now_"): return NICE.get(c[4:], c[4:])
    if c.startswith("at_"):  return NICE.get(c[3:], c[3:])
    return NICE.get(c, c)

def contrib_table(model, X):
    # exact split of each prediction into per-input contributions (+ the model's average value)
    c = model.predict(X, pred_contrib=True)
    return pd.DataFrame(c[:, :-1], columns=list(X.columns), index=X.index), c[:, -1]

def grouped(con):
    # add up inputs that mean the same thing in plain English (e.g. the three time-of-day encodings)
    return con.T.groupby([nice(c) for c in con.columns], sort=False).sum().T

def top_drivers(model, X, n=3, scale=1.0):
    con, _ = contrib_table(model, X)
    g = grouped(con.iloc[[0]] * scale).iloc[0]
    return [(name, float(g[name])) for name in g.abs().sort_values(ascending=False).index[:n]]

EXPLAINABLE = ("lgb_price", "lgb_change")

def explain_next(kind, obj, X, explain_kind=None, explain_obj=None):
    k, m = (kind, obj) if kind in EXPLAINABLE else (explain_kind, explain_obj)
    if m is None: return [], ""
    note = "" if k == kind else " (read from the LightGBM model)"
    return top_drivers(m, X[FEATURES]), note

def drivers_text(drivers):
    return [f"{name}: {'+' if v >= 0 else '-'}GBP{abs(v):,.0f}/MWh" for name, v in drivers]

def live_plan(f, now, kind, obj, res_q, mh_model, alpha, beta, kappa, soc, explain_kind=None, explain_obj=None):
    # Shared by this notebook and the website: forecast -> 22-hour plan -> decision for the coming half-hour
    pos = np.flatnonzero((f["startTime"] > now).values)
    if len(pos) == 0: return None
    j = int(pos[0]); X = f.iloc[[j]]
    FA = arrays(f)
    typical = float(np.nanmedian(FA["price"][-336:]))
    idx = np.minimum(j + np.arange(HORIZON), len(f) - 1)
    prof = FA["profile28"][idx]; prof = np.where(np.isfinite(prof), prof, typical)
    q50 = float(predict_cand(kind, obj, X)[0])
    if not np.isfinite(q50): q50 = float(prof[0])           # newest inputs not published yet
    fut = prof[1:]
    if mh_model is not None and beta > 0:
        mh = predict_mh(mh_model, FA, np.array([j]))[0]
        fut = np.where(np.isfinite(mh), prof[1:] + beta * (mh - prof[1:]), prof[1:])
    path = np.concatenate([[prof[0] + alpha * (q50 - prof[0])], fut])
    ch, dis = lp_schedule(soc, path + kappa, path - kappa)
    a = 0.0 if ch is None else (dis[0] - ch[0]) / E_STEP
    plan_mw = np.zeros(len(path)) if ch is None else (dis - ch) / DT
    act = "DISCHARGE (sell)" if a > 1e-6 else "CHARGE (buy)" if a < -1e-6 else "HOLD"
    sell_p = path[plan_mw > 0.5]; buy_p = path[plan_mw < -0.5]
    if a < -1e-6:
        why = (f"Expected price now (GBP{path[0]:,.0f}) is low compared with the later half-hours the plan sells in "
               f"(about GBP{sell_p.mean():,.0f})." if len(sell_p) else f"Expected price now (GBP{path[0]:,.0f}) is among the lowest of the next 22 hours.")
    elif a > 1e-6:
        why = (f"Expected price now (GBP{path[0]:,.0f}) is high compared with the half-hours the plan recharges in "
               f"(about GBP{buy_p.mean():,.0f})." if len(buy_p) else f"Expected price now (GBP{path[0]:,.0f}) is among the highest of the next 22 hours.")
    else:
        why = "Better opportunities are expected later, or the price gap is too small to cover losses and wear."
    try:
        drivers, dnote = explain_next(kind, obj, X, explain_kind, explain_obj)
    except Exception:
        drivers, dnote = [], ""
    return dict(drivers=drivers, drivers_note=dnote,
                time=X["startTime"].iloc[0].tz_convert("Europe/London"), q10=q50 + res_q[0], q50=q50,
                q90=q50 + res_q[1], typical=float(prof[0]), a=a, mw=abs(a) * P_MAX_MW, action=act, why=why,
                soc=soc, plan_times=pd.DatetimeIndex(f["startTime"].iloc[idx]), plan_price=path, plan_mw=plan_mw,
                recent=f.dropna(subset=["price"]).tail(96)[["startTime", "price"]])

def recommend(live_raw, now, current_soc=50.0, verbose=True):
    f = build_features(clean_data(live_raw, verbose=False))
    rec = live_plan(f, now, BEST_KIND, FINAL_OBJ, RES_Q, FINAL_MH, ALPHA, BETA, KAPPA, current_soc,
                    EXPLAIN_KIND, EXPLAIN_OBJ)
    if rec is None:
        if verbose: print("No upcoming period found")
        return None
    if verbose:
        print(f"\\nPeriod starting {rec['time']:%a %d %b %H:%M} (UK time)")
        print(f"Predicted imbalance price: GBP{rec['q50']:,.0f}/MWh  (80% range GBP{rec['q10']:,.0f} to GBP{rec['q90']:,.0f})")
        print(f"Typical price at this time of day: GBP{rec['typical']:,.0f}/MWh")
        print(f"Decision: {rec['action']} at {rec['mw']:.0f} MW   (battery at {current_soc:.0f} MWh)")
        print(f"Why: {rec['why']}")
        if rec["drivers"]:
            print(f"Main drivers of the forecast{rec['drivers_note']}: " + "; ".join(drivers_text(rec["drivers"])))
    return rec

def live_raw_data(max_age_s=900):
    if _cache["raw"] is None or time.time() - _cache["t"] > max_age_s:
        now = datetime.now(timezone.utc)
        prices, fuel, mid, windf, demf = fetch_all((now - timedelta(days=31)).date(), now.date())
        if prices.empty: raise RuntimeError("No live price data returned")
        _cache.update(t=time.time(), raw=combine(prices, fuel, mid, windf, demf, extend_periods=52))
    return _cache["raw"]

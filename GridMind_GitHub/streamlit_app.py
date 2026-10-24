# GridMind: decides every 30 minutes when a grid battery should charge or discharge,
# using real GB electricity market data from Elexon.  Run locally with:  streamlit run streamlit_app.py
#
# Tabs:   Live decision  -> what the battery should do in the next half-hour, and why
#         Replay a day   -> what GridMind did on any past test day (works offline)
#         Results        -> backtest, where the value comes from, is it luck, robustness
#         Explainability -> what drives the forecasts
#         How it works   -> the method in plain English
import json
import os
from datetime import datetime, timedelta, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import gridmind_core as core   # the notebook's own functions: data download, cleaning, features, models, optimiser

st.set_page_config(page_title="GridMind", page_icon="⚡", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, BLUE, CORAL, PURPLE, GRAY, AMBER = "#1D9E75", "#378ADD", "#D85A30", "#534AB7", "#888780", "#BA7517"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
UK = "Europe/London"


# ------------------------------------------------------------------ load saved files once (cached across visitors)
@st.cache_resource
def load_all():
    def js(name):
        p = os.path.join(HERE, name)
        return json.load(open(p)) if os.path.exists(p) else {}
    bundle = joblib.load(os.path.join(HERE, "gridmind_model.joblib"))   # trained models + settings chosen on validation
    rp_path = os.path.join(HERE, "gridmind_replay.csv")
    rp = pd.read_csv(rp_path, parse_dates=["startTime"]) if os.path.exists(rp_path) else pd.DataFrame()
    if not rp.empty:
        rp["day"] = rp.startTime.dt.tz_convert(UK).dt.date
    return bundle, js("gridmind_results.json"), js("gridmind_snapshot.json"), rp

B, R, SNAP, RP = load_all()
core.FEATURES = B["features"]


# ------------------------------------------------------------------ live decision
@st.cache_data(ttl=900, show_spinner=False)     # download at most every 15 minutes, shared by all visitors
def live_raw():
    now = datetime.now(timezone.utc)
    prices, fuel, mid, windf, demf = core.fetch_all((now - timedelta(days=31)).date(), now.date())
    if prices.empty:
        raise RuntimeError("Elexon returned no price data. Please try again in a minute.")
    return core.combine(prices, fuel, mid, windf, demf, extend_periods=52)

def recommend(raw, now, soc):
    f = core.build_features(core.clean_data(raw, verbose=False))
    return core.live_plan(f, now, B["kind"], B["model"], B["res_q"], B.get("mh_model"), B["alpha"],
                          B.get("beta", 0.0), B["kappa"], soc, B.get("explain_kind"), B.get("explain_model"))

def show_decision(rec, soc, note):
    icon = {"DISCHARGE (sell)": "🟢", "CHARGE (buy)": "🔵", "HOLD": "⚪"}[rec["action"]]
    st.subheader(f"{icon} {rec['action']} at {rec['mw']:.0f} MW")
    st.caption(f"Half-hour starting {rec['time'].tz_convert(UK):%a %d %b, %H:%M} (UK time) · battery at {soc:.0f} of 100 MWh")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted imbalance price", f"£{rec['q50']:,.0f}/MWh")
    c2.metric("80% range", f"£{rec['q10']:,.0f} – £{rec['q90']:,.0f}")
    c3.metric("Typical at this time of day", f"£{rec['typical']:,.0f}/MWh")
    st.markdown(f"**Why:** {rec['why']}")
    if rec.get("drivers"):
        st.markdown(f"**Main drivers of the forecast{rec.get('drivers_note', '')}:**\n" + "\n".join(
            f"- {'🔺' if v >= 0 else '🔻'} {n}: {'+' if v >= 0 else '−'}£{abs(v):,.0f}/MWh" for n, v in rec["drivers"]))
    st.pyplot(live_figure(rec))
    st.caption(note)

def live_figure(rec):
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    r = rec["recent"]
    ax[0].plot(r.startTime.dt.tz_convert(UK), r.price, color=CORAL, lw=1.8, label="Real imbalance price")
    ax[0].errorbar([rec["time"].tz_convert(UK)], [rec["q50"]], yerr=[[rec["q50"] - rec["q10"]], [rec["q90"] - rec["q50"]]],
                   fmt="o", color=PURPLE, capsize=6, label="Our forecast (80% range)")
    ax[0].set_title("Last 48 hours + next half-hour"); ax[0].set_ylabel("£/MWh")
    ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=30)
    pt = rec["plan_times"].tz_convert(UK)
    ax[1].bar(pt, rec["plan_mw"], width=0.018, color=np.where(rec["plan_mw"] > 0, GREEN, BLUE))
    ax[1].axhline(0, color="black", lw=0.6); ax[1].set_ylabel("Planned MW (+ sell, − buy)")
    tw = ax[1].twinx(); tw.plot(pt, rec["plan_price"], color=GRAY, lw=1.2); tw.set_ylabel("Expected £/MWh")
    tw.spines["right"].set_visible(True)
    ax[1].set_title("22-hour plan (re-planned every half-hour)"); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig

def snapshot_rec():
    rec = dict(SNAP)
    rec["time"] = pd.Timestamp(SNAP["time"])
    rec["plan_times"] = pd.DatetimeIndex(pd.to_datetime(SNAP["plan_times"], utc=True))
    rec["plan_mw"], rec["plan_price"] = np.array(SNAP["plan_mw"]), np.array(SNAP["plan_price"])
    rec["recent"] = pd.DataFrame({"startTime": pd.to_datetime(SNAP["recent_times"], utc=True), "price": SNAP["recent_price"]})
    return rec


# ------------------------------------------------------------------ replay a past day (no internet needed)
def replay_figure(d):
    t = d.startTime.dt.tz_convert(UK)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].fill_between(t, d.q10, d.q90, color=PURPLE, alpha=0.15, label="Forecast 80% range")
    ax[0].plot(t, d.q50, color=PURPLE, lw=1.2, label="Forecast")
    ax[0].plot(t, d.price, color=CORAL, lw=2, label="Actual price")
    ax[0].set_ylabel("£/MWh"); ax[0].legend(fontsize=8, loc="upper left")
    mw = d.gm_action * core.P_MAX_MW
    ax[1].bar(t, mw, width=0.018, color=np.where(mw > 0, GREEN, BLUE), label="GridMind (+ sell, − buy)")
    ax[1].step(t, d.fixed_action * core.P_MAX_MW, where="mid", color=GRAY, lw=1, label="Fixed schedule")
    ax[1].set_ylabel("MW"); ax[1].legend(fontsize=8, loc="upper left")
    tw = ax[1].twinx(); tw.plot(t, d.gm_soc, color="black", lw=1, ls=":"); tw.set_ylabel("Battery MWh")
    tw.spines["right"].set_visible(True); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ result charts
def cumulative_figure():
    cum = R.get("cumulative")
    if not cum: return None
    fig, ax = plt.subplots(figsize=(12, 4))
    dates = pd.to_datetime(cum["dates"]); colours = [GRAY, AMBER, PURPLE, GREEN, BLUE]
    for i, (name, vals) in enumerate(cum["series"].items()):
        hind = "hindsight" in name.lower()
        ax.plot(dates, np.array(vals) / 1000, label=name, color="black" if hind else colours[i % 5],
                lw=1 if hind else 2, ls="--" if hind else "-")
    ax.set_ylabel("Cumulative profit (£ thousands)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("50 MW / 100 MWh battery on unseen test months")
    fig.tight_layout(); return fig

def ablation_figure():
    ab = R.get("ablation")
    if not ab: return None
    fig, ax = plt.subplots(figsize=(12, 3.8)); prev = 0.0
    for k, row in enumerate(ab):
        tot = row["profit"]; inc = tot if k == 0 else tot - prev
        ax.bar(k, inc / 1000, bottom=0 if k == 0 else prev / 1000, color=GRAY if k == 0 else (GREEN if inc >= 0 else CORAL))
        ax.text(k, max(tot, prev) / 1000, f"{'' if k == 0 else ('+' if inc >= 0 else '')}{inc / 1000:,.0f}k",
                ha="center", va="bottom", fontsize=9)
        prev = tot
    ax.set_xticks(range(len(ab))); ax.set_xticklabels([r["step"] for r in ab], fontsize=8)
    ax.set_ylabel("Profit, £ thousands"); ax.set_title("Where the value comes from (test months)")
    fig.tight_layout(); return fig

def importance_figure():
    imp, imp2 = R.get("importance"), R.get("importance_mh")
    if not imp: return None
    fig, ax = plt.subplots(1, 2 if imp2 else 1, figsize=(13, 4.2), squeeze=False); ax = ax[0]
    ax[0].barh(imp["names"], imp["values"], color=GREEN)
    ax[0].set_title("Next half-hour forecast:\naverage impact, £/MWh", fontsize=10)
    if imp2:
        ax[1].barh(imp2["names"], imp2["values"], color=PURPLE)
        ax[1].set_title("22-hour plan forecast:\naverage impact, £/MWh", fontsize=10)
    for a in ax: a.tick_params(labelsize=8)
    fig.tight_layout(); return fig

def shape_figure():
    p = R.get("daily_shape")
    if not p: return None
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(p["hour"], p["weekday"], color=GREEN, lw=2, label="Weekday")
    ax.plot(p["hour"], p["weekend"], color=AMBER, lw=2, label="Weekend")
    ax.set_xlabel("Hour (UK time)"); ax.set_ylabel("Median £/MWh"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("When electricity is cheap and expensive (real data)")
    fig.tight_layout(); return fig

def show_table(title, key):
    rows = R.get(key)
    if rows:
        st.markdown(f"#### {title}")
        df = pd.DataFrame(rows)
        st.dataframe(df.astype(object).where(df.ne(""), None), hide_index=True)   # blank cells shown as empty


# ------------------------------------------------------------------ page
meta = R.get("meta", {})
st.title("⚡ GridMind")
st.markdown("**Decides every 30 minutes when a grid battery should charge or discharge, "
            "from live GB electricity market data, and explains why.**")
tab_live, tab_replay, tab_results, tab_explain, tab_how = st.tabs(
    ["Live decision", "Replay a day", "Results", "Explainability", "How it works"])

with tab_live:
    c1, c2 = st.columns([2, 1])
    soc = c1.slider("Battery charge now (MWh of 100)", 5, 95, 50)
    fresh = c2.button("Get a fresh decision from live data", type="primary")
    if fresh:
        try:
            with st.spinner("Downloading the latest Elexon data (up to a minute the first time)..."):
                raw = live_raw()
            rec = recommend(raw, datetime.now(timezone.utc), float(soc))
            if rec is None:
                st.warning("No upcoming half-hour in the latest data yet. Try again shortly.")
            else:
                show_decision(rec, soc, "Live from Elexon data, refreshed every 15 minutes.")
        except Exception as e:
            st.error(f"Live data unavailable right now: {e}. Try the Replay a day tab meanwhile.")
    elif SNAP:
        created = pd.Timestamp(SNAP["created"]).tz_convert(UK)
        show_decision(snapshot_rec(), SNAP.get("soc", 50),
                      f"Saved decision from {created:%a %d %b %H:%M}. Press the button for a fresh one from live data.")
    else:
        st.info("Press the button to get a live decision.")

with tab_replay:
    if RP.empty:
        st.info("Replay data not available.")
    else:
        days = sorted(RP.day.unique())
        default = RP.groupby("day").price.std().idxmax()          # the most volatile day: the best story
        day = st.selectbox("Pick a day from the test months (never used to train or tune anything)", days,
                           index=days.index(default), format_func=lambda d: f"{pd.Timestamp(d):%a %d %b %Y}")
        d = RP[RP.day == day]
        gm, fx = d.gm_pnl.sum(), d.fixed_pnl.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("GridMind profit", f"£{gm:,.0f}", f"{gm - fx:+,.0f} vs fixed")
        c2.metric("Fixed schedule profit", f"£{fx:,.0f}")
        c3.metric("Half-hours traded", f"{(d.gm_action.abs() > 0.01).sum()} of {len(d)}")
        st.pyplot(replay_figure(d))
        best = d.loc[d.gm_pnl.idxmax()]
        st.caption(f"Prices ranged from £{d.price.min():,.0f} to £{d.price.max():,.0f}/MWh. Best moment: "
                   f"{pd.Timestamp(best.startTime).tz_convert(UK):%H:%M}, "
                   f"{'sold' if best.gm_action > 0 else 'bought'} at £{best.price:,.0f}/MWh, earning £{best.gm_pnl:,.0f}.")

with tab_results:
    st.markdown(f"### Backtest on {meta.get('test_days', '?')} unseen days")
    st.markdown("\n".join(f"- {line}" for line in R.get("pitch", [])) or "_Run the notebook's export cell._")
    for fig in (cumulative_figure(), ablation_figure()):
        if fig is not None: st.pyplot(fig)
    show_table("Battery strategies compared", "backtest")
    show_table("Is the advantage real? (block bootstrap over days)", "significance")
    show_table("Robustness to battery assumptions (profit on test months, £)", "robustness")
    show_table("Forecasting models compared (validation months, chosen by profit)", "comparison")

with tab_explain:
    st.markdown("### What drives GridMind's forecasts\nEach forecast is split exactly into contributions from its inputs "
                "(TreeSHAP, built into LightGBM). Bars show the average size of each input's effect.")
    for fig in (importance_figure(), shape_figure()):
        if fig is not None: st.pyplot(fig)

with tab_how:
    st.markdown(f"""
### The problem
Britain's grid must balance supply and demand every second. When wind under-delivers or demand surprises, the grid
runs **short** and the half-hourly **imbalance price** spikes; with surplus power it falls, sometimes below zero.
Britain has over 8 GW of grid batteries that must decide **every 30 minutes** whether to charge, discharge or hold,
**before** that price is known. Many smaller sites still follow fixed timetables.

### How GridMind works
1. **Data:** real, public Elexon data: imbalance prices, generation by fuel, market prices, wind and demand forecasts.
   Only information published before each decision is used.
2. **Forecast:** {meta.get('best', 'a model')} predicts the next half-hour's price with an 80% range, and a multi-horizon
   LightGBM model predicts the following 22 hours from the official day-ahead wind and demand forecasts.
   Models are retrained regularly and chosen on validation months by the profit they produce.
3. **Decide:** an optimiser plans the next 22 hours for a 50 MW / 100 MWh battery (88% efficiency, wear cost,
   charge limits), acts on the first half-hour, then re-plans with fresh data.
4. **Prove:** tested on months never used for any choice, against a fixed schedule, a market-price forecast with the
   same optimiser, and perfect hindsight; plus an ablation, a significance test and robustness checks.
5. **Explain:** every forecast is split into plain-English drivers (TreeSHAP), shown with each decision.

### Honest limitations
Trades are settled at the imbalance price; a real battery also trades in other markets and needs an aggregator or
market licence. The battery is a price taker (50 MW vs ~30,000 MW national demand).

*Data: Elexon Insights Solution (BMRS), used under the BMRS Data Licence.*
""")

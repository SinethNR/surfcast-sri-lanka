"""
SurfCast Sri Lanka - Ensemble Learning Dashboard (Streamlit + Plotly)
Front end for the models trained in SurfCast_SriLanka.ipynb (saved with joblib in models/).

Run:  streamlit run app.py

AI-use declaration: parts of this dashboard were generated with Claude (Anthropic) and
reviewed/run by the student.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)

import features as F

BASE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: os.path.join(BASE, "results", f)
SPOT_COLORS = {"Arugam Bay": "#E07A1F", "Hikkaduwa": "#1F77B4"}
MODEL_ORDER = ["Decision Tree", "Bagged Trees", "Random Forest", "AdaBoost", "Gradient Boosting",
               "XGBoost", "LightGBM", "Stacking", "Soft Voting"]
FAMILY = {"Persistence (baseline)": "Baseline", "Decision Tree": "Baseline", "Bagged Trees": "Bagging",
          "Random Forest": "Bagging", "AdaBoost": "Boosting", "Gradient Boosting": "Boosting",
          "XGBoost": "Boosting", "LightGBM": "Boosting", "Stacking": "Stacking", "Soft Voting": "Voting"}
FAM_COLORS = {"Baseline": "#9aa5b1", "Bagging": "#2a9d8f", "Boosting": "#e76f51", "Stacking": "#6a4c93",
              "Voting": "#e9c46a"}

st.set_page_config(page_title="SurfCast Sri Lanka", page_icon="🏄", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background: #f4f8fb; border: 1px solid #e3ecf3; border-radius: 10px; padding: 10px 14px;}
.hero {background: linear-gradient(120deg, #0b3d5c 0%, #1f77b4 55%, #3fb0c9 100%); color: white;
       padding: 22px 28px; border-radius: 14px; margin-bottom: 14px;}
.hero h1 {color: white; margin: 0; font-size: 2.0rem;} .hero p {margin: 6px 0 0 0; opacity: .92;}
.small {font-size: .85rem; color: #5b6b79;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_data
def load_csv(name, dates=("time",)):
    df = pd.read_csv(R(name))
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data
def load_meta():
    return json.load(open(R("meta.json")))


@st.cache_resource
def load_models():
    meta = load_meta()
    out = {}
    for name, fn in meta["model_files"].items():
        p = os.path.join(BASE, "models", fn)
        if os.path.exists(p):
            out[name] = joblib.load(p)
    reg = joblib.load(os.path.join(BASE, "models", "hybrid_rf_mlp.pkl"))
    return out, reg


if not os.path.exists(R("meta.json")):
    st.error("Results not found. Run the notebook first (it saves models/ and results/).")
    st.stop()

meta = load_meta()
FEATS = meta["features"]
metrics = load_csv("metrics.csv")
preds = load_csv("test_predictions.csv")
daily = load_csv("data_daily.csv")

with st.sidebar:
    st.markdown("## 🏄 SurfCast Sri Lanka")
    st.caption("Ensemble Learning Dashboard: Bagging · Boosting · Stacking · Hybrids")
    st.markdown(f"**Data:** Open-Meteo marine + weather, {meta['date_min']} → {meta['date_max']}")
    st.markdown(f"**Test period:** {meta['test_start']} onwards ({meta['n_test']:,} spot-days)")
    st.markdown(f"**Best model (F1):** {meta['best_model']}")
    st.divider()
    st.markdown("<span class='small'>🤖 <b>AI-use declaration:</b> parts of this dashboard and the analysis "
                "code were generated with Claude (Anthropic). Student: <i>Dheeraka</i>.</span>",
                unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🏄 SurfCast Sri Lanka</h1>
<p>Will tomorrow be a good surf day at <b>Arugam Bay</b> or <b>Hikkaduwa</b>? Ensemble models trained on about 4 years (Oct 2021 to 2025) of
Sri Lankan wave and weather data.</p></div>""", unsafe_allow_html=True)

tabs = st.tabs(["🌊 Overview", "🔎 Data explorer", "📊 Model comparison", "🌲 Inside Random Forest",
                "🎯 Predict", "🔗 Hybrid models", "ℹ️ About"])

# ================================================================== OVERVIEW
with tabs[0]:
    best = meta["best_model"]
    mrow = metrics.set_index("Model")
    c = st.columns(5)
    c[0].metric("Spot-days of data", f"{len(daily):,}")
    c[1].metric("Good days: Arugam Bay", f"{daily.loc[daily.spot == 'Arugam Bay', 'good_surf'].mean():.0%}")
    c[2].metric("Good days: Hikkaduwa", f"{daily.loc[daily.spot == 'Hikkaduwa', 'good_surf'].mean():.0%}")
    c[3].metric(f"Best F1 ({best})", f"{mrow.loc[best, 'F1']:.3f}",
                f"{mrow.loc[best, 'F1'] - mrow.loc['Persistence (baseline)', 'F1']:+.3f} vs persistence")
    c[4].metric(f"ROC-AUC ({best})", f"{mrow.loc[best, 'ROC-AUC']:.3f}")

    left, right = st.columns([1, 1.35])
    with left:
        spots = pd.DataFrame([{"spot": s, "lat": v["lat"], "lon": v["lon"], "coast": v["coast"],
                               "good %": round(100 * daily.loc[daily.spot == s, "good_surf"].mean(), 1)}
                              for s, v in F.SPOTS.items()])
        fig = px.scatter_map(spots, lat="lat", lon="lon", color="spot", size=[22, 22], hover_name="spot",
                             hover_data={"coast": True, "good %": True, "lat": False, "lon": False},
                             color_discrete_map=SPOT_COLORS, zoom=6.2, center={"lat": 7.6, "lon": 80.8},
                             map_style="open-street-map", height=430)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=0.02, x=0.02))
        st.plotly_chart(fig, width="stretch")
    with right:
        mon = daily.assign(month=daily.time.dt.month).groupby(["spot", "month"])["good_surf"].mean().reset_index()
        mon["good %"] = mon["good_surf"] * 100
        fig = px.line(mon, x="month", y="good %", color="spot", markers=True, color_discrete_map=SPOT_COLORS,
                      title="Share of good surf days by month: the monsoon 'see-saw'", height=430)
        fig.update_xaxes(tickvals=list(range(1, 13)),
                         ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        st.plotly_chart(fig, width="stretch")
    st.info("**How a 'good surf day' is defined:** swell between "
            f"{F.RULES['swell_min_m']} and {F.RULES['swell_max_m']} m, swell period ≥ {F.RULES['period_min_s']} s, "
            "swell coming from the break's swell window, and wind lighter than "
            f"{F.RULES['light_wind_kmh']:.0f} km/h (or ≤ {F.RULES['offshore_wind_kmh']:.0f} km/h if offshore). "
            "The models predict this label for **tomorrow** using only today's information.")

# ================================================================== DATA EXPLORER
with tabs[1]:
    num_cols = [c for c in daily.columns if c not in ("time", "spot", "good_surf", "weather_code")
                and pd.api.types.is_numeric_dtype(daily[c])]
    a, b, c_ = st.columns([1, 1.3, 1.6])
    spot_sel = a.multiselect("Spot", list(F.SPOTS), default=list(F.SPOTS))
    var = b.selectbox("Variable", num_cols, index=num_cols.index("swell_wave_height_max"))
    dr = c_.slider("Date range", min_value=daily.time.min().date(), max_value=daily.time.max().date(),
                   value=(pd.Timestamp("2023-01-01").date(), daily.time.max().date()))
    d = daily[daily.spot.isin(spot_sel) & daily.time.between(pd.Timestamp(dr[0]), pd.Timestamp(dr[1]))]
    smooth = st.checkbox("Show 7-day rolling mean", value=True)
    plot_d = d.copy()
    if smooth:
        plot_d[var] = plot_d.groupby("spot")[var].transform(lambda s: s.rolling(7, min_periods=1).mean())
    fig = px.line(plot_d, x="time", y=var, color="spot", color_discrete_map=SPOT_COLORS, height=360,
                  title=f"{var} over time")
    st.plotly_chart(fig, width="stretch")
    l, r = st.columns(2)
    fig = px.box(d.assign(month=d.time.dt.month), x="month", y=var, color="spot", color_discrete_map=SPOT_COLORS,
                 title="Monthly distribution", height=360)
    l.plotly_chart(fig, width="stretch")
    fig = px.histogram(d, x=var, color=d["good_surf"].map({1: "good surf day", 0: "not good"}), barmode="overlay",
                       nbins=50, opacity=0.6, title="Distribution split by label", height=360,
                       color_discrete_map={"good surf day": "#2a9d8f", "not good": "#c9ccd1"})
    r.plotly_chart(fig, width="stretch")
    st.dataframe(d.head(500), width="stretch", height=260)
    st.download_button("⬇️ Download filtered data (CSV)", d.to_csv(index=False).encode(), "surfcast_filtered.csv")

# ================================================================== MODEL COMPARISON
with tabs[2]:
    st.subheader("Test-set performance (most recent full year, 2025)")
    show = metrics.copy()
    show.insert(1, "Family", show["Model"].map(FAMILY))
    st.dataframe(show.drop(columns=["Train accuracy", "Train F1"]).style.format(
        {c: "{:.3f}" for c in show.columns if c not in ("Model", "Family")}).highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], color="#c7f0d0"),
        width="stretch", hide_index=True)

    st.markdown("#### 🎚️ Live decision threshold")
    st.caption("Models output a probability. Days above the threshold are called 'good'. Lower threshold → catch more good "
               "days (recall ↑) but more false alarms (precision ↓).")
    thr = st.slider("Threshold", 0.05, 0.95, 0.50, 0.05)
    y = preds["actual"].values
    live = []
    for m in [c for c in preds.columns if c not in ("time", "spot", "actual")]:
        yh = (preds[m].values >= thr).astype(int)
        live.append({"Model": m, "Family": FAMILY.get(m.replace(" (tuned)", ""), "Tuned"),
                     "Accuracy": accuracy_score(y, yh), "Precision": precision_score(y, yh, zero_division=0),
                     "Recall": recall_score(y, yh), "F1": f1_score(y, yh)})
    live = pd.DataFrame(live)
    l, r = st.columns([1.5, 1])
    long = live.melt(id_vars=["Model", "Family"], value_vars=["Precision", "Recall", "F1"])
    fig = px.bar(long, x="Model", y="value", color="variable", barmode="group", height=420,
                 color_discrete_sequence=["#264653", "#2a9d8f", "#e9c46a"],
                 title=f"Precision / Recall / F1 at threshold {thr:.2f}")
    fig.update_yaxes(range=[0, 1]); fig.update_xaxes(tickangle=-35)
    l.plotly_chart(fig, width="stretch")
    with r:
        msel = st.selectbox("Confusion matrix for", live["Model"].tolist(),
                            index=live["Model"].tolist().index(meta["best_model"]))
        cm = confusion_matrix(y, (preds[msel].values >= thr).astype(int))
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", height=360,
                        x=["pred: not good", "pred: good"], y=["actual: not good", "actual: good"])
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(live.set_index("Model").loc[[msel]].round(3), width="stretch")

    l, r = st.columns(2)
    fig = go.Figure()
    for m in [c for c in preds.columns if c not in ("time", "spot", "actual") and not c.startswith("Persistence")
              and "(tuned)" not in c]:
        fpr, tpr, _ = roc_curve(y, preds[m])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=m))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420)
    l.plotly_chart(fig, width="stretch")
    gap = metrics[metrics.Model != "Persistence (baseline)"][["Model", "Train accuracy", "Accuracy"]]
    gap = gap.rename(columns={"Accuracy": "Test accuracy"}).melt(id_vars="Model")
    fig = px.bar(gap, x="Model", y="value", color="variable", barmode="group", height=420,
                 color_discrete_sequence=["#c9ccd1", "#2a9d8f"], title="Bias-variance: train vs test accuracy")
    fig.update_yaxes(range=[0.5, 1.01]); fig.update_xaxes(tickangle=-35)
    r.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    dtd = load_csv("dt_depth.csv")
    fig = px.line(dtd, x="max_depth", y=["train_acc", "test_acc"], markers=True, height=360,
                  title="Single tree: deeper = lower bias but higher variance")
    l.plotly_chart(fig, width="stretch")
    fig = px.bar(metrics, x="Model", y="Train time (s)", log_y=True, height=360, title="Training time (log scale)",
                 color=metrics["Model"].map(FAMILY), color_discrete_map=FAM_COLORS)
    fig.update_xaxes(tickangle=-35)
    r.plotly_chart(fig, width="stretch")
    st.markdown("**Stacking: blocked vs shuffled K-fold** (shuffling time-ordered data leaks neighbouring days "
                "into the meta-model's training folds)")
    st.dataframe(load_csv("stacking_cv.csv").round(4), width="stretch", hide_index=True)

# ================================================================== INSIDE RF
with tabs[3]:
    l, r = st.columns(2)
    oob = load_csv("oob_curve.csv")
    fig = px.line(oob, x="n_trees", y=["oob_error", "test_error"], markers=True, height=380,
                  title="Out-of-Bag error vs number of trees")
    fig.update_yaxes(title="error rate")
    l.plotly_chart(fig, width="stretch")
    mc = load_csv("max_features.csv")
    p = len(FEATS)
    fig = px.line(mc, x="m", y=["oob_error", "test_error"], markers=True, height=380,
                  title=f"Error vs m = max_features (p = {p})")
    fig.add_vline(x=np.sqrt(p), line_dash="dot", line_color="green", annotation_text="√p")
    fig.add_vline(x=p, line_dash="dot", line_color="red", annotation_text="m = p (bagging)")
    fig.update_yaxes(title="error rate")
    r.plotly_chart(fig, width="stretch")
    imp = load_csv("importance.csv")
    k = st.slider("Top features to show", 5, min(30, len(imp)), 15)
    l, r = st.columns(2)
    t1 = imp.nlargest(k, "RF_gini").iloc[::-1]
    fig = px.bar(t1, x="RF_gini", y="feature", orientation="h", height=30 * k + 120,
                 title="Random Forest: Gini importance", color_discrete_sequence=["#2a9d8f"])
    l.plotly_chart(fig, width="stretch")
    t2 = imp.nlargest(k, "XGB_gain").iloc[::-1]
    fig = px.bar(t2, x="XGB_gain", y="feature", orientation="h", height=30 * k + 120,
                 title="XGBoost: gain importance (normalised)", color_discrete_sequence=["#e76f51"])
    r.plotly_chart(fig, width="stretch")
    st.markdown("#### Hyperparameter tuning (RandomizedSearchCV, TimeSeriesSplit, scoring = F1)")
    st.dataframe(load_csv("tuning.csv").round(4), width="stretch", hide_index=True)

# ================================================================== PREDICT
with tabs[4]:
    clfs, reg = load_models()
    tf = load_csv("test_features.csv")
    a, b = st.columns([1, 2])
    spot = a.radio("Surf spot", list(F.SPOTS), horizontal=True)
    sub = tf[tf.spot == spot].reset_index(drop=True)
    dates = sub["time"].dt.date.tolist()
    day = b.select_slider("Pick a real day from the test period ('today')", options=dates, value=dates[len(dates) // 2])
    row = sub[sub["time"].dt.date == day].iloc[0]

    st.markdown(f"##### Conditions on {day} at {spot}. Move the sliders for a *what-if* scenario")
    snap = lambda v, step, lo, hi: float(min(hi, max(lo, round(round(float(v) / step) * step, 2))))
    defaults = {"swell_wave_height_max": snap(row["swell_wave_height_max"], 0.05, 0, 5),
                "swell_wave_period_max": snap(row["swell_wave_period_max"], 0.5, 3, 22),
                "swell_wave_direction_dominant": int(round(row["swell_wave_direction_dominant"])) % 360,
                "wind_speed_10m_max": snap(row["wind_speed_10m_max"], 0.5, 0, 70),
                "wind_direction_10m_dominant": int(round(row["wind_direction_10m_dominant"])) % 360}
    s = st.columns(5)
    k = f"{spot}_{day}"   # new day -> sliders reset to that day's real values
    sh = s[0].slider("Swell height (m)", 0.0, 5.0, defaults["swell_wave_height_max"], 0.05, key="sh" + k)
    sp = s[1].slider("Swell period (s)", 3.0, 22.0, defaults["swell_wave_period_max"], 0.5, key="sp" + k)
    sd = s[2].slider("Swell direction (° from)", 0, 359, defaults["swell_wave_direction_dominant"], key="sd" + k)
    ws = s[3].slider("Max wind (km/h)", 0.0, 70.0, defaults["wind_speed_10m_max"], 0.5, key="ws" + k)
    wd = s[4].slider("Wind direction (° from)", 0, 359, defaults["wind_direction_10m_dominant"], key="wd" + k)
    changed = {"swell_wave_height_max": sh, "swell_wave_period_max": sp, "swell_wave_direction_dominant": sd,
               "wind_speed_10m_max": ws, "wind_direction_10m_dominant": wd}
    is_whatif = any(abs(float(defaults[c]) - float(v)) > 1e-9 for c, v in changed.items())
    new = F.apply_what_if(row, **changed) if is_whatif else row    # untouched sliders -> exact real day
    X = pd.DataFrame([new[FEATS].astype(float).values], columns=FEATS)
    probs = pd.DataFrame([{"Model": n, "P(good surf tomorrow)": float(m.predict_proba(X)[0, 1]),
                           "Family": FAMILY.get(n.replace(" (tuned)", ""), "Tuned")} for n, m in clfs.items()])
    probs = probs.sort_values("P(good surf tomorrow)")

    l, r = st.columns([1.6, 1])
    fig = px.bar(probs, x="P(good surf tomorrow)", y="Model", orientation="h", color="Family",
                 color_discrete_map={**FAM_COLORS, "Tuned": "#577590"}, height=440, range_x=[0, 1],
                 title="Every saved model's probability that tomorrow is a good surf day", text_auto=".2f")
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
    l.plotly_chart(fig, width="stretch")
    with r:
        avg = probs["P(good surf tomorrow)"].mean()
        verdict = "🟢 GOOD surf expected" if avg >= 0.5 else "🔴 Not a good surf day"
        st.metric("Ensemble of all models (mean probability)", f"{avg:.0%}", verdict, delta_color="off")
        h_pred = reg["w_rf"] * reg["rf"].predict(X)[0] + (1 - reg["w_rf"]) * reg["mlp"].predict(X)[0]
        st.metric("Tomorrow's swell height (RF + MLP hybrid)", f"{h_pred:.2f} m")
        st.metric("Good surf TODAY by the rules", "yes ✅" if int(new["good_surf"]) else "no ❌")
        if is_whatif:
            st.warning("What-if scenario: sliders changed, so there is no real outcome to compare with.")
        else:
            act = int(row["good_surf_tomorrow"])
            st.success(f"**Actual outcome tomorrow:** {'GOOD surf day ✅' if act else 'not a good surf day ❌'}  \n"
                       f"Actual swell tomorrow: {row['swell_h_tomorrow']:.2f} m")
        off = F.SPOTS[spot]["offshore_from"]
        st.caption(f"Tip: at {spot} the wind is offshore when it blows from about {off}°. "
                   f"Swell window: {F.SPOTS[spot]['swell_window'][0]}–{F.SPOTS[spot]['swell_window'][1]}°.")

# ================================================================== HYBRIDS
with tabs[5]:
    st.subheader("Hybrid 1: RF + SARIMA (monthly mean swell height, one month ahead)")
    h1m, h1p = load_csv("hybrid_sarima_metrics.csv"), load_csv("hybrid_sarima_preds.csv")
    l, r = st.columns([1, 1.8])
    l.dataframe(h1m.round(4), width="stretch", hide_index=True)
    sp_ = r.radio("Spot ", list(F.SPOTS), horizontal=True, key="h1spot")
    d = h1p[h1p.spot == sp_].melt(id_vars=["time", "spot"], var_name="series", value_name="swell (m)")
    fig = px.line(d, x="time", y="swell (m)", color="series", markers=True, height=380,
                  color_discrete_map={"actual": "black", "SARIMA": "#e9c46a", "RF only": "#8d99ae",
                                      "RF + SARIMA (hybrid)": "#e76f51"})
    r.plotly_chart(fig, width="stretch")
    st.caption("SARIMA models the 12-month monsoon cycle; the Random Forest learns SARIMA's residual errors from lagged "
               "residuals, last month's wind and the season. Hybrid = SARIMA + RF correction.")
    st.divider()
    st.subheader("Hybrid 2: RF + ANN (MLP) regression of tomorrow's swell height")
    h2m, h2p = load_csv("hybrid_mlp_metrics.csv"), load_csv("hybrid_mlp_preds.csv")
    l, r = st.columns([1, 1.8])
    l.dataframe(h2m.round(4), width="stretch", hide_index=True)
    with r:
        sp2 = st.radio("Spot  ", list(F.SPOTS), horizontal=True, key="h2spot")
        n = st.slider("Days to show (most recent)", 30, 365, 120, 15)
        cols = [c for c in h2p.columns if c not in ("time", "spot")]
        pick = st.multiselect("Series", cols, default=[c for c in cols if c in ("actual", "Random Forest", "MLP (ANN)")]
                              + [c for c in cols if c.startswith("RF + MLP weighted")])
        d = h2p[h2p.spot == sp2].tail(n).melt(id_vars=["time", "spot"], value_vars=pick, var_name="series",
                                              value_name="swell (m)")
        fig = px.line(d, x="time", y="swell (m)", color="series", height=380)
        st.plotly_chart(fig, width="stretch")

# ================================================================== ABOUT
with tabs[6]:
    st.subheader("Methods")
    st.markdown("""
| | **Bagging** | **Boosting** | **Stacking** |
|---|---|---|---|
| Approach | Many models in **parallel** on bootstrap samples | Models trained **sequentially**, each fixing previous errors | Different models + a **meta-model** on their predictions |
| Base models | Same type (deep trees) | Same type (shallow trees / stumps) | Different types (RF, XGB, LGBM, LogReg) |
| Goal | Reduce **variance** | Reduce **bias** | Combine strengths |
| Combination | Vote / average | Weighted sum | Learned by meta-model |
| Here | Bagged Trees, Random Forest | AdaBoost, Gradient Boosting, XGBoost, LightGBM | StackingClassifier, Soft Voting |
""")
    st.markdown(f"""
**Data:** [Open-Meteo](https://open-meteo.com) Marine API (waves/swell) + Historical Weather API (ERA5), daily,
{meta['date_min']} → {meta['date_max']}, Arugam Bay and Hikkaduwa, Sri Lanka. Free, no login (CC BY 4.0).

**Validation:** time-based split, trained before {meta['test_start']} and tested on the most recent full year, 2025. Features use only
information known at the end of today (no leakage). Circular variables use sin/cos encoding. Tuning uses `TimeSeriesSplit`.
Random seed = 42.

**Hybrids:** RF + SARIMA (monthly swell, residual correction) and RF + MLP (daily swell-height regression, blend
weight chosen on the last training year).

**Backend:** models trained in `SurfCast_SriLanka.ipynb`, saved with joblib (`models/*.pkl`) and loaded here.
""")
    st.subheader("🤖 AI-use declaration")
    st.info("Parts of the code, dashboard and documentation for this project were generated with **Claude (Anthropic)**. "
            "I ran all the code, checked the outputs and understand the methods. All numbers shown come from running "
            "the notebook. Student: **Dheeraka**, Index: **[YOUR INDEX]**.")

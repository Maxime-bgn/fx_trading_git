"""
modules/market_analysis.py
---------------------------
Onglet "Stress Test & Marche" — branche sur risk_outputs/*.csv.
"""
import os, logging
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)
RISK_DIR   = "risk_outputs"
DAILY_CSV  = os.path.join(RISK_DIR, "risk_report_daily.csv")
SUMM_CSV   = os.path.join(RISK_DIR, "risk_summary.csv")
STRESS_CSV = os.path.join(RISK_DIR, "risk_stress_tests.csv")


def _clean_layout(fig, height=280, **kwargs):
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(color="#495057", gridcolor="#e9ecef"),
        yaxis=dict(color="#495057", gridcolor="#e9ecef"),
    )
    defaults.update(kwargs)
    fig.update_layout(**defaults)


@st.cache_data(ttl=300)
def _load_risk():
    daily = pd.read_csv(DAILY_CSV, sep=";", decimal=",", index_col=0, parse_dates=True) if os.path.exists(DAILY_CSV) else None
    summary = pd.read_csv(SUMM_CSV, sep=";", decimal=",", index_col=0) if os.path.exists(SUMM_CSV) else None
    stress = pd.read_csv(STRESS_CSV, sep=";", decimal=",", index_col=0) if os.path.exists(STRESS_CSV) else None
    return daily, summary, stress


@st.cache_data(ttl=300)
def _load_market():
    if not os.path.exists("dataset_ml_ready.csv"):
        return None
    return pd.read_csv("dataset_ml_ready.csv", sep=";", decimal=",", index_col=0, parse_dates=True)


def _val(summary, key, default=np.nan):
    try:
        return float(summary.loc[key, "value"])
    except Exception:
        return default


# ─── Sections ────────────────────────────────────────────────────────────────

def _render_kpis(daily, summary, risk_ok):
    c1, c2, c3, c4 = st.columns(4)
    if risk_ok:
        var_last = float(daily["VaR_hist"].dropna().iloc[-1]) if "VaR_hist" in daily.columns else np.nan
        c1.metric("Daily VaR (95%)", f"{var_last:.2%}" if not np.isnan(var_last) else "N/A")
        c2.metric("Max Drawdown", f"{_val(summary, 'max_drawdown'):.2%}")
        c3.metric("VaR Breach Rate", f"{_val(summary, 'VaR_backtest_breach_rate'):.1%}")
        c4.metric("Jours arret circuit", str(int(_val(summary, "stop_trading_days", 0))))
    else:
        st.warning("Donnees de risque non disponibles. Lancez run_pipeline.py.")
        for col in [c1, c2, c3, c4]:
            col.metric("-", "N/A")


def _render_drawdown_and_exposure(daily):
    if "drawdown" not in daily.columns:
        return
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Drawdown du portefeuille")
        fig = go.Figure(go.Scatter(
            x=daily.index, y=daily["drawdown"] * 100,
            fill="tozeroy", fillcolor="rgba(220,53,69,0.12)",
            line=dict(color="#dc3545", width=1.5),
        ))
        _clean_layout(fig, height=220, yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Drawdown (%)"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Multiplicateur d'exposition")
        fig = go.Figure()
        if "exposure_final" in daily.columns:
            fig.add_trace(go.Scatter(x=daily.index, y=daily["exposure_final"],
                                     line=dict(color="#0052cc", width=1.5), name="Exposition finale"))
        if "exposure_dd" in daily.columns:
            fig.add_trace(go.Scatter(x=daily.index, y=daily["exposure_dd"],
                                     line=dict(color="#6c757d", width=1, dash="dot"), name="DD seul"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="Circuit breaker -2%")
        _clean_layout(fig, height=220,
                      legend=dict(orientation="h", y=-0.3, font_size=11),
                      yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Exposition", tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)


def _render_var_es(daily):
    if "VaR_hist" not in daily.columns:
        return
    st.divider()
    st.markdown("#### VaR et ES roulants (252 jours)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["VaR_hist"] * 100,
                              line=dict(color="#dc3545", width=1.5), name="VaR 95%"))
    if "ES_hist" in daily.columns:
        fig.add_trace(go.Scatter(x=daily.index, y=daily["ES_hist"] * 100,
                                  line=dict(color="#ef4444", width=1, dash="dot"), name="ES 95%"))
    if "VaR_breach" in daily.columns:
        b_idx = daily[daily["VaR_breach"] == 1.0].index
        if len(b_idx):
            fig.add_trace(go.Scatter(x=b_idx, y=daily.loc[b_idx, "VaR_hist"] * 100,
                                      mode="markers", marker=dict(color="#dc3545", size=6, symbol="x"),
                                      name="Breach"))
    _clean_layout(fig, height=250, hovermode="x unified",
                  legend=dict(orientation="h", y=-0.3, font_size=11),
                  yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Perte (%)"))
    st.plotly_chart(fig, use_container_width=True)


def _render_stress_and_corr(daily, stress, risk_ok):
    if stress is None:
        return
    st.divider()
    col_st, col_corr = st.columns(2)
    with col_st:
        st.markdown("#### Stress tests")
        fig = px.bar(stress.reset_index(), x="index", y="stress_loss",
                     color="stress_loss", color_continuous_scale="Reds",
                     labels={"index": "Scenario", "stress_loss": "Perte estimee"})
        _clean_layout(fig, height=280, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_corr:
        st.markdown("#### Stress de correlation roulant")
        if risk_ok and "corr_abs_mean" in daily.columns:
            fig = go.Figure(go.Scatter(
                x=daily.index, y=daily["corr_abs_mean"],
                line=dict(color="#0052cc", width=1.5), name="Corr abs moy",
            ))
            fig.add_hline(y=0.35, line_dash="dash", line_color="#f59e0b", annotation_text="Seuil stress")
            _clean_layout(fig, height=280,
                          yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Correlation abs moy"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Donnees de correlation non disponibles.")


def _render_regime_map(df_market):
    if df_market is None:
        return
    st.divider()
    st.markdown("#### Regime map - Hurst vs Volatilite")
    latest = df_market.groupby("Pair").last().reset_index()
    size_col = "Vol_Regime" if "Vol_Regime" in latest.columns else "Vol_20"
    latest[size_col] = latest[size_col].clip(lower=0.01).fillna(0.5)
    fig = px.scatter(latest, x="Vol_20", y="Hurst", text="Pair", size=size_col,
                     color="Hurst", color_continuous_scale="Blues",
                     labels={"Vol_20": "Volatilite 20j", "Hurst": "Hurst"})
    fig.add_hrect(y0=0.55, y1=1.0, fillcolor="#28a745", opacity=0.05,
                  annotation_text="Trend -> Momentum / TSMOM")
    fig.add_hrect(y0=0.0, y1=0.45, fillcolor="#dc3545", opacity=0.05,
                  annotation_text="Range -> Mean-Reversion")
    fig.update_traces(textposition="top center")
    _clean_layout(fig, height=420, font_color="#495057", coloraxis_showscale=False)
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef")
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef")
    st.plotly_chart(fig, use_container_width=True)


def _render_alerts(summary, risk_ok, df_market):
    """Alertes systeme."""
    st.divider()
    st.markdown("#### Quant feed - alertes systeme")
    alerts = []

    if risk_ok:
        corr_avg = _val(summary, "corr_abs_mean_avg")
        breach_rate = _val(summary, "VaR_backtest_breach_rate")
        stop_days = int(_val(summary, "stop_trading_days", 0))
        if not np.isnan(corr_avg) and corr_avg > 0.6:
            alerts.append("CRITICAL: Correlation moyenne > 0.60 - risque contagion systemique.")
        if not np.isnan(breach_rate) and breach_rate > 0.07:
            alerts.append(f"WARNING: Breach rate VaR = {breach_rate:.1%} > 7%.")
        if stop_days > 5:
            alerts.append(f"INFO: {stop_days} jours d'arret circuit breaker.")

    if df_market is not None:
        latest = df_market.groupby("Pair").last()
        for pair, row in latest.iterrows():
            h = row.get("Hurst", np.nan)
            vr = row.get("Vol_Regime", np.nan)
            if not np.isnan(h) and h > 0.65:
                alerts.append(f"SIGNAL: Persistance forte sur {pair} (H={h:.2f}) - Momentum/TSMOM.")
            if not np.isnan(h) and h < 0.38:
                alerts.append(f"SIGNAL: Regime range sur {pair} (H={h:.2f}) - Mean-Reversion.")
            if not np.isnan(vr) and vr > 1.5:
                alerts.append(f"WARNING: Vol elevee sur {pair} (Vol_Regime={vr:.2f}x).")

    if not alerts:
        alerts.append("SYSTEM NORMAL: Aucune anomalie detectee.")
    for alert in alerts[-5:]:
        st.code(f"> {datetime.now().strftime('%H:%M:%S')} | {alert}")


# ─── Point d'entree ──────────────────────────────────────────────────────────

def show():
    st.markdown("<h2 style='color:#0052cc;margin-bottom:0'>SYSTEMIC RISK & REGIME RADAR</h2>",
                unsafe_allow_html=True)
    st.caption("Analyse des risques extremes et de la cohesion globale du portefeuille.")

    daily, summary, stress = _load_risk()
    df_market = _load_market()
    risk_ok = daily is not None and summary is not None

    _render_kpis(daily, summary, risk_ok)
    st.divider()

    if risk_ok:
        _render_drawdown_and_exposure(daily)
        _render_var_es(daily)

    _render_stress_and_corr(daily, stress, risk_ok)
    _render_regime_map(df_market)
    _render_alerts(summary, risk_ok, df_market)

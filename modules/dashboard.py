"""
modules/dashboard.py
--------------------
Onglet "Monitoring Portefeuille" — métriques, ordres, equity curve, backtest.
"""
import os, logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)
TRADING_DAYS = 252
STRAT_COLORS = {
    "Momentum":       "#0052cc",
    "Mean-Reversion": "#228b22",
    "Carry Trade":    "#f59e0b",
    "TSMOM":          "#ef4444",
    "Long-Term G10":  "#8b5cf6",
    "Portfolio":      "#6f42c1",
    "Benchmark":      "#6c757d",
}

# ─── Helpers plotly ──────────────────────────────────────────────────────────

def _clean_layout(fig, height=280, **kwargs):
    """Applique le style transparent commun à tous les graphiques."""
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(color="#495057", gridcolor="#e9ecef"),
        yaxis=dict(color="#495057", gridcolor="#e9ecef"),
    )
    defaults.update(kwargs)
    fig.update_layout(**defaults)
    return fig


# ─── Chargement ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    data = {
        "orders":           pd.DataFrame(),
        "weights":          pd.Series(dtype=float),
        "returns":          pd.DataFrame(),
        "backtest_metrics": pd.DataFrame(),
        "strategy_weights": pd.DataFrame(),
        "selected_pairs":   pd.DataFrame(),
        "mkw_weights":      pd.DataFrame(),
        "mkw_metrics":      pd.DataFrame(),
    }
    if os.path.exists("FINAL_ORDERS.csv"):
        data["orders"] = pd.read_csv("FINAL_ORDERS.csv")
    if os.path.exists("optimal_weights.csv"):
        data["weights"] = pd.read_csv("optimal_weights.csv", header=None, index_col=0)[1]
    if os.path.exists("strategies_returns.csv"):
        data["returns"] = pd.read_csv("strategies_returns.csv", index_col=0, parse_dates=True)
    if os.path.exists("backtest_metrics.csv"):
        data["backtest_metrics"] = pd.read_csv("backtest_metrics.csv")
    if os.path.exists("strategy_weights.csv"):
        data["strategy_weights"] = pd.read_csv("strategy_weights.csv")
    if os.path.exists("selected_pairs.csv"):
        data["selected_pairs"] = pd.read_csv("selected_pairs.csv", sep=";", decimal=",")
    if os.path.exists("markowitz_weights.csv"):
        data["mkw_weights"] = pd.read_csv("markowitz_weights.csv")
    if os.path.exists("markowitz_metrics.csv"):
        data["mkw_metrics"] = pd.read_csv("markowitz_metrics.csv")
    return data


def _rates_alert():
    try:
        from macro_data import check_rates_freshness, RATES_LAST_UPDATED
        days_old = check_rates_freshness()
        if days_old > 45:
            st.warning(f"Taux directeurs non mis a jour depuis {days_old} jours (dernier : {RATES_LAST_UPDATED}).", icon="⚠️")
        elif days_old > 20:
            st.info(f"Taux directeurs mis a jour il y a {days_old} jours.")
    except ImportError:
        pass


# ─── Calculs ─────────────────────────────────────────────────────────────────

def _bl_portfolio_returns(strat_returns, strat_w):
    """Rendements du portefeuille BL-pondere."""
    w_dict = strat_w.to_dict() if not strat_w.empty else {}
    w_sum = sum(w_dict.values()) or 1.0
    if w_dict:
        return sum(
            strat_returns[s] * w_dict.get(s, 0.0) / w_sum
            for s in strat_returns.columns if s in w_dict
        )
    return strat_returns.mean(axis=1)


def calculate_metrics(returns):
    """Metriques de performance par colonne de rendements (CAGR compose)."""
    rows = []
    for col in returns.columns:
        r_s = returns[col].dropna()
        if len(r_s) == 0:
            continue
        eq_s = (1 + r_s).cumprod()
        years = len(r_s) / TRADING_DAYS
        ar = (eq_s.iloc[-1] ** (1.0 / years) - 1.0) * 100 if years > 0 else 0.0
        av = r_s.std() * np.sqrt(TRADING_DAYS) * 100
        dd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100
        rows.append({
            "Strategie": col,
            "Ret Ann (%)": round(ar, 2),
            "Vol Ann (%)": round(av, 2),
            "Sharpe": round(ar / av, 3) if av > 0 else 0,
            "Max DD (%)": round(dd, 2),
            "Calmar": round(ar / abs(dd), 3) if dd != 0 else 0,
        })
    return pd.DataFrame(rows).set_index("Strategie")


def _cagr(daily_returns):
    """Calcule le CAGR (rendement compose annualise reel)."""
    r = daily_returns.dropna()
    if len(r) < 20:
        return 0.0
    cum = (1 + r).prod()
    years = len(r) / TRADING_DAYS
    return (cum ** (1.0 / years) - 1.0) * 100 if years > 0 else 0.0


def _compute_ann_return(mkw_weights, strat_returns, strat_w):
    """Return annualise CAGR du portefeuille Markowitz (ou BL fallback)."""
    # 1. Priorite : rendements Markowitz (pair-level, poids optimaux)
    if not mkw_weights.empty and os.path.exists("strategies_returns_by_pair.csv"):
        try:
            from portfolio_utils import compute_pair_returns
            pair_rets = compute_pair_returns()
            w_dict = dict(zip(mkw_weights["Pair"], mkw_weights["Final_Weight"]))
            port_daily = sum(
                pair_rets[p] * w_dict.get(p, 0.0)
                for p in pair_rets.columns if p in w_dict
            )
            return _cagr(port_daily)
        except Exception as e:
            logger.warning("Markowitz return fallback: %s", e)

    # 2. Fallback : rendements BL-ponderes
    if not strat_returns.empty:
        port_r = _bl_portfolio_returns(strat_returns, strat_w)
        return _cagr(port_r)

    return 0.0


def _resize_orders(orders_df, strat_w, capital, risk_pct, max_pos_pct, mkw_weights=None):
    """Recalcule Size_USD avec poids Markowitz x ML (ou fallback BL).
    Plafonne l'exposition totale au capital (pas de levier)."""
    if orders_df.empty or "Size_USD" not in orders_df.columns:
        return orders_df

    # Mode Markowitz : poids par (pair, strat) = Markowitz(pair) × ML(strat|pair)
    if mkw_weights is not None and not mkw_weights.empty:
        mkw_dict = dict(zip(mkw_weights["Pair"], mkw_weights["Final_Weight"]))
        sw_dict = {}
        if os.path.exists("strategy_weights.csv"):
            sw_df = pd.read_csv("strategy_weights.csv")
            if "Pair" in sw_df.columns:
                for _, row in sw_df.iterrows():
                    for s in [c for c in sw_df.columns if c != "Pair"]:
                        sw_dict[(row["Pair"], s)] = row.get(s, 0.0)

        def _calc_mkw(row):
            pair = str(row.get("Pair", ""))
            strat = str(row.get("Strategy", ""))
            pw = mkw_dict.get(pair, 0.0)
            sw = sw_dict.get((pair, strat), 0.25)
            return round(capital * pw * sw, 0)

        orders_df["Size_USD"] = orders_df.apply(_calc_mkw, axis=1)
    else:
        # Fallback BL
        w_dict = strat_w.to_dict() if not strat_w.empty else {
            s: 1.0 / max(len(orders_df["Strategy"].unique()), 1)
            for s in orders_df["Strategy"].unique()
        }
        def _calc_bl(row):
            sw = w_dict.get(str(row.get("Strategy", "")), 0.0)
            vol = max(float(row.get("Vol_20", 0.008)), 0.002)
            if sw <= 0:
                return 0.0
            return round(min(capital * sw * risk_pct / vol, capital * max_pos_pct), 0)
        orders_df["Size_USD"] = orders_df.apply(_calc_bl, axis=1)

    # Cap a 100% du capital
    total_exposure = orders_df["Size_USD"].sum()
    if total_exposure > capital and total_exposure > 0:
        scale = capital / total_exposure
        orders_df["Size_USD"] = (orders_df["Size_USD"] * scale).round(0)
    return orders_df


# ─── Sections d'affichage ────────────────────────────────────────────────────

def _render_kpis(orders_df, strat_returns, strat_w, capital, risk_pct, max_pos_pct, mkw_weights):
    """Ligne de KPIs en haut du dashboard."""
    exposure = orders_df["Size_USD"].sum() if not orders_df.empty else 0
    ann_ret = _compute_ann_return(mkw_weights, strat_returns, strat_w)
    sh = 0.0

    # Sharpe : Markowitz si dispo, sinon BL
    if not mkw_weights.empty and os.path.exists("strategies_returns_by_pair.csv"):
        try:
            from portfolio_utils import compute_pair_returns
            pair_rets = compute_pair_returns()
            w_dict = dict(zip(mkw_weights["Pair"], mkw_weights["Final_Weight"]))
            port_r = sum(pair_rets[p] * w_dict.get(p, 0.0)
                         for p in pair_rets.columns if p in w_dict)
            sh = (port_r.mean() / port_r.std()) * np.sqrt(TRADING_DAYS) if port_r.std() > 0 else 0
        except Exception:
            pass
    if sh == 0.0 and not strat_returns.empty:
        port_r = _bl_portfolio_returns(strat_returns, strat_w)
        sh = (port_r.mean() / port_r.std()) * np.sqrt(TRADING_DAYS) if port_r.std() > 0 else 0

    vol_p = 0.0
    if not orders_df.empty and "Vol_20" in orders_df.columns and capital > 0:
        w_pos = (orders_df["Size_USD"] / capital).values
        v_pos = orders_df["Vol_20"].values
        vol_p = float(np.sqrt(np.sum((w_pos * v_pos) ** 2)) * np.sqrt(TRADING_DAYS) * 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital alloue", f"${exposure:,.0f}", help=f"Exposition brute sur capital ${capital:,}")
    c2.metric("Signaux du jour", len(orders_df))
    c3.metric("Return annualise", f"{ann_ret:+.2f}%",
              help="Rendement Markowitz optimise (pair-level)")
    c4.metric("Sharpe (historique)", f"{sh:.2f}")
    c5.metric("Vol portefeuille (live)", f"{vol_p:.2f}%",
              help="Volatilite annualisee des positions actuelles")


def _render_orders_table(orders_df):
    """Tableau des ordres du jour avec carry et P&L estime."""
    st.markdown("### Ordres du jour")
    if orders_df.empty:
        st.info("Aucun signal actif aujourd'hui.")
        return
    from macro_data import get_interest_rate_differential
    disp = orders_df.copy()
    disp["Est. P&L/j ($)"] = (disp["Size_USD"] * disp["Vol_20"]).round(0).astype(int)

    def _carry(row):
        if row["Strategy"] not in ("Carry Trade", "Long-Term G10"):
            return np.nan
        pair = row["Pair"].replace("=X", "")
        diff = get_interest_rate_differential(pair)
        sign = 1 if row["Direction"] == "LONG" else -1
        return round(sign * diff / 100 * row["Size_USD"], 0)

    disp["Carry/an ($)"] = disp.apply(_carry, axis=1)

    fmt = {"Size_USD": "${:,.0f}", "Price": "{:.4f}", "Est. P&L/j ($)": "${:,.0f}"}
    style = (
        disp.style
        .map(lambda v: "color:#28a745;font-weight:700" if "LONG" in str(v) else
             ("color:#dc3545;font-weight:700" if "SHORT" in str(v) else ""),
             subset=["Direction"])
        .map(lambda v: f"color:{STRAT_COLORS.get(v, '#6c757d')};font-weight:600",
             subset=["Strategy"])
        .format(fmt)
    )
    if "Carry/an ($)" in disp.columns:
        style = (
            style
            .map(lambda v: ("color:#228b22;font-weight:600" if float(v) > 0 else "color:#dc3545")
                 if pd.notna(v) and str(v) not in ("", "nan") else "",
                 subset=["Carry/an ($)"])
            .format({"Carry/an ($)": "${:,.0f}"}, na_rep="-")
        )
    st.dataframe(style, use_container_width=True, height=360)


def _render_allocation_pie(strat_w):
    """Pie chart de l'allocation dynamique."""
    st.markdown("### Allocation dynamique")
    if strat_w.empty:
        st.warning("Allocation non disponible. Lancez black_litterman.py.")
        return
    alloc = strat_w.copy() * 0.70
    alloc["Long-Term G10"] = 0.30
    fig = go.Figure(go.Pie(
        labels=alloc.index, values=alloc.values, hole=0.55,
        marker_colors=[STRAT_COLORS.get(s, "#6c757d") for s in alloc.index],
        textinfo="label+percent", textfont_size=11,
    ))
    _clean_layout(fig, height=240, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="dash_alloc_pie")
    for strat, w in alloc.sort_values(ascending=False).items():
        c = STRAT_COLORS.get(strat, "#6c757d")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<span style='width:115px;font-size:11px;color:#495057'>{strat}</span>"
            f"<div style='flex:1;height:6px;background:#e9ecef;border-radius:3px'>"
            f"<div style='width:{w*100:.0f}%;height:100%;background:{c};border-radius:3px'></div></div>"
            f"<span style='font-size:11px;font-family:monospace;color:{c};font-weight:700'>{w*100:.0f}%</span>"
            f"</div>", unsafe_allow_html=True)


def _render_backtest_cards(bk_metrics):
    """Cartes Sharpe / SL / holding par strategie."""
    if bk_metrics.empty:
        return
    st.divider()
    st.markdown("### Performance backtest (apres couts de transaction)")
    st.caption("Spread, swap overnight et stop-loss ATR x 2 inclus")
    summary = bk_metrics.groupby("strategy").agg(
        sharpe_net=("sharpe_net", "first"),
        avg_holding=("avg_holding", "mean"),
        sl_trigger_rate=("sl_trigger_rate", "mean"),
        cost_drag_ann=("cost_drag_ann", "mean"),
        total_trades=("total_trades", "sum"),
    ).reset_index()
    cols = st.columns(len(summary))
    for i, row in summary.iterrows():
        c = STRAT_COLORS.get(row["strategy"], "#6c757d")
        cols[i].markdown(
            f"<div style='background:#f8f9fa;border-radius:8px;padding:12px;border-left:3px solid {c}'>"
            f"<p style='font-size:11px;color:#6c757d;margin:0'>{row['strategy']}</p>"
            f"<p style='font-size:20px;font-weight:700;color:{c};margin:4px 0'>{row['sharpe_net']:.2f}</p>"
            f"<p style='font-size:10px;color:#6c757d;margin:0'>Sharpe net</p>"
            f"<hr style='margin:8px 0;border-color:#e9ecef'>"
            f"<p style='font-size:10px;color:#495057;margin:2px 0'>SL declenche : {row['sl_trigger_rate']:.1f}%</p>"
            f"<p style='font-size:10px;color:#495057;margin:2px 0'>Detention moy : {row['avg_holding']:.1f}j</p>"
            f"<p style='font-size:10px;color:#495057;margin:2px 0'>Cout drag : {row['cost_drag_ann']:.3f}</p>"
            f"<p style='font-size:10px;color:#495057;margin:2px 0'>Trades : {int(row['total_trades'])}</p>"
            f"</div>", unsafe_allow_html=True)


def _render_equity_curve(strat_returns, strat_w):
    """Equity curve BL-ponderee."""
    if strat_returns.empty:
        return
    st.divider()
    st.markdown("### Equity curve - portefeuille cumule")
    port_r = _bl_portfolio_returns(strat_returns, strat_w)
    eq_port = (1 + port_r).cumprod() * 100

    fig = go.Figure()
    for col in strat_returns.columns:
        eq_s = (1 + strat_returns[col]).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=eq_s.index, y=eq_s.values, name=col, mode="lines",
            line=dict(color=STRAT_COLORS.get(col, "#6c757d"), width=1, dash="dot"),
            opacity=0.5,
        ))
    fig.add_trace(go.Scatter(
        x=eq_port.index, y=eq_port.values, name="Portfolio BL",
        line=dict(color="#6f42c1", width=2.5),
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="#aaa", line_width=1)
    _clean_layout(fig, height=280,
                  legend=dict(orientation="h", y=-0.25, font_size=11),
                  yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Valeur (base 100)"),
                  hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="dash_equity_curve")


def _render_exposure_bars(orders_df):
    """Barres d'exposition par strategie et par paire."""
    if orders_df.empty:
        return
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Exposition par strategie")
        exp_s = orders_df.groupby("Strategy")["Size_USD"].sum().reset_index()
        fig = px.bar(exp_s, x="Strategy", y="Size_USD", color="Strategy",
                     color_discrete_map=STRAT_COLORS)
        _clean_layout(fig, height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Exposition par paire")
        exp_p = orders_df.groupby("Pair")["Size_USD"].sum().reset_index()
        fig = px.bar(exp_p.sort_values("Size_USD", ascending=True),
                     x="Size_USD", y="Pair", orientation="h",
                     color_discrete_sequence=["#0052cc"])
        _clean_layout(fig, height=200)
        st.plotly_chart(fig, use_container_width=True)


def _render_strategy_weights(strat_weights_df, selected_pairs):
    """Poids ML par paire (couche 2)."""
    if strat_weights_df.empty:
        return
    st.divider()
    st.markdown("### Strategie ML par paire")
    st.caption("Poids optimal de chaque strategie par paire active")

    sw = strat_weights_df.set_index("Pair") if "Pair" in strat_weights_df.columns else strat_weights_df
    strat_cols = [c for c in ["Momentum", "Mean-Reversion", "TSMOM", "Carry Trade"] if c in sw.columns]
    if not strat_cols:
        return

    col_heat, col_detail = st.columns([3, 2])
    with col_heat:
        fig = go.Figure()
        for s in strat_cols:
            fig.add_trace(go.Bar(
                name=s, x=sw.index.tolist(), y=sw[s].values,
                marker_color=STRAT_COLORS.get(s, "#6c757d"),
            ))
        _clean_layout(fig, height=280, barmode="stack",
                      legend=dict(orientation="h", y=-0.25, font_color="#495057"),
                      yaxis=dict(color="#495057", gridcolor="#e9ecef",
                                 tickformat=".0%", range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    with col_detail:
        dominant = sw[strat_cols].idxmax(axis=1).rename("Strategie dominante")
        dom_weight = sw[strat_cols].max(axis=1).rename("Poids")
        detail_tbl = pd.concat([dominant, dom_weight], axis=1)
        if not selected_pairs.empty and "Score_ML" in selected_pairs.columns:
            sp_idx = selected_pairs.set_index("Pair")["Score_ML"]
            detail_tbl["Score ML"] = detail_tbl.index.map(sp_idx)
        st.dataframe(
            detail_tbl.style
            .map(lambda v: f"color:{STRAT_COLORS.get(v, '#495057')};font-weight:600",
                 subset=["Strategie dominante"])
            .format({"Poids": "{:.0%}", "Score ML": "{:.4f}"}),
            use_container_width=True, height=260,
        )


def _render_markowitz_section(mkw_weights, mkw_metrics):
    """Section Markowitz : allocation par paire + metriques."""
    if mkw_weights.empty:
        return
    st.divider()
    st.markdown("### Portefeuille Markowitz optimise")
    st.caption("Allocation directe par paire — Max Sharpe / Min Variance blend 60/40")

    col_pie, col_met = st.columns([1.2, 1])
    with col_pie:
        wf = mkw_weights[mkw_weights["Final_Weight"] > 0.005].copy()
        fig = go.Figure(go.Pie(
            labels=wf["Pair"], values=wf["Final_Weight"], hole=0.55,
            textinfo="label+percent", textfont_size=11,
            marker_colors=px.colors.qualitative.Set2[:len(wf)],
        ))
        _clean_layout(fig, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="mkw_pie")

    with col_met:
        if not mkw_metrics.empty:
            blend = mkw_metrics[mkw_metrics["Portfolio"] == "Blend_60_40"]
            if not blend.empty:
                row = blend.iloc[0]
                st.metric("Return annualise", f"{row.get('Ann. Return (%)', 0):+.2f}%")
                st.metric("Sharpe Ratio", f"{row.get('Sharpe Ratio', 0):.3f}")
                st.metric("Sortino Ratio", f"{row.get('Sortino Ratio', 0):.3f}")
                st.metric("Max Drawdown", f"{row.get('Max Drawdown (%)', 0):.2f}%")
                st.metric("VaR 95%", f"{row.get('VaR 95% (%)', 0)}%")

    # Comparaison Max Sharpe vs Min Var vs Blend
    if not mkw_metrics.empty and len(mkw_metrics) > 1:
        st.markdown("#### Comparaison des portefeuilles")
        disp = mkw_metrics.set_index("Portfolio")
        st.dataframe(
            disp.style
            .background_gradient(subset=["Ann. Return (%)"], cmap="RdYlGn")
            .background_gradient(subset=["Sharpe Ratio"], cmap="RdYlGn")
            .format("{:.2f}"),
            use_container_width=True)

    # Bar chart des poids par paire
    st.markdown("#### Poids Markowitz par paire")
    cols_w = ["Pair", "Max_Sharpe", "Min_Variance", "Final_Weight"]
    wdf = mkw_weights[[c for c in cols_w if c in mkw_weights.columns]].copy()
    fig = go.Figure()
    for col, color in [("Max_Sharpe", "#0052cc"), ("Min_Variance", "#228b22"), ("Final_Weight", "#6f42c1")]:
        if col in wdf.columns:
            fig.add_trace(go.Bar(name=col.replace("_", " "), x=wdf["Pair"], y=wdf[col],
                                 marker_color=color))
    _clean_layout(fig, height=260, barmode="group",
                  legend=dict(orientation="h", y=-0.25, font_size=11),
                  yaxis=dict(color="#495057", gridcolor="#e9ecef", tickformat=".0%", title="Poids"))
    st.plotly_chart(fig, use_container_width=True, key="mkw_weights_bar")


def _render_historical_perf(strat_returns, strat_w, mkw_weights):
    """Performances historiques + metriques de risque."""
    if strat_returns.empty:
        return
    st.divider()
    st.markdown("### Performances historiques")

    equity_df = (1 + strat_returns).cumprod() * 100
    equity_df["Portfolio (Eq. Weight)"] = (1 + strat_returns.mean(axis=1)).cumprod() * 100

    bl_port = None
    if not strat_w.empty:
        bl_port = _bl_portfolio_returns(strat_returns, strat_w)
        equity_df["Portfolio (BL)"] = (1 + bl_port).cumprod() * 100

    # Ajouter la courbe Markowitz si disponible
    mkw_port = None
    if not mkw_weights.empty and os.path.exists("strategies_returns_by_pair.csv"):
        try:
            from portfolio_utils import compute_pair_returns
            pair_rets = compute_pair_returns()
            w_dict = dict(zip(mkw_weights["Pair"], mkw_weights["Final_Weight"]))
            mkw_port = sum(pair_rets[p] * w_dict.get(p, 0.0)
                           for p in pair_rets.columns if p in w_dict)
            equity_df["Portfolio (Markowitz)"] = (1 + mkw_port).cumprod() * 100
        except Exception:
            pass

    fig = go.Figure()
    for col in equity_df.columns:
        is_mkw = "Markowitz" in col
        is_bl = "BL" in col
        fig.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df[col], name=col,
            line=dict(
                color="#e63946" if is_mkw else STRAT_COLORS.get(col, "#000000"),
                width=3.0 if is_mkw else (2.5 if is_bl else (2.0 if "Portfolio" in col else 1.5)),
                dash="solid" if (is_mkw or is_bl) else ("dash" if "Eq" in col else "solid"),
            ),
        ))
    _clean_layout(fig, height=350,
                  legend=dict(bgcolor="rgba(255,255,255,0.8)", font_color="#495057",
                              orientation="h", y=-0.2),
                  yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Base 100"),
                  hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Metriques de risque - par strategie")
    returns_for_metrics = strat_returns.copy()
    if bl_port is not None:
        returns_for_metrics["Portfolio (BL)"] = bl_port
    if mkw_port is not None:
        returns_for_metrics["Portfolio (Markowitz)"] = mkw_port
    returns_for_metrics["Portfolio (Eq.)"] = strat_returns.mean(axis=1)
    metrics_df = calculate_metrics(returns_for_metrics)
    st.dataframe(
        metrics_df.style
        .background_gradient(subset=["Ret Ann (%)"], cmap="RdYlGn")
        .background_gradient(subset=["Sharpe"], cmap="RdYlGn")
        .format("{:.2f}"),
        use_container_width=True)


# ─── Point d'entree ──────────────────────────────────────────────────────────

def show():
    _rates_alert()
    data = load_data()
    orders_df        = data["orders"].copy()
    strat_w          = data["weights"]
    strat_returns    = data["returns"]
    bk_metrics       = data["backtest_metrics"]
    strat_weights_df = data["strategy_weights"]
    selected_pairs   = data["selected_pairs"]
    mkw_weights      = data["mkw_weights"]
    mkw_metrics      = data["mkw_metrics"]

    capital     = st.session_state.get("capital",    100000)
    risk_pct    = st.session_state.get("risk_pct",   0.02)
    max_pos_pct = st.session_state.get("max_pos_pct", 0.15)

    orders_df = _resize_orders(orders_df, strat_w, capital, risk_pct, max_pos_pct, mkw_weights)

    _render_kpis(orders_df, strat_returns, strat_w, capital, risk_pct, max_pos_pct, mkw_weights)
    st.divider()

    col_l, col_r = st.columns([2, 1])
    with col_l:
        _render_orders_table(orders_df)
    with col_r:
        _render_allocation_pie(strat_w)

    _render_markowitz_section(mkw_weights, mkw_metrics)
    _render_backtest_cards(bk_metrics)
    _render_equity_curve(strat_returns, strat_w)
    _render_exposure_bars(orders_df)
    _render_strategy_weights(strat_weights_df, selected_pairs)
    _render_historical_perf(strat_returns, strat_w, mkw_weights)

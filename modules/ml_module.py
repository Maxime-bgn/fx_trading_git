"""
modules/ml_module.py
--------------------
Onglet "Machine Learning Core" — selection des paires + allocation par strategie.
"""
import os, logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

STRAT_COLORS = {
    "Momentum":       "#0052cc",
    "Mean-Reversion": "#228b22",
    "Carry Trade":    "#f59e0b",
    "TSMOM":          "#ef4444",
}


def _clean_layout(fig, height=280, **kwargs):
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(color="#495057", gridcolor="#e9ecef"),
        yaxis=dict(color="#495057", gridcolor="#e9ecef"),
    )
    defaults.update(kwargs)
    fig.update_layout(**defaults)


# ─── Chargement ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load():
    out = {k: pd.DataFrame() for k in
           ["pairs", "val_metrics", "feat_imp", "strat_w", "strat_metrics", "ic_detail", "market"]}
    files = {
        "pairs":        ("selected_pairs.csv",       dict(sep=";", decimal=",")),
        "val_metrics":  ("ml_validation_metrics.csv", {}),
        "feat_imp":     ("ml_feature_importance.csv", {}),
        "strat_w":      ("strategy_weights.csv",      {}),
        "strat_metrics":("strategy_ml_metrics.csv",   {}),
        "ic_detail":    ("ic_audit_pair_scores.csv",  {}),
    }
    for key, (path, kw) in files.items():
        if os.path.exists(path):
            out[key] = pd.read_csv(path, **kw)
    if os.path.exists("dataset_ml_ready.csv"):
        out["market"] = pd.read_csv("dataset_ml_ready.csv", sep=";", decimal=",",
                                    index_col=0, parse_dates=True)
    return out


# ─── Sections ────────────────────────────────────────────────────────────────

def _section_live_signals(pairs_df):
    """Signaux ML temps reel."""
    st.markdown("#### Signaux ML - Predictions temps reel")
    active = pairs_df[pairs_df["Status"] == "ACTIVE"].copy() if "Status" in pairs_df.columns else pairs_df.copy()
    if active.empty or "Score_ML" not in active.columns:
        st.info("Score_ML absent - lancez ml_pair_selection.py")
        return

    n_long = (active["Score_ML"] > 0).sum()
    n_short = (active["Score_ML"] < 0).sum()
    n_agree = active["Signal_Agree"].sum() if "Signal_Agree" in active.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Paires actives", len(active))
    k2.metric("Signaux LONG", int(n_long))
    k3.metric("Signaux SHORT", int(n_short))
    k4.metric("Ridge + LGBM d'accord", int(n_agree))

    sig = active[["Pair", "Score_ML"]].copy()
    for c in ["Signal_Agree", "Score_Ridge"]:
        if c in active.columns:
            sig[c] = active[c]
    sig["Direction"] = sig["Score_ML"].apply(
        lambda x: "LONG" if x > 0 else ("SHORT" if x < 0 else "FLAT"))
    sig = sig.sort_values("Score_ML", ascending=False)

    st.dataframe(
        sig.style
        .map(lambda v: "color:#228b22;font-weight:700" if v == "LONG" else
             ("color:#ef4444;font-weight:700" if v == "SHORT" else "color:#999"),
             subset=["Direction"])
        .map(lambda v: "background-color:#e8f5e9" if float(v) > 0.05 else
             ("background-color:#ffebee" if float(v) < -0.05 else "") if pd.notna(v) else "",
             subset=["Score_ML"]),
        use_container_width=True, height=280)

    fig = go.Figure(go.Bar(
        x=sig["Pair"], y=sig["Score_ML"],
        marker_color=["#228b22" if v > 0 else "#ef4444" for v in sig["Score_ML"]],
    ))
    fig.add_hline(y=0, line_color="#6c757d", line_width=1)
    _clean_layout(fig, height=220, yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Score ML"))
    st.plotly_chart(fig, use_container_width=True, key="ml_live_scores")

    if os.path.exists("optimal_weights.csv"):
        bl_w = pd.read_csv("optimal_weights.csv", header=None, names=["Strategy", "Weight"])
        st.markdown("**Allocation Black-Litterman -> live trading**")
        cols = st.columns(len(bl_w))
        for i, (_, row) in enumerate(bl_w.iterrows()):
            cols[i].metric(row["Strategy"], f"{row['Weight']:.1%}")


def _section_ic_validation(val_metrics):
    """Validation IC walk-forward."""
    st.markdown("#### Couche 1 - Validation IC walk-forward")
    if val_metrics.empty or "IC" not in val_metrics.columns:
        st.info("Lancez le pipeline pour generer ml_validation_metrics.csv")
        return

    wf = val_metrics[val_metrics["type"] == "walk_forward"]
    hld = val_metrics[val_metrics["type"] == "holdout"]

    col_ic, col_kpi = st.columns([2, 1])
    with col_ic:
        fig = go.Figure()
        if not wf.empty:
            fig.add_trace(go.Bar(
                x=wf["fold"].astype(str), y=wf["IC"], name="IC walk-forward",
                marker_color=["#228b22" if v > 0 else "#dc3545" for v in wf["IC"]],
            ))
        if not hld.empty:
            fig.add_trace(go.Scatter(
                x=["Holdout OOS"], y=hld["IC"].values, mode="markers",
                name="Holdout OOS", marker=dict(color="#ef4444", size=14, symbol="star"),
            ))
        fig.add_hline(y=0.05, line_dash="dash", line_color="#f59e0b", annotation_text="IC min (0.05)")
        fig.add_hline(y=0.0, line_dash="dot", line_color="#6c757d")
        _clean_layout(fig, height=260,
                      legend=dict(orientation="h", y=-0.3, font_size=11),
                      yaxis=dict(color="#495057", gridcolor="#e9ecef", title="IC (Spearman)"))
        st.plotly_chart(fig, use_container_width=True, key="ml_ic_walkforward")

    with col_kpi:
        st.caption("Metriques IC")
        if not wf.empty:
            st.metric("IC moyen (WF)", f"{wf['IC'].mean():.4f}")
            st.metric("% folds positifs", f"{(wf['IC'] > 0).mean():.0%}")
        if not hld.empty:
            h_ic = float(hld["IC"].iloc[0])
            st.metric("IC Holdout OOS", f"{h_ic:.4f}",
                      delta="Signal" if abs(h_ic) > 0.05 else "Faible",
                      delta_color="normal" if abs(h_ic) > 0.05 else "inverse")


def _section_pairs_table(pairs_df, ic_detail):
    """Tableau paires actives + IC audit."""
    st.markdown("#### Paires selectionnees - score ML + signal IC")
    col_pairs, col_ic = st.columns([1.2, 1])

    with col_pairs:
        if pairs_df.empty:
            st.info("Aucune paire disponible.")
            return
        disp_cols = ["Pair", "Status"]
        for c in ["Score", "Score_ML", "Carry_Real", "Holdout_IC", "Rank"]:
            if c in pairs_df.columns:
                disp_cols.append(c)
        fmt = {}
        for c in ["Score", "Score_ML"]:
            if c in disp_cols:
                fmt[c] = "{:.4f}"
        if "Carry_Real" in disp_cols:
            fmt["Carry_Real"] = "{:+.2f}%"
        st.dataframe(
            pairs_df[disp_cols]
            .sort_values("Rank" if "Rank" in disp_cols else disp_cols[0])
            .style
            .map(lambda v: "color:#228b22;font-weight:700" if v == "ACTIVE" else "color:#999",
                 subset=["Status"])
            .format(fmt),
            use_container_width=True, height=320)

    with col_ic:
        if ic_detail.empty:
            st.info("ic_audit_pair_scores.csv manquant.")
            return
        st.caption("Audit IC/ICIR - resultats horizon 5j")
        disp = ic_detail.copy()
        for c in ["nb_garde", "best_ICIR", "carry"]:
            if c not in disp.columns:
                disp[c] = np.nan
        show_cols = ["Pair", "nb_garde", "best_ICIR"]
        if "carry" in disp.columns:
            show_cols.append("carry")
        fmt_ic = {"best_ICIR": "{:.3f}"}
        if "carry" in show_cols:
            fmt_ic["carry"] = "{:+.2f}%"
        st.dataframe(
            disp[show_cols].style
            .map(lambda v: f"color:#228b22;font-weight:700" if float(v) >= 0.5 else
                 (f"color:#0052cc" if float(v) >= 0.35 else "color:#999")
                 if pd.notna(v) else "",
                 subset=["best_ICIR"])
            .format(fmt_ic),
            use_container_width=True, height=320)


def _section_feature_importance(feat_imp, strat_met):
    """Feature importance + IC par strategie."""
    st.markdown("#### Feature importance - LightGBM couche 1")
    col_feat, col_strat = st.columns(2)

    with col_feat:
        if feat_imp.empty:
            st.info("ml_feature_importance.csv manquant.")
            return
        fi = feat_imp.sort_values("importance")
        fig = go.Figure(go.Bar(x=fi["importance"], y=fi["feature"],
                                orientation="h", marker_color="#0052cc"))
        _clean_layout(fig, height=320, xaxis=dict(color="#495057", gridcolor="#e9ecef", title="Importance"))
        st.plotly_chart(fig, use_container_width=True, key="ml_feat_importance")

    with col_strat:
        if strat_met.empty or "IC_mean" not in strat_met.columns:
            st.info("strategy_ml_metrics.csv manquant.")
            return
        st.caption("IC moyen par strategie - couche 2")
        for strat in ["Momentum", "Mean-Reversion", "TSMOM", "Carry Trade"]:
            sub = strat_met[strat_met["Strategy"] == strat]
            if sub.empty:
                continue
            ic_val = sub["IC_mean"].mean()
            color = STRAT_COLORS.get(strat, "#6c757d")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                f"<span style='width:130px;font-size:12px;color:{color};font-weight:600'>{strat}</span>"
                f"<span style='font-size:13px;font-family:monospace'>IC = {ic_val:+.4f}</span>"
                f"</div>", unsafe_allow_html=True)


def _section_strategy_weights(strat_w):
    """Poids strategie par paire (stacked bar)."""
    st.markdown("#### Couche 2 - Poids strategie par paire active")
    if strat_w.empty:
        st.info("strategy_weights.csv manquant - lancez strategy_ml.py.")
        return
    sw = strat_w.set_index("Pair") if "Pair" in strat_w.columns else strat_w
    strat_cols = [c for c in ["Momentum", "Mean-Reversion", "TSMOM", "Carry Trade"] if c in sw.columns]
    if not strat_cols:
        return
    fig = go.Figure()
    for s in strat_cols:
        fig.add_trace(go.Bar(name=s, x=sw.index.tolist(), y=sw[s].values,
                              marker_color=STRAT_COLORS.get(s, "#6c757d")))
    _clean_layout(fig, height=280, barmode="stack",
                  legend=dict(orientation="h", y=-0.25, font_color="#495057"),
                  yaxis=dict(color="#495057", gridcolor="#e9ecef", tickformat=".0%", range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True, key="ml_strat_weights")


def _section_regime_map(df_market, pairs_df, ic_detail):
    """Regime map Hurst vs Vol."""
    st.markdown("#### Regime map - Hurst vs Volatilite")
    st.caption("Bleu = actif (IC selectionne) - Gris = ecarte")

    if df_market.empty:
        return
    active_set = set(
        pairs_df[pairs_df["Status"] == "ACTIVE"]["Pair"].tolist()
    ) if not pairs_df.empty else set()

    latest = df_market.groupby("Pair").last().reset_index()
    if latest.empty or "Hurst" not in latest.columns or "Vol_20" not in latest.columns:
        return

    latest["Statut"] = latest["Pair"].apply(
        lambda p: "Actif" if p in active_set else "Ecarte")
    if not ic_detail.empty and "best_ICIR" in ic_detail.columns:
        latest = latest.merge(ic_detail[["Pair", "best_ICIR"]], on="Pair", how="left")
        latest["best_ICIR"] = latest["best_ICIR"].fillna(0).round(2)

    color_map = {"Actif": "#0052cc", "Ecarte": "#adb5bd"}
    fig = go.Figure()
    for statut, color in color_map.items():
        sub = latest[latest["Statut"] == statut]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Vol_20"], y=sub["Hurst"], mode="markers+text",
            name=statut, text=sub["Pair"], textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(color=color, size=sub["Vol_20"] * 2000 + 6,
                        opacity=0.85 if statut == "Actif" else 0.45,
                        line=dict(width=1, color="white")),
        ))
    fig.add_hrect(y0=0.55, y1=1.05, fillcolor="#0052cc", opacity=0.03,
                  annotation_text="Trend -> Momentum/TSMOM", annotation_position="top right")
    fig.add_hrect(y0=0.0, y1=0.45, fillcolor="#228b22", opacity=0.03,
                  annotation_text="Range -> Mean-Reversion", annotation_position="bottom right")
    _clean_layout(fig, height=420, showlegend=True,
                  legend=dict(orientation="h", y=-0.12, font_size=11),
                  xaxis=dict(color="#495057", gridcolor="#e9ecef", title="Volatilite 20j"),
                  yaxis=dict(color="#495057", gridcolor="#e9ecef", title="Hurst"))
    st.plotly_chart(fig, use_container_width=True, key="ml_regime_map")


# ─── Point d'entree ──────────────────────────────────────────────────────────

def show():
    st.markdown("### Machine Learning Core - Selection & Allocation")

    d = _load()
    pairs_df = d["pairs"]
    if pairs_df.empty:
        st.error("'selected_pairs.csv' introuvable. Lancez le pipeline.")
        return

    _section_live_signals(pairs_df)
    st.divider()
    _section_ic_validation(d["val_metrics"])
    st.divider()
    _section_pairs_table(pairs_df, d["ic_detail"])
    st.divider()
    _section_feature_importance(d["feat_imp"], d["strat_metrics"])
    st.divider()
    _section_strategy_weights(d["strat_w"])
    st.divider()
    _section_regime_map(d["market"], pairs_df, d["ic_detail"])
    st.divider()

    with st.expander("Architecture du pipeline ML"):
        st.markdown("""
**Couche 1 - Selection des paires** (`ml_pair_selection.py`)
- Univers : paires ayant |ICIR| >= 0.35 avec signal stable
- Z-score cross-sectionnel + encodage signe IC
- Modele : **LightGBM regression** sur `Future_Return` (5j)
- Carry overlay : score final = Score_ML + 0.10 x Carry_Reel_Z

**Couche 2 - Poids par strategie** (`strategy_ml.py`)
- Target : Sharpe rolling 20j normalise en poids
- Features : Hurst, Vol_Regime, Trend, Mom_20, BB_Width, Dist_MA20, Vol_20
- Modele : **4 LightGBM** separes (un par strategie)
        """)

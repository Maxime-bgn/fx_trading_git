"""
app.py
------
Point d'entrée Streamlit — FX Quant Terminal.

Changements vs version initiale :
  - Checklist intégrité fichiers étendue aux outputs de risk_manager et backtest
  - Bouton "Lancer le pipeline" dans la sidebar (appelle run_pipeline.py via subprocess)
  - Alerte pipeline.log si le dernier run a échoué
"""
import os, subprocess, sys, logging
from datetime import datetime
import streamlit as st
from modules import dashboard, ml_module, derivatives, market_analysis

logger = logging.getLogger(__name__)

st.set_page_config(page_title="QUANT FX TERMINAL", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e0e0e0; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #fcfcfc; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #ffffff; color: #666666;
        border: 1px solid #e0e0e0; border-radius: 4px; padding: 0px 30px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background-color: #0052cc !important; color: #ffffff !important; border-color: #0052cc !important; }
    .main-header { background-color: #ffffff; padding: 25px; border-radius: 4px; border: 1px solid #e0e0e0; margin-bottom: 25px; }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 4px; }
    [data-testid="stMetricValue"] { color: #0052cc; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("<h1 style='color:#0052cc;font-size:20px;font-weight:800'>FX TERMINAL v2.1</h1>",
                unsafe_allow_html=True)

    last_sync = "Inconnue"
    if os.path.exists("FINAL_ORDERS.csv"):
        mod_time  = os.path.getmtime("FINAL_ORDERS.csv")
        last_sync = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    st.text(f"Dernier calcul : {last_sync}")
    st.divider()

    # ── BOUTON PIPELINE ───────────────────────────────────────────────────────
    st.markdown("### Pipeline")
    if st.button("Lancer le pipeline", use_container_width=True, type="primary"):
        with st.spinner("Pipeline en cours..."):
            try:
                result = subprocess.run(
                    [sys.executable, "run_pipeline.py"],
                    capture_output=True, text=True, timeout=1800,
                )
                if result.returncode == 0:
                    st.success("Pipeline terminé avec succès.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Erreur dans le pipeline. Voir pipeline.log.")
                    if result.stderr:
                        with st.expander("Détail de l'erreur"):
                            st.code(result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                st.error("Timeout (30 min dépassées).")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")

    # Alerte si pipeline.log contient une erreur récente
    if os.path.exists("pipeline.log"):
        with open("pipeline.log", encoding="utf-8") as f:
            last_lines = f.readlines()[-10:]
        if any("FAILED" in l or "TIMEOUT" in l or "ERREUR" in l for l in last_lines):
            st.warning("Le dernier run pipeline contient des erreurs. Vérifiez pipeline.log.")

    st.divider()

    # ── CONFIGURATION DU RISQUE ───────────────────────────────────────────────
    # Paramètres réels par profil : (risk_per_trade_pct, max_position_pct, max_leverage)
    RISK_PROFILES = {
        "Prudent":   {"risk_pct": 0.01, "max_pos": 0.10, "max_lev": 0.8,  "desc": "1% / trade, max 10% / paire"},
        "Modéré":    {"risk_pct": 0.02, "max_pos": 0.15, "max_lev": 1.2,  "desc": "2% / trade, max 15% / paire"},
        "Dynamique": {"risk_pct": 0.035,"max_pos": 0.25, "max_lev": 2.0,  "desc": "3.5% / trade, max 25% / paire"},
        "Agressif":  {"risk_pct": 0.06, "max_pos": 0.40, "max_lev": 3.5,  "desc": "6% / trade, max 40% / paire"},
    }
    st.markdown("### Configuration du risque")
    st.session_state["capital"] = st.number_input("Capital de référence ($)",
                                                   value=st.session_state.get("capital", 100000),
                                                   step=10000, min_value=10000)
    profile_name = st.selectbox("Profil d'aversion au risque",
                                list(RISK_PROFILES.keys()),
                                index=list(RISK_PROFILES.keys()).index(
                                    st.session_state.get("risk_profile", "Modéré")
                                ))
    profile = RISK_PROFILES[profile_name]
    st.session_state["risk_profile"]  = profile_name
    st.session_state["risk_pct"]      = profile["risk_pct"]
    st.session_state["max_pos_pct"]   = profile["max_pos"]
    st.session_state["max_lev"]       = profile["max_lev"]

    col_a, col_b = st.columns(2)
    col_a.metric("Risk/trade", f"{profile['risk_pct']*100:.1f}%")
    col_b.metric("Max / paire", f"{profile['max_pos']*100:.0f}%")
    st.caption(f"Levier max : {profile['max_lev']}× — {profile['desc']}")
    st.divider()

    # ── INTÉGRITÉ DES FICHIERS ────────────────────────────────────────────────
    st.markdown("### Intégrité des fichiers")
    FILES_TO_CHECK = [
        ("dataset_ml_ready.csv",                    "Data"),
        ("ic_audit_pair_scores.csv",                "IC Audit"),
        ("selected_pairs.csv",                       "ML sélection"),
        ("strategy_weights.csv",                     "ML stratégies"),
        ("backtest_metrics.csv",                     "Backtest"),
        ("ml_validation_metrics.csv",                "ML validation"),
        ("FINAL_ORDERS.csv",                         "Ordres"),
        ("risk_outputs/risk_report_daily.csv",       "Risque daily"),
        ("risk_outputs/risk_summary.csv",            "Risque summary"),
    ]
    for fpath, label in FILES_TO_CHECK:
        present = os.path.exists(fpath)
        color   = "#228b22" if present else "#cc0000"
        status  = "OK" if present else "ABSENT"
        st.markdown(
            f"<small style='display:flex;justify-content:space-between'>"
            f"<span>{label}</span>"
            f"<b style='color:{color}'>{status}</b></small>",
            unsafe_allow_html=True,
        )

# =============================================================================
# CONTENU PRINCIPAL
# =============================================================================
st.markdown("""
<div class="main-header">
    <h2 style="margin:0;font-weight:700;letter-spacing:-0.5px">ANALYSE QUANTITATIVE FX</h2>
    <p style="color:#666666;font-size:13px;margin:5px 0 0 0">
        Intelligence Artificielle — Gestion de Portefeuille Multi-Actifs — Équipe 487
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "MONITORING PORTEFEUILLE",
    "MACHINE LEARNING CORE",
    "DÉRIVÉS & OPTIONS",
    "STRESS TEST & MARCHÉ",
])

with tabs[0]: dashboard.show()
with tabs[1]: ml_module.show()
with tabs[2]: derivatives.show()
with tabs[3]: market_analysis.show()

st.markdown(
    "<div style='text-align:center;color:#999999;font-size:10px;margin-top:40px;"
    "border-top:1px solid #e0e0e0;padding-top:20px'>"
    "DOCUMENT CONFIDENTIEL — USAGE PROFESSIONNEL UNIQUEMENT"
    "</div>",
    unsafe_allow_html=True,
)
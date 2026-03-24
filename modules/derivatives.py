"""
Module 4 — Options & Derivatives
Outil interactif de pricing d'options FX via le modele de Garman-Kohlhagen.
"""
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from typing import Dict, Union

# =============================================================================
# MOTEUR DE PRICING (GARMAN-KOHLHAGEN)
# =============================================================================

def garman_kohlhagen(
    S: float, K: float, T: float, rd: float, rf: float, sigma: float, option_type: str = "call"
) -> Dict[str, float]:
    """
    Calcule le prix et les Grecques d'une option de change europeenne.
    rd: Taux domestique (ex: USD)
    rf: Taux etranger (ex: EUR)
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return dict(price=intrinsic, delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
        
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)
        delta = np.exp(-rf * T) * norm.cdf(d1)
    else:
        price = K * np.exp(-rd * T) * norm.cdf(-d2) - S * np.exp(-rf * T) * norm.cdf(-d1)
        delta = -np.exp(-rf * T) * norm.cdf(-d1)
        
    gamma = np.exp(-rf * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-rf * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    theta_part1 = -(S * norm.pdf(d1) * sigma * np.exp(-rf * T)) / (2 * np.sqrt(T))
    if option_type == "call":
        theta_part2 = -rd * K * np.exp(-rd * T) * norm.cdf(d2) + rf * S * np.exp(-rf * T) * norm.cdf(d1)
    else:
        theta_part2 = rd * K * np.exp(-rd * T) * norm.cdf(-d2) - rf * S * np.exp(-rf * T) * norm.cdf(-d1)
        
    theta = (theta_part1 + theta_part2) / 365
    
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)

# =============================================================================
# VUE PRINCIPALE (UI)
# =============================================================================

def show():
    st.markdown("### Pricer d'Options FX (Modèle Garman-Kohlhagen)")
    st.caption("Outil de simulation de couverture et de stratégies optionnelles")

    # ── PARAMETRES DE SIMULATION ──────────────────────────────────────────────
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Cours Spot Actuel (S)", value=1.1000, step=0.0010, format="%.4f")
            K = st.number_input("Prix d'Exercice (Strike K)", value=1.1000, step=0.0010, format="%.4f")
            T_days = st.slider("Maturité (en jours)", min_value=1, max_value=365, value=30)
            sigma = st.slider("Volatilité Implicite (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5) / 100.0
        with c2:
            rd = st.slider("Taux Domestique (Quote - ex: USD) (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25) / 100.0
            rf = st.slider("Taux Étranger (Base - ex: EUR) (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.25) / 100.0
            notional = st.number_input("Notionnel (Volume en Base)", min_value=10_000, max_value=10_000_000, value=1_000_000, step=10_000)
            strategy = st.selectbox("Structure Optionnelle", ["Long Call", "Long Put", "Straddle", "Risk Reversal (Zero Cost)"])

    T = T_days / 365.0
    
    # ── CALCULS DES PRIMES ET GRECQUES ────────────────────────────────────────
    call = garman_kohlhagen(S, K, T, rd, rf, sigma, "call")
    put = garman_kohlhagen(S, K, T, rd, rf, sigma, "put")

    st.divider()
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Prime Call (Pips)", f"{call['price']*10000:.1f}")
    g2.metric("Prime Put (Pips)", f"{put['price']*10000:.1f}")
    g3.metric("Delta Call", f"{call['delta']:+.4f}")
    g4.metric("Gamma (Call/Put)", f"{call['gamma']:.5f}")
    g5.metric("Vega (1%)", f"{call['vega']:+.4f}")

    # ── GENERATION DU PAYOFF ──────────────────────────────────────────────────
    spots = np.linspace(S * 0.90, S * 1.10, 400)
    
    if strategy == "Long Call":
        pnl = (np.maximum(spots - K, 0) - call["price"]) * notional
    elif strategy == "Long Put":
        pnl = (np.maximum(K - spots, 0) - put["price"]) * notional
    elif strategy == "Straddle":
        pnl = (np.maximum(spots - K, 0) + np.maximum(K - spots, 0) - call["price"] - put["price"]) * notional
    elif strategy == "Risk Reversal (Zero Cost)":
        # Achat Call OTM (Strike + 2%) et Vente Put OTM (Strike - 2%)
        Kc = K * 1.02
        Kp = K * 0.98
        pc2 = garman_kohlhagen(S, Kc, T, rd, rf, sigma, "call")["price"]
        pp2 = garman_kohlhagen(S, Kp, T, rd, rf, sigma, "put")["price"]
        
        payoff_call = np.maximum(spots - Kc, 0) - pc2
        payoff_put_short = -(np.maximum(Kp - spots, 0) - pp2) # Vente du Put
        pnl = (payoff_call + payoff_put_short) * notional

    # ── GRAPHIQUE DU PAYOFF ───────────────────────────────────────────────────
    fig = go.Figure()
    
    # Remplissage vert (Gains) et rouge (Pertes)
    fig.add_trace(go.Scatter(x=spots, y=np.maximum(pnl, 0), fill="tozeroy", fillcolor="rgba(40,167,69,0.15)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=spots, y=np.minimum(pnl, 0), fill="tozeroy", fillcolor="rgba(220,53,69,0.15)", line=dict(width=0), showlegend=False))
    
    # Ligne du PnL (Bleue)
    fig.add_trace(go.Scatter(x=spots, y=pnl, mode="lines", name=strategy, line=dict(color="#0052cc", width=2.5), hovertemplate="Spot: %{x:.4f}<br>P&L: %{y:,.0f}<extra></extra>"))
    
    # Lignes de reperes
    fig.add_hline(y=0, line_color="#495057", line_width=1)
    fig.add_vline(x=S, line_dash="dash", line_color="#f59e0b", annotation_text=f"Spot Actuel", annotation_position="top left", annotation_font_color="#f59e0b")
    fig.add_vline(x=K, line_dash="dot", line_color="#495057", annotation_text=f"Strike", annotation_position="bottom right", annotation_font_color="#495057")
    
    if strategy == "Risk Reversal (Zero Cost)":
        fig.add_vline(x=Kc, line_dash="dot", line_color="#28a745", annotation_text=f"Call Strike ({Kc:.4f})")
        fig.add_vline(x=Kp, line_dash="dot", line_color="#dc3545", annotation_text=f"Put Strike ({Kp:.4f})")

    # --- LE FIX EST ICI : Transparence et couleurs claires ---
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(color="#495057", gridcolor="#e9ecef", title="Prix du Sous-Jacent à l'Echéance (Spot)"),
        yaxis=dict(color="#495057", gridcolor="#e9ecef", title="P&L Net (Devise de Cotation)"),
        hovermode="x unified"
    )
    # ---------------------------------------------------------
    st.plotly_chart(fig, use_container_width=True)

    # ── ANALYSE DES RISQUES ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Gain Maximum Théorique", "Illimité" if strategy in ["Long Call", "Straddle", "Risk Reversal (Zero Cost)"] else f"{np.max(pnl):,.0f}")
    c2.metric("Perte Maximum Théorique", f"{np.min(pnl):,.0f}" if strategy != "Risk Reversal (Zero Cost)" else "Illimitée")
    
    # Calcul des points morts (Break-even)
    zero_crossings = np.where(np.diff(np.signbit(pnl)))[0]
    break_evens = [spots[i] for i in zero_crossings]
    c3.metric("Point(s) Mort(s)", " / ".join([f"{b:.4f}" for b in break_evens]) if break_evens else "N/A")
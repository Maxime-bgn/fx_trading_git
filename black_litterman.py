"""
black_litterman.py
------------------
Moteur Black-Litterman : fusionne prior historique + vues ML pour generer
les poids optimaux des 4 strategies (optimal_weights.csv).
"""

import logging
import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRADING_DAYS = 252
MIN_WEIGHT = 0.05       # plancher diversification
MAX_WEIGHT = 0.50       # plafond par strategie
VIEW_SCALE = 0.40       # excess_weight 10% -> vue +4% rendement annualise
OMEGA_SCALE = 2.0       # std 10% -> omega 0.20


# ─── Moteur BL ───────────────────────────────────────────────────────────────

def black_litterman_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tau: float = 0.05,
    P: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    Omega: Optional[np.ndarray] = None,
    max_w: float = MAX_WEIGHT,
) -> np.ndarray:
    """Poids optimaux BL via Bayes + max Sharpe."""
    n = len(mu)
    mu = np.asarray(mu).reshape(-1)
    Sigma = np.asarray(Sigma)
    fallback = np.full(n, 1.0 / n)

    if P is None or Q is None:
        P = np.eye(n)
        Q = mu.reshape(-1, 1)
    if Omega is None:
        Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))

    tauSigma = tau * Sigma
    middle = np.linalg.inv(P @ tauSigma @ P.T + Omega)
    mu_bl = mu.reshape(-1, 1) + tauSigma @ P.T @ middle @ (Q - P @ mu.reshape(-1, 1))
    mu_bl = mu_bl.flatten()

    def neg_sharpe(w):
        p_vol = float(np.sqrt(w @ Sigma @ w))
        return -(float(mu_bl @ w)) / p_vol if p_vol > 1e-8 else 0.0

    bounds = [(0.0, max_w)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(neg_sharpe, fallback, method="SLSQP", bounds=bounds, constraints=cons)

    if res.success:
        w_bl = np.clip(res.x, 0, None)
        if w_bl.sum() > 0:
            w_bl /= w_bl.sum()
        return w_bl
    return fallback


# ─── Generateur de vues ──────────────────────────────────────────────────────

def _views_from_ml(strat_names: List[str], strategy_weights_file: str) -> Optional[Tuple]:
    """Genere les vues BL depuis strategy_weights.csv (poids ML par paire)."""
    if not os.path.exists(strategy_weights_file):
        return None
    try:
        sw = pd.read_csv(strategy_weights_file)
        sw_cols = [c for c in strat_names if c in sw.columns]
        if not sw_cols:
            return None

        n = len(strat_names)
        equal_w = 1.0 / n
        avg_w = sw[sw_cols].mean()
        std_w = sw[sw_cols].std().fillna(0.02)

        P = np.eye(n)
        Q = np.zeros((n, 1))
        omega_diag = np.zeros(n)

        for i, s in enumerate(strat_names):
            excess = float(avg_w.get(s, equal_w)) - equal_w
            Q[i, 0] = excess * VIEW_SCALE
            omega_diag[i] = max(float(std_w.get(s, 0.02)) * OMEGA_SCALE, 1e-4)

        logging.info("Vues BL depuis ML : %s",
                     {s: f"{avg_w.get(s, equal_w):.2%}" for s in strat_names})
        return P, Q, np.diag(omega_diag)
    except Exception as exc:
        logging.warning("Echec lecture strategy_weights.csv (%s).", exc)
        return None


def _views_from_hurst(df_latest: pd.DataFrame, strat_names: List[str]) -> Tuple:
    """Fallback : vues basees sur Hurst et Vol moyens."""
    n = len(strat_names)
    avg_hurst = df_latest["Hurst"].mean() if "Hurst" in df_latest.columns else 0.5
    avg_vol = df_latest["Vol_20"].mean() if "Vol_20" in df_latest.columns else 0.01

    logging.info("Vues BL (fallback Hurst) - H=%.3f | Vol=%.3f", avg_hurst, avg_vol)

    views_P, views_Q, views_Omega = [], [], []

    def s_idx(name):
        return strat_names.index(name)

    if "Momentum" in strat_names and "TSMOM" in strat_names:
        p = np.zeros(n)
        p[s_idx("Momentum")] = 0.5
        p[s_idx("TSMOM")] = 0.5
        views_P.append(p)
        views_Q.append((avg_hurst - 0.5) * 0.30)
        views_Omega.append(0.05 / (abs(avg_hurst - 0.5) + 1e-4))

    if "Mean-Reversion" in strat_names:
        p = np.zeros(n)
        p[s_idx("Mean-Reversion")] = 1.0
        views_P.append(p)
        views_Q.append((0.5 - avg_hurst) * 0.30)
        views_Omega.append(0.05 / (abs(avg_hurst - 0.5) + 1e-4))

    if "Carry Trade" in strat_names:
        p = np.zeros(n)
        p[s_idx("Carry Trade")] = 1.0
        views_P.append(p)
        views_Q.append(0.10 if avg_vol < 0.01 else -0.05)
        views_Omega.append(max(avg_vol * 10, 1e-4))

    return (np.array(views_P),
            np.array(views_Q).reshape(-1, 1),
            np.diag(views_Omega))


def generate_views_from_regime(
    df_latest: pd.DataFrame,
    strat_names: List[str],
    strategy_weights_file: str = "strategy_weights.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vues BL : ML si dispo, sinon fallback Hurst."""
    result = _views_from_ml(strat_names, strategy_weights_file)
    if result is not None:
        return result
    return _views_from_hurst(df_latest, strat_names)


# ─── Integration haut niveau ─────────────────────────────────────────────────

def get_dynamic_bl_allocation(
    strat_returns: pd.DataFrame,
    latest_features: pd.DataFrame,
) -> pd.Series:
    """Allocation BL complete : prior + vues -> poids optimaux."""
    strat_names = strat_returns.columns.tolist()
    mu_prior = strat_returns.mean().values * TRADING_DAYS
    Sigma_prior = strat_returns.cov().values * TRADING_DAYS

    P, Q, Omega = generate_views_from_regime(latest_features, strat_names)

    return pd.Series(
        black_litterman_weights(mu_prior, Sigma_prior, tau=0.05, P=P, Q=Q, Omega=Omega),
        index=strat_names,
    )


# ─── Point d'entree pipeline ─────────────────────────────────────────────────

if __name__ == "__main__":
    RETURNS_FILE = "strategies_returns.csv"
    OUTPUT_FILE = "optimal_weights.csv"

    if not os.path.exists(RETURNS_FILE):
        logging.critical("Fichier manquant : %s", RETURNS_FILE)
        raise SystemExit(1)

    strat_returns = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True).fillna(0.0)
    logging.info("Returns charges : %d jours x %d strategies", *strat_returns.shape)

    allocation = get_dynamic_bl_allocation(strat_returns, pd.DataFrame())

    # Penaliser les strategies a Sharpe tres negatif, garder un plancher
    for strat in allocation.index:
        if strat in strat_returns.columns:
            s = strat_returns[strat].dropna()
            if s.std() > 0 and (s.mean() / s.std() * np.sqrt(TRADING_DAYS)) < -0.30:
                allocation[strat] = 0.0

    # Plancher minimum par strategie
    for strat in allocation.index:
        allocation[strat] = max(float(allocation[strat]), MIN_WEIGHT)
    # Cap TSMOM (Sharpe historique faible)
    if "TSMOM" in allocation.index:
        allocation["TSMOM"] = min(float(allocation["TSMOM"]), 0.20)
    if allocation.sum() > 0:
        allocation = allocation / allocation.sum()

    allocation.to_csv(OUTPUT_FILE, header=False)
    logging.info("Allocation BL sauvegardee -> %s", OUTPUT_FILE)

    print("\n" + "=" * 45)
    print("  ALLOCATION BLACK-LITTERMAN - LIVE")
    print("=" * 45)
    for name, w in allocation.items():
        bar = "#" * int(w * 40)
        print(f"  {name:20s} {w:6.1%}  {bar}")
    print("=" * 45)

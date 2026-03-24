"""
portfolio_utils.py
------------------
Optimisation Markowitz (Max Sharpe / Min Variance) sur les rendements FX.
Génère les poids optimaux par paire et les métriques de risque du portefeuille.

Place dans le pipeline : après backtest_strategies.py, avant main_trading.py.
Outputs :
  - markowitz_weights.csv   : poids optimaux par paire
  - markowitz_metrics.csv   : métriques de risque du portefeuille optimal
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

TRADING_DAYS = 252
MIN_WEIGHT = 0.0       # Poids minimum par paire (0 = on peut exclure)
MAX_WEIGHT = 0.35      # Max 35% sur une seule paire
RISK_FREE_RATE = 0.02  # ~2% annuel (taux sans risque — conservateur pour FX)

STRATEGIES = ["Momentum", "Mean-Reversion", "TSMOM", "Carry Trade"]

RETURNS_BY_PAIR_FILE = "strategies_returns_by_pair.csv"
SELECTED_PAIRS_FILE  = "selected_pairs.csv"
WEIGHTS_FILE         = "optimal_weights.csv"        # poids BL par stratégie
STRAT_WEIGHTS_FILE   = "strategy_weights.csv"       # poids ML par paire
OUTPUT_WEIGHTS       = "markowitz_weights.csv"
OUTPUT_METRICS       = "markowitz_metrics.csv"


# ─── Calcul des rendements par paire ─────────────────────────────────────────

def compute_pair_returns() -> pd.DataFrame:
    """
    Calcule le rendement quotidien net par paire en agrégeant les 4 stratégies
    pondérées par les poids ML par paire (strategy_weights.csv).

    Priorité :
      1. strategy_weights.csv (poids ML optimaux par paire — couche 2)
      2. optimal_weights.csv (poids BL globaux — fallback)
      3. Équipondération

    Retourne un DataFrame (date x paires) de rendements quotidiens.
    """
    if not os.path.exists(RETURNS_BY_PAIR_FILE):
        raise FileNotFoundError(f"{RETURNS_BY_PAIR_FILE} introuvable.")

    rp = pd.read_csv(RETURNS_BY_PAIR_FILE, parse_dates=["Date"])

    # 1. Charger les poids ML par paire (strategy_weights.csv)
    ml_pair_weights = {}
    if os.path.exists(STRAT_WEIGHTS_FILE):
        sw = pd.read_csv(STRAT_WEIGHTS_FILE)
        if "Pair" in sw.columns:
            for _, row in sw.iterrows():
                pair = row["Pair"]
                ml_pair_weights[pair] = {
                    s: row.get(s, 0.0) for s in STRATEGIES if s in row.index
                }

    # 2. Fallback : poids BL globaux
    bl_weights = {}
    if os.path.exists(WEIGHTS_FILE):
        w = pd.read_csv(WEIGHTS_FILE, header=None, index_col=0)[1]
        total = w.sum()
        bl_weights = (w / total).to_dict() if total > 0 else {}

    pairs = rp["Pair"].unique()
    strategies = rp["Strategy"].unique()
    default_w = {s: 1.0 / len(strategies) for s in strategies}

    # Pivoter : une colonne par paire, rendement pondéré par ML weights
    records = []
    for pair in pairs:
        pair_data = rp[rp["Pair"] == pair]
        pair_pivot = pair_data.pivot_table(
            index="Date", columns="Strategy", values="Return"
        ).fillna(0.0)

        # Choix des poids : ML pair > BL global > équipondéré
        w = ml_pair_weights.get(pair, bl_weights if bl_weights else default_w)

        weighted_ret = sum(
            pair_pivot[s] * w.get(s, 0.0)
            for s in pair_pivot.columns if s in w
        )
        records.append(weighted_ret.rename(pair))

    returns_df = pd.concat(records, axis=1).sort_index().dropna(how="all")
    return returns_df


# ─── Fonctions d'optimisation ────────────────────────────────────────────────

def _portfolio_stats(weights, mean_returns, cov_matrix):
    """Retourne (rendement annuel, volatilité annuelle, Sharpe)."""
    ret = np.dot(weights, mean_returns) * TRADING_DAYS
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * TRADING_DAYS, weights)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def optimize_max_sharpe(mean_returns, cov_matrix, n_assets):
    """Maximise le ratio de Sharpe (minimise -Sharpe)."""
    def neg_sharpe(w):
        r, v, _ = _portfolio_stats(w, mean_returns, cov_matrix)
        return -(r - RISK_FREE_RATE) / v if v > 1e-10 else 0.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)

    result = minimize(neg_sharpe, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})
    return result.x if result.success else x0


def optimize_min_variance(mean_returns, cov_matrix, n_assets):
    """Minimise la variance du portefeuille."""
    def portfolio_var(w):
        return np.dot(w.T, np.dot(cov_matrix * TRADING_DAYS, w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)

    result = minimize(portfolio_var, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})
    return result.x if result.success else x0


def compute_efficient_frontier(mean_returns, cov_matrix, n_assets, n_points=30):
    """Calcule la frontière efficiente pour visualisation."""
    # Bornes de rendement
    w_min = optimize_min_variance(mean_returns, cov_matrix, n_assets)
    w_max = optimize_max_sharpe(mean_returns, cov_matrix, n_assets)
    ret_min = np.dot(w_min, mean_returns) * TRADING_DAYS
    ret_max = np.dot(w_max, mean_returns) * TRADING_DAYS

    target_rets = np.linspace(ret_min, ret_max * 1.1, n_points)
    frontier = []

    for target in target_rets:
        def port_var(w):
            return np.dot(w.T, np.dot(cov_matrix * TRADING_DAYS, w))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) * TRADING_DAYS - t},
        ]
        bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n_assets
        x0 = np.full(n_assets, 1.0 / n_assets)

        result = minimize(port_var, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 500})
        if result.success:
            vol = np.sqrt(result.fun)
            frontier.append({"Return": target * 100, "Volatility": vol * 100})

    return pd.DataFrame(frontier)


# ─── Métriques de risque ─────────────────────────────────────────────────────

def compute_risk_metrics(returns_series: pd.Series) -> dict:
    """Calcule les métriques de risque pour une série de rendements de portefeuille."""
    r = returns_series.dropna()
    if len(r) < 20:
        return {}

    cum = (1 + r).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()

    # CAGR (compound annual growth rate) = vrai rendement gagné
    years = len(r) / TRADING_DAYS
    total_return = cum.iloc[-1]
    ann_ret = total_return ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - RISK_FREE_RATE) / downside if downside > 0 else 0.0

    # VaR et CVaR (95%)
    var_95 = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()

    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    return {
        "Ann. Return (%)":   round(ann_ret * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio":      round(sharpe, 3),
        "Sortino Ratio":     round(sortino, 3),
        "Max Drawdown (%)":  round(max_dd * 100, 2),
        "Calmar Ratio":      round(calmar, 3),
        "VaR 95% (%)":       round(var_95 * 100, 4),
        "CVaR 95% (%)":      round(cvar_95 * 100, 4) if not np.isnan(cvar_95) else "N/A",
    }


# ─── Optimisation complète ───────────────────────────────────────────────────

def run_markowitz() -> dict:
    """
    Point d'entrée : calcule les poids Markowitz Max Sharpe et Min Variance.
    Exporte markowitz_weights.csv et markowitz_metrics.csv.
    Retourne un dict avec les résultats pour le dashboard.
    """
    logger.info("=== MARKOWITZ OPTIMIZATION ===")

    # 1. Rendements par paire
    returns_df = compute_pair_returns()
    pairs = returns_df.columns.tolist()
    n = len(pairs)
    logger.info("Paires : %s (%d)", pairs, n)

    if n < 2:
        logger.warning("Moins de 2 paires — Markowitz impossible.")
        return {}

    # 2. Statistiques
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    logger.info("Rendements annualisés par paire :")
    for i, p in enumerate(pairs):
        logger.info("  %s: %.2f%%", p, mean_returns[i] * TRADING_DAYS * 100)

    # 3. Optimisation Max Sharpe
    w_sharpe = optimize_max_sharpe(mean_returns, cov_matrix, n)
    ret_s, vol_s, sh_s = _portfolio_stats(w_sharpe, mean_returns, cov_matrix)
    logger.info("MAX SHARPE  — Ret: %.2f%% | Vol: %.2f%% | Sharpe: %.3f",
                ret_s * 100, vol_s * 100, sh_s)

    # 4. Optimisation Min Variance
    w_minvar = optimize_min_variance(mean_returns, cov_matrix, n)
    ret_mv, vol_mv, sh_mv = _portfolio_stats(w_minvar, mean_returns, cov_matrix)
    logger.info("MIN VARIANCE — Ret: %.2f%% | Vol: %.2f%% | Sharpe: %.3f",
                ret_mv * 100, vol_mv * 100, sh_mv)

    # 5. Portefeuille final : blend 80% Max Sharpe + 20% Min Variance
    #    (orienté performance, légère stabilisation)
    w_final = 0.8 * w_sharpe + 0.2 * w_minvar
    w_final = w_final / w_final.sum()  # re-normaliser
    ret_f, vol_f, sh_f = _portfolio_stats(w_final, mean_returns, cov_matrix)
    logger.info("BLEND 60/40 — Ret: %.2f%% | Vol: %.2f%% | Sharpe: %.3f",
                ret_f * 100, vol_f * 100, sh_f)

    # 6. Rendements historiques du portefeuille optimal
    port_returns = returns_df.dot(w_final)
    metrics = compute_risk_metrics(port_returns)

    # 7. Export poids
    weights_df = pd.DataFrame({
        "Pair": pairs,
        "Max_Sharpe": w_sharpe,
        "Min_Variance": w_minvar,
        "Final_Weight": w_final,
    })
    weights_df.to_csv(OUTPUT_WEIGHTS, index=False)
    logger.info("Poids exportés -> %s", OUTPUT_WEIGHTS)

    # 8. Export métriques
    metrics_rows = []
    for label, w in [("Max_Sharpe", w_sharpe), ("Min_Variance", w_minvar), ("Blend_60_40", w_final)]:
        pr = returns_df.dot(w)
        m = compute_risk_metrics(pr)
        m["Portfolio"] = label
        metrics_rows.append(m)
    pd.DataFrame(metrics_rows).to_csv(OUTPUT_METRICS, index=False)

    # Résumé console
    print(f"\n{'='*60}")
    print("MARKOWITZ PORTFOLIO OPTIMIZATION")
    print(f"{'='*60}")
    print(weights_df.to_string(index=False, float_format="%.4f"))
    print(f"\nBlend 60/40 — Return: {ret_f*100:.2f}% | Vol: {vol_f*100:.2f}% | Sharpe: {sh_f:.3f}")
    print(f"{'='*60}\n")

    return {
        "weights": weights_df,
        "metrics": metrics,
        "port_returns": port_returns,
        "pairs": pairs,
        "w_final": w_final,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    run_markowitz()

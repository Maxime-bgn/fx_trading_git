"""
risk_manager.py
---------------
Module de gestion du risque FX — version pipeline.

Expose run(cfg) qui :
  - Charge les rendements (strategies_returns.csv ou fx_returns.csv)
  - Calcule VaR/ES roulants, drawdown, circuit breakers, stress de correlation
  - Exporte dans risk_outputs/ : risk_report_daily.csv, risk_summary.csv, risk_stress_tests.csv
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    var_window: int = 252
    corr_window: int = 63
    alpha: float = 0.95
    corr_abs_mean_threshold: float = 0.35
    daily_loss_cut: float = -0.02
    daily_loss_flat: float = -0.04
    dd_exposure_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.00, 1.00), (0.05, 0.80), (0.10, 0.60), (0.15, 0.40), (0.20, 0.00),
    ])
    cost_bps_per_turnover: float = 2.0
    stress_n_worst_days: int = 20
    shock_sigma_levels: List[int] = field(default_factory=lambda: [2, 3])
    returns_path: str = "fx_returns.csv"
    weights_path: str = "weights.csv"
    out_dir: str = "risk_outputs"


DEFAULT_CONFIG = RiskConfig()


# ─── Chargement ──────────────────────────────────────────────────────────────

def _load_weights(path: str, columns: list, index: pd.DatetimeIndex) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    w = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True).sort_index()
    w = w.reindex(columns=columns).reindex(index=index).ffill().fillna(0.0)
    s = w.sum(axis=1).replace(0.0, np.nan)
    return w.div(s, axis=0).fillna(0.0)


# ─── Metriques ───────────────────────────────────────────────────────────────

def _rolling_var_es(r: pd.Series, window: int, alpha: float) -> pd.DataFrame:
    """VaR et ES historiques roulants via pandas rolling."""
    q_level = 1.0 - alpha

    var_hist = r.rolling(window, min_periods=max(30, window // 3)).quantile(q_level).mul(-1)
    var_hist.name = "VaR_hist"

    def _es_func(w):
        q = np.quantile(w, q_level)
        tail = w[w <= q]
        return -tail.mean() if len(tail) > 0 else np.nan

    es_hist = r.rolling(window, min_periods=max(30, window // 3)).apply(_es_func, raw=True)
    es_hist.name = "ES_hist"

    return pd.concat([var_hist, es_hist], axis=1)


def _var_breaches(r: pd.Series, var_series: pd.Series) -> pd.Series:
    breach = (r < -var_series).astype(float)
    breach[var_series.isna()] = np.nan
    breach.name = "VaR_breach"
    return breach


def _equity_and_drawdown(r_log: pd.Series) -> Tuple[pd.Series, pd.Series]:
    equity = np.exp(r_log.cumsum())
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return equity, dd


def _exposure_from_dd(dd_mag: float, tiers: list) -> float:
    expo = tiers[0][1]
    for thr, m in tiers:
        if dd_mag >= thr:
            expo = m
    return float(expo)


def _risk_controller(port_net: pd.Series, cfg: RiskConfig) -> pd.DataFrame:
    equity, dd = _equity_and_drawdown(port_net)
    dd_mag = (-dd).clip(lower=0.0)

    expo_dd = dd_mag.apply(lambda x: _exposure_from_dd(float(x), cfg.dd_exposure_tiers))

    simple = np.exp(port_net) - 1.0
    expo_cb = pd.Series(1.0, index=simple.index)
    expo_cb[simple <= cfg.daily_loss_flat] = 0.0
    expo_cb[(simple <= cfg.daily_loss_cut) & (simple > cfg.daily_loss_flat)] = 0.5

    expo_final = np.minimum(expo_dd, expo_cb)

    return pd.DataFrame({
        "equity": equity, "drawdown": dd, "drawdown_mag": dd_mag,
        "exposure_dd": expo_dd, "exposure_cb": expo_cb, "exposure_final": expo_final,
    })


def _rolling_corr_stress(returns_df: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    """Stress de correlation roulant."""
    min_periods = max(30, window // 3)
    corr_vals = []
    for i in range(len(returns_df)):
        if i + 1 < window:
            corr_vals.append(np.nan)
            continue
        w = returns_df.iloc[i + 1 - window: i + 1]
        w = w.dropna(axis=1, thresh=window // 2).dropna(how="any")
        if w.shape[0] < min_periods or w.shape[1] < 2:
            corr_vals.append(np.nan)
            continue
        corr = w.corr().values
        mask = ~np.eye(corr.shape[0], dtype=bool)
        corr_vals.append(float(np.nanmean(np.abs(corr[mask]))))

    corr_series = pd.Series(corr_vals, index=returns_df.index, name="corr_abs_mean")
    stress_flag = (corr_series > threshold).astype(float)
    stress_flag[corr_series.isna()] = np.nan
    stress_flag.name = "corr_stress_flag"
    return pd.concat([corr_series, stress_flag], axis=1)


def _stress_tests(port_simple: pd.Series, n_worst: int, shock_sigmas: list) -> pd.DataFrame:
    r = port_simple.dropna()
    if len(r) < 30:
        return pd.DataFrame({"stress_loss": [np.nan]}, index=["insufficient_data"])
    worst_mean = -r.nsmallest(min(n_worst, len(r))).mean()
    sigma = r.std()
    rows = {"worst_days_mean_loss": float(worst_mean),
            **{f"shock_-{k}sigma": float(k * sigma) for k in shock_sigmas}}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["stress_loss"])


# ─── Point d'entree ──────────────────────────────────────────────────────────

def run(cfg=None) -> None:
    """Lance le calcul de risque complet et exporte les CSV."""
    if cfg is None:
        cfg = DEFAULT_CONFIG
    # Support ancien format dict
    if isinstance(cfg, dict):
        cfg = RiskConfig(**{k: v for k, v in cfg.items() if k in RiskConfig.__dataclass_fields__})

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Trouver la source de rendements
    returns_path = cfg.returns_path
    if not os.path.exists(returns_path):
        alt = "strategies_returns.csv"
        if os.path.exists(alt):
            logger.warning("'%s' introuvable - fallback sur '%s'.", returns_path, alt)
            returns_path = alt
        else:
            logger.error("Aucune source de rendements disponible.")
            return

    logger.info("Chargement des rendements depuis '%s'.", returns_path)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True).sort_index()
    returns_df = returns_df.dropna(axis=1, thresh=200).dropna()

    w = _load_weights(cfg.weights_path, list(returns_df.columns), returns_df.index)
    if w is None:
        logger.warning("weights.csv absent - ponderation egale utilisee.")
        w = pd.DataFrame(1.0 / len(returns_df.columns),
                         index=returns_df.index, columns=returns_df.columns)

    # Rendements portefeuille
    r_aligned = returns_df.reindex(index=w.index, columns=w.columns)
    port_gross = (r_aligned * w).sum(axis=1).rename("port_log_returns_gross")
    to = (w.diff().abs().sum(axis=1) / 2.0).rename("turnover")
    cost = (to * (cfg.cost_bps_per_turnover / 10000.0)).rename("cost")
    port_net = (port_gross - cost.fillna(0.0)).rename("port_log_returns_net")

    # VaR / ES
    logger.info("Calcul VaR/ES roulants (fenetre=%d, alpha=%.2f).", cfg.var_window, cfg.alpha)
    var_es = _rolling_var_es(port_net, cfg.var_window, cfg.alpha)
    breaches = _var_breaches(port_net, var_es["VaR_hist"])

    # Risk controller
    logger.info("Calcul du risk controller (drawdown + circuit breakers).")
    controller = _risk_controller(port_net, cfg)

    # Correlation stress
    logger.info("Calcul du stress de correlation (fenetre=%d).", cfg.corr_window)
    corr_stress = _rolling_corr_stress(returns_df, cfg.corr_window, cfg.corr_abs_mean_threshold)

    # Export daily
    daily = pd.concat([port_gross, port_net, cost, to, var_es, breaches, controller, corr_stress], axis=1)
    daily.to_csv(out_dir / "risk_report_daily.csv", sep=";", decimal=",")
    logger.info("Rapport journalier exporte (%d lignes).", len(daily))

    # Resume
    breach_n = breaches.dropna()
    summary_rows = {
        "max_drawdown":          float(daily["drawdown"].min()) if "drawdown" in daily else np.nan,
        "corr_abs_mean_avg":     float(np.nanmean(daily["corr_abs_mean"].values)),
        "corr_stress_days":      int(np.nansum(daily["corr_stress_flag"].values)),
        "exposure_avg":          float(np.nanmean(daily["exposure_final"].values)),
        "stop_trading_days":     int(np.nansum((daily["exposure_final"] == 0.0).astype(float).values)),
        "cost_total":            float(np.nansum(daily["cost"].values)),
        "turnover_avg":          float(np.nanmean(daily["turnover"].values)),
        "VaR_backtest_n_obs":    int(len(breach_n)),
        "VaR_backtest_n_breaches": int(breach_n.sum()),
        "VaR_backtest_breach_rate": float(breach_n.mean()),
        "VaR_backtest_expected_rate": float(1.0 - cfg.alpha),
    }
    pd.DataFrame.from_dict(summary_rows, orient="index", columns=["value"]).to_csv(
        out_dir / "risk_summary.csv", sep=";", decimal=",")
    logger.info("Resume exporte.")

    # Stress tests
    stress = _stress_tests(np.exp(port_net) - 1.0, cfg.stress_n_worst_days, cfg.shock_sigma_levels)
    stress.to_csv(out_dir / "risk_stress_tests.csv", sep=";", decimal=",")
    logger.info("Stress tests exportes.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run()

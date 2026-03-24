"""
Portfolio Optimizer - ML-Based Dynamic Allocation
==================================================
Pipeline : features multi-timeframe -> targets Markowitz (Ledoit-Wolf) ->
walk-forward validation -> stacking RF + MLP -> optimal_weights.csv
"""

import os
import logging
import warnings
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    input_file: str = "strategies_returns.csv"
    output_file: str = "optimal_weights.csv"
    max_weight: float = 0.50
    min_weight: float = 0.00
    risk_free_rate: float = 0.00
    trading_days: int = 252
    feature_windows: List[int] = field(default_factory=lambda: [20, 40, 60])
    autocorr_lags: List[int] = field(default_factory=lambda: [1, 5, 10])
    window_target: int = 20
    min_train_samples: int = 100
    n_folds: int = 5
    rf_n_estimators: int = 150
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 10
    rf_random_state: int = 42
    mlp_hidden_layers: Tuple[int, ...] = (64, 32)
    mlp_max_iter: int = 1000
    mlp_random_state: int = 42
    mlp_early_stopping: bool = True
    ridge_alpha: float = 1.0


# ─── Markowitz robuste (targets) ─────────────────────────────────────────────

def get_optimal_weights(returns: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Max Sharpe sous contraintes via Ledoit-Wolf."""
    n = returns.shape[1]
    fallback = np.full(n, 1.0 / n)
    if len(returns) < 5:
        return fallback

    mu = returns.mean().values * cfg.trading_days
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns.values)
    sigma = lw.covariance_ * cfg.trading_days

    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
        return fallback

    def neg_sharpe(w):
        p_vol = np.sqrt(w @ sigma @ w)
        return -(mu @ w - cfg.risk_free_rate) / p_vol if p_vol > 1e-8 else 0.0

    try:
        res = minimize(neg_sharpe, fallback, method="SLSQP",
                       bounds=[(cfg.min_weight, cfg.max_weight)] * n,
                       constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                       options={"ftol": 1e-9, "maxiter": 1000})
        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, cfg.min_weight, cfg.max_weight)
            w /= w.sum()
            return w
    except Exception:
        pass
    return fallback


# ─── Features multi-timeframe ────────────────────────────────────────────────

def _compute_window_features(window: pd.DataFrame, cfg: Config, n_strats: int) -> np.ndarray:
    mu = (window.mean() * cfg.trading_days).values
    vol = (window.std() * np.sqrt(cfg.trading_days)).values
    autocorrs = [
        window.apply(lambda s: s.autocorr(lag) if s.std() > 1e-10 else 0.0).fillna(0.0).values
        for lag in cfg.autocorr_lags
    ]
    corr = window.corr().fillna(0.0).values
    cross_corr = (corr.sum(axis=1) - 1.0) / max(n_strats - 1, 1)
    skew = window.skew().fillna(0.0).values
    max_dd = window.apply(lambda s: (s.cumsum() - s.cumsum().cummax()).min()).fillna(0.0).values
    return np.concatenate([mu, vol, cross_corr, skew, max_dd] + autocorrs)


def _build_features_for_date(returns: pd.DataFrame, t: int, cfg: Config) -> Optional[np.ndarray]:
    n_strats = returns.shape[1]
    parts = []
    for w in cfg.feature_windows:
        if t - w < 0:
            return None
        parts.append(_compute_window_features(returns.iloc[t - w: t], cfg, n_strats))
    vec = np.concatenate(parts)
    return vec if np.all(np.isfinite(vec)) else None


def build_ml_dataset(returns: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit (X, Y, x_latest) sans data leakage."""
    logging.info("Construction du dataset MTF (fenetres : %s)...", cfg.feature_windows)
    n = len(returns)
    max_window = max(cfg.feature_windows)
    X_list, Y_list = [], []

    for t in range(max_window, n - cfg.window_target):
        vec = _build_features_for_date(returns, t, cfg)
        if vec is None:
            continue
        target = get_optimal_weights(returns.iloc[t: t + cfg.window_target], cfg)
        if np.all(np.isfinite(target)):
            X_list.append(vec)
            Y_list.append(target)

    X, Y = np.array(X_list), np.array(Y_list)
    x_latest = _build_features_for_date(returns, n, cfg)
    if x_latest is None:
        x_latest = X[-1] if len(X) > 0 else np.zeros(X.shape[1] if len(X) > 0 else 1)
    x_latest = x_latest.reshape(1, -1)
    logging.info("Dataset : %d samples, %d features, %d assets.", len(X), X.shape[1], Y.shape[1])
    return X, Y, x_latest


# ─── Helpers modeles ─────────────────────────────────────────────────────────

def _make_rf(cfg: Config):
    return RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf, random_state=cfg.rf_random_state)


def _make_mlp(cfg: Config):
    return MLPRegressor(
        hidden_layer_sizes=cfg.mlp_hidden_layers, max_iter=cfg.mlp_max_iter,
        random_state=cfg.mlp_random_state, early_stopping=cfg.mlp_early_stopping)


def _fit_predict(model, X_train, Y_train, X_test):
    model.fit(X_train, Y_train)
    return model.predict(X_test)


# ─── Walk-forward validation ─────────────────────────────────────────────────

def walk_forward_validate(X: np.ndarray, Y: np.ndarray, cfg: Config) -> Tuple[float, float]:
    """Validation walk-forward : toujours train sur passe, eval sur futur."""
    logging.info("Walk-forward validation (%d folds)...", cfg.n_folds)
    n = len(X)
    fold_size = n // (cfg.n_folds + 1)
    mae_rf_list, mae_mlp_list = [], []

    for fold in range(cfg.n_folds):
        train_end = fold_size * (fold + 1)
        test_end = min(train_end + fold_size, n)
        if train_end < cfg.min_train_samples or train_end >= test_end:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[:train_end])
        X_te = scaler.transform(X[train_end:test_end])
        Y_tr, Y_te = Y[:train_end], Y[train_end:test_end]

        preds_rf = _fit_predict(_make_rf(cfg), X_tr, Y_tr, X_te)
        preds_mlp = _fit_predict(_make_mlp(cfg), X_tr, Y_tr, X_te)

        mae_rf_list.append(mean_absolute_error(Y_te, preds_rf))
        mae_mlp_list.append(mean_absolute_error(Y_te, preds_mlp))
        logging.info("  Fold %d/%d - MAE RF: %.4f | MAE MLP: %.4f",
                     fold + 1, cfg.n_folds, mae_rf_list[-1], mae_mlp_list[-1])

    mae_rf = float(np.mean(mae_rf_list)) if mae_rf_list else 0.5
    mae_mlp = float(np.mean(mae_mlp_list)) if mae_mlp_list else 0.5
    logging.info("MAE moyenne - RF: %.4f | MLP: %.4f", mae_rf, mae_mlp)
    return mae_rf, mae_mlp


# ─── Prediction finale ───────────────────────────────────────────────────────

def train_and_predict(
    X: np.ndarray, Y: np.ndarray, x_latest: np.ndarray,
    strat_names: List[str], cfg: Config,
    mae_rf: float, mae_mlp: float,
) -> pd.Series:
    """Entraine RF + MLP, combine via stacking pondere par MAE."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    x_live = scaler.transform(x_latest)

    pred_rf = _fit_predict(_make_rf(cfg), X_s, Y, x_live)[0]
    pred_mlp = _fit_predict(_make_mlp(cfg), X_s, Y, x_live)[0]

    # Stacking inverse-MAE
    inv_rf, inv_mlp = 1.0 / (mae_rf + 1e-8), 1.0 / (mae_mlp + 1e-8)
    w_rf = inv_rf / (inv_rf + inv_mlp)
    logging.info("Poids stacking - RF: %.2f | MLP: %.2f", w_rf, 1 - w_rf)

    final = np.clip(w_rf * pred_rf + (1 - w_rf) * pred_mlp, cfg.min_weight, cfg.max_weight)
    total = final.sum()
    if total > 1e-8:
        final /= total
    else:
        final = np.full(len(strat_names), 1.0 / len(strat_names))
    return pd.Series(final, index=strat_names)


# ─── Metriques ───────────────────────────────────────────────────────────────

def compute_benchmark_metrics(returns: pd.DataFrame, allocation: pd.Series, cfg: Config):
    eq_w = np.full(len(allocation), 1.0 / len(allocation))
    ml_ret = (returns * allocation.values).sum(axis=1)
    eq_ret = (returns * eq_w).sum(axis=1)

    def sharpe(r):
        ann_vol = r.std() * np.sqrt(cfg.trading_days)
        return r.mean() * cfg.trading_days / ann_vol if ann_vol > 1e-8 else 0.0

    def max_dd(r):
        cum = (1 + r).cumprod()
        return float((cum / cum.cummax() - 1).min())

    print(f"\n{'='*50}")
    print(f"  METRIQUES HISTORIQUES")
    print(f"{'='*50}")
    print(f"{'':30s} {'ML':>8} {'Equal':>8}")
    print(f"{'Sharpe annualise':30s} {sharpe(ml_ret):>8.3f} {sharpe(eq_ret):>8.3f}")
    print(f"{'Max Drawdown':30s} {max_dd(ml_ret):>8.2%} {max_dd(eq_ret):>8.2%}")
    print(f"{'='*50}")


# ─── Validation donnees ──────────────────────────────────────────────────────

def validate_and_load(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.input_file):
        raise FileNotFoundError(f"Fichier introuvable : '{cfg.input_file}'")
    returns = pd.read_csv(cfg.input_file, index_col=0, parse_dates=True)
    if returns.empty:
        raise ValueError("Le fichier de rendements est vide.")
    if returns.isnull().any().any():
        n_miss = returns.isnull().sum().sum()
        logging.warning("%d valeurs manquantes - remplissage par 0.", n_miss)
        returns = returns.fillna(0.0)
    if not np.isfinite(returns.values).all():
        raise ValueError("Le fichier contient des valeurs infinies.")
    logging.info("Donnees : %d jours, %d strategies (%s -> %s).",
                 len(returns), returns.shape[1], returns.index[0].date(), returns.index[-1].date())
    return returns


# ─── Point d'entree ──────────────────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Portfolio Optimizer")
    parser.add_argument("--input", default="strategies_returns.csv")
    parser.add_argument("--output", default="optimal_weights.csv")
    parser.add_argument("--max-weight", type=float, default=0.50)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    return Config(input_file=args.input, output_file=args.output,
                  max_weight=args.max_weight, n_folds=args.folds)


def optimize_strategy_weights(cfg: Optional[Config] = None) -> Optional[pd.Series]:
    if cfg is None:
        cfg = parse_args()
    try:
        returns = validate_and_load(cfg)
        strat_names = returns.columns.tolist()
        X, Y, x_latest = build_ml_dataset(returns, cfg)

        if len(X) < cfg.min_train_samples:
            logging.warning("Historique insuffisant (%d < %d). Equiponderation.", len(X), cfg.min_train_samples)
            allocation = pd.Series(np.full(len(strat_names), 1.0 / len(strat_names)), index=strat_names)
        else:
            mae_rf, mae_mlp = walk_forward_validate(X, Y, cfg)
            allocation = train_and_predict(X, Y, x_latest, strat_names, cfg, mae_rf, mae_mlp)

        # Zero out negative-Sharpe strategies
        for strat in allocation.index:
            if strat in returns.columns:
                s = returns[strat].dropna()
                if s.std() > 0 and s.mean() / s.std() * np.sqrt(cfg.trading_days) < 0:
                    allocation[strat] = 0.0
        if allocation.sum() > 0:
            allocation = allocation / allocation.sum()

        allocation.to_csv(cfg.output_file, header=False)
        logging.info("Allocation sauvegardee -> '%s'.", cfg.output_file)

        print(f"\n{'='*50}")
        print("  ALLOCATION ML DYNAMIQUE")
        print(f"{'='*50}")
        for name, w in allocation.items():
            print(f"  {name:20s} {w:6.1%}  {'#' * int(w * 40)}")
        print(f"{'='*50}")

        compute_benchmark_metrics(returns, allocation, cfg)
        return allocation

    except (FileNotFoundError, ValueError) as e:
        logging.critical("%s", e)
    except Exception as e:
        logging.critical("Erreur inattendue : %s", e, exc_info=True)
    return None


if __name__ == "__main__":
    optimize_strategy_weights()

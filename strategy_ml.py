"""
strategy_ml.py
--------------
Couche 2 du pipeline ML : pour chaque paire active, predit les poids optimaux
des 4 strategies en fonction du regime de marche courant.

Architecture :
  Target   : Sharpe rolling 20j normalise en poids (softmax positifs)
  Features : Hurst, Vol_Regime, Trend, Mom_20, BB_Width, Dist_MA20, Vol_20
  Modele   : LightGBM regression (1 par strategie), fallback Ridge
  Validation: Walk-forward, IC entre poids predits et Sharpe realise
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("lightgbm non installe - fallback Ridge.")
    LGBM_AVAILABLE = False
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

# ─── Parametres ──────────────────────────────────────────────────────────────

STRAT_RETURNS_FILE  = "strategies_returns_by_pair.csv"
FEATURES_FILE       = "dataset_ml_ready.csv"
SELECTED_PAIRS_FILE = "selected_pairs.csv"
OUTPUT_FILE         = "strategy_weights.csv"
METRICS_FILE        = "strategy_ml_metrics.csv"

STRATEGIES = ["Momentum", "Mean-Reversion", "TSMOM", "Carry Trade"]

SHARPE_WINDOW  = 20
PURGE_GAP_DAYS = 5
MIN_OBS        = 80

REGIME_FEATURES = [
    "Hurst", "Vol_Regime", "Trend", "Mom_20",
    "BB_Width", "Dist_MA20", "Vol_20",
]

LGBM_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "num_leaves": 15, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 15, "lambda_l1": 0.5, "lambda_l2": 0.5,
    "verbose": -1, "random_state": 42,
}
LGBM_ROUNDS = 300
LGBM_EARLY_STOP = 30


# ─── Targets ─────────────────────────────────────────────────────────────────

def compute_strategy_targets(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Sharpe rolling normalise en poids via softmax des positifs."""
    sharpe_df = pd.DataFrame(index=returns_wide.index, columns=STRATEGIES, dtype=float)
    for strat in STRATEGIES:
        if strat not in returns_wide.columns:
            sharpe_df[strat] = np.nan
            continue
        r = returns_wide[strat]
        roll_mean = r.rolling(SHARPE_WINDOW).mean()
        roll_std = r.rolling(SHARPE_WINDOW).std().replace(0, np.nan)
        sharpe_df[strat] = (roll_mean / roll_std * np.sqrt(252)).fillna(0.0)

    # Softmax des positifs -> poids
    weights = sharpe_df.copy()
    for idx in weights.index:
        row = weights.loc[idx].fillna(0.0).values.astype(float)
        pos = np.maximum(row, 0.0)
        total = pos.sum()
        weights.loc[idx] = pos / total if total > 1e-8 else 1.0 / len(STRATEGIES)
    return weights


# ─── Model helpers ───────────────────────────────────────────────────────────

def _train_lgbm(X_train, y_train, X_val=None, y_val=None):
    """Entraine un LightGBM. Si val fourni, utilise early stopping."""
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = dict(LGBM_PARAMS)
    kwargs = {"num_boost_round": LGBM_ROUNDS, "callbacks": [lgb.log_evaluation(-1)]}
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        kwargs["valid_sets"] = [dval]
        kwargs["callbacks"].append(lgb.early_stopping(LGBM_EARLY_STOP, verbose=False))
    return lgb.train(params, dtrain, **kwargs)


def _train_ridge(X_train, y_train):
    """Entraine Ridge avec normalisation."""
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)
    model.fit(scaler.fit_transform(X_train), y_train)
    return model, scaler


def _predict(model, X):
    """Prediction unifiee LGBM ou Ridge."""
    if LGBM_AVAILABLE:
        return model.predict(X)
    m, scaler = model
    return m.predict(scaler.transform(X))


# ─── Walk-forward par paire ──────────────────────────────────────────────────

def train_strategy_models(features_df, targets_df, pair):
    """Entraine un modele par strategie pour une paire. Retourne (models, metrics)."""
    available = [f for f in REGIME_FEATURES if f in features_df.columns]
    common_idx = features_df.index.intersection(targets_df.index)
    X_df = features_df.loc[common_idx, available].dropna()
    y_df = targets_df.loc[X_df.index]

    models, metrics = {}, {s: [] for s in STRATEGIES}
    n = len(X_df)
    fold_size = n // 5

    for strat in STRATEGIES:
        if strat not in y_df.columns:
            continue

        X, y = X_df.values, y_df[strat].values

        # Walk-forward : 3 folds
        for fold in range(1, 4):
            train_end = fold * fold_size
            test_end = min((fold + 1) * fold_size, n)
            if train_end < MIN_OBS or (test_end - train_end - PURGE_GAP_DAYS) < 20:
                continue
            t_idx = np.arange(train_end)
            v_idx = np.arange(train_end + PURGE_GAP_DAYS, test_end)

            if LGBM_AVAILABLE:
                m = _train_lgbm(X[t_idx], y[t_idx], X[v_idx], y[v_idx])
            else:
                m = _train_ridge(X[t_idx], y[t_idx])
            preds = _predict(m, X[v_idx])

            ic, _ = spearmanr(preds, y[v_idx])
            metrics[strat].append(float(ic) if np.isfinite(ic) else 0.0)

        # Modele final sur tout le dataset
        if LGBM_AVAILABLE:
            models[strat] = _train_lgbm(X, y)
        else:
            models[strat] = _train_ridge(X, y)

    return models, metrics


# ─── Scoring ─────────────────────────────────────────────────────────────────

def predict_weights(models, features_latest):
    """Score la derniere observation avec les 4 modeles -> poids normalises."""
    available = [f for f in REGIME_FEATURES if f in features_latest.index]
    X = features_latest[available].values.reshape(1, -1)

    raw = {}
    for strat, model in models.items():
        try:
            raw[strat] = float(_predict(model, X)[0])
        except Exception:
            raw[strat] = 0.25

    scores = np.array([raw.get(s, 0.0) for s in STRATEGIES])
    pos = np.maximum(scores, 0.0)
    total = pos.sum()
    weights = pos / total if total > 1e-8 else np.full(len(STRATEGIES), 1.0 / len(STRATEGIES))
    return pd.Series(dict(zip(STRATEGIES, weights)))


# ─── Execution principale ────────────────────────────────────────────────────

def run_strategy_ml() -> None:
    try:
        for fpath in [STRAT_RETURNS_FILE, FEATURES_FILE, SELECTED_PAIRS_FILE]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"{fpath} introuvable.")

        active_pairs = (
            pd.read_csv(SELECTED_PAIRS_FILE, sep=";", decimal=",")
            .query("Status == 'ACTIVE'")["Pair"].tolist()
        )
        if not active_pairs:
            raise ValueError("Aucune paire ACTIVE dans selected_pairs.csv.")
        logger.info("Paires actives : %s", active_pairs)

        features_all = pd.read_csv(FEATURES_FILE, sep=";", decimal=",",
                                   index_col=0, parse_dates=True)
        returns_long = pd.read_csv(STRAT_RETURNS_FILE, parse_dates=["Date"])
        logger.info("Rendements : %d lignes | %d paires | %d strategies",
                     len(returns_long), returns_long["Pair"].nunique(),
                     returns_long["Strategy"].nunique())

        all_weights, all_metrics = [], []

        for pair in active_pairs:
            logger.info("--- %s ---", pair)
            pair_rets = returns_long[returns_long["Pair"] == pair]
            if pair_rets.empty:
                logger.warning("%s absent du backtest - equiponderation.", pair)
                all_weights.append(pd.Series({s: 0.25 for s in STRATEGIES}, name=pair))
                continue

            rets_wide = pair_rets.pivot_table(
                index="Date", columns="Strategy", values="Return"
            ).sort_index()

            pair_feats = features_all[features_all["Pair"] == pair]
            if pair_feats.empty:
                all_weights.append(pd.Series({s: 0.25 for s in STRATEGIES}, name=pair))
                continue

            targets = compute_strategy_targets(rets_wide)
            pair_feats_aligned = pair_feats.loc[pair_feats.index.isin(targets.index)]
            if len(pair_feats_aligned) < MIN_OBS:
                all_weights.append(pd.Series({s: 0.25 for s in STRATEGIES}, name=pair))
                continue

            models, fold_metrics = train_strategy_models(pair_feats_aligned, targets, pair)

            for strat, ics in fold_metrics.items():
                if ics:
                    mean_ic = float(np.mean(ics))
                    logger.info("  %s - IC moyen : %.4f", strat, mean_ic)
                    all_metrics.append({"Pair": pair, "Strategy": strat, "IC_mean": round(mean_ic, 4)})

            pair_weights = predict_weights(models, pair_feats.iloc[-1])
            pair_weights.name = pair
            all_weights.append(pair_weights)
            logger.info("  Poids : %s",
                        " | ".join(f"{s[:4]}={v:.2f}" for s, v in pair_weights.items()))

        # Export
        weights_df = pd.DataFrame(all_weights)
        weights_df.index.name = "Pair"
        weights_df.reset_index(inplace=True)
        weights_df.to_csv(OUTPUT_FILE, index=False)
        logger.info("Poids strategies exportes -> %s", OUTPUT_FILE)

        print(f"\n{'='*70}\nSTRATEGY WEIGHTS PAR PAIRE\n{'='*70}")
        print(weights_df.set_index("Pair").round(3).to_string())

        if all_metrics:
            pd.DataFrame(all_metrics).to_csv(METRICS_FILE, index=False)

    except Exception as e:
        logger.critical("Erreur strategy_ml : %s", e, exc_info=True)


if __name__ == "__main__":
    run_strategy_ml()

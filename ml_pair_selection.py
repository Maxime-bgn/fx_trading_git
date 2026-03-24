"""
ml_pair_selection.py
--------------------
Scoring des paires FX selectionnees par l'audit IC/ICIR bilateral.

Architecture :
  1. Univers = toutes les paires ayant passe le filtre |ICIR| >= ICIR_MIN
     dans ic_audit.py (ic_selected_pairs.csv). Pas de cap arbitraire : un signal
     inversé stable (ICIR = -1.55) est traite exactement comme un signal direct.
  2. Encodage du signe IC : chaque feature est multipliee par sign(IC_moyen)
     de la paire avant d'entrer dans le modele.
     => apres encodage, feature_haute = retour attendu haussier, pour toutes les paires.
  3. LightGBM regression (target = Future_Return continu).
  4. Metrique de validation : IC (Spearman) entre score predit et retour reel.
  5. Walk-Forward purge + holdout OOS 6 mois.

Outputs :
  selected_pairs.csv          — paires actives avec score et rang
  ml_validation_metrics.csv   — IC par fold + holdout OOS
  ml_feature_importance.csv   — importance LGBM
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from macro_data import get_real_rate_differential

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── DEPENDANCE OPTIONNELLE LightGBM ──────────────────────────────────────────
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("lightgbm non installe — fallback RandomForest.")
    LGBM_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# PARAMETRES
# =============================================================================

DATA_INPUT_FILE     = "dataset_ml_ready.csv"
SIGNAL_DIR_FILE     = "ic_signal_directions.csv"
SELECTED_PAIRS_FILE = "ic_selected_pairs.csv"
OUTPUT_FILE         = "selected_pairs.csv"

# Top N paires envoyees au backtest / orders — toujours <= nb paires selectionnees
N_ACTIVE_PAIRS  = 99   # toutes les paires passant l'IC audit sont actives
PURGE_GAP_DAYS  = 10
HOLDOUT_DAYS    = 126   # ~6 mois

# Features disponibles dans dataset_ml_ready.csv (apres features.py mis a jour)
ML_FEATURES = [
    "Vol_20", "Hurst",
    "Mom_20", "Mom_60", "Mom_120", "Mom_Ratio",
    "BB_Width", "Dist_MA20",
    "Vol_Regime",   # vol relative = proxy regime
    "Trend",        # MA20/MA50 - 1
]
# RSI exclu : IC ~ 0 sur toutes les paires FX, p-value > 0.4 systematiquement.
# En FX les tendances persistent trop longtemps pour que les niveaux
# overbought/oversold aient un sens — RSI est calibre pour les actions.
# Carry, Real_Rate_Diff, Mom_Carry sont exclus du training :
# constantes par paire sur l'historique => IC time-series = 0 => bruit pur.
# Le carry est utilise comme overlay sur le score FINAL uniquement (signal cross-sectionnel).

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective"        : "regression",
    "metric"           : "rmse",
    "num_leaves"       : 31,
    "learning_rate"    : 0.05,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 5,
    "min_child_samples": 20,
    "lambda_l1"        : 0.1,
    "lambda_l2"        : 0.1,
    "verbose"          : -1,
    "n_jobs"           : -1,
    "random_state"     : 42,
}
LGBM_N_ROUNDS   = 400
LGBM_EARLY_STOP = 40

# =============================================================================
# METRIQUE IC CUSTOM POUR LIGHTGBM
# =============================================================================

def ic_eval(y_pred: np.ndarray, dataset) -> tuple[str, float, bool]:
    """
    Metrique custom LightGBM : IC (Spearman) entre score predit et retour reel.
    Signature attendue par LightGBM : (y_pred, dataset) -> (name, value, higher_is_better)
    """
    y_true = dataset.get_label()
    valid  = np.isfinite(y_pred) & np.isfinite(y_true)
    if valid.sum() < 10:
        return "IC", 0.0, True
    ic, _ = spearmanr(y_pred[valid], y_true[valid])
    return "IC", float(ic) if np.isfinite(ic) else 0.0, True


# =============================================================================
# Z-SCORE CROSS-SECTIONNEL
# =============================================================================

def apply_cross_sectional_zscore(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Normalise chaque feature entre toutes les paires a chaque date.

    Pourquoi cross-sectionnel et non time-series :
      - On compare les paires entre elles a un instant t (ranking relatif)
      - Une Vol_20 de 0.008 sur USDTRY n'a pas le meme sens que 0.008 sur EURUSD
      - Apres z-score : +1.5 signifie "50% au-dessus de la moyenne des paires ce jour"
      - C'est ce que le modele doit apprendre pour ranger les paires entre elles

    Le z-score est applique APRES l'encodage du signe pour ne pas inverser
    la normalisation. Seules les features dans ML_FEATURES sont normalisees ;
    Carry et Real_Rate_Diff sont deja des différentiels — on les laisse tels quels.

    NaN preserves : si moins de 2 paires valides sur une date, la feature reste NaN
    (sera droppee par dropna() en aval).
    """
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        # Groupby date => mean/std cross-sectionnel
        grp  = df.groupby(df.index)[feat]
        mean = grp.transform("mean")
        std  = grp.transform("std")
        # Eviter la division par zero (std = 0 quand une seule paire par date)
        df[feat] = (df[feat] - mean) / std.replace(0, np.nan)
    return df


# =============================================================================
# ENCODAGE DU SIGNE IC
# =============================================================================

def apply_sign_encoding(
    df: pd.DataFrame,
    sign_map: dict[tuple[str, str], int],
    features: list[str],
) -> pd.DataFrame:
    """
    Multiplie chaque feature par sign(IC_moyen) de la paire correspondante.

    Apres encodage :
      - USDMXN Vol_20 * (-1)  => une volatilite elevee signifie maintenant "retour attendu positif"
      - GBPUSD Vol_20 * (+1)  => inchange

    Le modele voit un signal directionnel coherent pour toutes les paires,
    quelle que soit la direction naturelle de chaque feature sur chaque paire.
    """
    df = df.copy()
    for pair in df["Pair"].unique():
        mask = df["Pair"] == pair
        for feat in features:
            sign = sign_map.get((pair, feat), 1)
            if sign == -1:
                df.loc[mask, feat] = df.loc[mask, feat] * -1
    return df


# =============================================================================
# WALK-FORWARD + HOLDOUT OOS
# =============================================================================

def train_and_score(
    df: pd.DataFrame,
    sign_map: dict[tuple[str, str], int],
) -> pd.DataFrame:
    """
    Walk-Forward purge sur les paires selectionnees avec LightGBM regression.
    Target    : Future_Return (continu — pas de seuil, pas de binarisation).
    Metrique  : IC (Spearman) entre score predit et retour reel.
    """
    available_features = [f for f in ML_FEATURES if f in df.columns]
    if not available_features:
        raise ValueError("Aucune feature ML disponible dans le dataset.")

    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        logger.warning("Features absentes du dataset (ajoutez-les dans features.py) : %s", missing)

    # 1. Encodage signe (retourne les features a ICIR negatif)
    df_enc = apply_sign_encoding(df, sign_map, available_features)

    # 2. Z-score cross-sectionnel (normalise entre paires a chaque date)
    #    Carry et Real_Rate_Diff exclus : ce sont des differentiels deja relatifs
    features_to_zscore = [f for f in available_features if f not in ("Carry", "Real_Rate_Diff")]
    df_enc = apply_cross_sectional_zscore(df_enc, features_to_zscore)

    # Target : Sharpe par periode = Future_Return / Vol_20
    # Normalise entre paires (USDTRY ±3%/j vs USDHKD ±0.07%/j)
    # Le modele predit de l'alpha pur, pas de la volatilite brute.
    df_enc["Target"] = df_enc["Future_Return"] / (df_enc["Vol_20"].replace(0, np.nan))

    df_clean = df_enc.dropna(subset=available_features + ["Target"]).copy()
    df_clean = df_clean.sort_index()

    if df_clean.empty:
        raise ValueError("Dataset vide apres nettoyage.")

    pairs_in_train = df_clean["Pair"].unique().tolist()
    logger.info(
        "Univers ML : %d paires | %d observations | features : %s",
        len(pairs_in_train), len(df_clean), available_features,
    )

    # ── Split holdout OOS ────────────────────────────────────────────────────
    unique_dates  = df_clean.index.unique().sort_values()
    holdout_start = unique_dates[-HOLDOUT_DAYS] if len(unique_dates) > HOLDOUT_DAYS else unique_dates[0]

    df_tv   = df_clean[df_clean.index < holdout_start]
    df_hold = df_clean[df_clean.index >= holdout_start]

    logger.info(
        "Train/val jusqu'au %s | holdout OOS %s -> %s (%d jours)",
        holdout_start.date(),
        df_hold.index.min().date() if not df_hold.empty else "N/A",
        df_hold.index.max().date() if not df_hold.empty else "N/A",
        len(df_hold.index.unique()),
    )

    X_tv = df_tv[available_features].values
    y_tv = df_tv["Target"].values

    # ── Walk-Forward 4 folds ─────────────────────────────────────────────────
    n         = len(X_tv)
    fold_size = n // 5
    fold_metrics: list[dict] = []

    logger.info("Walk-Forward avec purging gap de %d jours...", PURGE_GAP_DAYS)

    for fold in range(1, 5):
        train_end = fold * fold_size
        test_end  = min((fold + 1) * fold_size, n)
        if test_end - train_end < PURGE_GAP_DAYS + 30:
            continue

        train_idx = np.arange(train_end)
        test_idx  = np.arange(train_end + PURGE_GAP_DAYS, test_end)

        if LGBM_AVAILABLE:
            dtrain = lgb.Dataset(X_tv[train_idx], label=y_tv[train_idx])
            dval   = lgb.Dataset(X_tv[test_idx],  label=y_tv[test_idx], reference=dtrain)
            m = lgb.train(
                LGBM_PARAMS,
                dtrain,
                num_boost_round=LGBM_N_ROUNDS,
                valid_sets=[dval],
                feval=ic_eval,
                callbacks=[
                    lgb.early_stopping(LGBM_EARLY_STOP, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            preds = m.predict(X_tv[test_idx])
        else:
            m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
            m.fit(X_tv[train_idx], y_tv[train_idx])
            preds = m.predict(X_tv[test_idx])

        ic_fold, pval = spearmanr(preds, y_tv[test_idx])
        ic_fold = float(ic_fold) if np.isfinite(ic_fold) else 0.0
        logger.info("Fold %d — IC: %.4f (p=%.3f)", fold, ic_fold, pval)
        fold_metrics.append({
            "fold": fold, "IC": round(ic_fold, 4),
            "p_value": round(pval, 4), "type": "walk_forward",
        })

    # ── Modele final sur tout df_tv ──────────────────────────────────────────
    logger.info("Entrainement modele final sur train+val (holdout exclu).")
    if LGBM_AVAILABLE:
        model_final = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tv, label=y_tv),
            num_boost_round=LGBM_N_ROUNDS,
            callbacks=[lgb.log_evaluation(-1)],
        )
    else:
        model_final = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        model_final.fit(X_tv, y_tv)

    # ── Ridge sanity check (modele lineaire de reference) ───────────────────
    logger.info("Entrainement Ridge (alpha=1.0) pour sanity check...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tv, y_tv)

    # ── Holdout OOS ─────────────────────────────────────────────────────────
    holdout_ic = None
    if not df_hold.empty:
        X_h   = df_hold[available_features].values
        y_h   = df_hold["Target"].values
        p_h   = model_final.predict(X_h)
        valid = np.isfinite(p_h) & np.isfinite(y_h)
        if valid.sum() >= 30:
            holdout_ic, hpval = spearmanr(p_h[valid], y_h[valid])
            holdout_ic = float(holdout_ic) if np.isfinite(holdout_ic) else 0.0
            logger.info("Holdout OOS (6 mois) — IC: %.4f (p=%.3f)", holdout_ic, hpval)
            fold_metrics.append({
                "fold": "holdout_oos", "IC": round(holdout_ic, 4),
                "p_value": round(hpval, 4), "type": "holdout",
            })
            if abs(holdout_ic) < 0.05:
                logger.warning("IC holdout = %.4f — signal tres faible. Verifier la stabilite.", holdout_ic)

    # ── Exports metriques & importance ──────────────────────────────────────
    pd.DataFrame(fold_metrics).to_csv("ml_validation_metrics.csv", index=False)
    logger.info("Metriques -> ml_validation_metrics.csv")

    if LGBM_AVAILABLE:
        imp = pd.DataFrame({
            "feature"    : available_features,
            "importance" : model_final.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
    else:
        imp = pd.DataFrame({
            "feature"    : available_features,
            "importance" : model_final.feature_importances_,
        }).sort_values("importance", ascending=False)
    imp.to_csv("ml_feature_importance.csv", index=False)
    logger.info("Importance features -> ml_feature_importance.csv")

    # ── Scoring derniere observation par paire ───────────────────────────────
    latest    = [g.iloc[[-1]] for _, g in df_clean.groupby("Pair")]
    df_latest = pd.concat(latest).copy()
    X_latest  = df_latest[available_features].values

    df_latest["Score_ML"] = model_final.predict(X_latest)

    # Ridge predictions sur la derniere observation
    ridge_preds_latest = ridge.predict(X_latest)
    df_latest["Score_Ridge"] = ridge_preds_latest

    # Signal_Agree : 1 si LightGBM et Ridge predisent le meme signe
    df_latest["Signal_Agree"] = np.where(
        np.sign(df_latest["Score_ML"]) == np.sign(df_latest["Score_Ridge"]),
        1, 0
    )

    # ── Carry overlay (signal cross-sectionnel, taux actuels uniquement) ─────
    # Le ML capte les patterns time-series (momentum, vol, mean-rev).
    # Le carry reel s'ajoute comme biais de positionnement cross-sectionnel :
    # a signal ML egal, on prefere la paire avec le meilleur carry reel.
    # Alpha = 0.1 : le carry pese 10% du signal total, le ML en pese 90%.
    # Ajustable selon le regime (alpha plus eleve en periode de stabilite).
    CARRY_ALPHA = 0.10

    carry_scores = []
    for pair in df_latest["Pair"].values:
        carry_scores.append(get_real_rate_differential(pair))
    carry_arr = np.array(carry_scores, dtype=float)

    # Z-score du carry entre paires (meme echelle que le score ML)
    carry_std = carry_arr.std()
    if carry_std > 0:
        carry_z = (carry_arr - carry_arr.mean()) / carry_std
    else:
        carry_z = np.zeros_like(carry_arr)

    df_latest["Carry_Real"]   = carry_arr
    df_latest["Score_Carry_Z"] = carry_z
    df_latest["Score"] = df_latest["Score_ML"] + CARRY_ALPHA * carry_z

    results = df_latest[["Pair", "Score", "Score_ML", "Score_Ridge", "Signal_Agree", "Score_Carry_Z", "Carry_Real"]].copy()
    results = results.sort_values("Score", ascending=False).reset_index(drop=True)
    results["Rank"]   = results.index + 1

    n_active = min(N_ACTIVE_PAIRS, len(results))
    results["Status"] = np.where(results["Rank"] <= n_active, "ACTIVE", "INACTIVE")

    if holdout_ic is not None:
        results["Holdout_IC"] = round(holdout_ic, 4)

    return results


# =============================================================================
# EXECUTION PRINCIPALE
# =============================================================================

def run_pair_selection() -> None:
    try:
        # ── Chargement dataset ───────────────────────────────────────────────
        if not os.path.exists(DATA_INPUT_FILE):
            raise FileNotFoundError(f"{DATA_INPUT_FILE} introuvable. Executez features.py d'abord.")

        df = pd.read_csv(DATA_INPUT_FILE, sep=";", decimal=",", index_col=0, parse_dates=True)

        # ── Chargement paires selectionnees (toutes celles passant |ICIR|>=0.35) ──
        if not os.path.exists(SELECTED_PAIRS_FILE):
            raise FileNotFoundError(
                f"{SELECTED_PAIRS_FILE} introuvable. Executez ic_audit.py d'abord."
            )
        selected_pairs = pd.read_csv(SELECTED_PAIRS_FILE)["Pair"].tolist()
        logger.info(
            "Univers IC : %d paires — %s",
            len(selected_pairs), selected_pairs,
        )

        df = df[df["Pair"].isin(selected_pairs)].copy()
        if df.empty:
            raise ValueError("Aucune donnee pour les paires selectionnees.")

        # ── Chargement directions de signal ─────────────────────────────────
        if not os.path.exists(SIGNAL_DIR_FILE):
            raise FileNotFoundError(
                f"{SIGNAL_DIR_FILE} introuvable. Executez ic_audit.py d'abord."
            )
        sig_df   = pd.read_csv(SIGNAL_DIR_FILE)
        sign_map = {(row.Pair, row.Feature): int(row.IC_sign) for row in sig_df.itertuples()}

        n_inv = sum(1 for v in sign_map.values() if v == -1)
        logger.info("Sign map : %d entrees | %d features inversees", len(sign_map), n_inv)

        # ── Training + scoring ───────────────────────────────────────────────
        results = train_and_score(df, sign_map)

        results.to_csv(OUTPUT_FILE, sep=";", decimal=",", index=False)
        logger.info("Selection exportee -> %s", OUTPUT_FILE)

        active = results[results["Status"] == "ACTIVE"]
        logger.info(
            "Top %d paires actives :\n%s",
            len(active),
            active[["Pair", "Score", "Rank"]].to_string(index=False),
        )

    except Exception as e:
        logger.critical("Erreur selection ML : %s", e, exc_info=True)


if __name__ == "__main__":
    run_pair_selection()

"""
backtest_strategies.py
----------------------
Simule les rendements historiques des 4 strategies avec :
  - Couts de transaction realistes (spread par categorie de paire)
  - Swap overnight sur la strategie Carry Trade
  - Stop-loss base sur ATR x 2
  - Max holding period de 5 jours
  - Agregation ponderee par volatilite inverse (vol-adjusted)

Fournit la base historique (covariance / moyenne) pour Black-Litterman.

Features exportees vers backtest_metrics.csv pour le dashboard Streamlit :
  - sharpe_net_{strategy}     : Sharpe apres couts pour chaque strategie
  - avg_holding_{strategy}    : Duree moyenne de detention (jours)
  - sl_trigger_rate_{strategy}: Taux de declenchement du stop-loss (%)
  - cost_drag_{strategy}      : Cout moyen annualise en points de rendement
  - total_trades_{strategy}   : Nombre de trades simules
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd

from strategies import (
    add_strategy_indicators,
    strat_1_momentum,
    strat_2_mean_reversion,
    strat_3_tsmom,
    strat_4_carry_trade,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETRES DE COUTS DE TRANSACTION
# Spreads en points de rendement (proportion du prix, pas en pips).
# Calibres sur les spreads moyens observes en execution retail/prime broker.
# Source de reference : histdata.com, dukascopy spread statistics
# =============================================================================

# Spread aller-retour par categorie (en % du prix, applique a chaque entree ET sortie)
SPREAD_COST = {
    'major':  0.0001,   # 1 bps aller-retour (hypothese utilisateur)
    'cross':  0.0001,
    'exotic': 0.0001,
}

SWAP_COST_ANNUAL = {
    'major':  0.0001,
    'cross':  0.0001,
    'exotic': 0.0001,
}

# Paires classees par categorie
EXOTIC_PAIRS = {'TRY', 'BRL', 'MXN', 'ZAR', 'INR', 'CNY', 'KRW', 'IDR', 'RUB'}
MAJOR_QUOTE  = {'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD'}

# Stop-loss et holding period
ATR_SL_MULTIPLIER  = 2.0   # Stop = entree +/- ATR_SL_MULTIPLIER * ATR
MAX_HOLDING_DAYS   = 5     # Sortie forcee apres N jours quelle que soit la position

# =============================================================================
# UTILITAIRES
# =============================================================================

def _classify_pair(pair: str) -> str:
    """Retourne 'major', 'cross' ou 'exotic' pour une paire FX."""
    pair = pair.upper()
    currencies = {pair[:3], pair[3:]}
    if currencies & EXOTIC_PAIRS:
        return 'exotic'
    if 'USD' in currencies:
        return 'major'
    return 'cross'


def _compute_atr(group: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcule l'ATR (Average True Range) sur une serie OHLC ou approxime
    depuis Close uniquement si High/Low sont absents.
    """
    if {'High', 'Low'}.issubset(group.columns):
        prev_close = group['Close'].shift(1)
        tr = pd.concat([
            group['High'] - group['Low'],
            (group['High'] - prev_close).abs(),
            (group['Low']  - prev_close).abs(),
        ], axis=1).max(axis=1)
    else:
        # Approximation : ATR ~ volatilite realisee sur periode courte
        tr = group['Close'].pct_change().abs() * group['Close']

    return tr.rolling(period, min_periods=period).mean()


# =============================================================================
# SIMULATION D'UNE STRATEGIE AVEC COUTS ET STOP-LOSS
# =============================================================================

def _simulate_strategy(
    group: pd.DataFrame,
    signal_col: str,
) -> pd.DataFrame:
    """
    Simule le PnL net d'une strategie pour une paire donnee en appliquant :
      1. Spread aller-retour a chaque changement de signal
      2. Swap overnight (Carry Trade uniquement)
      3. Stop-loss ATR x ATR_SL_MULTIPLIER
      4. Sortie forcee apres MAX_HOLDING_DAYS jours

    Retourne un DataFrame avec les colonnes :
      pnl_net, holding_days, sl_triggered, cost_total
    """
    pair_name  = group['Pair'].iloc[0]
    pair_class = _classify_pair(pair_name)
    spread     = SPREAD_COST[pair_class]

    atr = _compute_atr(group)

    results = []
    position      = 0
    entry_price   = 0.0
    entry_atr     = 0.0
    holding_days  = 0

    closes  = group['Close'].values
    signals = group[signal_col].values
    atrs    = atr.values
    next_rets = group['Next_Return'].values

    for i in range(len(group)):
        raw_signal = int(signals[i])
        close      = closes[i]
        atr_val    = atrs[i] if not np.isnan(atrs[i]) else close * 0.005
        next_ret   = next_rets[i] if not np.isnan(next_rets[i]) else 0.0

        sl_triggered  = False
        forced_exit   = False
        pnl_net       = 0.0
        cost_total    = 0.0

        # Sortie forcee sur max holding period
        if position != 0 and holding_days >= MAX_HOLDING_DAYS:
            forced_exit = True
            raw_signal  = 0

        # Sortie forcee sur stop-loss
        if position != 0:
            sl_distance = ATR_SL_MULTIPLIER * entry_atr
            price_move  = (close - entry_price) * position
            if price_move < -sl_distance:
                sl_triggered = True
                raw_signal   = 0

        # Gestion des transitions de position
        if raw_signal != position:
            if position != 0:
                # Cloture de la position existante
                cost_total += spread
                position    = 0
                holding_days = 0

            if raw_signal != 0 and not forced_exit and not sl_triggered:
                # Ouverture d'une nouvelle position
                cost_total  += spread
                position     = raw_signal
                entry_price  = close
                entry_atr    = atr_val
                holding_days = 0

        # PnL brut de la position courante
        if position != 0:
            pnl_gross = position * next_ret
            holding_days += 1

            pnl_net = pnl_gross - cost_total
        else:
            pnl_net = -cost_total

        results.append({
            'pnl_net':      pnl_net,
            'holding_days': holding_days,
            'sl_triggered': int(sl_triggered),
            'cost_total':   cost_total,
        })

    return pd.DataFrame(results, index=group.index)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def get_strategy_returns() -> pd.DataFrame:
    """
    Simule les rendements nets de chaque strategie sur l'historique complet.
    Agregation ponderee par volatilite inverse (les paires calmes pesent plus).
    Exporte les metriques de risque dans backtest_metrics.csv.

    Retourne un DataFrame de rendements quotidiens nets par strategie,
    pret a etre consomme par black_litterman.py.
    """
    logger.info("Debut de la simulation backtest avec couts de transaction.")

    try:
        df = pd.read_csv(
            "dataset_ml_ready.csv", sep=";", decimal=",",
            index_col=0, parse_dates=True,
        )
    except FileNotFoundError:
        logger.error("Fichier 'dataset_ml_ready.csv' manquant.")
        return None

    if 'Pair' not in df.columns:
        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.set_index('Date')

    # Filtrer aux paires selectionnees par l'audit IC
    # Priorite : ic_selected_pairs.csv (audit) > selected_pairs.csv (ML scoring)
    for pairs_file in ["ic_selected_pairs.csv", "selected_pairs.csv"]:
        if os.path.exists(pairs_file):
            try:
                sep = ";" if pairs_file == "selected_pairs.csv" else ","
                sel = pd.read_csv(pairs_file, sep=sep)
                selected = sel["Pair"].tolist()
                df = df[df["Pair"].isin(selected)].copy()
                logger.info(
                    "Backtest filtre sur %d paires selectionnees (%s) : %s",
                    len(selected), pairs_file, selected,
                )
            except Exception as e:
                logger.warning("Impossible de lire %s : %s — backtest sur l'univers complet.", pairs_file, e)
            break
    else:
        logger.info("Aucun fichier de selection trouve — backtest sur l'univers complet (%d paires).",
                    df["Pair"].nunique())

    # Rendement futur T+1
    df['Next_Return'] = df.groupby('Pair')['Close'].pct_change().shift(-1)

    # Filtre qualité données : clip les rendements aberrants (erreurs de données type USDCLP 5→663)
    # Seuil ±15% : aucun FX majeur/cross ne bouge de plus de 15% en un jour hors data error
    MAX_DAILY_RETURN = 0.15
    n_outliers = (df['Next_Return'].abs() > MAX_DAILY_RETURN).sum()
    if n_outliers > 0:
        logger.warning(
            "Filtre qualite données : %d rendements aberrants (>±%.0f%%) mis à zéro.",
            n_outliers, MAX_DAILY_RETURN * 100,
        )
        df['Next_Return'] = df['Next_Return'].clip(-MAX_DAILY_RETURN, MAX_DAILY_RETURN)

    pairs_backup = df['Pair'].copy()
    df = df.groupby('Pair', group_keys=False).apply(add_strategy_indicators)
    df['Pair'] = pairs_backup

    logger.info("Generation des signaux bruts...")

    df['Sig_Mom'] = df.apply(strat_1_momentum, axis=1)
    df['Sig_MR']  = df.apply(strat_2_mean_reversion, axis=1)
    df['Sig_Brk'] = df.apply(strat_3_tsmom, axis=1)
    df['Sig_Cry'] = df.apply(lambda row: strat_4_carry_trade(row, row['Pair']), axis=1)

    # Aligner les signaux avec la direction IC de chaque paire.
    # Si IC_sign = -1 (signal inversé), le signal brut est retourné :
    # un signal LONG sur une paire INVERTED devient SHORT, et vice-versa.
    # Cela aligne backtest et ML qui utilisent tous les deux le signe IC.
    if os.path.exists("ic_signal_directions.csv"):
        sig_dir = pd.read_csv("ic_signal_directions.csv")
        # IC_sign dominant par paire = vote majoritaire sur toutes les features
        pair_ic_sign = (
            sig_dir.groupby("Pair")["IC_sign"]
            .apply(lambda x: 1 if x.sum() >= 0 else -1)
            .to_dict()
        )
        for sig_col in ["Sig_Mom", "Sig_MR"]:  # Sig_Brk (TSMOM) et Sig_Cry exclus : signal = direction intrinsèque
            df[sig_col] = df.apply(
                lambda row: row[sig_col] * pair_ic_sign.get(row["Pair"], 1), axis=1
            )

    strategies = {
        'Momentum':       'Sig_Mom',
        'Mean-Reversion': 'Sig_MR',
        'TSMOM':          'Sig_Brk',
        'Carry Trade':    'Sig_Cry',
    }

    # -------------------------------------------------------------------------
    # Simulation par paire et par strategie
    # -------------------------------------------------------------------------
    all_pnl          = {s: [] for s in strategies}
    all_metrics      = {s: [] for s in strategies}
    detail_rows      = []   # format long (Date, Pair, Strategy, Return) -> strategy_ml.py

    FLIP_WINDOW   = 252   # fenetre rolling pour evaluer le Sharpe du signal
    FLIP_THRESHOLD = -0.3  # si Sharpe roulant < seuil -> on inverse le signal

    for pair, group in df.groupby('Pair'):
        group = group.copy()
        vol_pair = group['Next_Return'].std()
        if vol_pair == 0 or np.isnan(vol_pair):
            continue

        for strat_name, sig_col in strategies.items():
            # Flip dynamique : rolling Sharpe du signal sur FLIP_WINDOW jours
            # Si le signal a systematiquement perdu sur la periode precedente,
            # on inverse (alpha inversé = alpha recuperé, logique bilatérale IC).
            raw_pnl = group[sig_col] * group['Next_Return']
            rolling_sharpe = (
                raw_pnl.rolling(FLIP_WINDOW, min_periods=60).mean()
                / (raw_pnl.rolling(FLIP_WINDOW, min_periods=60).std() + 1e-8)
            ) * np.sqrt(252)
            # Flip = -1 quand Sharpe roulant < seuil, +1 sinon
            flip = rolling_sharpe.apply(
                lambda s: -1 if (not np.isnan(s) and s < FLIP_THRESHOLD) else 1
            ).shift(1).fillna(1)  # shift(1) : pas de look-ahead
            group[sig_col] = group[sig_col] * flip

            sim = _simulate_strategy(group, sig_col)
            sim['Pair'] = pair
            # Vol clippée [0.003, 0.015] : évite que USDHKD (vol=0.07%) écrase avec un poids 17×
            # et que USDZAR (vol=1.2%) domine avec ses gros swings.
            vol_clipped = np.clip(vol_pair, 0.003, 0.015)
            sim['vol_weight'] = 1.0 / vol_clipped
            all_pnl[strat_name].append(sim[['pnl_net', 'vol_weight']])
            all_metrics[strat_name].append({
                'pair':            pair,
                'avg_holding':     sim['holding_days'].replace(0, np.nan).mean(),
                'sl_trigger_rate': sim['sl_triggered'].mean() * 100,
                'cost_drag_ann':   sim['cost_total'].sum() * 252 / max(len(sim), 1),
                'total_trades':    (sim['cost_total'] > 0).sum(),
            })

            # Accumulation format long pour strategy_ml.py
            tmp = sim[['pnl_net']].copy()
            tmp.index.name = 'Date'
            tmp['Pair']     = pair
            tmp['Strategy'] = strat_name
            tmp.rename(columns={'pnl_net': 'Return'}, inplace=True)
            detail_rows.append(tmp)

    # -------------------------------------------------------------------------
    # Agregation ponderee par volatilite inverse
    # -------------------------------------------------------------------------
    logger.info("Agregation des PnL ponderee par volatilite inverse...")

    daily_returns = {}

    for strat_name in strategies:
        combined = pd.concat(all_pnl[strat_name])
        combined.index.name = 'Date'

        # Poids normalises par date
        weight_sum = combined.groupby('Date')['vol_weight'].transform('sum')
        combined['weighted_pnl'] = combined['pnl_net'] * combined['vol_weight'] / weight_sum

        daily_returns[strat_name] = combined.groupby('Date')['weighted_pnl'].sum()

    returns_df = pd.DataFrame(daily_returns).dropna()

    # -------------------------------------------------------------------------
    # Export des metriques pour le dashboard Streamlit
    # -------------------------------------------------------------------------
    metrics_rows = []
    for strat_name in strategies:
        ret_series = returns_df[strat_name]
        ann_ret    = ret_series.mean() * 252
        ann_vol    = ret_series.std() * np.sqrt(252)
        sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0

        for row in all_metrics[strat_name]:
            row['strategy']   = strat_name
            row['sharpe_net'] = round(sharpe, 3)
        metrics_rows.extend(all_metrics[strat_name])

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("backtest_metrics.csv", index=False)
    logger.info("Metriques exportees dans backtest_metrics.csv.")

    # Export format long pour strategy_ml.py
    if detail_rows:
        detail_df = pd.concat(detail_rows).reset_index()
        detail_df.to_csv("strategies_returns_by_pair.csv", index=False)
        logger.info(
            "Rendements par (paire, strategie) exportes dans strategies_returns_by_pair.csv "
            "(%d lignes, %d paires, %d strategies).",
            len(detail_df),
            detail_df['Pair'].nunique(),
            detail_df['Strategy'].nunique(),
        )

    return returns_df


if __name__ == "__main__":
    rets = get_strategy_returns()
    if rets is not None:
        rets.to_csv("strategies_returns.csv")
        logger.info("Fichier 'strategies_returns.csv' genere.")
        logger.info("Correlation entre strategies :\n%s", rets.corr().round(3).to_string())
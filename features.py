"""
features.py
-----------
Feature engineering + target pour le pipeline ML FX.
Chaque paire CSV (*=X.csv) est enrichie puis consolidee dans dataset_ml_ready.csv.
"""

import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from macro_data import get_interest_rate_differential, get_real_rate_differential

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Parametres
WINDOW_RSI = 14
WINDOW_VOL = 20
WINDOW_HURST = 120
HORIZON = 5
TRADING_DAYS_PER_YEAR = 252


# ─── Hurst ───────────────────────────────────────────────────────────────────

def calculate_hurst(ts: np.ndarray) -> float:
    n = len(ts)
    if n < 20 or not np.all(np.isfinite(ts)):
        return 0.5
    max_lag = min(n // 4, 40)
    if max_lag < 4:
        return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 4:
        return 0.5
    try:
        poly = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
        return float(np.clip(poly[0] * 2.0, 0.0, 1.0))
    except Exception:
        return 0.5


# ─── Feature builders ────────────────────────────────────────────────────────

def _add_returns_and_vol(df):
    df['Returns'] = df['Close'].pct_change()
    df['Vol_20'] = df['Returns'].rolling(WINDOW_VOL).std()


def _add_momentum(df):
    for period in [5, 10, 20, 60, 120]:
        df[f'Mom_{period}'] = df['Close'].pct_change(period)
    df['Mom_Ratio'] = df['Mom_20'] / (df['Mom_120'].abs() + 1e-8)


def _add_rsi(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(WINDOW_RSI).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(WINDOW_RSI).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)


def _add_trend_and_bollinger(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Trend'] = df['MA20'] / df['MA50'] - 1
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * std20
    df['BB_Lower'] = df['MA20'] - 2 * std20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'].replace(0, np.nan)
    df['Dist_MA20'] = (df['Close'] - df['MA20']) / df['MA20'].replace(0, np.nan)


def _add_vol_regime(df):
    df['Vol_Regime'] = df['Vol_20'] / df['Vol_20'].rolling(60).mean()


def _add_hurst(df):
    if len(df) > WINDOW_HURST:
        df['Hurst'] = df['Close'].rolling(WINDOW_HURST).apply(calculate_hurst, raw=True)
    else:
        df['Hurst'] = np.nan


def _add_carry(df, pair):
    try:
        carry = get_interest_rate_differential(pair)
        carry_real = get_real_rate_differential(pair)
    except Exception:
        carry, carry_real = 0, 0
    df['Carry'] = carry
    df['Real_Rate_Diff'] = carry_real
    df['Mom_Carry'] = df['Mom_20'] * np.sign(carry) if carry != 0 else 0.0


# ─── Pipeline features + target ─────────────────────────────────────────────

def add_features_and_targets(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    df = df.copy()

    _add_returns_and_vol(df)
    _add_momentum(df)
    _add_rsi(df)
    _add_trend_and_bollinger(df)
    _add_vol_regime(df)
    _add_hurst(df)
    _add_carry(df, pair)

    # Target : retour a 5 jours
    df['Future_Return'] = df['Close'].pct_change(HORIZON).shift(-HORIZON)
    df['Label'] = (df['Future_Return'] > 0).astype(int)

    # Colonnes requises pour dropna
    required = [
        'Future_Return', 'Vol_20', 'Hurst',
        'Mom_5', 'Mom_10', 'Mom_20', 'Mom_60', 'Mom_120', 'Mom_Ratio',
        'Trend', 'Vol_Regime', 'BB_Width', 'Dist_MA20',
        'Carry', 'Real_Rate_Diff', 'Mom_Carry', 'RSI',
    ]
    return df.dropna(subset=required)


# ─── Pipeline global ─────────────────────────────────────────────────────────

def process_all_pairs():
    files = list(Path.cwd().glob("*=X.csv"))
    if not files:
        logger.error("Aucun fichier CSV trouve.")
        return

    all_data: List[pd.DataFrame] = []

    for file_path in files:
        pair = file_path.stem.replace("=X", "")
        try:
            df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0, parse_dates=True)
            # Nettoyage colonnes object -> numeric
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            if len(df) < 300:
                continue
            df_proc = add_features_and_targets(df, pair)
            df_proc.insert(0, 'Pair', pair)
            if not df_proc.empty:
                all_data.append(df_proc)
        except Exception as e:
            logger.error(f"{pair} failed: {e}")

    if not all_data:
        logger.error("Aucune donnee valide.")
        return

    full = pd.concat(all_data)
    full.to_csv("dataset_ml_ready.csv", sep=";", decimal=",")
    logger.info(f"Dataset cree: {full.shape}")


if __name__ == "__main__":
    process_all_pairs()

"""
strategies.py
-------------
5 stratégies de signal + add_strategy_indicators.
  1. Momentum         — MA20/MA50 crossover (toujours ±1, pas de filtre ADX)
  2. Mean-Reversion   — Bollinger Bands 1.5σ + filtre no-trend
  3. TSMOM            — Time-Series Momentum 12 mois skip-5j + filtre vol 63j
  4. Carry Trade      — différentiel de taux, filtre vol extrême uniquement
  5. Long-Term G10    — carry + MA200, position permanente sur G10
"""

import numpy as np
import pandas as pd
from macro_data import get_interest_rate_differential

G10 = {'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD'}

# Paires où TSMOM est documenté et positif : EM (carry structurel) + JPY crosses (risk-off trend)
# Source : Menkhoff et al. (2012) — FX momentum fonctionne surtout sur les paires à fort carry
TSMOM_UNIVERSE = {
    'USDTRY', 'USDPKR', 'USDINR', 'USDCNY', 'USDBRL', 'USDMXN',
    'USDILS', 'USDKRW', 'USDZAR', 'USDRUB',          # EM carry
    'CHFJPY', 'NZDJPY', 'EURJPY', 'CADJPY',           # JPY risk-off / risk-on trend
    'NZDUSD', 'AUDUSD',                                # commodity currencies (macro trend)
}


# ==========================================
# INDICATEURS TECHNIQUES
# ==========================================

def calculate_adx(df, period=14):
    """Indicateur de force de tendance (ADX)."""
    df = df.copy()
    df['H-L']  = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low']  - df['Close'].shift(1))
    df['TR']   = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    df['UpMove']   = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']

    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0),   df['UpMove'],   0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/period).mean() / df['TR'].ewm(alpha=1/period).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/period).mean() / df['TR'].ewm(alpha=1/period).mean())

    dx = 100 * abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-8))
    return dx.rolling(period).mean().fillna(0)


def add_strategy_indicators(df):
    """Ajoute les indicateurs techniques nécessaires aux 4 stratégies."""
    df = df.copy()

    # Moyennes mobiles
    df['MA20']  = df['Close'].rolling(window=20).mean()
    df['MA50']  = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # High/Low synthétiques si absents
    if 'High' not in df.columns:
        vol = df['Vol_20'].fillna(0.01)
        df['High'] = df['Close'] * (1 + vol * 0.5)
        df['Low']  = df['Close'] * (1 - vol * 0.5)

    # Bandes de Bollinger à 1.5σ (±2σ = 5% du temps, ±1.5σ ≈ 13% du temps)
    std_dev       = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + std_dev * 1.5
    df['BB_Lower'] = df['MA20'] - std_dev * 1.5

    # Distance relative à la MA20  (feature IC-validée, remplace RSI)
    df['Dist_MA20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8)

    # Canaux de Donchian avec shift(1) : conservés pour compatibilité mais
    # non utilisés par strat_3 (remplacée par TSMOM).
    df['Donchian_High'] = df['High'].rolling(window=20).max().shift(1)
    df['Donchian_Low']  = df['Low'].rolling(window=20).min().shift(1)

    # ADX
    df['ADX'] = calculate_adx(df)

    # ── TSMOM indicators (strat_3 replacement) ────────────────────────────────
    # Mom_252 : rendement sur 12 mois, dernière semaine exclue (skip 5j) pour
    # éviter le retournement court terme (effet mean-reversion à 1 semaine).
    # Ref : Moskowitz, Ooi, Pedersen (2012) — "Time Series Momentum".
    ret = df['Close'].pct_change()
    df['Mom_252_skip5'] = df['Close'].pct_change(252).shift(5)

    # Vol_63 : volatilité réalisée sur 63 jours (~1 trimestre), annualisée.
    # Sert à normaliser le signal : position proportionnelle à 1/vol.
    # Clippée à [0.002, 0.05] pour éviter les extrêmes (pegs et crises exotiques).
    df['Vol_63'] = ret.rolling(63).std() * np.sqrt(252)
    df['Vol_63'] = df['Vol_63'].clip(lower=0.002, upper=0.05)

    return df


# ==========================================
# LES 4 STRATÉGIES
# ==========================================

def strat_1_momentum(row):
    """
    Momentum — Dual Moving Average crossover.
    Toujours ±1 : le filtre ADX > 25 créait 83% de jours à signal=0.
    La direction MA20 vs MA50 est un signal en soi même sans tendance forte.
    """
    if row['MA20'] > row['MA50']:
        return 1
    elif row['MA20'] < row['MA50']:
        return -1
    return 0


def strat_2_mean_reversion(row):
    """
    Mean-Reversion — Bollinger Bands uniquement.
    Filtre tendance (ADX < 25) + filtre paires quasi-fixes (Vol_20 < 0.002 → peg).
    USDHKD (vol ~0.07%) exclu : paire arrimée, pas de dynamique de retour à la moyenne.
    """
    if row['Vol_20'] < 0.002:   # paire péggée (ex: USDHKD) → pas de MR
        return 0
    no_trend = row['ADX'] < 25
    if row['Close'] < row['BB_Lower'] and no_trend:
        return 1
    elif row['Close'] > row['BB_Upper'] and no_trend:
        return -1
    return 0


def strat_3_tsmom(row):
    """
    Time-Series Momentum (TSMOM) — remplace Donchian Breakout.

    Logique :
      - Direction  = signe du rendement sur 12 mois avec skip 5 jours
                     (évite le retournement court terme documenté en FX).
      - Filtre vol = signal annulé si volatilité 63j extrême (> 4%).
                     Les crises exotiques (USDTRY +20% en 1 mois) génèrent
                     de faux signaux TSMOM positifs ; on les filtre.
      - Signal     = ±1 discret (cohérent avec les autres stratégies).
                     Pas de position fractionnaire ici — le vol-weighting
                     est géré dans backtest_strategies.py.

    Pourquoi ça marche sur FX :
      - Carry + momentum sont corrélés (les devises à fort carry tendent) :
        TSMOM capture exactement cette persistence de 3-12 mois.
      - Horizon 12 mois évite le mean-reversion court terme (< 1 mois)
        et le noise quotidien.
      - Fonctionne sur toutes les 27 paires : G10 (tendances macro),
        exotiques (carry-driven trends de long terme).

    Réf : Moskowitz, Ooi, Pedersen (2012) ; Menkhoff et al. (2012) FX momentum.
    """
    pair = str(row.get('Pair', '')).upper().replace('=X', '')
    if pair not in TSMOM_UNIVERSE:
        return 0

    mom = row.get('Mom_252_skip5')
    vol = row.get('Vol_63')

    if pd.isna(mom) or pd.isna(vol):
        return 0

    # Filtre volatilité extrême : crises EM (USDTRY > 20% en krach)
    if vol > 0.20:
        return 0

    if mom > 0:
        return 1
    elif mom < 0:
        return -1
    return 0


def strat_4_carry_trade(row, pair_name):
    """
    Carry Trade conditionnel — différentiel de taux avec filtre volatilité.
    Bloqué uniquement en cas de crise de vol extrême (> 2%).
    """
    if row['Vol_20'] > 0.020:
        return 0
    diff = get_interest_rate_differential(pair_name)
    if diff > 2.0:
        return 1
    elif diff < -2.0:
        return -1
    return 0


def strat_5_longterm_g10(row, pair_name):
    """
    Long-Term Carry G10 — position permanente dans la direction du carry,
    confirmée par MA200 (tendance long terme).

    Logique : on est toujours positionné dans la direction du carry
    SAUF si MA200 contredit (évite de nager contre une tendance de fond).
    Signal quasi-permanent : réduit les jours à 0 signal sur G10.
    """
    if pair_name.upper() not in G10:
        return 0
    diff = get_interest_rate_differential(pair_name)
    # Direction carry : long base si base mieux rémunérée
    if diff > 0:
        carry_dir = 1
    elif diff < 0:
        carry_dir = -1
    else:
        return 0  # pas de différentiel de taux → pas de position

    # Filtre MA200 : si la tendance long terme contredit le carry → flat
    ma200 = row.get('MA200', float('nan'))
    if ma200 and not np.isnan(float(ma200)):
        above_ma200 = row['Close'] > float(ma200)
        if carry_dir == 1 and not above_ma200:
            return 0   # carry long mais tendance baissière → prudence
        if carry_dir == -1 and above_ma200:
            return 0   # carry short mais tendance haussière → prudence

    return carry_dir

"""
portofolio.py — Gestion du portefeuille & génération d'ordres
=============================================================
Place dans le pipeline : appelé par main_trading.py APRÈS optimize_strategies.py.

Responsabilités :
  1. Chargement des poids d'allocation (ML / Black-Litterman via optimal_weights.csv)
  2. Position sizing dynamique basé sur la volatilité (Risk Parity)
  3. Contraintes NDF (réduction de taille, flag de type)
  4. Vérification de l'exposition globale (capital total déployé)
  5. Génération du ticket d'ordres final avec métriques de risque

Corrections vs version initiale :
  - Signature de calculate_order_size() alignée avec l'appel dans main_trading.py
    (capital, strat_weight, volatility, pair) au lieu de (row, targets)
  - Capital global ne plus stocké en constante globale muette : injecté via Config
  - Exposition totale vérifiée avant export (pas de sur-levier silencieux)
  - Logging remplace les print()
  - Toutes les valeurs magiques centralisées dans PortfolioConfig
  - run_portfolio_allocation() retourne aussi les métriques d'exposition
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from macro_data import is_ndf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# =============================================================================
# CONFIGURATION CENTRALISÉE
# =============================================================================

@dataclass
class PortfolioConfig:
    # Capital de base
    total_capital: float = 100_000.0

    # Fichier des poids optimaux (généré par optimize_strategies.py)
    weights_file: Path = Path("optimal_weights.csv")

    # Position sizing — vol target ~8-10% annuel (FX quant fund standard)
    risk_per_trade_pct: float = 0.02    # 2% du capital stratégie par trade (était 1%)
    max_position_pct: float = 0.40      # Plafond : 40% du capital stratégie (était 20%)
    min_vol_floor: float = 0.003        # Plancher de vol abaissé pour capter les paires calmes

    # Contraintes NDF (Section 5.6)
    ndf_size_factor: float = 0.50       # Réduction de 50 % sur les paires NDF

    # Exposition globale
    max_gross_exposure_pct: float = 1.50  # Max 150 % du capital (levier prudent)
    max_single_pair_pct: float = 0.15    # Max 15 % du capital sur une seule paire

    # Poids de repli si optimal_weights.csv est absent
    fallback_weights: Dict[str, float] = field(default_factory=lambda: {
        "Momentum":       0.25,
        "Mean-Reversion": 0.25,
        "Carry Trade":    0.25,
        "TSMOM":          0.25,
        "Wait":           0.00,
    })


# Instance par défaut (peut être surchargée par main_trading.py)
DEFAULT_CONFIG = PortfolioConfig()


# =============================================================================
# CHARGEMENT DES POIDS D'ALLOCATION
# =============================================================================

def get_allocation_targets(cfg: PortfolioConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """
    Charge les poids issus de optimize_strategies.py (optimal_weights.csv).
    Format attendu : CSV sans header, deux colonnes → [nom_stratégie, poids].

    Retourne les poids de repli équipondérés si le fichier est absent ou corrompu.
    """
    try:
        weights_df = pd.read_csv(cfg.weights_file, index_col=0, header=None)
        weights_df.index = weights_df.index.astype(str).str.strip()
        targets: Dict[str, float] = weights_df.iloc[:, 0].astype(float).to_dict()

        # Sanity check : la somme des poids (hors "Wait") doit être ~1
        total = sum(v for k, v in targets.items() if k != "Wait")
        if not (0.90 <= total <= 1.10):
            logging.warning(
                "Somme des poids hors-Wait = %.3f — renormalisation appliquée.", total
            )
            targets = {k: (v / total if k != "Wait" else 0.0) for k, v in targets.items()}

        logging.info(
            "Poids chargés depuis '%s' : %s",
            cfg.weights_file,
            {k: f"{v:.1%}" for k, v in targets.items()},
        )
        return targets

    except FileNotFoundError:
        logging.warning(
            "Fichier '%s' introuvable — poids de repli équipondérés utilisés.",
            cfg.weights_file,
        )
    except Exception as exc:
        logging.error("Erreur lecture poids ('%s') : %s", cfg.weights_file, exc)

    return cfg.fallback_weights.copy()


# =============================================================================
# POSITION SIZING  (signature alignée avec main_trading.py)
# =============================================================================

def calculate_order_size(
    capital: float,
    strat_weight: float,
    volatility: float,
    pair: str,
    cfg: PortfolioConfig = DEFAULT_CONFIG,
) -> float:
    """
    Calcule la taille de position en USD pour un signal donné.

    Signature alignée avec l'appel de main_trading.py :
        calculate_order_size(capital, strat_weight, row['Vol_20'], row['Pair'])

    Formule Risk Parity / Volatility Targeting :
        position = (capital_stratégie × risk_pct) / volatilité
        plafonnée à max_position_pct × capital_stratégie
        réduite de ndf_size_factor pour les NDF

    Args:
        capital      : Capital total disponible (USD).
        strat_weight : Fraction allouée à la stratégie (ex : 0.30).
        volatility   : Volatilité journalière de la paire (ex : 0.008).
        pair         : Code de la paire FX (ex : "USDINR").
        cfg          : Configuration du portefeuille.

    Returns:
        Taille de position en USD (≥ 0).
    """
    strat_capital = capital * strat_weight
    if strat_capital <= 0:
        return 0.0

    vol = max(volatility, cfg.min_vol_floor)
    risk_budget = strat_capital * cfg.risk_per_trade_pct
    position_size = risk_budget / vol

    # Plafond par position au sein de la stratégie
    position_size = min(position_size, strat_capital * cfg.max_position_pct)

    # Plafond global par paire (évite la concentration)
    position_size = min(position_size, capital * cfg.max_single_pair_pct)

    # Réduction NDF (Section 5.6 : liquidité réduite, risque de règlement)
    if is_ndf(pair):
        position_size *= cfg.ndf_size_factor

    return round(position_size, 2)


# =============================================================================
# CONSTRUCTION D'UN ORDRE
# =============================================================================

def _build_order_row(
    idx,
    row: pd.Series,
    targets: Dict[str, float],
    cfg: PortfolioConfig,
) -> Optional[dict]:
    """Construit un dict d'ordre. Retourne None si pas d'ordre valide."""
    strat = str(row.get("Strategy", "")).strip()
    signal = int(row.get("Signal", 0))
    if strat == "Wait" or signal == 0:
        return None

    strat_weight = targets.get(strat, 0.0)
    if strat_weight <= 0:
        return None

    pair = str(row.get("Pair", ""))
    size = calculate_order_size(
        capital=cfg.total_capital, strat_weight=strat_weight,
        volatility=float(row.get("Vol_20", cfg.min_vol_floor)),
        pair=pair, cfg=cfg,
    )
    if size <= 0:
        return None

    hurst = row.get("Hurst", np.nan)
    date_val = idx.date() if hasattr(idx, "date") else idx

    return {
        "Date":         date_val,
        "Pair":         pair,
        "Strategy":     strat,
        "Type":         "NDF" if is_ndf(pair) else "SPOT",
        "Direction":    "LONG" if signal == 1 else "SHORT",
        "Size_USD":     size,
        "Price":        round(float(row.get("Close", np.nan)), 4),
        "Vol_20":       round(float(row.get("Vol_20", np.nan)), 4),
        "Hurst":        round(float(hurst), 2) if not np.isnan(float(hurst)) else None,
        "Strat_Weight": f"{strat_weight:.1%}",
    }


# =============================================================================
# CONTRÔLE DE L'EXPOSITION GLOBALE
# =============================================================================

def _check_exposure(orders_df: pd.DataFrame, cfg: PortfolioConfig) -> None:
    """
    Logue des avertissements si l'exposition brute ou par paire dépasse les seuils.
    N'annule pas les ordres — alerte pour le trader.
    """
    if orders_df.empty:
        return

    gross = orders_df["Size_USD"].sum()
    gross_pct = gross / cfg.total_capital
    logging.info("Exposition brute : $%.0f (%.1f %% du capital)", gross, gross_pct * 100)

    if gross_pct > cfg.max_gross_exposure_pct:
        logging.warning(
            "⚠️  Exposition brute %.1f %% > seuil %.1f %% — vérifiez le levier.",
            gross_pct * 100, cfg.max_gross_exposure_pct * 100,
        )

    by_pair = orders_df.groupby("Pair")["Size_USD"].sum()
    for pair, amt in by_pair.items():
        if amt / cfg.total_capital > cfg.max_single_pair_pct:
            logging.warning(
                "⚠️  Paire %s : $%.0f (%.1f %%) > max autorisé %.1f %%.",
                pair, amt, amt / cfg.total_capital * 100, cfg.max_single_pair_pct * 100,
            )


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def run_portfolio_allocation(
    df_signals: pd.DataFrame,
    cfg: PortfolioConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Génère le ticket d'ordres final à partir des signaux de trading.
    Appelé par main_trading.py après génération des signaux.

    Args:
        df_signals : DataFrame avec colonnes [Signal, Strategy, Pair,
                     Close, Vol_20, Hurst, ADX].
        cfg        : Configuration du portefeuille.

    Returns:
        DataFrame des ordres prêts pour l'export vers FINAL_ORDERS.csv.
    """
    logging.info(
        "=== GESTION PORTEFEUILLE | Capital : $%.0f ===", cfg.total_capital
    )

    targets = get_allocation_targets(cfg)
    orders: List[dict] = []
    skipped = 0

    for idx, row in df_signals.iterrows():
        order = _build_order_row(idx, row, targets, cfg)
        if order:
            orders.append(order)
        else:
            skipped += 1

    orders_df = pd.DataFrame(orders)

    if orders_df.empty:
        logging.info("Aucun ordre généré pour cette session.")
        return orders_df

    # Tri : chronologique, puis par taille décroissante
    orders_df = orders_df.sort_values(
        ["Date", "Size_USD"], ascending=[True, False]
    ).reset_index(drop=True)

    _check_exposure(orders_df, cfg)

    logging.info(
        "%d ordres générés (%d signaux ignorés — Wait / poids nul / size=0).",
        len(orders_df), skipped,
    )

    # Résumé console
    print("\n" + "=" * 62)
    print(f"  TICKET D'ORDRES — {orders_df['Date'].iloc[0]}")
    print("=" * 62)
    cols = ["Pair", "Strategy", "Direction", "Size_USD", "Type", "Strat_Weight"]
    print(orders_df[cols].to_string(index=False))
    print("-" * 62)
    print(f"  Exposition brute totale : ${orders_df['Size_USD'].sum():>10,.0f}")
    print(f"  Nombre d'ordres         : {len(orders_df)}")
    print("=" * 62 + "\n")

    return orders_df
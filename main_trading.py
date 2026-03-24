import os
import logging
import warnings
import pandas as pd
import numpy as np
from strategies import (
    add_strategy_indicators,
    strat_1_momentum, strat_2_mean_reversion,
    strat_3_tsmom, strat_4_carry_trade,
    strat_5_longterm_g10, G10,
)

from portofolio import calculate_order_size

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE            = "dataset_ml_ready.csv"
SELECTED_PAIRS_FILE  = "selected_pairs.csv"
WEIGHTS_FILE         = "optimal_weights.csv"       # poids BL par strategie
MARKOWITZ_FILE       = "markowitz_weights.csv"     # poids Markowitz par paire
STRAT_WEIGHTS_FILE   = "strategy_weights.csv"      # poids ML par paire x strategie
FINAL_ORDERS_FILE    = "FINAL_ORDERS.csv"

DEFAULT_CAPITAL      = 100000
SIGNAL_THRESHOLD     = 0.15
LONGTERM_BUDGET      = 0.30   # 30% capital reservé à la strat Long-Term G10
LONGTERM_PAIRS_MAX   = 7      # max G10 pairs actives simultanément

# =============================================================================
# CORE LOGIC
# =============================================================================

def load_trading_universe():
    """Charge les donnees et filtre l'univers actif determine par le ML."""
    if not all(os.path.exists(f) for f in [DATA_FILE, SELECTED_PAIRS_FILE]):
        raise FileNotFoundError("Donnees sources ou selection ML manquantes.")

    df_data = pd.read_csv(DATA_FILE, sep=";", decimal=",", index_col=0, parse_dates=True)
    df_selected = pd.read_csv(SELECTED_PAIRS_FILE, sep=";", decimal=",")
    
    active_pairs = df_selected[df_selected['Status'] == 'ACTIVE']['Pair'].tolist()
    if not active_pairs:
        raise ValueError("Aucune paire active detectee dans selected_pairs.csv.")

    logging.info(f"Univers de trading identifie : {active_pairs}")
    return df_data[df_data['Pair'].isin(active_pairs)].copy()



def _load_ml_scores() -> dict:
    """
    Charge selected_pairs.csv -> dict {pair: {"Score_ML": float, "Signal_Agree": int}}.
    Retourne un dict vide si le fichier est absent ou incomplet.
    """
    if not os.path.exists(SELECTED_PAIRS_FILE):
        logging.warning("selected_pairs.csv absent — ML score overlay desactive.")
        return {}
    df = pd.read_csv(SELECTED_PAIRS_FILE, sep=";", decimal=",")
    if "Score_ML" not in df.columns or "Signal_Agree" not in df.columns:
        logging.warning("Colonnes Score_ML / Signal_Agree absentes — ML score overlay desactive.")
        return {}
    result = {}
    for _, row in df.iterrows():
        result[str(row["Pair"])] = {
            "Score_ML":     float(row["Score_ML"]),
            "Signal_Agree": int(row["Signal_Agree"]),
        }
    logging.info("ML scores charges pour %d paires.", len(result))
    return result


def process_signals(df):
    """
    Genere un ordre par (paire x strategie) ayant un signal non-nul.

    Avant : blend des 4 strategies -> 1 ordre par paire -> conflits annulent tout.
    Maintenant : chaque strategie vote independamment -> jusqu'a 4 ordres par paire.

    Ex : MR dit LONG, Momentum dit SHORT sur GBPUSD -> 2 ordres distincts,
         chacun size par son poids BL (MR 43%, Momentum 9%).

    Overlay ML (Score_ML) : si Signal_Agree == 1 et Score_ML contredit le signal
    de la strategie, ce signal est abandonne (filtre de coherence ML).
    """
    logging.info("Generation des signaux par (paire x strategie)...")
    ml_scores = _load_ml_scores()

    strat_fns = {
        "Momentum":       strat_1_momentum,
        "Mean-Reversion": strat_2_mean_reversion,
        "TSMOM":          strat_3_tsmom,
    }

    df_results = []
    for pair, group in df.groupby("Pair"):
        group = add_strategy_indicators(group.copy())
        ml_pair   = ml_scores.get(pair, {})
        score_ml  = ml_pair.get("Score_ML",     None)
        sig_agree = ml_pair.get("Signal_Agree", 0)

        # Une ligne par (date x strategie) avec signal != 0
        rows_per_strat = {s: [] for s in list(strat_fns.keys()) + ["Carry Trade"]}

        for _, row in group.iterrows():
            strat_signals = {}
            for strat, fn in strat_fns.items():
                strat_signals[strat] = fn(row)
            strat_signals["Carry Trade"] = strat_4_carry_trade(row, pair)

            for strat, sig in strat_signals.items():
                # Filtre coherence ML : si Ridge+LGBM d'accord et contredisent ce signal -> drop
                if sig_agree == 1 and score_ml is not None:
                    if (score_ml > 0 and sig < 0) or (score_ml < 0 and sig > 0):
                        sig = 0  # ML contredit cette strategie -> pas d'ordre

                rows_per_strat[strat].append({
                    "Signal":   sig,
                    "Strategy": strat,
                })

        # Reconstituer un DataFrame multi-lignes (une par strategie)
        for strat in list(strat_fns.keys()) + ["Carry Trade"]:
            dec = pd.DataFrame(rows_per_strat[strat], index=group.index)
            grp_strat = pd.concat([group, dec[["Signal", "Strategy"]]], axis=1)
            df_results.append(grp_strat)

    return pd.concat(df_results)

def _load_pair_strat_weights():
    """
    Charge les poids combines Markowitz (par paire) x ML (par strategie dans la paire).
    Retourne un dict {(pair, strategy): weight} ou weight = markowitz_pair * ml_strat.

    Chaine de priorite :
      1. markowitz_weights.csv x strategy_weights.csv → poids (pair, strat) optimaux
      2. optimal_weights.csv (BL global) → fallback strategy-level
      3. Equiponderation
    """
    combined = {}

    # 1. Markowitz pair weights
    mkw = {}
    if os.path.exists(MARKOWITZ_FILE):
        df = pd.read_csv(MARKOWITZ_FILE)
        if "Pair" in df.columns and "Final_Weight" in df.columns:
            mkw = dict(zip(df["Pair"], df["Final_Weight"]))
            logging.info("Poids Markowitz charges : %s", {k: f"{v:.1%}" for k, v in mkw.items()})

    # 2. ML strategy weights per pair
    ml_sw = {}
    if os.path.exists(STRAT_WEIGHTS_FILE):
        sw = pd.read_csv(STRAT_WEIGHTS_FILE)
        if "Pair" in sw.columns:
            strats = [c for c in sw.columns if c != "Pair"]
            for _, row in sw.iterrows():
                for s in strats:
                    ml_sw[(row["Pair"], s)] = row.get(s, 0.0)

    # Combine : poids final = markowitz_pair_weight * ml_strat_weight
    if mkw and ml_sw:
        for (pair, strat), sw_val in ml_sw.items():
            pw = mkw.get(pair, 0.0)
            combined[(pair, strat)] = pw * sw_val
        logging.info("Poids combines Markowitz x ML : %d entries", len(combined))
        return combined

    # Fallback : BL strategy-level weights
    if os.path.exists(WEIGHTS_FILE):
        bl = pd.read_csv(WEIGHTS_FILE, index_col=0, header=None).iloc[:, 0].to_dict()
        logging.info("Fallback BL weights : %s", {k: f"{v:.1%}" for k, v in bl.items()})
        return {"_bl": bl}  # signal special pour le mode BL

    return {}


def generate_execution_ticket(df_signals, capital=DEFAULT_CAPITAL):
    """
    Transforme les signaux en montants monetaires ($).

    Sizing : poids_final(pair, strat) = Markowitz(pair) × ML(strat|pair)
    → Size_USD = capital × poids_final
    → Cap total a 100% du capital (pas de levier)
    """
    last_date = df_signals.index.max()
    daily_data = df_signals[df_signals.index == last_date].copy()

    weights = _load_pair_strat_weights()
    is_bl_mode = "_bl" in weights

    orders = []
    for _, row in daily_data.iterrows():
        if row['Signal'] == 0:
            continue

        pair = row['Pair']
        strat = row['Strategy']

        if is_bl_mode:
            # Mode BL : ancien comportement (strategy-level)
            strat_weight = weights["_bl"].get(strat, 0.0)
            if strat_weight <= 0:
                continue
            size = calculate_order_size(
                capital=capital, strat_weight=strat_weight,
                volatility=row['Vol_20'], pair=pair,
            )
        else:
            # Mode Markowitz x ML : poids direct par (pair, strat)
            final_weight = weights.get((pair, strat), 0.0)
            if final_weight <= 0.001:
                continue
            size = int(capital * final_weight)

        if size <= 0:
            continue

        orders.append({
            "Date": last_date.date(),
            "Pair": pair,
            "Strategy": strat,
            "Direction": "LONG" if row['Signal'] == 1 else "SHORT",
            "Price": round(row['Close'], 4),
            "Size_USD": size,
            "Vol_20": round(row['Vol_20'], 4),
            "Hurst": round(row['Hurst'], 2)
        })

    orders_df = pd.DataFrame(orders)

    # Cap exposition totale a 100% du capital (pas de levier)
    if not orders_df.empty:
        total = orders_df["Size_USD"].sum()
        if total > capital:
            scale = capital / total
            orders_df["Size_USD"] = (orders_df["Size_USD"] * scale).astype(int)
            logging.info("Exposition cappee : $%d -> $%d (scale=%.2f)", total, capital, scale)

    return orders_df


def generate_longterm_g10_orders(capital=DEFAULT_CAPITAL):
    """
    Strat 5 — Long-Term Carry G10 : position permanente (30% du capital).
    Charge toutes les paires G10 depuis dataset_ml_ready.csv,
    indépendamment de la sélection IC (selected_pairs.csv).
    Budget par paire = (capital * LONGTERM_BUDGET) / nb_paires_signalées.
    """
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE, sep=";", decimal=",", index_col=0, parse_dates=True)
    df_g10 = df[df["Pair"].str.upper().isin(G10)].copy()
    if df_g10.empty:
        # Essai avec format sans "=" (ex: "EURUSD" vs "EURUSD=X")
        df_g10 = df[df["Pair"].str.replace("=X","").str.upper().isin(G10)].copy()
    if df_g10.empty:
        logging.warning("Aucune paire G10 trouvée dans %s", DATA_FILE)
        return pd.DataFrame()

    last_date = df_g10.index.max()
    orders = []

    for pair, grp in df_g10.groupby("Pair"):
        grp = add_strategy_indicators(grp.copy())
        last_row = grp[grp.index == last_date]
        if last_row.empty:
            continue
        row = last_row.iloc[0]
        clean_pair = pair.replace("=X", "")
        sig = strat_5_longterm_g10(row, clean_pair)
        if sig == 0:
            continue
        orders.append({
            "pair": clean_pair,
            "signal": sig,
            "vol": row["Vol_20"],
            "close": row["Close"],
            "hurst": row.get("Hurst", 0.5),
        })

    if not orders:
        return pd.DataFrame()

    budget_per_pair = (capital * LONGTERM_BUDGET) / len(orders)
    rows = []
    for o in orders:
        size = int(budget_per_pair / max(o["vol"] * 10, 0.001)) if o["vol"] > 0 else int(budget_per_pair)
        size = min(size, int(budget_per_pair * 2))   # cap à 2× la part équitable
        rows.append({
            "Date":      last_date.date(),
            "Pair":      o["pair"],
            "Strategy":  "Long-Term G10",
            "Direction": "LONG" if o["signal"] == 1 else "SHORT",
            "Price":     round(o["close"], 4),
            "Size_USD":  size,
            "Vol_20":    round(o["vol"], 4),
            "Hurst":     round(o["hurst"], 2),
        })
    logging.info("Long-Term G10 : %d positions generees (budget %.0f$)", len(rows), capital * LONGTERM_BUDGET)
    return pd.DataFrame(rows)


# =============================================================================
# EXECUTION
# =============================================================================

def run_trading_session():
    try:
        logging.info("--- DEBUT DE LA SESSION DE GENERATION D'ORDRES ---")

        # 1. Preparation de l'univers tactique (IC-sélectionné)
        df_universe = load_trading_universe()

        # 2. Signaux tactiques (70% capital)
        df_signals = process_signals(df_universe)
        tactical_orders = generate_execution_ticket(df_signals, capital=int(DEFAULT_CAPITAL * (1 - LONGTERM_BUDGET)))

        # 3. Signaux long terme G10 (30% capital)
        longterm_orders = generate_longterm_g10_orders(capital=DEFAULT_CAPITAL)

        # 4. Fusion
        final_orders = pd.concat([tactical_orders, longterm_orders], ignore_index=True)
        
        # 4. Sauvegarde et Export
        if not final_orders.empty:
            final_orders.to_csv(FINAL_ORDERS_FILE, index=False)
            logging.info(f"Session terminee : {len(final_orders)} ordres exportes vers {FINAL_ORDERS_FILE}")
            print("\n" + final_orders.to_string() + "\n")
        else:
            logging.info("Session terminee : Aucun signal d'entree detecte aujourd'hui.")
            # On cree un fichier vide pour eviter les erreurs de lecture du dashboard
            pd.DataFrame().to_csv(FINAL_ORDERS_FILE, index=False)

    except Exception as e:
        logging.error(f"Echec de la session : {e}")

if __name__ == "__main__":
    run_trading_session()
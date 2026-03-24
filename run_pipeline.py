"""
run_pipeline.py
---------------
FX QUANT SYSTEM — MASTER PIPELINE v2.1
Orchestre la sequence complète : Data -> Features -> ML -> Backtest -> Risk -> Allocation -> Orders.

Corrections vs v2.0 :
  - os.system() remplacé par subprocess.run() avec capture stdout/stderr
  - Timeout explicite par étape
  - Logs écrits dans pipeline.log en plus de la console
  - risk_manager.py ajouté après backtest_strategies.py
"""

import sys
import time
import logging
import subprocess
from datetime import datetime

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_fmt)
logger.addHandler(_console)

_file = logging.FileHandler("pipeline.log", encoding="utf-8")
_file.setFormatter(_fmt)
logger.addHandler(_file)

STEP_TIMEOUTS = {
    "data.py":                 300,
    "features.py":             180,
    "ic_audit.py":             120,
    "ml_pair_selection.py":    300,
    "backtest_strategies.py":  240,
    "strategy_ml.py":          300,
    "risk_manager.py":         240,
    "black_litterman.py":      120,
    "portfolio_utils.py":      120,
    "main_trading.py":          60,
}

def run_step(script_name: str, description: str) -> bool:
    timeout = STEP_TIMEOUTS.get(script_name, 180)
    logger.info(f"STARTING  [{script_name}] {description} (timeout={timeout}s)")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True, text=True, timeout=timeout,
        )
        duration = time.time() - start
        if result.returncode == 0:
            logger.info(f"SUCCESS   [{script_name}] terminé en {duration:.1f}s")
            for line in result.stdout.strip().splitlines()[-5:]:
                if line.strip():
                    logger.info(f"  stdout | {line}")
            return True
        logger.error(f"FAILED    [{script_name}] code={result.returncode} ({duration:.1f}s)")
        for line in result.stderr.strip().splitlines():
            logger.error(f"  {line}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT   [{script_name}] dépasse {timeout}s")
        return False
    except FileNotFoundError:
        logger.error(f"NOT FOUND [{script_name}] script introuvable")
        return False
    except Exception as exc:
        logger.error(f"EXCEPTION [{script_name}] {type(exc).__name__}: {exc}")
        return False

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"{'='*60}")
    logger.info(f"FX TRADING SYSTEM — PIPELINE v2.1  (run_id={run_id})")
    logger.info(f"{'='*60}")

    pipeline = [
        ("data.py",                "Ingestion des données (Yahoo Finance)"),
        ("features.py",            "Engineering des indicateurs + nouvelles features"),
        ("ic_audit.py",            "Audit IC/ICIR bilatéral — sélection univers + directions"),
        ("ml_pair_selection.py",   "ML couche 1 : sélection et scoring des paires actives"),
        ("backtest_strategies.py", "Simulation historique (PnL net + coûts + export par paire)"),
        ("strategy_ml.py",         "ML couche 2 : poids optimal par (paire × stratégie)"),
        ("risk_manager.py",        "Risque : VaR, drawdown, stress tests"),
        ("black_litterman.py",     "Allocation globale Black-Litterman -> optimal_weights.csv"),
        ("portfolio_utils.py",     "Optimisation Markowitz Max Sharpe / Min Variance"),
        ("main_trading.py",        "Génération du ticket d'ordres final"),
    ]

    overall_start = time.time()
    success_count = 0

    for script, desc in pipeline:
        if run_step(script, desc):
            success_count += 1
        else:
            logger.error(f"PIPELINE INTERROMPU : {desc} — voir pipeline.log")
            sys.exit(1)

    total = time.time() - overall_start
    logger.info(f"{'='*60}")
    logger.info(f"PIPELINE COMPLET en {total/60:.2f} min — {success_count}/{len(pipeline)} étapes OK")
    logger.info(f"Outputs : FINAL_ORDERS.csv | backtest_metrics.csv | risk_outputs/*.csv")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
"""
ic_audit.py
-----------
Audit IC/ICIR sur l'univers FX complet avec horizon dual G10/EM.

Critere de selection GARDE (bilateral) :
  |ICIR| >= ICIR_MIN  ET  (pct_pos < PCT_POS_LOW  OU  pct_pos > PCT_POS_HIGH)

Logique : un signal inversé stable (ICIR = -1.55) est aussi de l'alpha qu'un signal
direct (ICIR = +0.57). On rejette uniquement les signaux instables (pct_pos ~ 50%).

Horizon dual :
  G10 — FORWARD_MAIN=20j, seuils de stabilite relaches (PCT_POS 0.35 / 0.65)
  EM  — FORWARD_MAIN=5j,  seuils de stabilite stricts   (PCT_POS 0.30 / 0.70)
  ICIR_MIN=0.20 pour les deux groupes.

Exports :
  ic_audit_pair_detail.csv   — detail par (paire, feature, horizon)
  ic_audit_pair_scores.csv   — synthese par paire sur horizon propre (G10=20j, EM=5j)
  ic_signal_directions.csv   — signe IC moyen par (paire, feature) pour le pipeline ML
  ic_selected_pairs.csv      — les 8 paires retenues avec leur direction de signal
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

from macro_data import get_interest_rate_differential

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH     = "dataset_ml_ready.csv"
FORWARDS     = [5, 20]
ICIR_MIN     = 0.20       # commun G10 et EM
MIN_OBS      = 100

# G10 — horizon 20j, seuils de stabilite relaches
G10_FORWARD      = 20
G10_PCT_POS_LOW  = 0.35
G10_PCT_POS_HIGH = 0.65

# EM — horizon 5j, seuils de stabilite stricts
EM_FORWARD       = 5
PCT_POS_LOW      = 0.30   # EM : signal inversé stable pct_pos < 30 %
PCT_POS_HIGH     = 0.70   # EM : signal direct  stable pct_pos > 70 %

G10 = {"EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY",
       "USDCHF","USDCAD","USDSEK","USDNOK","USDDKK"}

PAIRS = [
    "EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCNY","USDCHF",
    "USDCAD","USDMXN","USDINR","USDBRL","USDRUB","USDKRW",
    "USDTRY","USDSEK","USDPLN","USDNOK","USDZAR","USDDKK","USDSGD",
    "USDILS","USDHKD","USDCLP","USDPKR","USDCZK","USDHUF",
]

# Features auditees (noms alignes avec dataset_ml_ready.csv apres maj features.py)
ALL_FEATURES = [
    "Vol_20", "Hurst",
    "Mom_20", "Mom_60", "Mom_120", "Mom_Ratio",
    "BB_Width", "Dist_MA20",
]
# RSI exclu : IC ~ 0 / p-value > 0.4 sur toutes les paires FX.
# Carry, Real_Rate_Diff, Mom_Carry exclus : constantes par paire,
# IC time-series = 0. Carry utilise comme overlay cross-sectionnel uniquement.
EXISTING = {"Vol_20", "Hurst"}

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
print("Chargement dataset_ml_ready.csv ...")
df = pd.read_csv(CSV_PATH, sep=";", parse_dates=["Date"], decimal=",")
df = df[df["Pair"].isin(PAIRS)].copy()

cols_num = [c for c in ALL_FEATURES + ["Adj Close", "Close", "Returns"] if c in df.columns]
df[cols_num] = df[cols_num].apply(pd.to_numeric, errors="coerce")
df.sort_values(["Pair", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  {len(df):,} lignes | {df['Pair'].nunique()} paires | "
      f"{df['Date'].min().date()} -> {df['Date'].max().date()}")

# ── TARGETS FORWARD ───────────────────────────────────────────────────────────
print("\nConstruction des cibles forward...")
for p, g in df.groupby("Pair"):
    idx = g.index
    for fwd in FORWARDS:
        df.loc[idx, f"fwd_{fwd}d"] = g["Returns"].shift(-fwd)

# ── CALCUL IC ─────────────────────────────────────────────────────────────────

def compute_ic(feat_series: pd.Series, fwd_series: pd.Series) -> dict | None:
    """
    IC (Spearman global) + ICIR via rolling windows annuelles (stride 20j).
    Retourne None si moins de MIN_OBS observations valides.
    """
    tmp = pd.DataFrame({"f": feat_series, "r": fwd_series}).dropna()
    if len(tmp) < MIN_OBS:
        return None

    rho, _ = stats.spearmanr(tmp["f"], tmp["r"])

    # Rolling IC — fenetre 252j, stride 20j
    ics = []
    for i in range(60, len(tmp), 20):
        w = tmp.iloc[max(0, i - 252):i]
        if len(w) >= 60:
            r, _ = stats.spearmanr(w["f"], w["r"])
            ics.append(r)

    if len(ics) > 5:
        ic_arr  = np.array(ics)
        icir    = ic_arr.mean() / (ic_arr.std(ddof=1) + 1e-8)
        pct_pos = float((ic_arr > 0).mean())
    else:
        icir    = rho / 0.3
        pct_pos = 0.5

    n      = len(tmp)
    t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2 + 1e-8))
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

    return {
        "IC"      : round(rho, 4),
        "ICIR"    : round(icir, 3),
        "IC_sign" : int(np.sign(rho)) if rho != 0 else 1,
        "p-value" : round(p_val, 4),
        "n_obs"   : n,
        "pct_pos" : pct_pos,
    }


print("Calcul IC paire par paire...")
rows = []

for pair, g in df.groupby("Pair"):
    groupe = "G10" if pair in G10 else "EM"
    carry  = get_interest_rate_differential(pair)

    for fwd in FORWARDS:
        fwd_col = f"fwd_{fwd}d"
        if fwd_col not in df.columns:
            continue

        for feat in ALL_FEATURES:
            if feat not in df.columns:
                continue

            res = compute_ic(g[feat], g[fwd_col])
            if res is None:
                continue

            pct_pos  = res["pct_pos"]
            abs_icir = abs(res["ICIR"])
            pval     = res["p-value"]

            # Seuils dependant du groupe (G10 vs EM)
            if pair in G10:
                pct_low  = G10_PCT_POS_LOW
                pct_high = G10_PCT_POS_HIGH
            else:
                pct_low  = PCT_POS_LOW
                pct_high = PCT_POS_HIGH

            # Critere bilateral : signal fort ET stable dans un sens
            stable_signal = (pct_pos < pct_low) or (pct_pos > pct_high)
            if abs_icir >= ICIR_MIN and pval < 0.05 and stable_signal:
                verdict = "GARDE"
            elif abs_icir >= 0.20 and pval < 0.15:
                verdict = "FILTRE"
            else:
                verdict = "JETTE"

            rows.append({
                "Pair"         : pair,
                "Groupe"       : groupe,
                "Feature"      : feat,
                "Type"         : "Existante" if feat in EXISTING else "Nouvelle",
                "Forward"      : f"{fwd}j",
                "IC"           : res["IC"],
                "ICIR"         : res["ICIR"],
                "IC_sign"      : res["IC_sign"],
                "p-value"      : res["p-value"],
                "n_obs"        : res["n_obs"],
                "pct_pos"      : f"{pct_pos * 100:.0f}%",
                "pct_pos_raw"  : round(pct_pos, 4),
                "Carry"        : round(carry, 4),
                "Verdict"      : verdict,
            })

detail = pd.DataFrame(rows)

# ── AFFICHAGE PAR PAIRE ───────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

print("\n" + "=" * 140)
print("AUDIT IC/ICIR — BILATERAL (signal direct ET inversé)")
print("=" * 140)

for pair in PAIRS:
    sub = detail[detail["Pair"] == pair]
    if sub.empty:
        continue
    ng = (sub["Verdict"] == "GARDE").sum()
    nf = (sub["Verdict"] == "FILTRE").sum()
    print(f"\n{'─' * 140}")
    carry_str = f"Carry={get_interest_rate_differential(pair):+.2f}%"
    print(f"  {pair} [{sub['Groupe'].iloc[0]}]  {carry_str}   GARDE: {ng}   FILTRE: {nf}")
    print(f"{'─' * 140}")
    print(sub[["Feature", "Type", "Forward", "IC", "ICIR", "IC_sign", "p-value", "pct_pos", "Verdict"]]
          .sort_values(["Forward", "ICIR"], key=lambda s: s if s.dtype != object else s.map(
              lambda x: abs(float(x)) if isinstance(x, (int, float)) else x), ascending=[True, False])
          .to_string(index=False))

# ── SYNTHESE ─────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 90)
print("SYNTHESE — nb paires GARDE par feature et horizon")
print("=" * 90)
pivot = (detail[detail["Verdict"] == "GARDE"]
         .groupby(["Feature", "Forward"])["Pair"]
         .count()
         .unstack(fill_value=0))
if not pivot.empty:
    print(pivot.sort_values(pivot.columns[-1], ascending=False).to_string())
else:
    print("Aucune feature GARDE.")

# ── SCORES PAR PAIRE (horizon propre G10=20j / EM=5j) ────────────────────────
print(f"\n\n" + "=" * 90)
print(f"TOP PAIRES — features GARDE sur horizon propre (G10={G10_FORWARD}j / EM={EM_FORWARD}j)")
print("=" * 90)

# detail_main : pour chaque ligne, conserver seulement l'horizon propre a la paire
detail_main = detail[detail.apply(
    lambda r: r["Forward"] == f"{G10_FORWARD}j" if r["Pair"] in G10 else r["Forward"] == f"{EM_FORWARD}j",
    axis=1
)].copy()

score = (detail_main[detail_main["Verdict"].isin(["GARDE", "FILTRE"])]
         .groupby(["Pair", "Groupe"])
         .agg(
             nb_garde    = ("Verdict",  lambda x: (x == "GARDE").sum()),
             nb_filtre   = ("Verdict",  lambda x: (x == "FILTRE").sum()),
             best_ICIR   = ("ICIR",     lambda x: x.abs().max()),
             mean_ICIR   = ("ICIR",     "mean"),
             carry       = ("Carry",    "first"),
         )
         .sort_values(["nb_garde", "best_ICIR"], ascending=False)
         .reset_index())
print(score.to_string(index=False))

# ── SELECTION DES 8 PAIRES ────────────────────────────────────────────────────
print(f"\n\n" + "=" * 90)
print(f"PAIRES SELECTIONNEES — |ICIR| >= {ICIR_MIN} | G10 pct_pos ({G10_PCT_POS_LOW*100:.0f}%-{G10_PCT_POS_HIGH*100:.0f}%) | EM ({PCT_POS_LOW*100:.0f}%-{PCT_POS_HIGH*100:.0f}%)")
print("=" * 90)

garde_main = detail_main[detail_main["Verdict"] == "GARDE"]
selected_pairs_list = garde_main["Pair"].unique().tolist()

selected_summary = []
for pair in selected_pairs_list:
    sub = garde_main[garde_main["Pair"] == pair]
    mean_icir = sub["ICIR"].mean()
    direction = "POSITIVE" if mean_icir >= 0 else "INVERTED"
    fwd_used = G10_FORWARD if pair in G10 else EM_FORWARD
    selected_summary.append({
        "Pair"         : pair,
        "Groupe"       : sub["Groupe"].iloc[0],
        "Forward_used" : fwd_used,
        "nb_garde"     : len(sub),
        "mean_ICIR"    : round(mean_icir, 3),
        "best_absICIR" : round(sub["ICIR"].abs().max(), 3),
        "signal_dir"   : direction,
        "carry"        : round(get_interest_rate_differential(pair), 4),
    })

selected_df = pd.DataFrame(selected_summary).sort_values("best_absICIR", ascending=False)
print(selected_df.to_string(index=False))

# ── EXPORT ic_signal_directions.csv ──────────────────────────────────────────
# Signe IC moyen par (paire, feature) sur l'horizon propre a chaque paire.
# G10 -> G10_FORWARD (20j) | EM -> EM_FORWARD (5j)
# Utilise par ml_pair_selection.py pour retourner les features avant entree modele.
print(f"\nConstruction ic_signal_directions.csv (G10={G10_FORWARD}j / EM={EM_FORWARD}j)...")

dir_rows = []
for pair in selected_pairs_list:
    # Choisir l'horizon propre a la paire
    fwd_label = f"{G10_FORWARD}j" if pair in G10 else f"{EM_FORWARD}j"
    sub_fwd = detail[(detail["Pair"] == pair) & (detail["Forward"] == fwd_label)]
    for feat in ALL_FEATURES:
        feat_rows = sub_fwd[sub_fwd["Feature"] == feat]
        if feat_rows.empty:
            ic_sign = 1  # pas de signal => on ne retourne pas
        else:
            ic_mean = feat_rows["IC"].mean()
            ic_sign = int(np.sign(ic_mean)) if ic_mean != 0 else 1
        dir_rows.append({
            "Pair"    : pair,
            "Feature" : feat,
            "Forward" : fwd_label,
            "IC_mean" : round(feat_rows["IC"].mean(), 4) if not feat_rows.empty else 0.0,
            "IC_sign" : ic_sign,
            "ICIR"    : round(feat_rows["ICIR"].mean(), 3) if not feat_rows.empty else 0.0,
        })

signal_dir_df = pd.DataFrame(dir_rows)

# ── EXPORTS ───────────────────────────────────────────────────────────────────
detail.to_csv("ic_audit_pair_detail.csv", index=False)
score.to_csv("ic_audit_pair_scores.csv", index=False)
signal_dir_df.to_csv("ic_signal_directions.csv", index=False)
selected_df.to_csv("ic_selected_pairs.csv", index=False)

print("\nExportes :")
print("  ic_audit_pair_detail.csv")
print("  ic_audit_pair_scores.csv")
print("  ic_signal_directions.csv  <-- signe IC par (paire, feature) pour ML")
print("  ic_selected_pairs.csv     <-- 8 paires retenues")

"""
macro_data.py
-------------
Taux directeurs des banques centrales — mise a jour manuelle.

Source de reference : tradingeconomics.com/country-list/interest-rate
Derniere mise a jour : 23 mars 2026

Prochaines reunions :
  Fed   : 6-7 mai 2026
  BCE   : 30 avril 2026
  BoE   : 30 avril 2026
  BoJ   : 28 avril 2026
  RBA   : 5-6 mai 2026
  BoC   : 29 avril 2026
"""

import logging
from datetime import date

logger = logging.getLogger(__name__)

# =============================================================================
# DATE DE DERNIERE MISE A JOUR
# Modifier cette date a chaque revision du dictionnaire.
# Un warning est emis si les taux ont plus de STALENESS_WARNING_DAYS jours.
# =============================================================================

RATES_LAST_UPDATED = date(2026, 3, 23)
STALENESS_WARNING_DAYS = 45

# =============================================================================
# TAUX DIRECTEURS — 23 MARS 2026
#
# Decisions de la semaine du 17-19 mars 2026 :
#   USD : maintenu 3.625% (fourchette 3.50-3.75%)  — Fed 18/03
#   EUR : maintenu 2.00% (taux de depot)            — BCE 19/03
#   GBP : maintenu 3.75%                            — BoE 19/03
#   JPY : maintenu 0.75%                            — BoJ 19/03  (etait 0.10)
#   CHF : maintenu 0.00%                            — SNB 19/03  (etait 1.75)
#   AUD : hausse +25bp a 4.10%                      — RBA 17/03  (etait 3.85)
#   CAD : maintenu 2.25%                            — BoC 18/03  (etait 5.00)
#   NZD : estime 3.50% (plusieurs baisses depuis 5.50%)
#   TRY : estime 42.50% (cycle de baisse depuis 50%)
#   BRL : estime 14.75% (cycle de hausse 2025)
#   MXN : estime 9.00% (plusieurs baisses 2025-2026)
#   ZAR : estime 7.50%
#   INR : estime 6.25% (coupe 25bp dec 2025)
# =============================================================================

CENTRAL_BANK_RATES = {
    # Majeurs
    'USD': 3.625,
    'EUR': 2.00,
    'GBP': 3.75,
    'JPY': 0.75,
    'CHF': 0.00,
    'AUD': 4.10,
    'CAD': 2.25,
    'NZD': 3.50,
    # Haut rendement / Carry
    'TRY': 42.50,
    'BRL': 14.75,
    'MXN': 9.00,
    'ZAR': 7.50,
    'INR': 6.25,
    # Asie / Emergents supplementaires
    'CNY': 3.10,   # PBOC, mars 2026
    'HKD': 5.25,   # HKMA — arrime au USD
    'KRW': 2.75,   # BoK, mars 2026
    'SGD': 3.50,   # MAS, estimation mars 2026
    'PKR': 13.00,  # SBP, mars 2026 (cycle de baisse depuis 22%)
    'ILS': 4.50,   # BoI, mars 2026
    # Europe / Nordiques
    'NOK': 4.25,   # Norges Bank, mars 2026
    'SEK': 2.50,   # Riksbank, mars 2026
    'PLN': 5.25,   # NBP, mars 2026
    'CZK': 3.75,   # CNB, mars 2026
    'HUF': 6.50,   # MNB, mars 2026
    'DKK': 2.10,   # DN (suit la BCE), mars 2026
    # Ameriques / Autres
    'CLP': 5.00,   # BCCh, mars 2026
    'RUB': 21.00,  # CBR, mars 2026
}


# =============================================================================
# CPI ANNUEL ESTIME — 23 MARS 2026
#
# Inflation glissante 12 mois par devise (sources : Banques centrales / Eurostat / BLS).
# Mise a jour manuelle recommandee tous les 2 mois.
#
# Logique real_rate = taux_nominal - cpi
# Ex : TRY  real = 42.50 - 38.00 = +4.50%   (taux tres eleve mais inflation aussi)
#      CHF  real =  0.00 -  0.70 = -0.70%   (taux negatif en termes reels)
#      USD  real =  3.625 - 2.80 = +0.825%
# =============================================================================

CPI_RATES = {
    # Majeurs
    'USD': 2.80,   # BLS CPI, fevrier 2026
    'EUR': 2.30,   # Eurostat HICP, fevrier 2026
    'GBP': 3.00,   # ONS CPI, fevrier 2026
    'JPY': 2.50,   # BoJ CPI, fevrier 2026
    'CHF': 0.70,   # OFS IPC, fevrier 2026
    'AUD': 3.20,   # ABS CPI, T4 2025 annualise
    'CAD': 2.60,   # StatCan IPC, fevrier 2026
    'NZD': 2.50,   # Stats NZ CPI, T4 2025
    # Emergents
    'TRY': 38.00,  # TUIK, fevrier 2026 (cycle de desinflation depuis 85%)
    'BRL':  5.50,  # IBGE IPCA, fevrier 2026
    'MXN':  4.50,  # INEGI, fevrier 2026
    'ZAR':  4.80,  # Stats SA, fevrier 2026
    'INR':  4.50,  # MoSPI, fevrier 2026
    'CNY':  0.50,  # NBS, fevrier 2026 (risque deflation)
    'HKD':  2.50,  # C&SD HK, fevrier 2026
    'KRW':  2.00,  # Statistics Korea, fevrier 2026
    'SGD':  2.20,  # DOS Singapore, fevrier 2026
    'PKR': 10.00,  # PBS Pakistan, mars 2026 (chute depuis 30%)
    'ILS':  3.50,  # CBS Israel, fevrier 2026
    # Europe / Nordiques
    'NOK':  2.80,  # SSB, fevrier 2026
    'SEK':  1.80,  # SCB, fevrier 2026
    'PLN':  4.50,  # GUS, fevrier 2026
    'CZK':  2.50,  # CSU, fevrier 2026
    'HUF':  4.50,  # KSH, fevrier 2026
    'DKK':  2.20,  # DST, fevrier 2026
    # Autres
    'CLP':  4.20,  # INE Chile, fevrier 2026
    'RUB':  9.00,  # Rosstat, fevrier 2026
}


def get_real_rate(currency: str) -> float:
    """
    Taux reel annualise (%) = taux_nominal - CPI.
    Retourne 0.0 si la devise est inconnue.
    """
    nominal = CENTRAL_BANK_RATES.get(currency.upper())
    cpi     = CPI_RATES.get(currency.upper())
    if nominal is None or cpi is None:
        return 0.0
    return round(nominal - cpi, 4)


def get_real_rate_differential(pair: str) -> float:
    """
    Differentiel de taux reels annualise (%) entre devise base et devise cotee.
    Plus robuste que le carry nominal car integre l'inflation.
    Ex : USDTRY -> real_USD - real_TRY = 0.825 - 4.50 = -3.675
         => USD moins attractif en termes reels malgre un carry nominal positif.
    """
    base  = pair[:3].upper()
    quote = pair[3:].upper()
    return round(get_real_rate(base) - get_real_rate(quote), 4)


def check_rates_freshness() -> int:
    """
    Retourne le nombre de jours depuis la derniere mise a jour des taux.
    Emet un warning logger si superieur a STALENESS_WARNING_DAYS.
    Appele a l'import — le Streamlit dashboard peut recuperer cette valeur.
    """
    delta = (date.today() - RATES_LAST_UPDATED).days
    if delta > STALENESS_WARNING_DAYS:
        logger.warning(
            "Taux potentiellement obsoletes : derniere mise a jour il y a %d jours (%s).",
            delta,
            RATES_LAST_UPDATED.strftime("%d/%m/%Y"),
        )
    return delta


def get_interest_rate_differential(pair: str) -> float:
    """
    Differentiel de taux annualise (%) entre devise base et devise cotee.
    Exemple : get_interest_rate_differential('USDJPY') -> 3.625 - 0.75 = 2.875
    Retourne 0.0 si l'une des devises est inconnue.
    """
    base = pair[:3].upper()
    quote = pair[3:].upper()

    base_rate = CENTRAL_BANK_RATES.get(base)
    quote_rate = CENTRAL_BANK_RATES.get(quote)

    if base_rate is None:
        logger.warning("Taux inconnu pour '%s' (paire %s).", base, pair)
        return 0.0
    if quote_rate is None:
        logger.warning("Taux inconnu pour '%s' (paire %s).", quote, pair)
        return 0.0

    return round(base_rate - quote_rate, 4)


def get_carry_score(pair: str) -> float:
    """
    Alias de get_interest_rate_differential.
    Positif = favorable au LONG base / SHORT quote.
    """
    return get_interest_rate_differential(pair)


def is_ndf(pair: str) -> bool:
    """
    True si la paire est reglee comme NDF (Non-Deliverable Forward).
    Impacte le position sizing dans portofolio.py (reduction de 50%).
    """
    ndf_currencies = {'INR', 'CNY', 'BRL', 'KRW', 'IDR', 'RUB', 'TWD', 'CLP', 'COP'}
    return any(ccy in pair.upper() for ccy in ndf_currencies)


# Verification a l'import — visible dans les logs du pipeline, pas en console.
check_rates_freshness()
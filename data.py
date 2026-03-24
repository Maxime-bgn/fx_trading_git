import yfinance as yf

pairs_list = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCNY", "USDCHF",
    "USDCAD", "USDMXN", "USDINR", "USDBRL", "USDRUB", "USDKRW",
    "USDTRY", "USDSEK", "USDPLN", "USDNOK", "USDZAR", "USDDKK", "USDSGD",
    "USDILS", "USDHKD", "USDCLP", "USDPKR", "USDCZK", "USDHUF"
]

def download_fx_data(pairs, period="10y"):
    for pair in pairs:
        ticker = f"{pair}=X"
        print(f"Telechargement {ticker} ...")

        df = yf.download(
            ticker,
            period=period,
            auto_adjust=False,
            group_by="column",
            progress=False
        )

        if df.empty:
            print(f"Aucune donnee pour {ticker}")
            continue

        #  FIX CRUCIAL : enlever MultiIndex si présent
        if isinstance(df.columns, tuple) or hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        df.to_csv(
            f"{ticker}.csv",
            sep=";",
            decimal=","
        )

        print(f"saved : {ticker}.csv")

if __name__ == "__main__":
    download_fx_data(pairs_list)

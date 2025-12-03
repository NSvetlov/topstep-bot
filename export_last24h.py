import os
from typing import List

import pandas as pd
import yfinance as yf


def download_1m_last24h(ticker: str) -> pd.DataFrame:
    # Download 1m data; 2d window ensures we can slice last 24h robustly
    df = yf.download(ticker, interval="1m", period="2d", group_by="column", auto_adjust=False)
    if df.empty:
        return df
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df.droplevel(1, axis=1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, level=0, axis=1)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # Ensure tz -> US/Central
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        df.index = df.index.tz_convert("US/Central")
    cutoff = df.index.max() - pd.Timedelta(hours=24)
    df = df[df.index >= cutoff].copy()
    df.insert(0, "timestamp", df.index.strftime("%Y-%m-%d %H:%M:%S%z"))
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    # ENV: EXPORT_TICKERS or BACKTEST_TICKERS; default MNQ=F
    raw = os.environ.get("EXPORT_TICKERS") or os.environ.get("BACKTEST_TICKERS") or "MNQ=F"
    tickers: List[str] = [t.strip() for t in raw.split(",") if t.strip()]
    for tkr in tickers:
        try:
            print(f"Fetching 1m last 24h for {tkr}...")
            df = download_1m_last24h(tkr)
            if df.empty:
                print(f"No data for {tkr}")
                continue
            sym = (tkr.split("=")[0] if "=" in tkr else tkr).strip().upper()
            out = f"{sym.lower()}_1m_last24h.csv"
            df.to_csv(out, index=False)
            print(f"Saved {out} ({len(df)} rows)")
        except Exception as e:
            print(f"Error exporting {tkr}: {e}")


if __name__ == "__main__":
    main()


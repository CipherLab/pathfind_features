#!/usr/bin/env python3
"""
Lightweight market data helper: fetch daily prices for any tickers via yfinance
and map them to Numerai-style eras (0001..NNNN) using a deterministic mapping.

We don't know true calendar dates behind eras here, so we provide two mappings:
- ordinal mapping: assign the k-th unique date to era k zero-padded (e.g., 0001)
- custom mapping: accept a CSV mapping file era,date to use user-provided dates

This keeps the pipeline flexible: you can try any ticker, or an ensemble.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
import os
import pickle
from pathlib import Path


# Global cache for fetched ticker data
_TICKER_CACHE: Dict[str, pd.DataFrame] = {}


def fetch_ticker_history(ticker: str, period: str = "max", interval: str = "1d",
                        cache_dir: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Fetch OHLCV for a ticker using yfinance. Requires internet access.

    Caches results to avoid re-downloading.
    """
    cache_key = f"{ticker}_{period}_{interval}"
    cache_file = None

    if cache_dir and not refresh:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # Fall through to fetch

    if cache_key in _TICKER_CACHE and not refresh:
        return _TICKER_CACHE[cache_key].copy()

    import yfinance as yf

    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=False)
    if not isinstance(hist.index, pd.DatetimeIndex) or hist.empty:
        raise ValueError(f"No data returned for {ticker}")
    hist = hist.reset_index().rename(columns={"Date": "date"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.date

    # Cache in memory and disk
    _TICKER_CACHE[cache_key] = hist.copy()
    if cache_file:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(hist, f)
        except Exception:
            pass  # Ignore cache write failures

    return hist


def map_dates_to_eras(dates: pd.Series,
                      mode: Literal["ordinal", "custom"] = "ordinal",
                      mapping_csv: Optional[str] = None) -> pd.DataFrame:
    """Create an era mapping DataFrame with columns ['era','date'].

    - ordinal: sort unique dates ascending and label as 0001..NNNN
    - custom: read mapping_csv with columns ['era','date'] (date parseable)
    """
    if mode == "custom":
        if not mapping_csv:
            raise ValueError("mapping_csv is required for custom mode")
        m = pd.read_csv(mapping_csv)
        if not {"era", "date"}.issubset(m.columns):
            raise ValueError("custom mapping must have columns: era,date")
        m["date"] = pd.to_datetime(m["date"]).dt.date
        m["era"] = m["era"].astype(str).str.zfill(4)
        return m[["era", "date"]].drop_duplicates()

    # ordinal
    u = pd.Series(pd.to_datetime(pd.Series(dates)).dt.date.unique())
    u = pd.Series(sorted(u.dropna().tolist()))
    eras = [str(i + 1).zfill(4) for i in range(len(u))]
    return pd.DataFrame({"era": eras, "date": u.values})


def build_era_ticker_features(eras: pd.Series,
                              tickers: List[str],
                              agg: Literal["close", "ret", "zscore", "ensemble_ret"] = "ret",
                              period: str = "max",
                              interval: str = "1d",
                              mapping_mode: Literal["ordinal", "custom"] = "ordinal",
                              mapping_csv: Optional[str] = None,
                              cache_dir: Optional[str] = None,
                              refresh: bool = False) -> pd.DataFrame:
    """Create a DataFrame with ['era'] + per-ticker features aligned to eras.

    Caches fetched data to avoid re-downloading.
    """
    if not len(tickers):
        raise ValueError("tickers list is empty")

    # Build mapping from dates to eras
    # We use the union of all dates across tickers to form a common mapping
    all_dates = []
    per_ticker = {}
    for t in tickers:
        df = fetch_ticker_history(t, period=period, interval=interval,
                                 cache_dir=cache_dir, refresh=refresh)
        per_ticker[t] = df
        all_dates.append(df["date"])

    all_dates = pd.concat([pd.Series(d) for d in all_dates], ignore_index=True).drop_duplicates()
    era_map = map_dates_to_eras(all_dates, mode=mapping_mode, mapping_csv=mapping_csv)

    out = era_map.copy()

    # Compute per-ticker feature
    feat_cols = []
    for t, df in per_ticker.items():
        df2 = df.merge(era_map, on="date", how="inner")
        if agg == "close":
            s = df2[["era", "Close"]].rename(columns={"Close": f"{t}_close"})
        else:
            ret = df2["Close"].pct_change()
            if agg == "zscore":
                r = (ret - ret.mean()) / (ret.std(ddof=1) if ret.std(ddof=1) else 1.0)
                s = pd.DataFrame({"era": df2["era"], f"{t}_zret": r.values})
            else:  # ret or ensemble_ret share same per-ticker base
                s = pd.DataFrame({"era": df2["era"], f"{t}_ret": ret.values})

        out = out.merge(s, on="era", how="left")
        feat_cols.extend([c for c in out.columns if c not in ["era", "date"]])

    # If ensemble requested, create mean return column
    if agg == "ensemble_ret":
        ret_cols = [c for c in out.columns if c.endswith("_ret")]
        if ret_cols:
            out["ensemble_ret"] = out[ret_cols].mean(axis=1)

    # Keep only era + features (drop date)
    out = out.drop(columns=["date"]).drop_duplicates(subset=["era"])
    # Align to provided eras list if needed
    if eras is not None:
        eras = pd.Series(eras).astype(str).str.zfill(4)
        out = out.merge(pd.DataFrame({"era": eras.unique()}), on="era", how="right")
        out = out.sort_values("era").reset_index(drop=True)

    return out

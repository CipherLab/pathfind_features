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
import logging

logger = logging.getLogger(__name__)


# Global cache for fetched ticker data
_TICKER_CACHE: Dict[str, pd.DataFrame] = {}


def _check_internet_connectivity() -> bool:
    """Check if we have internet connectivity by trying multiple reliable hosts."""
    import socket
    
    test_hosts = [
        ('8.8.8.8', 53),        # Google DNS
        ('1.1.1.1', 53),        # Cloudflare DNS
        ('208.67.222.222', 53), # OpenDNS
    ]
    
    # First check basic internet connectivity
    basic_connectivity = False
    for host, port in test_hosts:
        try:
            socket.setdefaulttimeout(3)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            basic_connectivity = True
            break
        except socket.error:
            continue
    
    if not basic_connectivity:
        return False
    
    # Now check Yahoo Finance specifically
    try:
        # Try to resolve Yahoo Finance hostname
        socket.gethostbyname('fc.yahoo.com')
        
        # Try to connect to Yahoo Finance
        yahoo_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        yahoo_socket.settimeout(5)
        yahoo_socket.connect(('fc.yahoo.com', 443))
        yahoo_socket.close()
        return True
    except (socket.gaierror, socket.error):
        logger.warning("Yahoo Finance servers are not accessible, but basic internet connectivity is available")
        return False
    
    # Fallback to HTTP check if DNS fails
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        return True
    except:
        pass
    
    return False


def fetch_ticker_history(ticker: str, period: str = "max", interval: str = "1d",
                        cache_dir: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Fetch OHLCV for a ticker using yfinance. Requires internet access.

    Caches results to avoid re-downloading. Errors if no data available from API or cache.
    """
    cache_key = f"{ticker}_{period}_{interval}"
    cache_file = None

    if cache_dir and not refresh:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached data for {ticker} from {cache_file}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")

    if cache_key in _TICKER_CACHE and not refresh:
        logger.info(f"Using in-memory cached data for {ticker}")
        return _TICKER_CACHE[cache_key].copy()

    # Check connectivity BEFORE attempting to fetch
    if not _check_internet_connectivity():
        raise ValueError(f"No internet connection available. Cannot fetch data for {ticker}")

    logger.info(f"Fetching fresh data for {ticker} from yfinance (period={period}, interval={interval})")
    
    import yfinance as yf

    # Try to fetch data with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to fetch {ticker}")
            
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval, auto_adjust=False)

            if not isinstance(hist.index, pd.DatetimeIndex) or hist.empty:
                raise ValueError(f"No data returned for {ticker}")

            hist = hist.reset_index().rename(columns={"Date": "date"})
            hist["date"] = pd.to_datetime(hist["date"]).dt.date

            # Validate we have the expected columns
            required_cols = ["date", "Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in hist.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

            logger.info(f"Successfully fetched {len(hist)} rows of data for {ticker}")

            # Cache in memory and disk
            _TICKER_CACHE[cache_key] = hist.copy()
            if cache_file:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(hist, f)
                    logger.info(f"Cached data for {ticker} to {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to save cache for {ticker}: {e}")

            return hist

        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for specific error types
            if "failed to connect" in error_msg or "could not connect" in error_msg:
                logger.warning(f"Connection failed for {ticker} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"All connection attempts failed for {ticker}")
            elif "no data" in error_msg or "empty" in error_msg:
                logger.warning(f"No data available for {ticker}: {e}")
            else:
                logger.error(f"Unexpected error fetching {ticker}: {e}")
            
            # If this is the last attempt, raise the error
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts: {e}")
    
    # This should never be reached, but just in case
    raise ValueError(f"Failed to fetch data for {ticker} after all attempts")


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

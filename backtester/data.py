"""Price loading (Yahoo Finance via yfinance) with an explicit missing-data report."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .universe import to_yf_ticker

# Single-day moves that are corporate actions Yahoo does not adjust for, not
# real P&L. The return on that day is replaced (and reported).
# TMPV 2025-10-14: Tata Motors demerged its commercial-vehicle business (TMCV);
# TMPV's price fell ~40% ex-demerger while TMCV only listed on 2025-11-12.
CORPORATE_ACTION_RETURN_OVERRIDES: dict[tuple[str, str], float] = {
    ("TMPV", "2025-10-14"): 0.0,
}

# Optional stand-ins for delisted names with no Yahoo history. Only used when
# missing_policy="proxy". HDFC Ltd merged into HDFC Bank (2023-07-13, 42 HDFCBANK
# per 25 HDFC); HDFCBANK returns are a rough stand-in, not HDFC's own.
DEFAULT_PROXIES: dict[str, str] = {"HDFC": "HDFCBANK"}


@dataclass
class PriceData:
    close: pd.DataFrame                      # dates x NSE symbols (adjusted close)
    missing: list[str] = field(default_factory=list)   # symbols with no price history at all
    first_date: dict[str, pd.Timestamp] = field(default_factory=dict)
    last_date: dict[str, pd.Timestamp] = field(default_factory=dict)


def download_prices(symbols, start, end, cache_dir: str | Path | None = None) -> PriceData:
    """Download adjusted closes for NSE symbols. Never drops a symbol silently:
    anything without data is listed in ``PriceData.missing``."""
    import yfinance as yf

    symbols = sorted(set(symbols))
    cache = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = Path(cache_dir) / f"close_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.pkl"
        if cache.exists():
            close = pd.read_pickle(cache)
            if set(symbols) <= set(close.columns):
                return _summarise(close[symbols])

    tickers = [to_yf_ticker(s) for s in symbols]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=tickers)
    close.columns = symbols
    close.index = pd.to_datetime(close.index).tz_localize(None)
    if cache is not None:
        close.to_pickle(cache)
    return _summarise(close)


def _summarise(close: pd.DataFrame) -> PriceData:
    missing = [c for c in close.columns if close[c].dropna().empty]
    first = {c: close[c].first_valid_index() for c in close.columns if c not in missing}
    last = {c: close[c].last_valid_index() for c in close.columns if c not in missing}
    return PriceData(close=close, missing=missing, first_date=first, last_date=last)


def daily_returns(close: pd.DataFrame,
                  overrides: dict[tuple[str, str], float] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Simple daily returns (NaN where no price). Applies corporate-action overrides
    and returns the list of overrides actually applied."""
    rets = close.pct_change(fill_method=None)
    applied = []
    for (sym, day), val in (overrides if overrides is not None else CORPORATE_ACTION_RETURN_OVERRIDES).items():
        ts = pd.Timestamp(day)
        if sym in rets.columns and ts in rets.index:
            applied.append({"symbol": sym, "date": day, "raw_return": float(rets.at[ts, sym]), "used": val})
            rets.at[ts, sym] = val
    return rets, applied

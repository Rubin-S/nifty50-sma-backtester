"""Point-in-time (PIT) Nifty 50 universe.

Membership data: aditya-jha/nse-historical-membership
(index_history/data/index_membership_history.csv, CC BY 4.0), derived from
NSE press releases / NSE Exchange circulars. Only the Nifty 50 rows are
vendored in ``data/nifty50_membership_history.csv``.

Intervals are half-open ``[valid_from, valid_to)``; an empty ``valid_to``
means the stock is still a member as of the dataset's freshness date.
Symbols are canonicalised by the dataset to the *latest* name in any rename
chain (e.g. MINDTREE/LTI/LTIM -> LTM, ZOMATO -> ETERNAL, TATAMOTORS -> TMPV,
INFRATEL -> INDUSTOWER), which is also the symbol Yahoo Finance uses today.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_HISTORY_CSV = DATA_DIR / "nifty50_membership_history.csv"
DEFAULT_CURRENT_CSV = DATA_DIR / "nifty50_current_list.csv"
UPSTREAM_HISTORY_URL = (
    "https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/"
    "main/index_history/data/index_membership_history.csv"
)
NSE_CURRENT_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"

# Last date the vendored membership snapshot is known to be correct for
# (upstream COVERAGE.md: "index membership through 2026-05-15").
DATASET_COVERAGE_END = pd.Timestamp("2026-05-15")

# NSE symbol -> Yahoo Finance ticker, where it is not simply SYMBOL + ".NS".
YF_TICKER_OVERRIDES: dict[str, str] = {}


def to_yf_ticker(symbol: str) -> str:
    return YF_TICKER_OVERRIDES.get(symbol, f"{symbol}.NS")


class Nifty50Membership:
    """Answers "which stocks were in the Nifty 50 on date D?"."""

    def __init__(self, history: pd.DataFrame, index_name: str = "Nifty 50",
                 coverage_end: pd.Timestamp | None = DATASET_COVERAGE_END):
        required = {"index_name", "symbol", "valid_from", "valid_to"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"membership history missing columns: {sorted(missing)}")
        h = history[history["index_name"] == index_name].copy()
        if h.empty:
            raise ValueError(f"no rows for index {index_name!r}")
        h["valid_from"] = pd.to_datetime(h["valid_from"])
        h["valid_to"] = pd.to_datetime(h["valid_to"])  # NaT = still a member
        bad = h[h["valid_to"].notna() & (h["valid_to"] <= h["valid_from"])]
        if not bad.empty:
            raise ValueError(f"invalid intervals: {bad[['symbol', 'valid_from', 'valid_to']]}")
        self.history = h.reset_index(drop=True)
        self.index_name = index_name
        self.coverage_end = coverage_end
        self._warned_beyond = False

    # ------------------------------------------------------------------ io
    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_HISTORY_CSV, **kw) -> "Nifty50Membership":
        return cls(pd.read_csv(path), **kw)

    @classmethod
    def from_upstream(cls, url: str = UPSTREAM_HISTORY_URL, **kw) -> "Nifty50Membership":
        return cls(pd.read_csv(url), **kw)

    # ------------------------------------------------------------- queries
    def members_on(self, date) -> list[str]:
        d = pd.Timestamp(date)
        if self.coverage_end is not None and d > self.coverage_end and not self._warned_beyond:
            warnings.warn(
                f"{d.date()} is after the membership dataset's coverage end "
                f"({self.coverage_end.date()}); later index changes are not reflected.",
                stacklevel=2,
            )
            self._warned_beyond = True
        h = self.history
        mask = (h["valid_from"] <= d) & (h["valid_to"].isna() | (h["valid_to"] > d))
        return sorted(h.loc[mask, "symbol"].unique())

    def is_member(self, symbol: str, date) -> bool:
        return symbol in self.members_on(date)

    def current_members(self) -> list[str]:
        """Open intervals = members as of the dataset's freshness date."""
        return sorted(self.history.loc[self.history["valid_to"].isna(), "symbol"].unique())

    def all_symbols(self, start, end) -> list[str]:
        """Every symbol that was a member at any time in [start, end]."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        h = self.history
        mask = (h["valid_from"] <= e) & (h["valid_to"].isna() | (h["valid_to"] > s))
        return sorted(h.loc[mask, "symbol"].unique())

    def membership_matrix(self, dates: Iterable, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        """Boolean DataFrame (dates x symbols): True if member on that date."""
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        if symbols is None:
            symbols = self.all_symbols(idx.min(), idx.max())
        symbols = list(symbols)
        out = pd.DataFrame(False, index=idx, columns=symbols)
        for row in self.history.itertuples(index=False):
            if row.symbol not in out.columns:
                continue
            m = idx >= row.valid_from
            if pd.notna(row.valid_to):
                m &= idx < row.valid_to
            out.loc[m, row.symbol] = True
        if self.coverage_end is not None and len(idx) and idx.max() > self.coverage_end:
            warnings.warn(
                f"membership requested up to {idx.max().date()}, dataset coverage ends "
                f"{self.coverage_end.date()}; later index changes are not reflected.",
                stacklevel=2,
            )
        return out

    def changes(self, start, end) -> pd.DataFrame:
        """Index additions/deletions with effective date in (start, end]."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        h = self.history
        if "source" not in h.columns:
            h = h.assign(source=None)
        joins = h[(h["valid_from"] > s) & (h["valid_from"] <= e)][["symbol", "valid_from", "source"]]
        joins = joins.rename(columns={"valid_from": "date"}).assign(event="joined")
        leaves = h[h["valid_to"].notna() & (h["valid_to"] > s) & (h["valid_to"] <= e)][["symbol", "valid_to", "source"]]
        leaves = leaves.rename(columns={"valid_to": "date"}).assign(event="left")
        return pd.concat([joins, leaves]).sort_values(["date", "event", "symbol"]).reset_index(drop=True)


def load_current_list(path: str | Path = DEFAULT_CURRENT_CSV, refresh: bool = False) -> list[str]:
    """Today's Nifty 50 list (what a survivorship-biased backtest would use).

    Uses the vendored snapshot of NSE's ind_nifty50list.csv unless
    ``refresh=True``, in which case it downloads the live file from NSE.
    """
    if refresh:
        import urllib.request
        req = urllib.request.Request(NSE_CURRENT_LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            df = pd.read_csv(r)
    else:
        df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else "symbol"
    return sorted(df[col].astype(str).str.strip().unique())

"""Constituent-level dual-SMA backtest on a (point-in-time or fixed) Nifty 50 universe.

The notebooks in this repo trade the index itself (^NSEI). This module applies
the same 20/50 SMA rule to each Nifty 50 stock, so the choice of universe
matters: on every trading day t, only stocks that are index members on t are
eligible. Each eligible stock gets weight 1/N_t (N_t = number of members on t)
while its fast SMA is above its slow SMA, otherwise its slot sits in cash.
Signals on day t are traded on day t+1, with ``fee_bps`` charged on turnover,
mirroring the notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    positions: pd.DataFrame
    metrics: dict
    notes: list = field(default_factory=list)


def build_universe_mask(dates: pd.DatetimeIndex, symbols, mode: str,
                        membership=None, current_list=None) -> pd.DataFrame:
    """mode="pit": members on each date from the PIT dataset (default).
    mode="current": today's list applied to every date (survivorship-biased)."""
    symbols = list(symbols)
    if mode == "pit":
        if membership is None:
            raise ValueError("pit mode needs a Nifty50Membership")
        return membership.membership_matrix(dates, symbols)
    if mode == "current":
        if current_list is None:
            raise ValueError("current mode needs today's list")
        cur = set(current_list)
        return pd.DataFrame({s: s in cur for s in symbols}, index=dates)
    raise ValueError(f"unknown universe mode {mode!r}")


def apply_missing_policy(close: pd.DataFrame, mask: pd.DataFrame, missing: list[str],
                         policy: str = "cash", proxies: dict[str, str] | None = None):
    """Decide what to do with universe members that have no price data.

    cash  : keep them in N_t so their 1/N slot sits in cash (default; conservative,
            keeps the universe size honest).
    drop  : remove them from the universe; others are re-weighted 1/N over the rest.
    proxy : substitute a stand-in's prices (e.g. HDFC -> HDFCBANK); names with no proxy
            fall back to cash.
    Returns (close, mask, notes).
    """
    close, mask = close.copy(), mask.copy()
    notes = []
    for sym in missing:
        if sym not in mask.columns or not mask[sym].any():
            continue
        days = int(mask[sym].sum())
        if policy == "drop":
            mask[sym] = False
            notes.append(f"{sym}: no price data, dropped from universe ({days} member-days)")
        elif policy == "proxy" and proxies and sym in proxies and proxies[sym] in close.columns:
            close[sym] = close[proxies[sym]]
            notes.append(f"{sym}: no price data, proxied by {proxies[sym]} ({days} member-days)")
        else:
            notes.append(f"{sym}: no price data, slot held in cash ({days} member-days)")
    return close, mask, notes


def run_backtest(close: pd.DataFrame, mask: pd.DataFrame, start=None, end=None,
                 fast: int = 20, slow: int = 50, fee_bps: float = 5.0,
                 use_signal: bool = True, returns: pd.DataFrame | None = None) -> BacktestResult:
    """close: dates x symbols adjusted closes (with warm-up history before ``start``).
    mask : dates x symbols bool universe membership (same columns)."""
    cols = [c for c in mask.columns if c in close.columns]
    close = close[cols]
    mask = mask.reindex(index=close.index, columns=cols).fillna(False).astype(bool)

    if returns is None:
        rets = close.pct_change(fill_method=None)
    else:
        rets = returns.reindex(index=close.index, columns=cols)

    if use_signal:
        sma_f = close.rolling(fast).mean()
        sma_s = close.rolling(slow).mean()
        signal = (sma_f > sma_s) & sma_s.notna()
    else:
        signal = close.notna()

    n_members = mask.sum(axis=1).replace(0, np.nan)
    tradable = mask & close.notna() & signal
    target = tradable.div(n_members, axis=0).fillna(0.0)

    lo = pd.Timestamp(start) if start is not None else close.index[0]
    hi = pd.Timestamp(end) if end is not None else close.index[-1]
    target = target.loc[lo:hi]
    rets = rets.loc[lo:hi]

    pos = target.shift(1).fillna(0.0)               # trade at t+1
    trades = pos.diff().abs().fillna(pos.abs())      # first day: entering from cash
    cost = fee_bps / 10000.0
    port = (pos * rets.fillna(0.0)).sum(axis=1) - trades.sum(axis=1) * cost
    equity = (1 + port).cumprod()

    held = pos.columns[(pos > 0).any()]
    metrics = compute_metrics(port, equity)
    metrics.update({
        "tickers_traded": int(len(held)),
        "avg_gross_exposure": float(pos.sum(axis=1).mean()),
        "annual_turnover": float(trades.sum(axis=1).sum() / (len(port) / TRADING_DAYS)),
    })
    return BacktestResult(returns=port, equity=equity, positions=pos, metrics=metrics)


def compute_metrics(returns: pd.Series, equity: pd.Series) -> dict:
    n = len(returns)
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1)
    vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = cagr / vol if vol else 0.0                    # repo convention (CAGR / vol, rf = 0)
    sd = returns.std()
    sharpe_daily = float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd else 0.0
    dd = equity / equity.cummax() - 1
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "days": n,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(sharpe),
        "sharpe_mean_std": sharpe_daily,
        "max_drawdown": float(dd.min()),
    }

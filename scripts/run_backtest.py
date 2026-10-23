"""Run the constituent-level SMA backtest on a Nifty 50 universe.

Examples
--------
# default: point-in-time universe
python scripts/run_backtest.py
# old behaviour: today's Nifty 50 list on every date (survivorship-biased)
python scripts/run_backtest.py --universe current
# both, side by side, written to results/
python scripts/run_backtest.py --compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtester.universe import Nifty50Membership, load_current_list, DATASET_COVERAGE_END  # noqa: E402
from backtester.data import download_prices, daily_returns, DEFAULT_PROXIES  # noqa: E402
from backtester.strategy import build_universe_mask, apply_missing_policy, run_backtest  # noqa: E402

WARMUP_START = "2017-06-01"


def run_mode(mode, prices, rets, membership, current, start, end, policy, use_signal, fast, slow):
    dates = prices.close.index
    syms = list(prices.close.columns)
    mask = build_universe_mask(dates, syms, mode, membership=membership, current_list=current)
    close, mask, notes = apply_missing_policy(prices.close, mask, prices.missing, policy, DEFAULT_PROXIES)
    r = rets.copy()
    if policy == "proxy":
        for s, p in DEFAULT_PROXIES.items():
            if s in prices.missing and p in r.columns:
                r[s] = r[p]
    res = run_backtest(close, mask, start=start, end=end, fast=fast, slow=slow,
                       use_signal=use_signal, returns=r)
    res.notes = notes
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["pit", "current"], default="pit",
                    help="pit = point-in-time members (default); current = today's list on every date")
    ap.add_argument("--compare", action="store_true", help="run both universes and write results/")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=str(DATASET_COVERAGE_END.date()),
                    help="default = membership dataset coverage end")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--missing-policy", choices=["cash", "drop", "proxy"], default="cash")
    ap.add_argument("--refresh-current-list", action="store_true",
                    help="download today's list from NSE instead of the vendored snapshot")
    ap.add_argument("--cache-dir", default=str(ROOT / ".cache"))
    a = ap.parse_args(argv)

    membership = Nifty50Membership.from_csv()
    current = load_current_list(refresh=a.refresh_current_list)
    symbols = sorted(set(membership.all_symbols(a.start, a.end)) | set(current))
    end_excl = (pd.Timestamp(a.end) + pd.Timedelta(days=1)).date()
    prices = download_prices(symbols, WARMUP_START, end_excl, cache_dir=a.cache_dir)
    rets, ca = daily_returns(prices.close)

    print(f"Symbols needed: {len(symbols)} | no price data: {prices.missing or 'none'}")
    for c in ca:
        print(f"Corporate-action override: {c['symbol']} {c['date']} raw {c['raw_return']:+.1%} -> {c['used']:+.1%}")

    modes = ["pit", "current"] if a.compare else [a.universe]
    out = {}
    for m in modes:
        for strat, use_sig in (("sma", True), ("ew_hold", False)):
            res = run_mode(m, prices, rets, membership, current, a.start, a.end,
                           a.missing_policy, use_sig, a.fast, a.slow)
            out[f"{m}/{strat}"] = {**res.metrics, "notes": res.notes}
            print(f"[{m:7s} {strat:7s}] " + " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in res.metrics.items() if k not in ("start", "end")))
            for n in res.notes:
                print("    note:", n)

    if a.compare:
        ch = membership.changes(a.start, a.end)
        res_dir = ROOT / "results"
        res_dir.mkdir(exist_ok=True)
        payload = {
            "period": [a.start, a.end], "missing_policy": a.missing_policy,
            "fast": a.fast, "slow": a.slow,
            "missing_price_symbols": prices.missing, "corporate_action_overrides": ca,
            "index_changes": {"joined": int((ch.event == "joined").sum()),
                              "left": int((ch.event == "left").sum()),
                              "events": ch.assign(date=ch.date.dt.date.astype(str)).to_dict("records")},
            "current_list_not_in_pit_at_start": sorted(set(current) - set(membership.members_on(a.start))),
            "pit_at_start_not_in_current_list": sorted(set(membership.members_on(a.start)) - set(current)),
            "results": out,
        }
        suffix = "" if a.missing_policy == "cash" else f"_{a.missing_policy}"
        (res_dir / f"universe_comparison{suffix}.json").write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote results/universe_comparison{suffix}.json")
    return out


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest

from backtester.data import daily_returns, _summarise
from backtester.strategy import (apply_missing_policy, build_universe_mask,
                                 compute_metrics, run_backtest)
from backtester.universe import Nifty50Membership

DATES = pd.bdate_range("2020-01-01", periods=200)


def trending_prices():
    t = np.arange(len(DATES))
    return pd.DataFrame({
        "UP": 100 * 1.01 ** t,           # always uptrend -> signal on after warm-up
        "UP2": 50 * 1.005 ** t,
        "DOWN": 100 * 0.99 ** t,          # never signals
    }, index=DATES)


def mask_all(cols, value=True):
    return pd.DataFrame(value, index=DATES, columns=cols)


def test_non_member_never_held():
    close = trending_prices()
    mask = mask_all(close.columns)
    mask["UP2"] = False
    res = run_backtest(close, mask, fast=5, slow=10)
    assert (res.positions["UP2"] == 0).all()
    assert res.positions["UP"].max() == pytest.approx(1 / 2)   # N = 2 members
    assert (res.positions["DOWN"] == 0).all()


def test_membership_switch_mid_period():
    close = trending_prices()
    mask = mask_all(close.columns)
    mask.loc[DATES[100]:, "UP"] = False      # UP removed from index on day 100
    res = run_backtest(close, mask, fast=5, slow=10)
    assert res.positions["UP"].iloc[100] > 0      # still held on day 100 (decided at close of 99)
    assert (res.positions["UP"].iloc[101:] == 0).all()


def test_next_day_execution_no_lookahead():
    close = trending_prices()
    mask = mask_all(close.columns)
    res = run_backtest(close, mask, fast=5, slow=10, fee_bps=0)
    first = res.positions["UP"].ne(0).idxmax()
    # SMA(10) first exists on row 9 -> signal at close of row 9 -> position on row 10
    assert DATES.get_loc(first) == 10


def test_ew_hold_equals_average_return_without_fees():
    close = trending_prices()
    mask = mask_all(close.columns)
    res = run_backtest(close, mask, use_signal=False, fee_bps=0)
    expected = close.pct_change().iloc[2:].mean(axis=1)
    assert np.allclose(res.returns.iloc[2:], expected)


def test_fees_reduce_returns():
    close = trending_prices()
    mask = mask_all(close.columns)
    a = run_backtest(close, mask, fast=5, slow=10, fee_bps=0).metrics["total_return"]
    b = run_backtest(close, mask, fast=5, slow=10, fee_bps=50).metrics["total_return"]
    assert b < a


def test_missing_policies():
    close = trending_prices()
    close["GONE"] = np.nan
    mask = mask_all(close.columns)
    c_close, c_mask, notes = apply_missing_policy(close, mask, ["GONE"], "cash")
    assert c_mask["GONE"].all() and "cash" in notes[0]
    res_cash = run_backtest(c_close, c_mask, fast=5, slow=10)
    assert res_cash.positions["UP"].max() == pytest.approx(1 / 4)   # GONE still counts in N

    d_close, d_mask, notes = apply_missing_policy(close, mask, ["GONE"], "drop")
    assert not d_mask["GONE"].any() and "dropped" in notes[0]
    res_drop = run_backtest(d_close, d_mask, fast=5, slow=10)
    assert res_drop.positions["UP"].max() == pytest.approx(1 / 3)

    p_close, p_mask, notes = apply_missing_policy(close, mask, ["GONE"], "proxy", {"GONE": "UP2"})
    assert p_close["GONE"].equals(p_close["UP2"]) and "proxied by UP2" in notes[0]


def test_missing_symbol_is_reported():
    close = trending_prices()
    close["GONE"] = np.nan
    assert _summarise(close).missing == ["GONE"]


def test_corporate_action_override():
    close = trending_prices()
    r, applied = daily_returns(close, {("UP", str(DATES[50].date())): 0.0})
    assert r.loc[DATES[50], "UP"] == 0.0 and applied[0]["symbol"] == "UP"


def test_universe_modes_differ():
    hist = pd.DataFrame({"index_name": "Nifty 50", "symbol": ["UP", "DOWN"],
                         "valid_from": ["2019-01-01", "2019-01-01"],
                         "valid_to": [None, str(DATES[50].date())]})
    m = Nifty50Membership(hist, coverage_end=None)
    pit = build_universe_mask(DATES, ["UP", "DOWN", "UP2"], "pit", membership=m)
    cur = build_universe_mask(DATES, ["UP", "DOWN", "UP2"], "current", current_list=["UP", "UP2"])
    assert pit["DOWN"].iloc[49] and not pit["DOWN"].iloc[50] and not pit["UP2"].any()
    assert cur["UP2"].all() and not cur["DOWN"].any()
    with pytest.raises(ValueError):
        build_universe_mask(DATES, ["UP"], "bogus")


def test_metrics_known_series():
    r = pd.Series([0.1, -0.5, 0.2], index=DATES[:3])
    eq = (1 + r).cumprod()
    mt = compute_metrics(r, eq)
    assert mt["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert mt["max_drawdown"] == pytest.approx(-0.5)

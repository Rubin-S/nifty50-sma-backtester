import warnings

import pandas as pd
import pytest

from backtester.universe import Nifty50Membership, load_current_list, to_yf_ticker


@pytest.fixture(scope="module")
def m():
    return Nifty50Membership.from_csv()


def synthetic():
    return Nifty50Membership(pd.DataFrame({
        "index_name": ["Nifty 50"] * 3 + ["Nifty Bank"],
        "symbol": ["AAA", "BBB", "CCC", "ZZZ"],
        "valid_from": ["2020-01-01", "2020-01-01", "2020-06-01", "2020-01-01"],
        "valid_to": [None, "2020-06-01", None, None],
    }), coverage_end=pd.Timestamp("2021-01-01"))


@pytest.mark.parametrize("d", ["2018-01-01", "2019-06-15", "2020-09-25", "2023-07-12",
                               "2023-07-13", "2024-09-30", "2025-09-30", "2026-05-15"])
def test_always_fifty_members(m, d):
    assert len(m.members_on(d)) == 50


def test_hdfc_merger_transition(m):
    assert m.is_member("HDFC", "2023-07-12")
    assert not m.is_member("HDFC", "2023-07-13")
    assert not m.is_member("LTM", "2023-07-12")
    assert m.is_member("LTM", "2023-07-13")


@pytest.mark.parametrize("sym,before,after", [
    ("ETERNAL", "2025-03-27", "2025-03-28"),
    ("INDIGO", "2025-09-29", "2025-09-30"),
    ("BEL", "2024-09-27", "2024-09-30"),
])
def test_known_inclusions(m, sym, before, after):
    assert not m.is_member(sym, before) and m.is_member(sym, after)


@pytest.mark.parametrize("sym,last_in,first_out", [
    ("YESBANK", "2020-03-26", "2020-03-27"),
    ("INDUSINDBK", "2025-09-29", "2025-09-30"),
    ("HEROMOTOCO", "2025-09-29", "2025-09-30"),
])
def test_known_exclusions(m, sym, last_in, first_out):
    assert m.is_member(sym, last_in) and not m.is_member(sym, first_out)


def test_half_open_interval_semantics():
    s = synthetic()
    assert s.members_on("2020-05-31") == ["AAA", "BBB"]
    assert s.members_on("2020-06-01") == ["AAA", "CCC"]   # valid_to exclusive, valid_from inclusive
    assert s.members_on("2019-12-31") == []
    assert "ZZZ" not in s.all_symbols("2019-01-01", "2021-01-01")  # other index ignored


def test_membership_matrix_matches_members_on(m):
    dates = pd.bdate_range("2018-01-01", "2026-05-15", freq="21B")
    mat = m.membership_matrix(dates)
    for d in dates:
        assert sorted(mat.columns[mat.loc[d]]) == m.members_on(d)


def test_changes_synthetic():
    ch = synthetic().changes("2020-01-01", "2020-12-31")
    assert list(zip(ch.symbol, ch.event)) == [("CCC", "joined"), ("BBB", "left")]


def test_changes_real_period_balanced(m):
    ch = m.changes("2018-01-01", "2026-05-15")
    assert (ch.event == "joined").sum() == (ch.event == "left").sum()  # index stays at 50


def test_warns_beyond_coverage():
    s = synthetic()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.members_on("2022-01-01")
    assert any("coverage" in str(x.message) for x in w)


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        Nifty50Membership(pd.DataFrame({"index_name": ["Nifty 50"], "symbol": ["X"],
                                        "valid_from": ["2020-02-01"], "valid_to": ["2020-01-01"]}))


def test_current_list_snapshot():
    cur = load_current_list()
    assert len(cur) == 50 and "RELIANCE" in cur


def test_yf_ticker():
    assert to_yf_ticker("M&M") == "M&M.NS"
    assert to_yf_ticker("BAJAJ-AUTO") == "BAJAJ-AUTO.NS"

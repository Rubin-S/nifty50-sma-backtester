# nifty50-sma-backtester

Trend-following and risk-management experiments on the NIFTY 50.

## Notebooks (index-level, unchanged)

The Colab notebooks trade the index itself (`^NSEI` from yfinance), so a stock universe does not affect them:

* Dual SMA strategy vs buy & hold (2018 to present)
* Parameter optimisation & walk-forward validation
* Dynamic volatility targeting & risk management
* Adaptive ATR trailing stops & chandelier exits
* Institutional execution microstructure & statistical validation

## Constituent-level backtester with a point-in-time universe

`backtester/` applies the 20/50 SMA rule to each Nifty 50 **stock**. On every trading day it only considers
stocks that were in the index **on that day**, which removes survivorship bias.

```bash
pip install -r requirements.txt
python scripts/run_backtest.py                      # point-in-time universe (default)
python scripts/run_backtest.py --universe current   # old behaviour: today's list on every date
python scripts/run_backtest.py --compare            # both, written to results/
python scripts/run_backtest.py --compare --missing-policy proxy   # cash (default) | drop | proxy
pytest -q
```

```python
from backtester import Nifty50Membership
m = Nifty50Membership.from_csv()
m.members_on("2021-08-16")          # 50 symbols in the index that day
m.changes("2018-01-01", "2026-05-15")
```

* Prices come from yfinance (`SYMBOL.NS`). Symbols with **no price data are reported, never silently dropped**.
  `--missing-policy` chooses what happens to them: `cash` (slot held in cash, the default), `drop`, or `proxy`.
* The membership data ends on 2026-05-15. Asking for later dates gives a warning.

### Result: survivorship bias, 2018-01-01 to 2026-05-15

| 20/50 SMA per stock | Point-in-time | Today's list |
|---|---|---|
| CAGR | 7.90% | 12.33% |
| Sharpe (CAGR/vol) | 0.89 | 1.34 |
| Max drawdown | -15.47% | -12.87% |
| Total return | 86.5% | 159.5% |
| Tickers traded | 69 | 50 |

Using today's list overstates CAGR by about 4.4 pp a year. Full tables and caveats: [`results/universe_comparison.md`](results/universe_comparison.md).

## Data credits

Point-in-time index membership: **[aditya-jha/nse-historical-membership](https://github.com/aditya-jha/nse-historical-membership)**
(`index_history/data/index_membership_history.csv`), © Aditya Jha, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: NSE press releases / NSE Exchange circulars
(publicly published). A filtered copy (Nifty 50 rows only) is in `data/`. See `data/README.md`.

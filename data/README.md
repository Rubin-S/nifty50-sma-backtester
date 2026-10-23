# Data

* `nifty50_membership_history.csv`: Nifty 50 rows of
  [`index_history/data/index_membership_history.csv`](https://github.com/aditya-jha/nse-historical-membership/blob/main/index_history/data/index_membership_history.csv)
  from **aditya-jha/nse-historical-membership** (upstream commit `cc6f89f`, membership coverage through 2026-05-15).
  Licensed **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). The underlying source is NSE press
  releases / NSE Exchange circulars (publicly published). Only the rows were filtered; nothing else was changed.
  Columns: `index_id, index_name, symbol, valid_from, valid_to, weightage, source, source_url, notes`.
  Intervals are half-open `[valid_from, valid_to)`, and an empty `valid_to` means still a member.
* `nifty50_current_list.csv`: NSE's `ind_nifty50list.csv` (https://archives.nseindia.com/content/indices/ind_nifty50list.csv),
  downloaded 2026-10-23. Used only for the survivorship-biased `--universe current` mode.

To pull the full upstream file instead: `Nifty50Membership.from_upstream()`.

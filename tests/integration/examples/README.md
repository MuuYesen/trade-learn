# External strategy example tests

These tests cover strategy projects maintained outside the library repository.
They retain the source and behavior assertions previously located under
`tests/unit/examples`, which referred to untracked `zoo` deployment files.
The library's default test suite does not include this integration directory.

To run them, set `TRADELEARN_STRATEGY_EXAMPLES_ROOT` to a directory containing:

```text
tushare_sw_hs300/
  alpha101_hs300_backtest.py
  hs300_alpha101_strategy.py
wechat_docs/articles/alpha101-us-tech/outputs/examples/
  alpha101_us_tech_lite_backtest.py
  alpha101_us_tech_experiment.py
```

Keep any supporting modules and dependencies from those projects available.
Then run `python -m pytest tests/integration/examples`.

Without the environment variable, explicitly collecting this directory reports
three clear module skips. A configured root or required source file that does
not exist raises an error; configuration mistakes never silently skip tests.

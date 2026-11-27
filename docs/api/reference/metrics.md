# Metrics Reference

Return, risk, drawdown, trade, and factor evaluation metrics.

[Back to API Reference](../reference.md)

## Sharpe inference

Pass periodic excess returns and an explicit number of periods per year. Use
`sharpe_se(returns, periods, method="iid")` for the normal IID approximation.
For a regular chronological series with time dependence,
`sharpe_se(returns, periods, method="hac", bandwidth=...)` estimates a
Bartlett HAC standard error of the same annualized Sharpe estimator. Record
the bandwidth and the return sampling convention.

For the mean of a chronological IC or similar series, use
`hac_mean_se(values, bandwidth=...)` in the original series units.

`paired_sharpe_diff(left, right, periods, block_length=..., seed=...)` requires
identical, unique, increasing indexes and finite observations. It computes
the annualized Sharpe of the right series minus that of the left series and
returns the estimate, standard error, percentile limits, observation count,
and valid draw count. Each circular block resamples the same time rows on
both sides. Choose the block length before reviewing the result. Reconcile
missing observations before calling the function and retain the intersection
coverage as evidence. Report excluded zero-volatility draws when the valid
draw count is smaller than the requested count.

`probabilistic_sharpe` and `minimum_track_record_length` use a prespecified
per-period Sharpe benchmark. For a selected result from multiple search
trials, `deflated_sharpe` accepts the independent trial count and the
cross-trial variance of comparable **per-period** Sharpe estimates. These
are moment-based approximations. A single prespecified trial uses PSR
against its benchmark.

::: tradelearn.metrics
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: false
      show_root_full_path: false
      show_object_full_path: false
      show_bases: false
      show_signature_annotations: true
      separate_signature: true
      members_order: source

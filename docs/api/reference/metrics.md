# Metrics Reference

Return, risk, drawdown, trade, and factor evaluation metrics.

[Back to API Reference](../reference.md)

Inputs are periodic simple returns, with an explicit positive integer `periods` for annualization.
The caller owns cash-flow adjustment, valuation currency, calendar alignment, and coverage.
`simple_returns` does not forward-fill missing prices. For APIs with `nan_policy`, `propagate`
keeps scalar results unavailable if a required observation is missing, and cumulative curves
stay unavailable from the first gap onward. `drop` describes the observed sample; `zero`
is an explicit modeling assumption, not evidence of a zero return. These policies reject
infinite input observations. Undefined analytic results may be NaN; JSON adapters must use null.

Risk dispersion uses sample standard deviation (`ddof=1`). The current risk-free-rate convention
is nominal annual `rf / periods`; Sortino's `required` is a separate per-period target.
Drawdown values in this API are nonpositive; dashboards showing loss magnitudes convert the sign.


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

## Missing evidence

`simple_returns` and factor forward returns do not fill missing prices. Under
`nan_policy="propagate"`, missing returns remain unknown: scalar statistics
return NaN and cumulative curves stay unknown after the first gap. `drop` and
`zero` remain explicit alternatives; infinite observations are rejected rather
than silently interpreted as valid returns. Annualization requires a positive
integer period count.

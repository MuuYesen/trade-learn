# Metrics Reference

Return, risk, drawdown, trade, and factor evaluation metrics.

[Back to API Reference](../reference.md)

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

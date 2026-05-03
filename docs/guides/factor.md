# Factor Guide

`tradelearn.factor` 提供 alphalens 风格的因子清洗、分组收益、IC、换手和多周期报告。

## 标准入口

```python
from tradelearn.factor import FactorAnalyzer, clean_factor_and_forward_returns

clean = clean_factor_and_forward_returns(
    factors,
    factor="momentum",
    prices=prices,
    periods=(1, 5, 10),
    quantiles=5,
)

fa = FactorAnalyzer.from_clean_factor_data(clean)
fa.report("factor_report.html")
```

## 单因子与多因子

- 单因子会进入多周期分析。
- 多因子会进入多因子对比分析。
- 报告默认覆盖传入的多个 forward return period，不需要为每个周期单独生成报告。

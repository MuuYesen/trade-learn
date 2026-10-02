# 因子指南

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

## Alpha101 缺失值修正

`alpha002` 和 `alpha003` 在滚动窗口不足、输入缺失或序列方差为零时返回 `NaN`。
旧实现把未定义的相关系数填成了零；修正后这些位置的历史输出会改变。
训练时应根据训练窗口选择有观测值的因子，不能把未定义的因子当作已观测零值。

`alpha066` 的价格比率分母为零时也保留 `NaN`，避免把除零产生的无穷值转换为有效排名。

Alpha101/Alpha191 的加权滚动窗口保留 NumPy 参考计算的求和顺序，避免浮点累加差异经后续排名放大。
加权均值和时间序列排名均要求窗口内观测有限；`NaN` 或无穷值离开窗口后恢复计算。

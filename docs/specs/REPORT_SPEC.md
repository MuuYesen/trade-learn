# REPORT SPEC

`tradelearn.report` 模块规格——融合 pyfolio + quantstats 的策略报告引擎。

## 1. 模块布局

```
tradelearn/report/
├── __init__.py          # Reporter 门面
├── html.py              # HTML 报告(融合 pyfolio + quantstats)
├── excel.py             # Excel 多 sheet 导出(xlsxwriter)
├── explore.py           # pygwalker 交互式探索(lab 专用)
├── charts/              # 图表组件(bokeh)
│   ├── equity.py
│   ├── drawdown.py
│   ├── heatmap.py
│   └── ...
└── templates/           # HTML 模板(jinja2)
    └── tear_sheet.html
```

## 2. Reporter 门面 API

```python
from tradelearn.report import Reporter

stats = cerebro.run()
r = Reporter(stats)

# 核心方法
r.summary()                      # → dict 关键指标
r.html("report.html")            # 完整 HTML tear sheet
r.excel("report.xlsx")           # Excel 多 sheet
r.explore()                      # pygwalker 交互(仅 lab)

# 单独图表(bokeh,用于 notebook 内嵌)
r.equity_curve()
r.drawdown()
r.monthly_heatmap()
r.rolling_sharpe(window=126)
r.rolling_beta(benchmark, window=126)
r.trade_distribution()
r.exposure()                     # 多资产持仓暴露
```

## 3. Stats 输入契约

Reporter 接收 `cerebro.run()` 返回的 `Stats` 对象(见 CONTRACTS.md):

```python
@dataclass
class Stats:
    returns: pd.Series          # tz-aware UTC
    equity: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    summary: dict[str, float]
    analyzers: dict[str, Any]
    config: dict
```

## 4. HTML 报告(tear sheet)

### 4.1 必须包含的 8 部分

1. **Header**:策略名 / 时间范围 / 生成时间 / run_id(若有)
2. **Summary Stats 表**:16 个关键指标
3. **Equity Curve**:累计收益曲线(可选叠加基准)
4. **Drawdown Chart**:回撤曲线 + Top 10 回撤表
5. **Monthly Returns Heatmap**:月度收益热图
6. **Rolling Sharpe**(6 个月)
7. **Trade Distribution**:盈亏分布直方图
8. **Footer**:版本 / 生成时间 / configuration 摘要

### 4.2 多资产扩展(自动)

检测到 `positions.symbol.nunique() > 1`,增加:

9. **Correlation Matrix**:多资产相关性
10. **Exposure Chart**:每日持仓暴露

### 4.3 Summary Stats 列表(16 个)

| 指标 | 单位 | 来源 |
|---|---|---|
| Annual Return | % | metrics.annual_return |
| Cumulative Return | % | metrics.cum_returns.iloc[-1] |
| Annual Volatility | % | metrics.volatility |
| Sharpe Ratio | — | metrics.sharpe |
| Calmar Ratio | — | metrics.calmar |
| Sortino Ratio | — | metrics.sortino |
| Max Drawdown | % | metrics.max_drawdown |
| Max DD Duration | days | 计算 |
| Alpha (vs benchmark) | % | metrics.alpha |
| Beta (vs benchmark) | — | metrics.beta |
| Information Ratio | — | metrics.information_ratio |
| Win Rate | % | metrics.win_rate |
| Profit Factor | — | metrics.profit_factor |
| Avg Win / Avg Loss | — | metrics.avg_win / avg_loss |
| Total Trades | count | 计算 |
| Turnover | /year | 计算 |

**全部底层调 `tradelearn.metrics`,不重复实现**。

### 4.4 图表技术栈

- **bokeh**:所有交互式图(equity / drawdown / heatmap / rolling)
- **不用 matplotlib**:pyfolio 原版用 matplotlib,融合时统一 bokeh

### 4.5 HTML 模板引擎

- jinja2 模板 `templates/tear_sheet.html`
- 内嵌 bokeh JSON(客户端渲染)
- 单文件 HTML(可分享,不依赖外部资源)

## 5. Excel 导出

### 5.1 Sheets

| sheet | 内容 |
|---|---|
| `summary` | 16 个关键指标表格 |
| `trades` | 逐笔交易明细 |
| `daily_returns` | 日收益序列 |
| `monthly_returns` | 月度收益矩阵(含条件格式热图) |
| `drawdowns` | Top 10 回撤表(含恢复时间) |
| `positions` | 每日持仓快照 |
| `orders` | 订单历史 |
| `config` | 回测配置 + 策略参数 |

### 5.2 技术栈

- `xlsxwriter`(不用 openpyxl,xlsxwriter 写入更快且格式丰富)
- 条件格式:月度热图用 xlsxwriter 原生 heat map,不嵌图片
- 所有数字精度保留 6 位小数(Excel 可配置)

## 6. pygwalker 交互探索

```python
r.explore()                    # 打开 pygwalker UI(仅 notebook)
```

### 实现

```python
def explore(self):
    try:
        import pygwalker as pyg
    except ImportError:
        raise ImportError(
            "explore() requires pygwalker. Install with: "
            "pip install trade-learn[lab]"
        )

    # 默认探索 trades 表
    return pyg.walk(self.stats.trades)
```

**脚本场景不装 pygwalker → 调用报错有提示**。

## 7. 图表详细规格

### 7.1 Equity Curve

- x 轴:日期
- y 轴:归一化净值(1.0 起点)
- 可选叠加:基准、买入持有对照
- 标记点:Top 5 回撤 peak / valley
- 交互:hover 显示具体日期和数值

### 7.2 Drawdown Chart

- 上图:equity curve(高亮回撤期间)
- 下图:drawdown % 填充面积图
- 颜色:回撤深度 → 红色深浅

### 7.3 Monthly Heatmap

- 行:年份
- 列:月份(1-12)
- 值:该月收益 %
- 颜色:绿(正) / 红(负),色阶 ±5%
- 行尾:year-total 列
- 列尾:month-avg 行

### 7.4 Rolling Sharpe

- x 轴:日期
- y 轴:滚动 Sharpe
- 水平线:全期 Sharpe
- 默认窗口:126 个交易日(约 6 个月)

### 7.5 Trade Distribution

- 直方图:PnL 分布
- 分桶:20 个
- 红绿区分盈亏
- 叠加:mean / median 垂直线

## 8. 基准对比(可选)

```python
r.html("report.html", benchmark='HS300')
```

框架自动:
- 拉取基准行情(通过 `Query.history_ohlc(symbol='HS300', engine='tdx')`)
- 对齐时间段
- 计算 alpha / beta / information_ratio
- 所有图表叠加基准线

### 预设基准

| 代码 | 市场 | 来源 |
|---|---|---|
| `HS300` | A 股 | 通达信 |
| `SP500` | 美股 | TV |
| `NASDAQ` | 美股 | TV |
| `BTC` | 加密 | TV |

## 9. 输出路径约定

```python
Reporter(stats).html("report.html")
# 默认输出:
# ./reports/{strategy_name}/{run_id_or_timestamp}/
#   ├── report.html
#   ├── report.xlsx
#   ├── equity.parquet
#   ├── trades.parquet
#   └── stats.json
```

MLflowAnalyzer 附加 artifacts 时用同一路径。

## 10. 性能要求

- HTML tear sheet 生成(10 年日线)< 2 秒
- Excel 生成 < 5 秒
- 输出文件大小:HTML < 5 MB / Excel < 2 MB

## 11. 融合时砍掉的部分

从 pyfolio 原版砍掉:

- ❌ Bayesian tear sheet(依赖 pymc,极少用)
- ❌ Round trip 分析(和 trade_distribution 重合)
- ❌ Capacity analysis(太专业,研究员少用)
- ❌ 所有 matplotlib 图(统一 bokeh)

从 quantstats 原版保留:

- ✅ HTML 模板结构(现代、整洁)
- ✅ 月度热图的视觉风格

## 12. 金标对照

### 12.1 对照源

- pyfolio 原版(dev 依赖)
- quantstats 原版(dev 依赖)

### 12.2 数值对照

```python
# tests/consistency/test_report_summary.py
def test_summary_matches_pyfolio():
    stats = load_golden_stats()
    ours = Reporter(stats).summary()

    import pyfolio as pf
    theirs = pf.timeseries.perf_stats(stats.returns)

    common = ["annual_return", "sharpe_ratio", "max_drawdown", ...]
    for key in common:
        assert np.isclose(ours[key], theirs[key], rtol=1e-6)
```

### 12.3 视觉对照

不做自动化,**手工目视对比金标截图**。

## 13. CLI 集成

```bash
# 从 MLflow artifact 重新生成报告(用户修改样式后重新出图)
tradelearn run --report stats.json --output report.html

# 对比多次 run 的报告
tradelearn run --compare run_id_1,run_id_2
```

1.0 可选做,1.1 按需。

## 14. 错误处理

```python
class ReportError(TradelearnError): ...

# 数据不足
>>> ReportError: Cannot generate monthly heatmap with < 2 months of data.

# 基准数据拉取失败
>>> ReportError: Failed to load benchmark 'HS300':
    connection timeout. Report generated without benchmark comparison.

# 输出路径不可写
>>> ReportError: Cannot write to './reports/...':
    permission denied. Check directory permissions.
```

## 15. 不做的事

- ❌ PDF 导出(用户可用浏览器打印 HTML 为 PDF)
- ❌ 视频 / 动画报告
- ❌ 自定义模板(1.0 固定模板;1.1 按反馈开放)
- ❌ 实时刷新报告(静态生成)
- ❌ 多策略同 HTML 对比(用 MLflow UI)

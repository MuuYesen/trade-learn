# MLflow 与本地研究环境

## 运行后记录已有结果

源码入口为 `tradelearn.integrations.mlflow.log`。先完成回测/研究，再记录已有结果；不会重新回测、训练、补算指标、读取行情或自动登记模型。

```python
from tradelearn.integrations.mlflow import log

# Lite
result = backtest.run()
logged = log(result, research_result=research_result,
             experiment_name="research", run_name="validation")
# 等价便利入口：backtest.log_mlflow(...)

# Engine，普通评价 Analyzer 保留
strategies = cerebro.run()
logged = log(strategies, research_result=research_result)
print(logged.run_id, logged.status, logged.error)
```

也可直接 `log(research_result)`，仅记录已提供的研究名称、参数、真实步骤和组件元数据。不存在的训练、评分、分配或验证记录不会自动补造；步骤完整性仍由 ResearchRun 的实际调用记录保证。多个 Engine 策略写入同一个 Run，指标前缀为 `strategy.0.`、`strategy.1.`。单结果保持原 summary 指标名称和数值，不上传非有限数值。已有`metric_metadata`单位/缺失原因及显式 `pnl_basis`、`periods`、`period_source` 通过原生 tags 记录；未提供单位为unspecified，不猜测业务口径。

未指定 `uri` 时沿用原生 MLflow 配置，可使用本地 tracking 后端，不要求先启动服务。存在原生活动 Run 时复用同一 `run_id`，记录后保持打开；显式 uri、experiment_name 或 run_name 与活动 Run 冲突时在写入前拒绝。集成只关闭自己创建的 Run。`LogResult` 返回真实 `run_id`、`status`、`error`、`reused`、`run_status`；外部 Run 成功记录返回 RUNNING，自己创建并成功关闭返回 FINISHED，记录失败返回 FAILED 及错误；`run_status`另记实际已知的Run状态，外部Run记录失败时仍为RUNNING，关闭失败且状态未知时为None。非法输入/配置直接抛错；缺 MLflow 提示安装 `trade-learn[research]`。

默认只记录已有数值、已提供的配置/策略参数和受限研究元数据，排除凭据键、原始 features/target/model 对象；这不是对任意用户字符串的自动脱敏保证。不要把秘密写入实验名称或普通参数。报告和本地文件显式启用：

```python
logged = log(result, log_report=True,
             attachments=["deployment/manifest.json", "report.xlsx"])
```

`log_report=True` 使用 Reporter，默认不传原行情给图表。附件接受显式选择的文件或完整bundle目录；同名根附件冲突在写入前拒绝。目录通过原生log_artifacts上传，文件通过log_artifact上传。附件可能包含敏感业务内容，上传目的地沿实际 MLflow 配置。模型登记/部署继续通过显式 ModelRegistry 流程；应保全拟合 Pipeline、FeatureSet、输入 schema、标签/purge 配置与版本，不以裸模型 pickle 声称完整可部署。

迁移旧调用时，用上述运行后 log 替代 TL 专属 `MLflowAnalyzer`；`nested`、`mlflow_module`、隐式全量附件及旧固定字符串返回不属于新接口。普通 Returns/Sharpe/Drawdown 等评价 Analyzer 继续保留。调用迁移及旧 Analyzer 退役进度见本轮交付记录；源码入口可用不等于已经公开发行。

## 研究环境与可选工具

目标安装组统一为 default / research / nautilus / tl-store，原 lab 依赖并入 research。研究环境采用 JupyterLab、Jupyter AI Lite、Skills、MLflow；不默认附加 jupyterlab-git、PyGWalker ；TL自建MCP服务与公开入口已删除。环境计划/CLI 的迁移和真实 Notebook/provider 验收须单独通过，安装包存在不证明已连接付费 provider。

上游 [Jupyter AI Lite 安装与环境要求](https://pypi.org/project/jupyter-ai-lite/)为 JupyterLab >=4.5；本轮候选固定 0.21.0，实际解析/安装证据以交付记录为准。[上游技能机制](https://jupyter-ai.readthedocs.io/en/latest/users/browser/index.html)从工作区 `.agents/skills/` 或 `_agents/skills/` 加载；`tradelearn new`生成可由默认Jupyter Contents发现的`_agents/skills/tradelearn-research/SKILL.md`，不要求开放全部隐藏文件；TL 技能使用当前正式 API，不自动执行实盘。API/provider 配置由用户在原生界面显式完成，不在仓库或报告内保存密钥。TL不再提供MCP启动入口或自建MCP服务。第三方AI工具自身的MCP依赖不受此删除影响。


CLI仅保留`tradelearn new 项目名`、`tradelearn lab`、`tradelearn doctor`及`--help/--version`。基础命令和new不要求Typer/PyYAML；new创建策略定义、说明、Notebook起点及版本匹配Skills，不强制生成YAML。策略定义需由用户给出Bars后加入Cerebro，不宣传为自动联网或实盘脚本。

`tradelearn doctor`自动列出Python/TL版本及当前已安装组件，可选组件缺失单独说明，不把未安装research/NA/Store判为基础失败，不读取provider密钥或调用模型。显式已有YAML配置仍需PyYAML。

`tradelearn lab`只启动当前解释器的JupyterLab，沿已有tracking配置传入Notebook；默认`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`，在当前工作目录本地记录，不要求启动服务。显式环境变量/配置URI继续有效。删除`tradelearn mcp`、`tradelearn mlflow`、doctor旧组件选择参数及lab的`--with-mlflow/--mlflow-host/--mlflow-port`，不保兼容别名。确需独立MLflow服务时直接使用上游`mlflow server`，TL不再编排后台服务。退出或异常仅终止并等待自己启动的Jupyter进程。


参数网格位于 `tradelearn.integrations.search.grid_search`，独立于 Engine 和共同运行时核心。所有组合完成后，显式 `mlflow={...}` 会把已有策略列表一次交给 `log`，共享一个原生 Run；不会为每个组合隐式创建嵌套 Run。返回的每项策略引用同一 `mlflow_result`。

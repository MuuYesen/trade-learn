# 策略编写入口

当前源码候选的两种外观共用运行内核。完整公共边界以[系统规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#engine-lite-facade-20261112)为唯一规则，不要求策略类原样互换。

- Engine使用params/self.p与__init__，固定BT接口、原生订单控制，见[Engine指南](engine.md)和[实际API](engine-api.md)。
- Lite使用类属性与init，保留window/record/equity/target_weights等便利入口，见[Lite指南](lite.md)和[实际API](lite-api.md)。
- 指标使用同一个Indicator与共同Lines，选择真实合格的来源，见[指标指南](indicators.md)。旧I与PTA已退出本轮候选。
- 研究组件在外定义也须显式登记及执行，用户参数/bundle显式接入；不由基类注入research_result，见[研究指南](research.md)。
- 实时从tradelearn_runtime.NautilusStore组装，沿getdata/getbroker与Cerebro.run使用唯一官方NA宿主；带监控的独立配置用store.run(engine,monitor=...)，见[实时指南](live.md)。

notify_order/notify_trade可以覆盖，框架订单状态和独立观察链不依赖super调用。持仓只反映成交；在途意图与活动订单分开。错误、未知提交、取消中和恢复未决不能当成交或自动重发。运行后独立Reporter/MLflow记录已有结果，不在导出时重跑策略或补算未提供输入。

源码/离线fixture/目标构建/官方NA回放/生产与自然业务验收分别记录，不能以方法存在或本地单元测试代替券商能力资格。

# Store公开入口

用户通过`from tradelearn import engine as bt`与`bt.stores.NautilusStore`装配，唯一启动入口为`cerebro.run()`。`tradelearn_runtime`属于内部宿主，不是用户接线入口。

参见[Stores API](stores.md)、[执行指南](../../guides/live.md)与[API索引](../reference.md)。

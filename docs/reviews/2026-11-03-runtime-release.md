# 0.2.9 运行组件发布验证记录

本页保留按阶段适用的历史事实，当前安装/API以[Runtime](../concepts/runtime.md)为准。

### 0.2.8安装与原生回测（历史）

上一轮0.2.8为本地源码候选，未自动发布PyPI或Windows wheel；普通pip安装已发布旧版本不会自动获得本API。外部broker驱动的正式Runtime不启动Rust模拟撮合；原生研究回测需要同时匹配Python ABI/平台及0.2.8 Rust扩展。不能把源码树旧cp310二进制改名作为cp312的新版本。

源码用户按现有pyproject/maturin和原Cargo.lock在独立源码副本或构建目录生成本机wheel/扩展，再执行`scripts/check_order_lifecycle_wheel.py`核`tradelearn.__version__ == _rust.tradelearn_rust_version() == '0.2.8'`。本地验收保留macOS arm64 / CPython3.12匹配扩展及构建SHA回执，原engine/backtest/lite全目录776项通过；该本机构建不等于Windows/Linux研究wheel或公开发布。


### 预览依赖安全范围

早期预览提交26df434/9d19ff8沿用上游锁，含已知漏洞依赖，不推荐采用其构建产物。修补候选为发送路径加入证书、HTTP与IDNA依赖的最低修复版本，并成对升级PyO3/numpy Rust绑定及许可材料；最终可安装产物必须以修补后的CI/构建回执为准。安全查询覆盖发行依赖的精确PyPI版本和Cargo.lock闭包，不等于审计官方Nautilus二进制内嵌全部依赖或承诺没有未知漏洞。


Windows现有回测wheel继续验收；新增持久Recorder目前明确不支持Windows，构造时在创建状态/线程前拒绝，不跳过目录同步假称写入成功。Windows产物检查包含这个明确拒绝反例，而非记录/flush/恢复通过；仅希望运行Windows回测不需要启用日志。Mac/Linux仍要求真实写入、flush、close、恢复及原始许可成员验收。

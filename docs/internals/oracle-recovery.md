# Restoring the frozen test oracle

The frozen 1.x sources were retained by commit
`e28ae1556d637975b95f1cd4952e9e090af8a461` and removed from Git tracking by
`3ac1c4f67162211552353bbb16abe54e2e4b7b9c` ("keep local only"). A fresh checkout
therefore needs an explicit restore before running oracle comparisons:

```sh
python scripts/restore_oracle.py
python scripts/check_oracle.py
python -m pytest tests/consistency/test_metrics.py tests/unit/factor/test_alpha101.py tests/unit/factor/test_alpha191.py tests/golden
```

`scripts/oracle_manifest.json` records the source commit, Git blob IDs, and SHA-256
checksums for the 13 required Query, Alpha101/191, Empyrical, and Alphalens files.
The restore uses local Git history, verifies all bytes before writing, and refuses
to overwrite a modified oracle. Re-running it is read-only when all files exist.
The restored `reference/` stays ignored and excluded from package discovery.
For a shallow clone missing the source commit, first fetch repository history
from the configured origin (`git fetch --unshallow origin`). No current production
calculation is used to generate these reference sources.

The tests use deterministic small inputs to compare algorithms. This does not
restore historical market parquet files or the 100 real Golden outputs.
`python scripts/check_golden_readiness.py --json` reports those separately;
passing the Golden test suite does not imply those real-data artifacts exist.

The TL-owned MCP implementation and its tests were retired by the final CLI decision on 2026-11-12. AI Lite uses its upstream tools and version-matched Skills; do not reinstall the old MCP test dependency.

## Real-data release gates and test discovery

The current checkout contains five TV parquet datasets covering 2020–2024 and
50 expected files marked `source_engine=backtrader`. An actual comparison of all
50 pairs with `scripts/compare_golden.py --engine tv --rtol 1e-6 --json` passed;
this is separate from running the Golden tooling tests. These are the small,
deterministic strategy adapters in `tests/golden/strategies`, including proxy
adapters for portfolio/ML names, not a certification of every production strategy.

The five TDX datasets (`000001`, `600519`, `SH.000300`, `510300`, `159919`) and
50 corresponding expected results remain absent. Reachable repository history
only added `.gitkeep` files to those artifact directories in commits `9803db1`
and `0b23c53`; it contains no historical TDX parquet or expected-result blobs to
restore. These absent artifacts are outside the current stage release scope: `benchmarks/stage3_migration_blockers.json` records the decision to abandon TDX acquisition and use the TV subset only. Do not block the current release on TDX, or claim that its absent artifacts passed verification.
A differently dated, one-row test-generated parquet is not a substitute.

Only if TDX real-data validation is explicitly brought back into scope, obtain the manifest's genuine TDX data from a
traceable source (or fetch it using the provider), validate its symbol, dates,
adjustment policy and timezone, and record its hashes. Then generate new results
with the independent Backtrader executor into a new output directory and review
that provenance. Do not use the default `--oracle tradelearn` to establish an
independent baseline, and do not overwrite an existing accepted archive:

```sh
# Network acquisition, only when the source is available and being validated:
python scripts/build_golden.py --version 1.x --engine tdx --datasets-only \
  --datasets-root /path/to/reviewed-datasets --out /path/to/new-expected
# Independent computation after genuine data are available:
python scripts/build_golden.py --version 1.x --engine tdx --oracle backtrader \
  --datasets-root /path/to/reviewed-datasets --out /path/to/new-expected
python scripts/compare_golden.py --engine tdx --rtol 1e-6 --json \
  --datasets-root /path/to/reviewed-datasets --expected-root /path/to/new-expected
```

Readiness currently checks artifact existence; it is not itself a provenance,
coverage, or numerical-comparison check. Full manifest readiness remains
`datasets=5/10`, `expected=50/100` until the missing TDX evidence exists.

The default pytest `testpaths` now include all of `tests`, including top-level
provider/transport tests and the Charting review/indicator tests. Previously the
three selected directories (`unit`, `golden`, `consistency`) omitted those files.
The two `tests/smoke/test_*.py` files contain runnable helpers rather than test
functions; unit example tests invoke their quickstart, migration, and tutorial
entry points. Run `python -m pytest` for the complete default collection.

Seven unconditional future-stage consistency placeholders remain skips. Existing
indicator, factor, report, and Backtrader tests cover portions of those goals, but
that does not prove every originally proposed external-oracle/E2E check exists.
The separate `design/PROJECT_STRUCTURE.md` skip concerns a missing documentation
artifact; executable layering tests still enforce import/module boundaries.
Neither kind of skip should be counted as a passing numerical check.

2026-11-12：比较器统一使用TL比率口径。Backtrader Returns 的 `rnorm` 映射 `annual_return`，`rnorm100` 是百分数，不直接与比率比较；`return_pct` 则保持百分数。只修对照输出适配，不修改固定oracle源码、收益公式或比较容差。golden消费者使用显式PandasData和当前指标输出名。


2026-11-12 Pandas 3测试兼容：Alpha191冻结原件仍按原SHA保存，不修改/删除公式。旧原件47处numeric `loc/iloc[...] = None`在Pandas 3拒绝；测试装载器只将这些赋值缺失标记转为NaN，其余AST不变。已在原Pandas2.3.3对184公式、240行/186列比较原装载与兼容装载：数值、缺失与顺序逐值完全一致（dtype沿原比较规则排除），随后Pandas3.1真实旧oracle对当前实现通过。详见整合ORACLE-ALPHA191-SENTINEL-COMPAT收据；不替换基准数据、公式、窗口或放宽数值容差。

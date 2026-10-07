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

MCP tests require the optional lab dependency declared in `pyproject.toml`.
The minimum-supported SDK can be checked explicitly in the project environment:

```sh
uv pip install --python .venv/bin/python 'mcp==1.8.0'
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_mcp.py
```

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

The default pytest `testpaths` cover `unit`, `golden`, and `consistency`.
Run `python -m pytest tests` to include top-level provider/transport tests and
the Charting review/indicator tests. The two `tests/smoke/test_*.py` files
contain runnable helpers rather than test functions; unit example tests invoke
their quickstart, migration, and tutorial entry points.

Seven unconditional future-stage consistency placeholders remain skips. Existing
indicator, factor, report, and Backtrader tests cover portions of those goals, but
that does not prove every originally proposed external-oracle/E2E check exists.
The separate `design/PROJECT_STRUCTURE.md` skip concerns a missing documentation
artifact; executable layering tests still enforce import/module boundaries.
Neither kind of skip should be counted as a passing numerical check.

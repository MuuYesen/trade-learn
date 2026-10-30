# Restoring the frozen test oracle

The frozen 1.x sources are present under `reference/tradelearn_1x/` in the
public repository. `tests/golden/fixture_checksums.json` and
`scripts/oracle_manifest.json` identify their expected SHA-256 content. The
reference is test-only and excluded from the distributed package; it is not
regenerated from the current implementation. If a checkout is missing the files,
restore them with `python scripts/restore_oracle.py` from the verified historical
Git blobs; the restore refuses to overwrite modified files. A shallow checkout
without those blobs must first fetch the required history.

CI keeps a shallow checkout and fetches the exact historical commit declared in
`scripts/oracle_manifest.json` before tests that exercise restoration from Git.
It does not fetch unrelated repository history or replace the frozen files.

CI installs the dedicated `ci` and `oracle` dependency groups plus the documented
indicator, factor, Parquet, report, ML, DuckDB and research extras before running
`python scripts/check_oracle.py`, the consistency suite and the full tests. Tests
that reference LeafQuant-private research/article files report a skip when those
files are absent from a standalone public checkout; they remain runnable against
their actual sources. The existing oracle and coverage gates remain in force.

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

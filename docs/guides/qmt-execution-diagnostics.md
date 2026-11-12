# QMT execution diagnostics

The feedback contract `2026-09-08-v2` distinguishes bridge submission, QMT order state, and execution completion. A bridge acknowledgement is not an exchange acceptance or a fill. The executor must not advance a rebalance from a first/partial fill.

## Evidence retained in each run log

- `SIGNAL_START` / `RUNTIME`: run ID, feedback contract version, execution flag and effective polling settings.
- `QMT_REQUEST`: unique client correlation ID, symbol, side, quantity, order type, price type and limit price. Credentials and session tokens are excluded.
- `QMT_ACK`: client ID, returned QMT ID, stable tracking ID, raw `passorder` return if supplied, and whether a non-client ID was resolved. Unresolved IDs remain distinguishable.
- `QMT_ORDER_UPDATE`: state changes only, with tracking/QMT/client IDs, numeric state, reason availability and filled quantity. These are written during polling, so later account/position query failures do not erase observed rejections.
- `QMT_QUERY_ERROR` / `QMT_REQUEST_ERROR`: method, endpoint name, exception type, elapsed time and whether submission is uncertain. No automatic order retry occurs.
- `POLL_DIAGNOSTICS`: configured timeout/interval, raw versus matched order/trade counts, missing tracking IDs, and up to 30 observed order IDs. Zero matched orders must not be interpreted as zero broker orders.
- `RESPONSE` / `ORDER_STATUS`: matched final feedback. Status 57 is rejected. A missing rejection message is explicitly recorded as `QMT未返回废单原因`, not replaced with a fictitious cause.
- `SELL_PHASE` / `EXECUTION_FAILURE`: missing sell/buy IDs, rejection counts and available rejection reasons. `RESULT` records final outcome.

The finalizer covers startup failures and HOLD/ALREADY_EXECUTED/NO_ACCOUNT exits as well as execution. It records a terminal RESULT and attempts to upload its tagged text log to the configured S3 prefix on failure as well as success. A disconnect failure is recorded and does not prevent the upload attempt. An S3 upload failure remains visible in container output; it cannot be guaranteed that a log reaches S3 during an S3 outage.

## Discord statistics

The existing webhook schema is retained. `buy_orders` and `sell_orders` now represent ACTION records (actual order attempts), rather than the pre-trade plan. `REBALANCE` still retains planned counts in the original run log. Repeated audit, response and status entries count a rejected order once. Per-order `reason` carries the rejection message or the explicit missing-reason marker. Position deltas are diagnostic evidence only and never confirm individual orders. Fill quantities and fees accumulate once per unique trade ID within each order; broker cumulative quantities are reconciled separately. Partial fills remain PARTIAL. The parser retains the original SIGNAL_START run ID and uses a deterministic content-based fallback for legacy logs, so replaying the same file preserves its identity.

## Submission and restart recovery

Batch acknowledgements retain one ACCEPTED, REJECTED or UNKNOWN outcome per request. ACCEPTED means the bridge returned an identity; it does not prove execution. A timeout after POST is UNKNOWN, with no automatic POST retry. ACTION retains the client ID, broker ID, outcome and reason even when persistence fails after the broker response. Unconfirmed or rejected sells block buying; unconfirmed buys block the successful execution marker.

The following describes the future canonical source, not a restored production strategy. On 2026-10-25 the retired Windows legacy host launch/reactivation entries were made fail-closed; the stopped legacy image was not changed. Any future restoration requires a reviewed release using the new namespace, preserving the original account hash, identity and journal contents.

Before POST, the executor conditionally persists PREPARED requests to `s3://<configured-bucket>/Other/恢复记录/qmt-execution-journals/<first-24-hex-of-sha256-account-id>.json`. This namespace is independent of the strategy's S3_PREFIX. After POST and polling, it persists outcomes and terminal states. Failure to read or write the journal blocks further submission. Restart reconciles unresolved requests with broker identities and terminal statuses; missing, ambiguous or active outcomes block submission rather than trigger a retry. Other active account orders also block entry.

A nonblocking process lock and persistent `res/executor_host_id` enforce one executor volume owner. Keep the same bind-mounted `res` directory across restarts. A different host identity requires explicit reconciliation and migration; deleting the journal or regenerating host identity is not a recovery procedure. All executors for an account must use one canonical S3 bucket. This is a single-executor-host design, not a multi-host trading coordinator. The runner refuses to remove a running container.

Required deployment capability: S3 GetObject/PutObject on the account journal namespace, conditional PUT support (`IfMatch`/`IfNoneMatch`) in the installed SDK and S3 endpoint, plus the persistent writable volume. These have been tested offline with conditional-write failure injection, not against production writes.

## September 8 regression

Sanitized captured QMT rows are in `tests/unit/strategy/fixtures/qmt_20260908_rejections.json`. The offline test passes them through the adapter, execution poller, text logger and notification parser. Expected result: submitted 12, buy 0, sell 12, rejected 12, fills 0; all details REJECTED, no TIMEOUT, no confirmed execution. The fixture has no credentials and does not query QMT or send notifications.

Run:

```sh
.venv/bin/python -m pytest tests/unit/strategy --ignore=tests/unit/strategy/test_no_vendor_reports.py -q
```

## Coordinated rollout

1. Replace the QMT-hosted bridge script with `zoo/deploy/adapters/qmt/qmt-internal-http-bridge/qmt_internal_bridge.py`. Its order endpoint now preserves local/system/client IDs, side/date, and `m_strCancelInfo` / `m_strErrorMsg` fallback reasons.
2. Build the executor image with the updated adapter, common strategy/logger, executor, `execution_journal.py` and monitor, and install the updated runner. The Dockerfile copies these sources. Preserve the mounted `res` directory and configure the account journal permissions described above. Both the bridge and adapter directories are ignored by this release repository; include their actual local contents in the deployment artifact.
3. Verify the next run has `Feedback_Contract: 2026-09-08-v2`, correlation events, and a reason or explicit missing-reason marker. An older bridge can expose xt query IDs but may still omit rejection detail.
4. Diagnose the upstream rejection before authorizing another order run. The repair does not establish why QMT rejected the September 8 orders and does not make those orders executable. Inspect QMT's rejection column or `logs/qmt_internal_bridge_2026-09-08.log` on the QMT host if the reason remains absent.

Do not replay the live executor to test logging. Use the offline fixture. Do not overwrite historical production logs to make them appear to contain information unavailable at their original observation time.

## Local validation status

The strategy suite passes 119 tests with `test_no_vendor_reports.py` excluded. Its existing legacy-package assertion fails because the local `tradelearn/strategy` directory exists; that directory was not deleted as part of this repair. Production bridge/executor/runner deployment and production conditional-write validation have not been performed.

Bridge exception boundary: validation failures before calling QMT are REJECTED. Exceptions during or after `passorder` return per-item UNKNOWN with the original client identity and any captured order identity. Batch summaries count unknown separately from rejected; the adapter prioritizes explicit UNKNOWN over the legacy success flag. Offline tests verify persistence and blocked restart/resubmission after this response. Deploy bridge and adapter together; the previous adapter interprets success=false as rejection.

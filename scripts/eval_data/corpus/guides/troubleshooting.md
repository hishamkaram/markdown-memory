![runbook](https://img.shields.io/badge/runbook-live-blue.svg)
Pasted from the on-call wiki; formatting is rough.

# Troubleshooting

#### Consumer stuck at the same offset

Check for a poison message. Skip it with the replay flag after capturing the payload.

## Out Of Memory Kills

The container is OOM-killed when batch size times record size exceeds the memory limit.

```bash
kubectl top pod -l app=helios

# list the largest recent batches
helios debug batches --top 10

## Slow Queries After Upgrade

Statistics are stale after a major version bump. Run `ANALYZE` on the fact tables.

### Verifying The Fix

Compare p99 latency before and after in the dashboard.

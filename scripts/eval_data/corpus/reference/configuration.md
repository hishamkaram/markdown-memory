---
title: Helios Configuration Reference
owner: platform-team
---

Every option can be set in `helios.toml`, by environment variable, or by flag.
Flags win over environment variables, which win over the file.

# Helios Configuration Reference

## Environment Variables

### Ingest

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `HELIOS_INGEST_BATCH_SIZE` | int | `500` | Records pulled from the broker per poll |
| `HELIOS_INGEST_WORKERS` | int | `4` | Parallel decoder goroutines |
| `HELIOS_INGEST_MAX_LAG_MS` | int | `15000` | Lag that trips the backpressure circuit |

### Storage

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `HELIOS_STORE_DSN` | string | *(required)* | PostgreSQL connection string |
| `HELIOS_STORE_POOL_MAX` | int | `32` | Upper bound of pooled connections |
| `HELIOS_WAL_SEGMENT_MB` | int | `64` | Size at which a write-ahead segment rolls |

## Command Line Flags

```text
helios serve [flags]

  --config string               path to helios.toml (default "/etc/helios/helios.toml")
  --replay-from-offset int      re-consume the topic starting at this offset
  --dry-run                     decode and validate without writing to storage
  --metrics-addr string         Prometheus listener (default ":9102")
```

`--replay-from-offset` is destructive when combined with `--truncate`; take a snapshot first.

## File Format

```toml
[ingest]
batch_size = 500
workers = 4

[store]
dsn = "postgres://helios@db.internal/helios"
pool_max = 32
```

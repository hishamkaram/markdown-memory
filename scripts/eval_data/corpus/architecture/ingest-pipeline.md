# Ingest Pipeline Architecture

Status: **accepted** - revision 7.

## Overview

Helios consumes change events from the broker, normalises them, and writes them to the
columnar store. The pipeline is a chain of bounded stages connected by channels.

## Stages

### Decoder

The decoder turns Avro payloads into the internal `Record` struct. Schema lookups are
cached for ten minutes; a cache miss costs one round trip to the schema registry.

```go
type Record struct {
    Key       []byte
    Timestamp time.Time
    Fields    map[string]Value
}
```

### Deduplicator

Records are deduplicated on `(key, timestamp)` with a sliding window of two minutes
backed by a cuckoo filter. False positives drop at most 0.01% of legitimate records.

### Writer

The writer groups records into row groups of 64 MiB and flushes them with a two-phase
commit against the metadata catalog.

## Backpressure

When the writer falls behind, channel buffers fill up and the decoder blocks. Once lag
exceeds the configured ceiling the consumer stops polling the broker entirely, which lets
the broker retain the data instead of the process exhausting its memory. Polling resumes
after lag falls under half of the ceiling.

## Failure Modes

| Failure | Detection | Automatic response |
| --- | --- | --- |
| Schema registry down | lookup timeout | serve from cache, then pause the partition |
| Catalog commit conflict | optimistic lock error | retry with jitter, max 5 attempts |
| Disk full | `ENOSPC` on flush | stop ingest, page on-call |

## Capacity Planning

**Scenario 1.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 2.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 3.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 4.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 5.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 6.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 7.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 8.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 9.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

**Scenario 10.** A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. A cluster sized for this scenario needs headroom for replays, compaction and traffic spikes, so provision for twice the steady-state throughput and verify it with a synthetic load test before onboarding new tenants. 

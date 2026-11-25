---
title: Runbook — Telemetry Ingestion Outage
department: engineering
doc_type: engineering
updated: 2026-09-02
---

# Runbook: Telemetry Ingestion Outage

Use this runbook when the **telemetry-ingest** service stops accepting data from the
fleet, or when ingestion lag exceeds the alert threshold.

## Alert definition

The `TelemetryIngestLagHigh` alert fires when the 5-minute p95 ingestion lag exceeds
**90 seconds** for 3 consecutive evaluation periods. A separate critical alert,
`TelemetryIngestDown`, fires when the accepted-messages rate drops to zero for 2 minutes.

## Severity

- Lag above 90 seconds but data still flowing: **SEV-3**.
- Lag above 10 minutes, or more than 25% of gateways disconnected: **SEV-2**.
- `TelemetryIngestDown` firing, or any data loss confirmed: **SEV-1**.

A SEV-1 requires an Incident Commander within 15 minutes and a status-page update within
30 minutes. SEV-2 requires a status-page update within 60 minutes. SEV-3 needs no
status-page update.

## First five minutes

1. Acknowledge the page in PagerDuty so it does not escalate to the secondary on-call.
2. Open the "Ingest Overview" Grafana dashboard and check the Kafka consumer lag panel.
3. Check whether the broker cluster `kafka-telemetry-prod` is healthy. If two or more
   brokers are offline, escalate immediately to the data-platform team.
4. Check recent deploys in the #deploys channel. More than half of past ingestion
   outages were caused by a deploy in the preceding 2 hours.

## Common causes and fixes

**Bad deploy.** Roll back with `kctl rollback telemetry-ingest --to previous`. Rollbacks
take about 4 minutes to complete across all three regions.

**Consumer group rebalance storm.** Symptoms: lag sawtooths and consumers restart
repeatedly. Fix: scale the consumer deployment down to 0, wait 60 seconds, then scale
back to the normal 24 replicas.

**Disk full on brokers.** Kafka retention is 72 hours. If disk usage exceeds 85%, reduce
retention temporarily to 24 hours with the `retention-emergency` script. Never delete
segment files by hand.

## Data loss and replay

Gateways buffer up to **6 hours** of telemetry locally (see the Vela gateway
specifications). If an outage lasts less than 6 hours, no data is lost: gateways replay
their buffer automatically once ingestion recovers. Beyond 6 hours, the oldest buffered
data is overwritten and the incident must be treated as SEV-1 data loss.

## After the incident

A blameless post-mortem is required for every SEV-1 and SEV-2, due within 5 business
days. Post-mortems are filed in the Incident Reviews space and reviewed at the Thursday
reliability meeting.

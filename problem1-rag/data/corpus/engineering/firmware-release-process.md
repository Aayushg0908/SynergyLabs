---
title: Firmware Release Process
department: engineering
doc_type: engineering
updated: 2026-07-21
---

# Firmware Release Process

This document describes how firmware for the KX-200 drone and the Vela gateway moves
from a merged pull request to the customer fleet.

## Versioning

Firmware uses semantic versioning: `MAJOR.MINOR.PATCH`. A MAJOR bump is required for any
change to the flight-control protocol or the on-disk log format. Release candidates are
tagged with an `-rc.N` suffix, for example `4.2.0-rc.3`.

## Release train

Kestrel ships a minor firmware release every **six weeks**. Code freeze happens on the
Monday two weeks before the release date. After code freeze, only fixes approved by the
release captain may be merged into the release branch.

## Required testing

Every release candidate must pass, in order:

1. The automated hardware-in-the-loop (HIL) suite on the test rack in Building C. The suite
   contains about 1,800 test cases and takes roughly 5 hours.
2. A **72-hour soak test** on a fleet of 20 internal drones running continuous simulated
   missions.
3. Field flight testing: at least **40 flight hours** across at least 3 different weather
   conditions, logged by the flight-test team.

A release candidate that fails any step is discarded and a new `-rc` is cut.

## Staged rollout

Approved releases go out in four stages:

| Stage | Share of fleet | Minimum duration |
|-------|----------------|------------------|
| Canary | 1% | 48 hours |
| Early | 10% | 72 hours |
| Broad | 50% | 72 hours |
| General availability | 100% | — |

Customers enrolled in the "Stable channel" receive updates only at the General
availability stage. Rollout is paused automatically if the crash-report rate in any stage
exceeds **0.5%** of devices in that stage within 24 hours.

## Rollback

Every device keeps the previous firmware image in a secondary partition (A/B
partitioning). A rollback can be pushed to the fleet in under 30 minutes. Downgrading
across a MAJOR version is not supported because of log-format changes.

## Sign-off

A release needs sign-off from three people: the release captain, the QA lead and the
head of flight safety. The current head of flight safety is Tomás Okafor.

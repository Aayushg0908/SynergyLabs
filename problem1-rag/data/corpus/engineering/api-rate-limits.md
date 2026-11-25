---
title: Kestrel Cloud API — Rate Limits and Quotas
department: engineering
doc_type: engineering
updated: 2026-08-14
---

# Kestrel Cloud API — Rate Limits and Quotas

The Kestrel Cloud API is the public REST interface customers use to pull telemetry from
KX-series drones and Vela gateways. This page is the source of truth for rate limits.
If a support article disagrees with this page, this page wins.

## Plans and request limits

Rate limits are enforced per API key using a token-bucket algorithm. Each bucket refills
continuously, so a client that waits will regain capacity gradually rather than all at once
at the top of the minute.

| Plan       | Sustained rate        | Burst capacity | Monthly request quota |
|------------|-----------------------|----------------|-----------------------|
| Starter    | 60 requests / minute  | 120 requests   | 500,000               |
| Team       | 600 requests / minute | 1,200 requests | 10,000,000            |
| Enterprise | 3,000 requests / minute | 6,000 requests | Negotiated in contract |

The telemetry streaming endpoint (`/v2/streams`) is not counted against the per-minute
limit. Instead, each plan has a cap on concurrent open streams: Starter allows 2, Team
allows 25 and Enterprise allows 200.

## What happens when you exceed a limit

When the bucket is empty the API returns HTTP **429 Too Many Requests**. The response
always includes a `Retry-After` header with the number of seconds to wait, and the
headers `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`.

Clients must back off exponentially, starting at the `Retry-After` value and doubling up
to a maximum wait of 64 seconds, with random jitter of up to 20%. The official Python and
Go SDKs implement this automatically and give up after 6 retries.

If an API key receives more than 10,000 HTTP 429 responses in a rolling 24-hour window,
it is automatically placed in a "cool-down" state for 15 minutes, during which every
request returns HTTP 403 with error code `KEY_COOLDOWN`. Repeated cool-downs page the
on-call API engineer.

## Monthly quota overages

Starter accounts that exhaust their monthly quota are hard-stopped until the first day
of the next calendar month (UTC). Team accounts are never hard-stopped; overage is billed
at $0.40 per additional 10,000 requests. Enterprise overage terms are defined in each
contract.

## Requesting a limit increase

Team customers may request a temporary burst increase of up to 3x for a planned event
(for example, a large survey flight) by opening a ticket at least 5 business days in
advance. Increases are approved by the API platform team lead, currently Priya Raman.
Permanent increases require an upgrade to Enterprise.

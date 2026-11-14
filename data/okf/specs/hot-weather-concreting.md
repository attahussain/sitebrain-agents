---
type: Specification Requirement
title: Hot-weather concreting
description: Current limits for concrete temperature and time from batching to discharge.
tags: [concrete, hot-weather, discharge-time, temperature, spec-03300]
sources:
  - id: spec-03300
    resource: data/docs/spec_03300_cast_in_place_concrete.md
    title: Specification Section 03300, Hot-weather concreting
    author: team:design-consultant
  - id: rfi-021
    resource: /decisions/rfi-021-retarder-discharge-time.md
    title: RFI-021 response
    author: team:design-consultant
generated: { by: sitebrain/0.4, at: 2026-10-16T09:00:00Z }
verified: { by: human:qa-manager, at: 2026-10-16T14:00:00Z }
status: stable
stale_after: 2027-04-30T00:00:00Z
---
# Requirement
- Concrete temperature at the point of discharge must not exceed 32 C.[^spec-03300]
- Standard mixes must be discharged within 90 minutes of batching; later loads are rejected.[^spec-03300]
- Exception: mixes with an approved retarding admixture may be discharged up to 120 minutes after
  batching. See [RFI-021](/decisions/rfi-021-retarder-discharge-time.md).[^rfi-021]

This revision replaces [Rev 0](/specs/hot-weather-concreting-rev0.md).

[^spec-03300]: Specification Section 03300, Hot-weather concreting
[^rfi-021]: RFI-021 response from the consultant

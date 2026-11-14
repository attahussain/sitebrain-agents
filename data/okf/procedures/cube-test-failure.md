---
type: Playbook
title: Failed cube test
description: Steps when 28-day cube results fall below the specified characteristic strength.
tags: [concrete, testing, cubes, quality]
sources:
  - id: qa-plan
    resource: project quality plan, section 7.3
    title: Project Quality Plan, non-conforming concrete
    author: team:qa
generated: { by: human:qa-manager, at: 2026-09-02T10:00:00Z }
verified: { by: human:qa-manager, at: 2026-09-02T10:00:00Z }
status: stable
---
# Trigger
Any 28-day cube result below the specified grade (C50 for columns and core walls, C40 for slabs).

# Steps
1. Raise a non-conformance report within 24 hours and notify the consultant.[^qa-plan]
2. Take three cores from the element within 7 days and test them at an approved laboratory.[^qa-plan]
3. If the cores also fail, the structural engineer decides between strengthening and demolition.[^qa-plan]

[^qa-plan]: Project Quality Plan, section 7.3

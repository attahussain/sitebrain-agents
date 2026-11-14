---
type: Specification Requirement
title: Concrete cover
description: Minimum concrete cover to reinforcement for each structural element.
tags: [concrete, cover, rebar, spec-03300]
sources:
  - id: spec-03300
    resource: data/docs/spec_03300_cast_in_place_concrete.md
    title: Specification Section 03300, Cast-in-Place Concrete
    author: team:design-consultant
generated: { by: sitebrain/0.4, at: 2026-10-14T08:00:00Z }
verified: { by: human:qa-manager, at: 2026-10-14T12:00:00Z }
status: stable
---
# Requirement

| Element | Minimum cover |
|---|---|
| Columns | 40 mm |
| Beams | 35 mm |
| Slabs | 25 mm |
| Foundations and raft | 50 mm |

Values are taken from Section 03300.[^spec-03300]

# Related
Cover is checked at every rebar inspection. After the failed Level 15 inspection the site introduced a
spacer rule; see the [Level 15 column cover lesson](/lessons/level-15-cover-failure.md).

[^spec-03300]: Specification Section 03300, Concrete cover

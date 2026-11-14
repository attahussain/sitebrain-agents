"""Open Knowledge Format (OKF v0.2) bundles: curated, linked project knowledge.

A bundle is a directory of markdown concepts with YAML frontmatter (spec:
github.com/GoogleCloudPlatform/open-knowledge-format). SiteBrain reads it as a second, trusted
source next to the raw documents: each concept becomes one searchable row that carries its trust
tier, and links between concepts are followed at query time.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

RESERVED = {"index.md", "log.md"}  # spec §3.1: never concept documents
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+\.md)(?:#[^)]*)?\)")
_FOOTNOTE_DEF = re.compile(r"^\[\^[^\]]+\]:.*$", re.M)
_FOOTNOTE_REF = re.compile(r"\[\^[^\]]+\]")


@dataclass
class Concept:
    concept_id: str  # bundle path without ".md" (spec §2)
    type: str
    title: str
    description: str
    body: str
    status: str  # draft | stable | deprecated; absent means stable (spec §5.4)
    trust: str  # unverified | machine-confirmed | human-reviewed (spec §5.3)
    stale: bool
    links: list[str] = field(default_factory=list)  # concept ids this concept links to
    sources: list[dict] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n---", 1)
    if len(parts) < 2:
        return {}, text
    meta = yaml.safe_load(parts[0][3:]) or {}
    return (meta if isinstance(meta, dict) else {}), parts[1].lstrip("-").lstrip("\n")


def trust_tier(verified) -> str:
    if not verified:
        return "unverified"
    events = verified if isinstance(verified, list) else [verified]  # a bare mapping is a one-element list
    actors = [str(e.get("by", "")) for e in events if isinstance(e, dict)]
    return "human-reviewed" if any(a.startswith("human:") for a in actors) else "machine-confirmed"


def _as_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def resolve_link(from_id: str, href: str) -> str | None:
    """Turn a markdown link target into a concept id. Absolute links start at the bundle root."""
    if "://" in href:
        return None
    if href.startswith("/"):
        path = PurePosixPath(href.lstrip("/"))
    else:
        path = PurePosixPath(from_id).parent / href
    parts: list[str] = []
    for p in path.parts:
        if p == "..":
            if parts:
                parts.pop()
        elif p != ".":
            parts.append(p)
    return "/".join(parts)[:-3] if parts else None


def load_bundle(root: Path, now: datetime | None = None) -> dict[str, Concept]:
    """Every concept in the bundle, keyed by concept id. Concepts without `type` are skipped."""
    now = now or datetime.now(timezone.utc)
    concepts: dict[str, Concept] = {}
    if not root.is_dir():
        return concepts
    for path in sorted(root.rglob("*.md")):
        if path.name in RESERVED:
            continue
        cid = path.relative_to(root).with_suffix("").as_posix()
        meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        if not meta.get("type"):
            continue  # not a conformant concept (spec §11)
        stale_after = _as_datetime(meta.get("stale_after"))
        links = []
        for href in _LINK.findall(body):
            target = resolve_link(cid, href)
            if target and target != cid and target not in links:
                links.append(target)
        concepts[cid] = Concept(
            concept_id=cid,
            type=str(meta["type"]),
            title=str(meta.get("title") or path.stem.replace("-", " ")),
            description=str(meta.get("description") or ""),
            body=body.strip(),
            status=str(meta.get("status") or "stable"),
            trust=trust_tier(meta.get("verified")),
            stale=bool(stale_after and now >= stale_after),
            links=links,
            sources=list(meta.get("sources") or []),
            frontmatter=meta,
        )
    return concepts


def provenance(c: Concept) -> str:
    """One line a reader (or model) can cite: trust tier, who verified it and when, lifecycle status."""
    verified = c.frontmatter.get("verified") or []
    events = verified if isinstance(verified, list) else [verified]
    checks = ", ".join(f"{e.get('by')} on {str(e.get('at', ''))[:10]}" for e in events if isinstance(e, dict))
    origin = [s.get("title") or s.get("resource") for s in c.sources if isinstance(s, dict)]
    parts = [f"Trust: {c.trust}" + (" (stale)" if c.stale else ""), f"status: {c.status}"]
    if checks:
        parts.append(f"verified by {checks}")
    if origin:
        parts.append("derived from " + "; ".join(str(o) for o in origin))
    return ". ".join(parts) + "."


def to_row(c: Concept) -> dict:
    """One search row per concept. Concepts are short and self-contained, so they are not split."""
    body = _FOOTNOTE_REF.sub("", _FOOTNOTE_DEF.sub("", c.body)).strip()
    return {
        "chunk_id": f"okf:{c.concept_id}",
        "doc_id": f"okf:{c.concept_id}",
        "title": c.title,
        "text": f"{c.title} ({c.type})\n{c.description}\n{body}\n{provenance(c)}".strip(),
        "kind": "okf",
        "trust": c.trust + (", stale" if c.stale else ""),
        "status": c.status,
    }


def searchable(concepts: dict[str, Concept]) -> list[Concept]:
    """Deprecated concepts stay in the bundle for history and links but are not retrieved."""
    return [c for c in concepts.values() if c.status != "deprecated"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "rfi"


def record_rfi(root: Path, subject: str, body: str, citations: list[str], reviewer: str,
               now: datetime | None = None) -> Path:
    """Write an approved RFI back into the bundle as a draft concept.

    The reviewer approved releasing the RFI, not its eventual answer, so the concept is `draft`
    and carries no `verified` entry; it records that the question is open, so later questions
    on the same topic see it instead of raising a duplicate.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    sources = [{"id": f"s{i}", "resource": f"/{c[4:]}.md" if c.startswith("okf:") else c}
               for i, c in enumerate(citations, 1)]
    meta = {
        "type": "Open RFI",
        "title": subject,
        "description": f"RFI raised on {now:%d %b %Y}; awaiting the consultant's response.",
        "tags": ["rfi", "open"],
        "sources": sources,
        "generated": {"by": "sitebrain/0.4", "at": stamp},
        "approved_for_release": {"by": f"human:{_slug(reviewer) or 'reviewer'}", "at": stamp},
        "status": "draft",
    }
    path = root / "rfis" / f"{now:%Y%m%d}-{_slug(subject)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{front}\n---\n# RFI\n{body.strip()}\n", encoding="utf-8")
    return path

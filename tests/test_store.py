from sitebrain.config import ROOT
from sitebrain.store import chunk_markdown


def test_sections_become_separate_chunks_with_context_prefix():
    text = (ROOT / "data" / "docs" / "hse_heat_stress_policy.md").read_text(encoding="utf-8")
    chunks = chunk_markdown("hse", text)

    scaffold = [c for c in chunks if "inspection tag" in c["text"]]
    assert len(scaffold) == 1
    # The scaffold rule is not diluted by the heat-stress sections...
    assert "midday" not in scaffold[0]["text"].lower()
    # ...and still carries its document and section context.
    assert scaffold[0]["text"].startswith("HSE Policy - Heat Stress Management (fictional demo project) - Scaffolding")


def test_oversized_section_is_split_by_paragraph():
    body = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(4))
    chunks = chunk_markdown("doc", f"# Title\n\n## Long section\n{body}", max_chars=700)
    assert len(chunks) == 2
    assert all(c["text"].startswith("Title - Long section") for c in chunks)


def _hits(ids, source):
    return [{"chunk_id": i, "doc_id": i, "title": "", "text": "", "score": 0.0, "sources": [source]} for i in ids]


def test_identifier_detection():
    from sitebrain.store import has_identifier

    for q in ["What did inspection INS-1007 find?", "Why was TC-2 stopped?", "Rework on C15-01?", "Section 03300 cover"]:
        assert has_identifier(q), q
    for q in ["How often must scaffolds be re-inspected?", "What is the cover on Level 15?", "re-inspection rules"]:
        assert not has_identifier(q), q


def test_rrf_rewards_agreement_between_retrievers():
    from sitebrain.store import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([_hits(["a", "b", "c"], "vector"), _hits(["b", "d", "a"], "keyword")], k=3)
    assert [h["chunk_id"] for h in fused][:2] == ["b", "a"]  # found by both lists beat single-list hits
    assert fused[0]["sources"] == ["keyword", "vector"]


def test_lexical_first_keeps_keyword_order_and_backfills():
    from sitebrain.store import lexical_first

    out = lexical_first(_hits(["k1", "k2"], "keyword"), _hits(["v1", "k1", "v2"], "vector"), k=4)
    assert [h["chunk_id"] for h in out] == ["k1", "k2", "v1", "v2"]

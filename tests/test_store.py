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

"""Create red-difference Word copies against the preceding theory drafts."""
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


DOWNLOADS = Path("C:/Users/Admin/Downloads")
MAIN_IN = DOWNLOADS / "GB-LCSS_experiments_revised_20260919.docx"
SUPP_IN = DOWNLOADS / "GB-LCSS_supplementary_experiments_20260919.docx"
MAIN_OUT = DOWNLOADS / "GB-LCSS_experiments_revised_20260919_red.docx"
SUPP_OUT = DOWNLOADS / "GB-LCSS_supplementary_experiments_20260919_red.docx"

RED = "FF0000"
BLACK = "000000"


def set_xml_color(rpr, color):
    tag = qn("w:color")
    node = rpr.find(tag)
    if node is None:
        node = OxmlElement("w:color")
        rpr.append(node)
    node.set(qn("w:val"), color)


def mark_container_red(container):
    """Color Word runs and native math runs while preserving their formatting."""
    for rpr in container._p.xpath(".//w:rPr"):
        set_xml_color(rpr, RED)
    for rpr in container._p.xpath(".//m:rPr"):
        node = rpr.find(qn("m:color"))
        if node is None:
            node = OxmlElement("m:color")
            rpr.append(node)
        node.set(qn("m:val"), RED)


def mark_table_red(table):
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                mark_container_red(paragraph)


def mark_phrase_red(paragraph, phrase):
    """Rebuild a plain-text paragraph with one changed phrase in red."""
    text = paragraph.text
    assert text.count(phrase) == 1, (phrase, text)
    before, after = text.split(phrase, 1)
    children = list(paragraph._p)
    ppr = paragraph._p.pPr
    bookmarks = [c for c in children if c.tag in {qn("w:bookmarkStart"), qn("w:bookmarkEnd")}]
    for child in children:
        if child is not ppr and child not in bookmarks:
            paragraph._p.remove(child)

    def add(text_part, color):
        if not text_part:
            return
        run = paragraph.add_run(text_part)
        rpr = run._r.get_or_add_rPr()
        set_xml_color(rpr, color)

    add(before, BLACK)
    add(phrase, RED)
    add(after, BLACK)


def mark_main():
    doc = Document(MAIN_IN)

    # One small edit in an otherwise unchanged dataset paragraph.
    dataset_paragraph = next(
        p for p in doc.paragraphs if p.text.startswith("We evaluate GB-LCSS on five")
    )
    mark_phrase_red(dataset_paragraph, "Appendices A–I")

    # These paragraphs were rewritten or newly inserted in the experiment update.
    changed_prefixes = [
        "Unless varied in Section 4.3.4",
        "Section 4.3.1 and Appendix I.2",
        "The supplementary runs under uniform 0.4",
        "The representation used for grouping determines",
        "Table 7. Classification accuracy",
        "Under the same construction settings",
        "4.3.4 Sensitivity to construction and target parameters",
        "We varied one parameter at a time",
        "The support and split thresholds have clearer failure cases",
    ]
    for paragraph in doc.paragraphs:
        if any(paragraph.text.startswith(prefix) for prefix in changed_prefixes):
            mark_container_red(paragraph)

    # Table 7 was replaced with the paired five-seed version.
    assert len(doc.tables) == 9
    mark_table_red(doc.tables[6])
    doc.save(MAIN_OUT)


def mark_supplement():
    doc = Document(SUPP_IN)
    appendix_start = next(
        i for i, p in enumerate(doc.paragraphs)
        if p.text.startswith("Appendix I. Parameter sensitivity")
    )
    for paragraph in doc.paragraphs[appendix_start:]:
        mark_container_red(paragraph)

    # Appendix I adds the last two tables after the fourteen original tables.
    assert len(doc.tables) == 16
    mark_table_red(doc.tables[14])
    mark_table_red(doc.tables[15])
    doc.save(SUPP_OUT)


if __name__ == "__main__":
    mark_main()
    mark_supplement()
    print(MAIN_OUT)
    print(SUPP_OUT)

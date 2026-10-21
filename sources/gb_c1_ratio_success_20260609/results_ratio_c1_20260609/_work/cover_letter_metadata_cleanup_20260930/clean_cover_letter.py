import hashlib
import io
import json
import posixpath
import zipfile
from pathlib import Path

from docx import Document
from lxml import etree as ET


SOURCE = Path(r"C:\Users\Admin\Downloads\cover_letter.docx")
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "cover_letter_metadata_clean.docx"
REPORT = Path(__file__).resolve().parent / "verification.json"
STAMP = (1980, 1, 1, 0, 0, 0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canon(data):
    tree = ET.fromstring(data)
    for element in list(tree.iter()):
        if ET.QName(element).localname in {"rsids", "rsid", "rsidRoot", "docId", "attachedTemplate", "savePreviewPicture"}:
            if element.getparent() is not None:
                element.getparent().remove(element)
            continue
        for attr in list(element.attrib):
            local = ET.QName(attr).localname
            if local.startswith("rsid") or local in {"paraId", "textId", "docId"}:
                del element.attrib[attr]
    return ET.tostring(tree, method="c14n")


def write_entry(zout, name, data):
    info = zipfile.ZipInfo(name, STAMP)
    info.create_system = 0
    info.external_attr = 32
    info.compress_type = zipfile.ZIP_DEFLATED
    zout.writestr(info, data)


def main():
    source_data = SOURCE.read_bytes()
    source = zipfile.ZipFile(io.BytesIO(source_data))
    assert source.testzip() is None
    original = {name: source.read(name) for name in source.namelist()}
    remove = {n for n in original if n.startswith(("docProps/", "customXml/"))}
    # No comments, tracked revisions, or content controls should be discarded.
    doc = ET.fromstring(original["word/document.xml"])
    assert not doc.xpath("//*[local-name()='ins' or local-name()='del' or local-name()='commentRangeStart' or local-name()='commentReference' or local-name()='dataBinding']")
    result = {}
    for name, data in original.items():
        if name in remove:
            continue
        if name.endswith((".xml", ".rels")):
            tree = ET.fromstring(data)
            changed = False
            for element in list(tree.iter()):
                if ET.QName(element).localname in {"rsids", "rsid", "rsidRoot", "docId", "attachedTemplate", "savePreviewPicture"}:
                    if element.getparent() is not None:
                        element.getparent().remove(element)
                        changed = True
                    continue
                for attr in list(element.attrib):
                    local = ET.QName(attr).localname
                    if local.startswith("rsid") or local in {"paraId", "textId", "docId"}:
                        del element.attrib[attr]
                        changed = True
            if name == "[Content_Types].xml":
                for child in list(tree):
                    if child.get("PartName", "").lstrip("/") in remove:
                        tree.remove(child)
                        changed = True
            if name.endswith(".rels"):
                base = "" if name == "_rels/.rels" else posixpath.dirname(posixpath.dirname(name))
                for child in list(tree):
                    target = posixpath.normpath(posixpath.join(base, child.get("Target", "")))
                    if target in remove:
                        tree.remove(child)
                        changed = True
            if changed:
                data = ET.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
        result[name] = data

    # Verify that all visible/editable Word content and formatting XML is unchanged
    # after excluding Word's document revision identifiers.
    for name in original:
        if name.startswith("word/") and name.endswith(".xml"):
            assert canon(original[name]) == canon(result[name]), name

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zout:
        for name, data in result.items():
            write_entry(zout, name, data)
    output_data = buffer.getvalue()
    check = zipfile.ZipFile(io.BytesIO(output_data))
    assert check.testzip() is None
    assert not any(n.startswith(("docProps/", "customXml/")) for n in check.namelist())
    assert not check.comment
    for info in check.infolist():
        assert info.date_time == STAMP and not info.extra and not info.comment
    before, after = Document(io.BytesIO(source_data)), Document(io.BytesIO(output_data))
    assert [p.text for p in before.paragraphs] == [p.text for p in after.paragraphs]
    assert len(before.sections) == len(after.sections)
    assert len(before.tables) == len(after.tables)
    assert len(before.inline_shapes) == len(after.inline_shapes)
    assert source_data and sha(SOURCE.read_bytes()) == sha(source_data)

    # Recheck properties are absent at the package level; Word will expose empty defaults.
    for n in check.namelist():
        assert not n.startswith("docProps/")
        if n.endswith((".xml", ".rels")):
            raw = check.read(n)
            assert b"rsidRoot" not in raw and b"python-docx" not in raw and b"WPS_" not in raw

    OUTPUT.write_bytes(output_data)
    REPORT.write_text(json.dumps({
        "source_sha256": sha(source_data), "output_sha256": sha(output_data),
        "source_unchanged": sha(SOURCE.read_bytes()) == sha(source_data),
        "removed_parts": sorted(remove), "body_text_identical": True,
        "word_content_and_formatting_xml_identical_except_revision_ids": True,
        "zip_timestamps_normalized": True, "output": str(OUTPUT)
    }, indent=2), encoding="utf-8")
    print(OUTPUT)
    print("Removed parts:", ", ".join(sorted(remove)))
    print("Paragraphs:", len(after.paragraphs), "Tables:", len(after.tables), "Sections:", len(after.sections))
    print("Original file unchanged; Word content/formatting XML verified.")


if __name__ == "__main__":
    main()

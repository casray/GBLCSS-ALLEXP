from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('C:/Users/Admin/Downloads/cover letter.docx')
OUTPUT = ROOT / 'GB-LCSS_EM_submission_20260926/cover_letter_template_revised.docx'
EXPECTED_HASH = 'd5e7df70685b920f9bde8a06fb7fe1139ca65329b8764ddc01062061fb793c19'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}

texts = {
    0: 'September 26, 2026',
    2: 'Editorial Office',
    3: 'Information Sciences',
    6: 'Dear Editors,',
    8: (
        'We would like to submit our manuscript entitled '
        '\u201cGranular ball local consensus for graph-free node classification under label noise\u201d '
        'for consideration as a Research Article in Information Sciences.'
    ),
    10: (
        'In this work, we propose Granular Ball Local-Consensus Soft Supervision '
        '(GB-LCSS), which uses the training graph to refine noisy supervision '
        'for an attribute-only classifier. Its key design separates label-free '
        'granular ball construction from label aggregation, so noisy annotations '
        'do not determine which nodes vote together. Leave-one-out consensus '
        "provides local evidence for refining each node's training target. We "
        'also derive a sufficient condition for the target to have lower expected '
        'squared error than the observed label under independent symmetric label noise.'
    ),
    12: (
        'Experiments using the same deployed MLP architecture, together with '
        'random-grouping and label-smoothing controls, support the contribution '
        'of local label evidence beyond target softening alone. The graph-teacher '
        'extension further shows that these targets can improve the distilled '
        'MLP. We believe this connection between granular ball computing and '
        'noise-robust learning will be of interest to Information Sciences readers.'
    ),
    14: 'Thank you for your time and consideration. We look forward to your response.',
    16: 'Yours sincerely,',
    18: 'Silei Cao, Kun Wang, Jingyu Zhu, Tianxing Wang, Bing Huang',
    19: 'Corresponding author: Tianxing Wang',
    20: 'E-mail: wangtianxing@nau.edu.cn',
}

assert sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_HASH
with ZipFile(REFERENCE) as reference:
    parts = {item.filename: reference.read(item.filename) for item in reference.infolist()}
    metadata = reference.infolist()

document = ET.fromstring(parts['word/document.xml'])
body = document.find('w:body', NS)
paragraphs = body.findall('w:p', NS)
assert len(paragraphs) == 21
section_before = ET.tostring(body.find('w:sectPr', NS))

for index, text in texts.items():
    paragraph = paragraphs[index]
    properties = paragraph.find('w:pPr', NS)
    original_runs = paragraph.findall('w:r', NS)
    assert original_runs
    new_run = deepcopy(original_runs[0])
    run_properties = new_run.find('w:rPr', NS)
    for child in list(new_run):
        if child is not run_properties:
            new_run.remove(child)
    ET.SubElement(new_run, f'{{{W}}}t').text = text
    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)
    paragraph.append(new_run)
body.remove(paragraphs[4])
declaration_text = (
    'The authors confirm that this manuscript has not been published previously '
    'and is not under consideration for publication elsewhere. '
    'The authors declare no conflicts of interest.'
)
declaration = deepcopy(paragraphs[12])
declaration.find('w:r/w:t', NS).text = declaration_text
paragraphs[13].addnext(declaration)
declaration.addnext(deepcopy(paragraphs[13]))
assert ET.tostring(body.find('w:sectPr', NS)) == section_before

parts['word/document.xml'] = ET.tostring(document, xml_declaration=True, encoding='UTF-8', standalone=True)
core = ET.fromstring(parts['docProps/core.xml'])
for tag in (
    '{http://purl.org/dc/elements/1.1/}creator',
    '{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy',
):
    element = core.find(tag)
    if element is not None:
        element.text = ''
title = core.find('{http://purl.org/dc/elements/1.1/}title')
if title is None:
    title = ET.SubElement(core, '{http://purl.org/dc/elements/1.1/}title')
title.text = 'Cover letter for GB-LCSS'
parts['docProps/core.xml'] = ET.tostring(core, xml_declaration=True, encoding='UTF-8', standalone=True)

with ZipFile(OUTPUT, 'w') as target:
    for item in metadata:
        target.writestr(item, parts[item.filename])

with ZipFile(REFERENCE) as reference, ZipFile(OUTPUT) as target:
    for name in reference.namelist():
        if name not in ('word/document.xml', 'docProps/core.xml'):
            assert reference.read(name) == target.read(name), name
    final = ET.fromstring(target.read('word/document.xml'))
    actual = [''.join(p.itertext()) for p in final.findall('w:body/w:p', NS)]
    visible = '\n'.join(final.xpath('//w:t/text()', namespaces=NS))
    for text in texts.values():
        assert text in visible
    assert declaration_text in visible
    assert not any(term in visible for term in ('IEEE', 'Shan Liu', 'IMVGCR', 'hbhuangbing'))
assert sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_HASH
print('Saved:', OUTPUT)
print('Total words:', len(visible.split()))
print('Template styles and page geometry preserved; author-confirmed declarations included.')

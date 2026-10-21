from pathlib import Path
import difflib
import json
import re
import shutil
import zipfile
import fitz

root = Path(__file__).resolve().parents[1]
source = root / 'GB-LCSS_submission_notation_final_20260925'
work = root / 'tmp/em_submission_build_20260926'
out = root / 'GB-LCSS_EM_submission_20260926'
backup = root / 'tmp/equation_punctuation_before_20260926'
display = re.compile(r'\\begin\{equation\}(.*?)\\end\{equation\}|\\\[(.*?)\\\]', re.S)

for entry in ('main', 'supplementary'):
    old = (backup / (entry + '.tex')).read_text(encoding='utf-8')
    new = (source / (entry + '.tex')).read_text(encoding='utf-8')
    changes = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != 'equal':
            assert set(old[a:b] + new[c:d]) <= {'.', ','}, (tag, old[a:b], new[c:d])
            changes.append((old[a:b], new[c:d]))
    formulas = list(display.finditer(new))
    assert all((m.group(1) or m.group(2)).rstrip()[-1] in '.,' for m in formulas)
    stage = work / entry
    log = (stage / (entry + '.log')).read_text(errors='replace')
    assert not re.search(r'(?:undefined references|Reference .+ undefined|Citation .+ undefined)', log, re.I)
    warnings = re.findall(r'.*Overfull.*', log)
    assert not warnings if entry == 'supplementary' else all('2.22221pt' in w for w in warnings)
    manifest = json.loads((work / (entry + '_manifest.json')).read_text())
    shutil.copy2(stage / (entry + '.pdf'), out / (entry + '.pdf'))
    with zipfile.ZipFile(out / (entry + '_source.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(manifest):
            z.write(stage / name, name)
    with zipfile.ZipFile(out / (entry + '_source.zip')) as z:
        assert all('/' not in name and '\\' not in name for name in z.namelist())
        assert z.read(entry + '.tex') == (stage / (entry + '.tex')).read_bytes()
    doc = fitz.open(stage / (entry + '.pdf'))
    assert len(doc) == (23 if entry == 'main' else 14)
    needle = '(10)' if entry == 'main' else '(H.4)'
    for index, page in enumerate(doc):
        if needle in page.get_text():
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(backup / (entry + '_equations.png'))
            print(entry, 'visual check page', index + 1)
            break
    print(entry, 'display equations:', len(formulas), 'punctuation edits:', len(changes), 'pages:', len(doc))
    # Keep the editable working copy consistent when the PDF is not locked.
    for extension in ('.pdf', '.bbl', '.aux', '.out', '.log', '.spl', '.blg'):
        generated = stage / (entry + extension)
        if generated.exists():
            try:
                shutil.copy2(generated, source / generated.name)
            except PermissionError:
                print('Original file locked; latest copy is in EM package:', generated.name)

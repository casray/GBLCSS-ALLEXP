from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
import pdfplumber
import pypdfium2 as pdfium

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
out = root / 'GB-LCSS_Elsevier_LaTeX_GPT_discussion_compiled_20260921'
original = Path('C:/Users/Admin/Downloads/GB-LCSS_final_submission_Discussion_revised.tex').read_text(encoding='utf-8-sig')
main = (out / 'main.tex').read_text(encoding='utf-8-sig')
assert main == original.replace('tab:graph_mlp', 'tab:3').replace('tab:grouping', 'tab:5')
inputs = [out / s for s in re.findall(r'\\input\{([^}]+)\}', main)]
text = main + '\n' + '\n'.join(p.read_text(encoding='utf-8-sig') for p in inputs)
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs = set(re.findall(r'\\(?:ref\*?|eqref)\{([^}]+)\}', text))
assert refs <= labels, refs - labels
aux = (out / 'main.aux').read_text(encoding='utf-8')
assert re.search(r'\\newlabel\{tab:3\}\{\{4\}', aux)
assert re.search(r'\\newlabel\{tab:5\}\{\{6\}', aux)
log = (out / 'main.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'undefined|multiply defined|LaTeX Error|Missing character', log, re.I)
render_pages = []
with pdfplumber.open(out / 'main.pdf') as pdf:
    for i, page in enumerate(pdf.pages):
        content = page.extract_text() or ''
        assert '??' not in content
        if '4.5. Discussion' in content or '5. Conclusion' in content:
            render_pages.append(i)
    print(f'PDF: {len(pdf.pages)} pages; all references resolved; only the two approved labels changed.')
doc = pdfium.PdfDocument(str(out / 'main.pdf'))
for i in render_pages:
    doc[i].render(scale=1.5).to_pil().save(work / f'page_{i+1}.png')
print('Rendered pages:', [i+1 for i in render_pages])
files = [out / name for name in ['main.tex', 'main.pdf', 'main.bbl', 'references.bib', 'gblcss-numeric.bst']]
files += inputs + [out / s for s in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', main)]
with ZipFile(out.with_suffix('.zip'), 'w', ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        z.write(p, p.relative_to(out).as_posix())
with ZipFile(out.with_suffix('.zip')) as z:
    assert z.testzip() is None
print('Main-paper source ZIP verified.')

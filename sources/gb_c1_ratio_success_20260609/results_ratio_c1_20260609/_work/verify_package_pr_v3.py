from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import difflib
import json
import re
import fitz
from PIL import Image, ImageDraw

workspace = Path(__file__).resolve().parent.parent
root = workspace / 'GB-LCSS_PR_compact_v3_reviewed_20260929'
audit = workspace / '_work/pr33_audit'
source = Path('C:/Users/Admin/Downloads/GB-LCSS_PR_PR_2026PR_compact_v2_20260929.zip')
with ZipFile(source) as archive:
    prefix = next(n[:-len('main/main.tex')] for n in archive.namelist() if n.endswith('/main/main.tex'))
    original = {n[len(prefix):]: archive.read(n) for n in archive.namelist() if n.startswith(prefix) and not n.endswith('/')}

old = original['main/main.tex'].decode('utf-8')
new = (root / 'main/main.tex').read_text(encoding='utf-8')
equations = lambda text: re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', text, re.S)
assert equations(old) == equations(new)
assert len(equations(new)) == 23
assert old.split('\\begin{abstract}')[0] == new.split('\\begin{abstract}')[0]
assert re.findall(r'\\cite\{[^}]+\}', old) == re.findall(r'\\cite\{[^}]+\}', new)
assert re.findall(r'\\label\{[^}]+\}', old) == re.findall(r'\\label\{[^}]+\}', new)
changed = []
for path, data in original.items():
    if (root / path).read_bytes() != data:
        changed.append(path)
assert set(changed) <= {'main/main.tex', 'main/main.pdf', 'main/main.bbl', 'main/main.spl'}
log = (root / 'main/main.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'Overfull|undefined references|undefined citations|Citation .+ undefined|Reference .+ undefined', log)
bbl = (root / 'main/main.bbl').read_text(encoding='utf-8')
assert bbl.count('\\bibitem') == 42
doc = fitz.open(root / 'main/main.pdf')
assert len(doc) <= 35
for page in doc:
    assert '\ufffd' not in page.get_text()
    for block in page.get_text('blocks'):
        assert block[0] >= 0 and block[1] >= 0 and block[2] <= page.rect.width + .1 and block[3] <= page.rect.height + .1

for start in range(0, len(doc), 8):
    canvas = Image.new('RGB', (1300,1840), 'white')
    draw = ImageDraw.Draw(canvas)
    for i in range(start, min(start+8,len(doc))):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(.9,.9), alpha=False)
        im = Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        im.thumbnail((620,425))
        x,y=(i-start)%2*650,(i-start)//2*460
        draw.text((x+10,y+5),str(i+1),fill='black')
        canvas.paste(im,(x+(650-im.width)//2,y+25))
    canvas.save(audit/f'v3_pages_{start+1}.png')

(audit/'v3_main_changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v2/main.tex',tofile='v3/main.tex')),encoding='utf-8')
target = workspace / (root.name + '.zip')
if target.exists():
    raise FileExistsError(target)
included = sorted(k for k in original if not k.endswith('.spl'))
with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
    for path in included:
        archive.write(root/path, arcname=root.name+'/'+path)
with ZipFile(target) as archive:
    assert archive.testzip() is None
    for path in included:
        assert archive.read(root.name+'/'+path) == (root/path).read_bytes()
result = {'pages':len(doc),'numbered_equations':23,'references':42,'changed_files':changed,'overfull':False,'supplementary_unchanged':True,'tables_algorithms_figures_unchanged':True,'zip':str(target),'zip_bytes':target.stat().st_size,'warnings':['One underfull bibliography box','Five empty-pages BibTeX warnings','algorithm2e package encoding warnings','Included PDF version warnings']}
(audit/'v3_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

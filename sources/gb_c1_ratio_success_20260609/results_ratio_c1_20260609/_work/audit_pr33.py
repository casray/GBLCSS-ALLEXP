from pathlib import Path
from zipfile import ZipFile
import difflib
import re
import json
import fitz
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parent / 'pr33_audit'
out.mkdir(exist_ok=True)
names = ['GB-LCSS_PR_PR_editor_template_compact_v1_20260929.zip', 'GB-LCSS_PR_PR_2026PR_compact_v2_20260929.zip']
packs = []
for tag, name in zip(['old', 'new'], names):
    with ZipFile(Path('C:/Users/Admin/Downloads') / name) as z:
        root = next(n[:-len('main/main.tex')] for n in z.namelist() if n.endswith('/main/main.tex'))
        files = {n[len(root):]: z.read(n) for n in z.namelist() if n.startswith(root) and not n.endswith('/')}
    packs.append(files)
    for path, data in files.items():
        if path.endswith(('.tex', '.bib', '.bbl', '.pdf', '.log')):
            dest = out / tag / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
a, b = packs
changed = [k for k in a.keys() & b.keys() if a[k] != b[k]]
print('CHANGED', sorted(changed))
print('ADDED', sorted(b.keys() - a.keys()))
print('REMOVED', sorted(a.keys() - b.keys()))
diff = []
for k in sorted(changed):
    if k.endswith(('.tex', '.bib')):
        diff.extend(difflib.unified_diff(a[k].decode('utf-8').splitlines(True), b[k].decode('utf-8').splitlines(True), fromfile='old/'+k, tofile='new/'+k, n=1))
(out/'changes.diff').write_text(''.join(diff), encoding='utf-8')
for tag, files in zip(['old', 'new'], packs):
    tex = files['main/main.tex'].decode('utf-8')
    equations = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', tex, re.S)
    (out/(tag+'_equations.json')).write_text(json.dumps(equations, indent=2), encoding='utf-8')
    doc = fitz.open(stream=files['main/main.pdf'], filetype='pdf')
    print(tag, 'pages', len(doc), 'equations', len(equations), 'bbl refs', files['main/main.bbl'].count(b'\\bibitem'))
    (out/(tag+'_pdf.txt')).write_text('\n'.join(f'PAGE {i+1}\n'+p.get_text() for i,p in enumerate(doc)), encoding='utf-8')
    if tag == 'new':
        for start in range(0,len(doc),8):
            canvas = Image.new('RGB',(1300,1840),'white')
            draw = ImageDraw.Draw(canvas)
            for i in range(start,min(start+8,len(doc))):
                pix=doc[i].get_pixmap(matrix=fitz.Matrix(.9,.9), alpha=False)
                im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
                im.thumbnail((620,425))
                x,y=(i-start)%2*650,(i-start)//2*460
                draw.text((x+10,y+5),f'Page {i+1}',fill='black')
                canvas.paste(im,(x+(650-im.width)//2,y+25))
            canvas.save(out/f'pages_{start+1}.png')
print('EQUATIONS IDENTICAL', (out/'old_equations.json').read_bytes()==(out/'new_equations.json').read_bytes())
oldtex, newtex = [p['main/main.tex'].decode('utf-8') for p in packs]
keys = [set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', p['main/main.bbl'].decode('utf-8'))) for p in packs]
print('REMOVED REFERENCES', sorted(keys[0]-keys[1]))
print('ADDED REFERENCES', sorted(keys[1]-keys[0]))
print('PREAMBLE IDENTICAL', oldtex.split('\\begin{abstract}')[0] == newtex.split('\\begin{abstract}')[0])
for tag, files in zip(['old','new'],packs):
    tex=files['main/main.tex'].decode('utf-8')
    print(tag, 'whitespace words',len(tex.split()))
    doc=fitz.open(stream=files['main/main.pdf'],filetype='pdf')
    for term in ['Existing noise-robust', 'The deployed two-layer', 'Granular ball computing represents', 'FF-G2M retains', 'confirm that', 'ball is eligible']:
        print(tag, term, [i+1 for i,p in enumerate(doc) if term in p.get_text()])
    for i in [1,len(doc)-2]:
        spans=[s for block in doc[i].get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for s in line['spans']]
        from collections import Counter
        print(tag, 'page',i+1,'fonts',Counter((s['font'],round(s['size'],2)) for s in spans).most_common(4))
    print(tag,'outside pages', [i+1 for i,p in enumerate(doc) if any(x[0]<0 or x[1]<0 or x[2]>p.rect.width+.1 or x[3]>p.rect.height+.1 for x in p.get_text('blocks'))])
    labels=re.findall(r'\\label\{([^}]+)\}',tex+'\n'+'\n'.join(v.decode('utf-8') for k,v in files.items() if k.startswith('main/') and k.endswith('.tex') and k!='main/main.tex'))
    refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex)
    print(tag,'unresolved label keys',sorted(set(refs)-set(labels)))

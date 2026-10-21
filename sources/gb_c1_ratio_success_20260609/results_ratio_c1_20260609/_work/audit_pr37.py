from pathlib import Path
from zipfile import ZipFile
import re
import fitz
from PIL import Image, ImageDraw

ARCHIVE = Path('C:/Users/Admin/Downloads/GB-LCSS_PR_PR_editor_template_compact_v1_20260929.zip')
OUT = Path(__file__).resolve().parent / 'pr37_audit'
OUT.mkdir(exist_ok=True)
with ZipFile(ARCHIVE) as z:
    name = next(n for n in z.namelist() if n.endswith('/main/main.tex'))
    tex = z.read(name).decode('utf-8')
    doc = fitz.open(stream=z.read(name.replace('.tex', '.pdf')), filetype='pdf')
    (OUT / 'main_readonly.txt').write_text(tex, encoding='utf-8')
    for name in ('algorithm1.tex', 'algorithm2.tex', 'notation.tex'):
        key = next(n for n in z.namelist() if n.endswith('/main/' + name))
        (OUT / (name + '.txt')).write_text(z.read(key).decode('utf-8'), encoding='utf-8')

for start in range(0, len(doc), 8):
    canvas = Image.new('RGB', (1300, 1840), 'white')
    draw = ImageDraw.Draw(canvas)
    for n in range(start, min(start + 8, len(doc))):
        pix = doc[n].get_pixmap(matrix=fitz.Matrix(0.8, 0.8), alpha=False)
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        im.thumbnail((620, 425))
        x, y = (n - start) % 2 * 650, (n - start) // 2 * 460
        draw.text((x + 10, y + 5), f'Page {n + 1}', fill='black')
        canvas.paste(im, (x + (650 - im.width) // 2, y + 25))
    canvas.save(OUT / f'pages_{start+1}.png')

def prose_words(s):
    s = re.sub(r'\\begin\{(equation|align|figure|enumerate)\*?\}.*?\\end\{\1\*?\}', '', s, flags=re.S)
    s = re.sub(r'\\\(.*?\\\)|\$[^$]*\$', '', s, flags=re.S)
    s = re.sub(r'\\(?:cite|label|ref|eqref|input|includegraphics|bibliography|bibliographystyle)(?:\[[^\]]*\])?\{[^}]*\}', '', s)
    s = re.sub(r'\\[A-Za-z*]+', '', s)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", s))

heads = list(re.finditer(r'\\(section|subsection|subsubsection)\{([^}]+)\}', tex))
for n, h in enumerate(heads):
    end = heads[n+1].start() if n+1 < len(heads) else len(tex)
    print(h.group(2), prose_words(tex[h.end():end]))

print('\nLOWER PAGE ENDS (body text, excluding page numbers)')
for i, page in enumerate(doc):
    blocks = [b for b in page.get_text('blocks') if b[4].strip() != str(i+1)]
    if blocks:
        print(i+1, round(max(b[3] for b in blocks),1), 'words', len(page.get_text('words')))

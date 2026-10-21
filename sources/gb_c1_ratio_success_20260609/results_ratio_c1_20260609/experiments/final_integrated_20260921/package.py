from pathlib import Path
import difflib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_checked_20260921'
NEW = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_integrated_20260921'
OUT = Path(__file__).resolve().parent

for name in ('main.tex', 'supplementary.tex', 'highlights.tex', 'tables/table_F_1.tex'):
    old = (OLD / name).read_text(encoding='utf-8-sig').splitlines(keepends=True)
    new = (NEW / name).read_text(encoding='utf-8-sig').splitlines(keepends=True)
    delta = ''.join(difflib.unified_diff(old, new, fromfile='reviewed/' + name,
                                       tofile='integrated/' + name))
    (OUT / (name.replace('/', '_') + '.patch')).write_text(delta, encoding='utf-8')

files = [NEW / name for name in ('main.tex', 'supplementary.tex', 'highlights.tex',
                                'main.pdf', 'supplementary.pdf', 'highlights.pdf',
                                'references.bib', 'gblcss-numeric.bst', 'README.md')]
for sub in ('tables', 'algorithms', 'figures'):
    files.extend(path for path in (NEW / sub).rglob('*') if path.is_file()
                 and path.suffix.lower() in ('.tex', '.pdf', '.drawio', '.png'))
assert (NEW / 'figures/source/fig1_v20_refined.drawio') in files
assert (NEW / 'figures/source/fig1_v20_refined.png') in files
archive = NEW.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path, path.relative_to(NEW).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert z.read(name) == (NEW / name).read_bytes()
print(f'{archive}\n{len(files)} files; {archive.stat().st_size} bytes; ZIP verified')

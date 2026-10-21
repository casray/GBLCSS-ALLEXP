from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent.parent
project = root / 'GB-LCSS_Elsevier_LaTeX_Template'
output = root / 'GB-LCSS_Elsevier_Overleaf_senior_style_20260908.zip'
names = ['main.tex', 'references.tex', 'highlights.tex', 'README.md',
         'elsart-num-sort.bst', 'elsarticle-num-fullhref.bst']
names.extend(p.relative_to(project).as_posix() for p in sorted((project / 'figures').glob('*.png')))
assert len(names) == 11
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name in names:
        archive.write(project / name, name)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == set(names)
    for name in names:
        assert archive.read(name) == (project / name).read_bytes()
print('ZIP verified:', output)
print('Contents:', '\n'.join(names))
print('ZIP size:', round(output.stat().st_size / 1024 / 1024, 2), 'MiB')

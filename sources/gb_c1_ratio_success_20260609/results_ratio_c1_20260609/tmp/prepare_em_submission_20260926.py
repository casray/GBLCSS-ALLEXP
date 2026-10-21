"""Build dependency-only, flat LaTeX archives without changing manuscript content."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'GB-LCSS_submission_notation_final_20260925'
WORK = ROOT / 'tmp' / 'em_submission_build_20260926'
OUT = ROOT / 'GB-LCSS_EM_submission_20260926'
DEPENDENCY = re.compile(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}')


def prepare(entry):
    stage = WORK / entry
    stage.mkdir(parents=True, exist_ok=True)
    seen = {}

    def copy_file(relative):
        source = SOURCE / relative
        assert source.is_file(), source
        name = source.name
        if name in seen:
            assert seen[name] == relative, (name, relative, seen[name])
            return
        seen[name] = relative
        if source.suffix == '.tex':
            original = source.read_text(encoding='utf-8')
            for dependency in DEPENDENCY.findall(original):
                copy_file(dependency)
            flattened = DEPENDENCY.sub(
                lambda m: m.group(0).replace(m.group(1), Path(m.group(1)).name), original)
            (stage / name).write_text(flattened, encoding='utf-8', newline='\n')
        else:
            shutil.copy2(source, stage / name)

    copy_file(entry + '.tex')
    if entry == 'main':
        for name in ('references.bib', 'gblcss-numeric.bst', 'main.bbl'):
            copy_file(name)
    (WORK / (entry + '_manifest.json')).write_text(json.dumps(seen, indent=2), encoding='utf-8')
    print(entry, 'source files:', len(seen))


def finalize():
    import fitz
    OUT.mkdir(exist_ok=True)
    report = {}
    for entry in ('main', 'supplementary', 'highlights'):
        stage = WORK / entry
        files = json.loads((WORK / (entry + '_manifest.json')).read_text())
        log = (stage / (entry + '.log')).read_text(errors='replace')
        assert not re.search(r'(?:undefined references|Reference .+ undefined|Citation .+ undefined)', log, re.I)
        for name, original in files.items():
            if name == 'main.bbl':
                continue
            source = SOURCE / original
            if source.suffix == '.tex':
                expected = DEPENDENCY.sub(
                    lambda m: m.group(0).replace(m.group(1), Path(m.group(1)).name),
                    source.read_text(encoding='utf-8'))
                assert (stage / name).read_text(encoding='utf-8') == expected
            else:
                assert (stage / name).read_bytes() == source.read_bytes()
        pdf = stage / (entry + '.pdf')
        baseline = (ROOT / 'supplementary_table_format_final.pdf') if entry == 'supplementary' else SOURCE / (entry + '.pdf')
        new, old = fitz.open(pdf), fitz.open(baseline)
        assert len(new) == len(old), (entry, len(new), len(old))
        same_text = all(a.get_text() == b.get_text() for a, b in zip(new, old))
        same_pixels = all(
            hashlib.sha256(a.get_pixmap().samples).digest() == hashlib.sha256(b.get_pixmap().samples).digest()
            for a, b in zip(new, old))
        assert same_text and same_pixels, (entry, same_text, same_pixels)
        shutil.copy2(pdf, OUT / pdf.name)
        if entry != 'highlights':
            archive = OUT / (entry + '_source.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
                for name in sorted(files):
                    z.write(stage / name, name)
            with zipfile.ZipFile(archive) as z:
                assert all('/' not in n and '\\' not in n for n in z.namelist())
        else:
            shutil.copy2(stage / 'highlights.tex', OUT / 'highlights.tex')
        report[entry] = {'pages': len(new), 'files': len(files), 'text_unchanged': same_text, 'render_unchanged': same_pixels}
    shutil.copy2(SOURCE / 'supplementary_configurations.zip', OUT / 'supplementary_configurations.zip')
    with zipfile.ZipFile(OUT / 'supplementary_configurations.zip') as z:
        assert z.testzip() is None
    assert len(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{', (WORK / 'main/main.bbl').read_text())) == 51
    (WORK / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        for entry in ('main', 'supplementary', 'highlights'):
            prepare(entry)
    elif sys.argv[1] == 'finalize':
        finalize()

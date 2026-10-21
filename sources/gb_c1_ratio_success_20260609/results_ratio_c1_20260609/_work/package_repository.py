"""Assemble the reviewer repository without modifying source/archive files."""
from pathlib import Path
import shutil
import zipfile

workspace = Path(__file__).resolve().parents[1]
source = workspace / 'GB-LCSS_code'
destination = workspace / 'GB-LCSS_GitHub'
archive = workspace / 'GB-LCSS_GitHub.zip'
if destination.exists() or archive.exists():
    raise SystemExit('Refusing to overwrite an existing delivery.')

files = [source / name for name in [
    'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', '.gitignore',
    'requirements.txt', 'pytest.ini', 'run.py', 'scripts/prepare_representation.py',
]]
for directory, extension in [('src', '*.py'), ('tests', '*.py'), ('configs', '*.yaml'),
                             ('representations', '*.pt'), ('licenses', '*.txt')]:
    files.extend(sorted((source / directory).rglob(extension)))
for item in files:
    relative = item.relative_to(source)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item, target)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
    for item in files:
        relative = item.relative_to(source)
        handle.write(destination / relative, Path('GB-LCSS') / relative)
with zipfile.ZipFile(archive) as handle:
    assert handle.testzip() is None
    assert not any('/results/' in name or '/figures/' in name or '__pycache__' in name
                   for name in handle.namelist())
print(f'{len(files)} files; {archive.stat().st_size / 1024**2:.1f} MiB')
print(destination)
print(archive)

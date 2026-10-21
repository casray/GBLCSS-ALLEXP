import shutil
import zipfile
from replay_gate import HERE

root=HERE.parents[1]
paper=root/'GB-LCSS_Elsevier_LaTeX_final_integrated_v2_20260921'
archive=paper.with_suffix('.zip')
with zipfile.ZipFile(archive) as old:
    names=old.namelist()
backup=HERE/'submission_package_before_replay.zip'
if not backup.exists():
    shutil.copy2(archive,backup)
temporary=HERE/'submission_package_new.zip'
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as output:
    for name in names:
        source=paper/name
        assert source.is_file(),source
        output.write(source,name)
with zipfile.ZipFile(temporary) as check:
    assert check.testzip() is None
    for name in names:
        assert check.read(name)==(paper/name).read_bytes()
shutil.move(temporary,archive)
print(f'Updated {archive.name}: {len(names)} verified entries')

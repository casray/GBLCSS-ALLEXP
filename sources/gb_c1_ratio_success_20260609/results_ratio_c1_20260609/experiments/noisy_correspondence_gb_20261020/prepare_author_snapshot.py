"""Unpack the user-provided MGBCC snapshot and its split Caltech archive."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


ARCHIVE = Path(r"C:\Users\Admin\Downloads\mgbcc_main-main(1).zip")
ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "mgbcc_main"
DATA = ROOT / "data" / "smalldata"
PREFIX = "mgbcc_main-main/"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename.startswith(PREFIX + "data/"):
                continue
            if not member.filename.startswith(PREFIX):
                raise ValueError(f"Unexpected archive member: {member.filename}")
            relative = Path(member.filename.removeprefix(PREFIX))
            destination = (VENDOR / relative).resolve()
            if not destination.is_relative_to(VENDOR.resolve()):
                raise ValueError(f"Archive path escapes vendor directory: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))

        parts = [
            archive.read(PREFIX + "data/smalldata/Caltech101-20.zip.001"),
            archive.read(PREFIX + "data/smalldata/Caltech101-20.zip.002"),
        ]
        with ZipFile(io.BytesIO(b"".join(parts))) as dataset_archive:
            members = dataset_archive.namelist()
            if members != ["Caltech101-20.mat"] or dataset_archive.testzip() is not None:
                raise ValueError(f"Unexpected or corrupt split dataset: {members}")
            output = DATA / members[0]
            output.write_bytes(dataset_archive.read(members[0]))

    manifest = {
        "source_archive": str(ARCHIVE),
        "source_sha256": digest(ARCHIVE),
        "author_repository": "https://github.com/poensu/mgbcc_main",
        "archive_snapshot_revision": "unknown: download ZIP did not include commit metadata",
        "caltech_mat_path": str(output),
        "caltech_mat_sha256": digest(output),
        "caltech_mat_bytes": output.stat().st_size,
    }
    (ROOT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

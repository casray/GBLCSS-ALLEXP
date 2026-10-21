from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _module():
    return importlib.import_module("scripts.build_anonymous_submission_artifact_20260716")


def test_forbidden_scan_detects_windows_user_paths_and_local_project_name():
    module = _module()
    text = r"C:\Users\Admin\Desktop\实验8去噪5.19-\file.csv"

    matches = module.find_forbidden_matches(text, hostname="WORKSTATION")

    assert any("C:\\Users" in item for item in matches)
    assert any("Admin" in item for item in matches)
    assert any("实验8" in item for item in matches)


def test_build_artifact_uses_allowlist_and_creates_no_git_history(tmp_path: Path):
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.py").write_text("print('safe')\n", encoding="utf-8")
    (source / "ignored.txt").write_text("not copied\n", encoding="utf-8")
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("safe.py\n", encoding="utf-8")
    output = tmp_path / "artifact"

    manifest = module.build_artifact(
        source_root=source,
        output_root=output,
        allowlist_path=allowlist,
        hostname="WORKSTATION",
    )

    assert (output / "safe.py").exists()
    assert not (output / "ignored.txt").exists()
    assert not (output / ".git").exists()
    assert manifest[0]["path"] == "safe.py"
    assert (output / "ARTIFACT_MANIFEST.csv").exists()


def test_build_artifact_rejects_identity_leakage(tmp_path: Path):
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "unsafe.txt").write_text(r"C:\Users\Admin\private", encoding="utf-8")
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("unsafe.txt\n", encoding="utf-8")

    output = tmp_path / "artifact"
    with pytest.raises(ValueError, match="forbidden identity/path token"):
        module.build_artifact(
            source_root=source,
            output_root=output,
            allowlist_path=allowlist,
            hostname="WORKSTATION",
        )
    assert not output.exists()

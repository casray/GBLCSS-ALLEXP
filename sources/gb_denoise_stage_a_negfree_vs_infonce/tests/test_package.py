import gb_stage_a


def test_scaffold_package_exports_no_missing_modules():
    assert gb_stage_a.__all__ == []

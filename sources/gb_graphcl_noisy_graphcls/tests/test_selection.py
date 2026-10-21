from gb_graphcl.selection import keep_ratio_at_epoch


def test_keep_ratio_schedule():
    assert keep_ratio_at_epoch(0, 20, 20, 0.6) == 1.0
    assert keep_ratio_at_epoch(19, 20, 20, 0.6) == 1.0
    assert abs(keep_ratio_at_epoch(30, 20, 20, 0.6) - 0.8) < 1e-6
    assert keep_ratio_at_epoch(40, 20, 20, 0.6) == 0.6


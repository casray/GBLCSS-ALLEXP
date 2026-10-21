from scripts.summarize_mlp57_loo_validation import kendall_tau


def test_kendall_tau_is_one_for_same_order_and_negative_one_for_reverse():
    methods = ("a", "b", "c", "d")
    assert kendall_tau(methods, methods, methods) == 1.0
    assert kendall_tau(methods, tuple(reversed(methods)), methods) == -1.0

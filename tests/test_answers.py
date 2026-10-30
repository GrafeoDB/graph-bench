import itertools

from graph_bench.answers import canonical, canonical_map, digest, verify


def test_canonical_numbers_and_lists() -> None:
    assert canonical([(2.0, [3, 1])], ordered=True) == [[2, [1, 3]]]


def test_canonical_unordered_sorts_rows() -> None:
    assert canonical([(2, "b"), (1, "a")], ordered=False) == canonical([(1, "a"), (2, "b")], ordered=True)
    assert canonical([(2, "b"), (1, "a")], ordered=False) == [[1, "a"], [2, "b"]]


def test_bfs_unreachable_forms_agree() -> None:
    assert (
        canonical_map({1: 0, 2: None}, drop_unreachable=True)
        == canonical_map({1: 0, 2: float("inf")}, drop_unreachable=True)
        == canonical_map({1: 0, 2: 9223372036854775807}, drop_unreachable=True)
        == {1: 0}
    )


def test_verify_epsilon() -> None:
    assert verify({1: 0.15001}, {1: 0.15}, rel_eps=1e-4) is None
    message = verify({1: 0.151}, {1: 0.15}, rel_eps=1e-4)
    assert message is not None
    assert "1" in message


def test_verify_reports_first_difference() -> None:
    answer = [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3]], [[1, 2, 3]], [[1, 2, 17]]]
    reference = [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3]], [[1, 2, 3]], [[1, 2, 18]]]
    message = verify(answer, reference, rel_eps=1e-9)
    assert message is not None
    assert "binding 3" in message
    assert "row 0" in message
    assert "col 2" in message


def test_verify_bool_is_not_a_number() -> None:
    assert verify({1: True}, {1: 1}, rel_eps=1e-9) is not None
    assert verify({1: True}, {1: True}, rel_eps=1e-9) is None


def test_digest_stable_across_float_noise() -> None:
    assert digest([0.1 + 0.2]) == digest([0.3])


def test_canonical_sort_ignores_float_noise() -> None:
    noisy = canonical([(0.30000000000000004, "a"), (0.3, "b")], ordered=False)
    clean = canonical([(0.3, "a"), (0.3, "b")], ordered=False)
    assert verify(noisy, clean, rel_eps=1e-9) is None


def test_canonical_is_permutation_invariant() -> None:
    pair = [(0.3000001, "a"), (0.3, "a")]
    mixed = [(0.3, "a"), (2.5, "z"), (0.3000001, "a")]
    for rows in (pair, mixed):
        outputs = [canonical(list(p), ordered=False) for p in itertools.permutations(rows)]
        assert all(out == outputs[0] for out in outputs)


def test_verify_int_float_equal() -> None:
    assert verify([[2]], [[2.0]], rel_eps=1e-9) is None


def test_verify_zero_reference_uses_absolute_tolerance() -> None:
    assert verify(1e-10, 0.0, rel_eps=1e-9) is None
    assert verify(1.0, 0.0, rel_eps=1e-9) is not None


def test_verify_inf_equals_inf() -> None:
    assert verify(float("inf"), float("inf"), rel_eps=1e-9) is None
    assert verify(float("inf"), 1.0, rel_eps=1e-9) is not None


def test_verify_length_mismatch_names_both_lengths() -> None:
    message = verify([[1], [2], [3]], [[1], [2]], rel_eps=1e-9)
    assert message == "3 bindings != 2 bindings"


def test_verify_missing_dict_key() -> None:
    assert verify({1: 1}, {1: 1, 2: 2}, rel_eps=1e-9) == "missing keys [2]"
    unexpected = verify({1: 1, 2: 2}, {1: 1}, rel_eps=1e-9)
    assert unexpected is not None
    assert "unexpected" in unexpected


def test_digest_int_dict_keys() -> None:
    assert digest({2: 0.5, 1: 0.25}) == digest({"1": 0.25, "2": 0.5})
    assert digest({1: 0.25}) != digest({1: 0.5})

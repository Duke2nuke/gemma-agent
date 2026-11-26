from bug_10_mutates_input import with_bonus


def test_bonus_added_and_input_unchanged():
    scores = [10, 20]
    assert with_bonus(scores, 5) == [15, 25]
    assert scores == [10, 20]


def test_bonus_added_larger_list():
    scores = [1, 2, 3]
    assert with_bonus(scores, 10) == [11, 12, 13]
    assert scores == [1, 2, 3]


def test_repeated_calls_same_input():
    scores = [4]
    first = with_bonus(scores, 1)
    second = with_bonus(scores, 1)
    assert first == [5]
    assert second == [5]
    assert scores == [4]

from bug_05_wrong_sort import top_scores


def test_top_three_highest_first():
    assert top_scores([5, 1, 9, 3]) == [9, 5, 3]


def test_fewer_than_three():
    assert top_scores([2, 7]) == [7, 2]


def test_already_sorted():
    assert top_scores([10, 8, 6, 4]) == [10, 8, 6]

from bug_01_off_by_one import total


def test_total_all_numbers():
    assert total([1, 2, 3]) == 6


def test_total_single():
    assert total([7]) == 7


def test_total_empty():
    assert total([]) == 0

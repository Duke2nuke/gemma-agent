from bug_09_slice_off_by_one import page


def test_first_page():
    assert page([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 3) == [1, 2, 3]


def test_second_page():
    assert page([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 3) == [4, 5, 6]


def test_second_page_size_two():
    assert page([1, 2, 3, 4, 5], 2, 2) == [3, 4]

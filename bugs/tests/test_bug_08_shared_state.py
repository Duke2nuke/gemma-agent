from bug_08_shared_state import evens


def test_evens_first_call():
    assert evens([1, 2, 3, 4]) == [2, 4]
    # A second identical call must give the same answer, not accumulate.
    assert evens([1, 2, 3, 4]) == [2, 4]


def test_evens_second_call_is_fresh():
    evens([1, 2, 3, 4])
    assert evens([5, 6]) == [6]


def test_evens_empty_after_calls():
    evens([2, 4])
    assert evens([]) == []

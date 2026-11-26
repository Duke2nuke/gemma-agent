from bug_02_wrong_comparison import is_adult


def test_eighteen_is_adult():
    assert is_adult(18) is True


def test_twenty_is_adult():
    assert is_adult(20) is True


def test_seventeen_is_not_adult():
    assert is_adult(17) is False

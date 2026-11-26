from bug_04_missing_key import get_price


def test_known_item():
    assert get_price("apple") == 10


def test_unknown_item_is_zero():
    assert get_price("mango") == 0


def test_another_unknown_item():
    assert get_price("kiwi") == 0

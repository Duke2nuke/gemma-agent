from bug_03_mutable_default import add_item


def test_first_call():
    assert add_item("apple") == ["apple"]


def test_carts_are_independent():
    a = add_item("apple")
    b = add_item("banana")
    assert a == ["apple"]
    assert b == ["banana"]


def test_passed_cart_is_used():
    assert add_item("pear", ["plum"]) == ["plum", "pear"]

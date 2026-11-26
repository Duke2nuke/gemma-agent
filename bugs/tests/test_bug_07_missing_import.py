from bug_07_missing_import import get_setting


def test_string_value():
    assert get_setting('{"theme": "dark"}', "theme") == "dark"


def test_int_value():
    assert get_setting('{"retries": 3}', "retries") == 3


def test_second_key():
    assert get_setting('{"a": 1, "b": 2}', "b") == 2

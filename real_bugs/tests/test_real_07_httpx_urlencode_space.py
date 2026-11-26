import real_07_httpx_urlencode_space as mod


def test_spaces_encoded_as_percent_20():
    assert mod.build_query_string([("q", "hello world")]) == "q=hello%20world"


def test_slash_left_alone():
    assert mod.build_query_string([("a", "x/y")]) == "a=x/y"


def test_plain_pairs():
    assert mod.build_query_string([("a", "1"), ("b", "2")]) == "a=1&b=2"

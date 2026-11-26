import real_02_requests_super_len as mod


def test_ascii_str():
    assert mod.super_len("hello") == 5


def test_multibyte_str_counts_bytes():
    assert mod.super_len("café") == 5


def test_emoji_counts_bytes():
    assert mod.super_len("😀") == 4

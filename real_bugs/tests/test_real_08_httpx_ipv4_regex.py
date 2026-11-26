import pytest

import real_08_httpx_ipv4_regex as mod


def test_ipv4_like_hostname_not_rejected():
    assert mod.encode_host("023b76x43144") == "023b76x43144"


def test_valid_ipv4():
    assert mod.encode_host("1.2.3.4") == "1.2.3.4"


def test_invalid_ipv4_and_plain_hostname():
    with pytest.raises(mod.InvalidURL):
        mod.encode_host("256.1.1.1")
    assert mod.encode_host("example.com") == "example.com"

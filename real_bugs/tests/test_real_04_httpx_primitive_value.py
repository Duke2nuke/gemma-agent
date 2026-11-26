import pytest

import real_04_httpx_primitive_value as mod


def test_valid_primitives():
    assert mod.primitive_value_to_str("a") == "a"
    assert mod.primitive_value_to_str(1) == "1"
    assert mod.primitive_value_to_str(1.5) == "1.5"
    assert mod.primitive_value_to_str(True) == "true"
    assert mod.primitive_value_to_str(False) == "false"
    assert mod.primitive_value_to_str(None) == ""


def test_bytes_raises():
    with pytest.raises(
        TypeError, match="Expected str, int, float, bool, or None. Got 'bytes'."
    ):
        mod.primitive_value_to_str(b"bytes")


def test_list_and_dict_raise():
    with pytest.raises(TypeError):
        mod.primitive_value_to_str([1, 2])
    with pytest.raises(TypeError):
        mod.primitive_value_to_str({"a": 1})

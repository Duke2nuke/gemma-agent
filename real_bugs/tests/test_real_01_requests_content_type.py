import real_01_requests_content_type as mod


def test_malformed_param_ignored():
    content_type, params = mod._parse_content_type_header(
        "multipart/form-data; boundary=something; no_equals"
    )
    assert content_type == "multipart/form-data"
    assert params == {"boundary": "something"}


def test_normal_params():
    content_type, params = mod._parse_content_type_header("text/html; charset=utf-8")
    assert content_type == "text/html"
    assert params == {"charset": "utf-8"}


def test_multiple_malformed_params():
    content_type, params = mod._parse_content_type_header(
        "application/json; foo; bar=baz; qux"
    )
    assert content_type == "application/json"
    assert params == {"bar": "baz"}

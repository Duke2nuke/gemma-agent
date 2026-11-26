import real_09_httpx_gen_delims as mod

PATH = "/!$&'()*+,;= abc ABC 123 :/[]@"


def test_path_percent_encoding():
    path, query, fragment = mod.encode_url_parts(PATH, None, None)
    assert path == "/!$&'()*+,;=%20abc%20ABC%20123%20:/[]@"


def test_query_percent_encoding():
    path, query, fragment = mod.encode_url_parts("/", PATH + "?", None)
    assert query == "/!$&'()*+,;=%20abc%20ABC%20123%20:/[]@?"


def test_fragment_percent_encoding():
    path, query, fragment = mod.encode_url_parts("/", None, PATH + "?#")
    assert fragment == "/!$&'()*+,;=%20abc%20ABC%20123%20:/[]@?#"

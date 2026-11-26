import real_11_rich_markup_parse as mod


def test_nested_bracket_not_one_tag():
    result = list(mod._parse("[foo bar [baz] qux]"))
    assert result == [
        (9, "[foo bar ", None),
        (9, None, mod.Tag("baz", None)),
        (14, " qux]", None),
    ]


def test_simple_tags():
    result = list(mod._parse("[bold]hi[/bold]"))
    assert result == [
        (0, None, mod.Tag("bold", None)),
        (8, "hi", None),
        (8, None, mod.Tag("/bold", None)),
    ]


def test_no_tags():
    assert list(mod._parse("no tags")) == [(0, "no tags", None)]

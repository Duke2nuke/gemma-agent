import real_10_rich_escape as mod


def test_at_tag_escaped():
    assert mod.escape("[@click]Hello[/@click]") == "\\[@click]Hello\\[/@click]"


def test_bold_tag_escaped():
    assert mod.escape("[bold]hi[/bold]") == "\\[bold]hi\\[/bold]"


def test_plain_text_unchanged():
    assert mod.escape("plain text") == "plain text"

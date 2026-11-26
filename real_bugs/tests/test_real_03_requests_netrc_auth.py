import real_03_requests_netrc_auth as mod


def test_empty_default_credentials_ignored(tmp_path, monkeypatch):
    netrc_path = tmp_path / ".netrc"
    monkeypatch.setenv("NETRC", str(netrc_path))
    netrc_path.write_text("machine example.com login user password pass\ndefault\n")
    assert mod.get_netrc_auth("http://httpbin.org/") is None


def test_matching_machine_returns_auth(tmp_path, monkeypatch):
    netrc_path = tmp_path / ".netrc"
    monkeypatch.setenv("NETRC", str(netrc_path))
    netrc_path.write_text("machine example.com login user password pass\n")
    assert mod.get_netrc_auth("http://example.com/") == ("user", "pass")


def test_no_netrc_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("NETRC", str(tmp_path / ".netrc"))
    assert mod.get_netrc_auth("http://example.com/") is None

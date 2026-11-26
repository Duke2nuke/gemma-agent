import real_06_httpx_no_proxy as mod

PROXY_VARS = [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
]


def _set_env(monkeypatch, env):
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_no_proxy_ipv4(monkeypatch):
    _set_env(monkeypatch, {"no_proxy": "127.0.0.1"})
    assert mod.get_environment_proxies() == {"all://127.0.0.1": None}


def test_no_proxy_ipv6_and_localhost(monkeypatch):
    _set_env(monkeypatch, {"no_proxy": "::1"})
    assert mod.get_environment_proxies() == {"all://[::1]": None}
    _set_env(monkeypatch, {"no_proxy": "localhost"})
    assert mod.get_environment_proxies() == {"all://localhost": None}


def test_no_proxy_domain_suffix(monkeypatch):
    _set_env(monkeypatch, {"no_proxy": "github.com"})
    assert mod.get_environment_proxies() == {"all://*github.com": None}

import bug_06_type_error as mod


def test_default_config():
    mod.CONFIG = {"price": "120", "discount_pct": 10}
    assert mod.final_price() == 108.0


def test_quarter_off():
    mod.CONFIG = {"price": "200", "discount_pct": 25}
    assert mod.final_price() == 150.0


def test_no_discount():
    mod.CONFIG = {"price": "50", "discount_pct": 0}
    assert mod.final_price() == 50.0

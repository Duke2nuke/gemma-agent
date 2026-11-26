"""Apply a discount to a price loaded from a config dict.

Prices in the config are stored as strings, so they must be converted to
numbers before doing arithmetic.
"""

CONFIG = {"price": "120", "discount_pct": 10}


def final_price():
    return CONFIG["price"] * (1 - CONFIG["discount_pct"] / 100)

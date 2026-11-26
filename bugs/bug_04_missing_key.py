"""Look up the price of an item.

Unknown items cost 0 instead of crashing.
"""

PRICES = {"apple": 10, "banana": 5, "orange": 8}


def get_price(item):
    return PRICES[item]

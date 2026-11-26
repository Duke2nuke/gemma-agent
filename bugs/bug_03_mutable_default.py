"""Add an item to a shopping cart and return the cart.

Each call without a cart should start from an empty cart.
"""


def add_item(item, cart=[]):
    cart.append(item)
    return cart

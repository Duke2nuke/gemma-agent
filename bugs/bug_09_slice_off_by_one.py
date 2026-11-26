"""Return page `page` (1-based) of `size` items from a list."""


def page(items, page, size):
    start = (page - 1) * size
    return items[start:start + size - 1]

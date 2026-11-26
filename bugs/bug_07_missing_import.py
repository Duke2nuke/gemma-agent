"""Load a JSON string and return the value for a key."""


def get_setting(payload, key):
    data = json.loads(payload)
    return data[key]

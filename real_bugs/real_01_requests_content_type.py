"""Parse a Content-Type header into (content_type, params).

Extracted from requests, src/requests/utils.py, as it was before
PR #7309 ("Fix malformed value parsing for Content-Type",
Nate Prewitt, 2026-03-30).

The bug: a parameter with no ``=`` in it (a malformed value such as
``; no_equals``) is stored in the params dict with the value ``True``
instead of being ignored.
"""


def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of
        parameters
    """

    tokens = header.split(";")
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value
    return content_type, params_dict

"""Small, testable checks used by scripts/verify_data.py."""


def box_overshoot(bbox, width, height):
    """How many pixels a COCO box [x, y, w, h] extends outside a width x height image.

    Returns 0.0 if the box lies fully inside the image.
    """
    x, y, w, h = bbox
    return max(0.0, -x, -y, (x + w) - width, (y + h) - height)


def size_check(expected_wh, actual_wh):
    """Compare (width, height) pairs: 'match', 'swap' (width and height reversed) or 'other'."""
    expected_wh, actual_wh = tuple(expected_wh), tuple(actual_wh)
    if expected_wh == actual_wh:
        return "match"
    if expected_wh == actual_wh[::-1]:
        return "swap"
    return "other"

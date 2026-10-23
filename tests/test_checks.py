from vdv.data.checks import box_overshoot, size_check


def test_box_inside_image_has_no_overshoot():
    assert box_overshoot([10, 10, 50, 50], width=100, height=100) == 0.0


def test_box_past_right_edge():
    assert box_overshoot([60, 10, 50, 20], width=100, height=100) == 10.0


def test_box_with_negative_corner():
    assert box_overshoot([-2.5, 5, 10, 10], width=100, height=100) == 2.5


def test_size_check_cases():
    assert size_check((1000, 750), (1000, 750)) == "match"
    assert size_check((1000, 750), (750, 1000)) == "swap"
    assert size_check((1000, 750), (1000, 667)) == "other"

from vdv.data.image_info import angle_bin, colour_bin


def test_colour_bins():
    for c in ["white", " Silver ", "champagne"]:
        assert colour_bin(c) == "light"
    for c in ["black", "gray"]:
        assert colour_bin(c) == "dark"
    for c in ["red", "blue", "mixture"]:
        assert colour_bin(c) == "coloured"
    assert colour_bin(float("nan")) == "unknown"
    assert colour_bin(None) == "unknown"


def test_angle_bins():
    for a in ["front", "side", "rear"]:
        assert angle_bin(a) == a
    assert angle_bin("inside") == "other-unknown"
    assert angle_bin(float("nan")) == "other-unknown"

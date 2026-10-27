import numpy as np

from vdv.data.augment import BRIGHTNESS_RANGE, adjust_brightness, augment_train, hflip


def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (40, 90, 3), dtype=np.uint8)


def test_flip_mirrors_and_twice_is_identity():
    img = image()
    assert np.array_equal(hflip(img)[:, 0], img[:, -1])
    assert np.array_equal(hflip(hflip(img)), img)


def test_brightness_identity_scaling_and_clipping():
    img = np.array([[[100, 200, 250]]], dtype=np.uint8)
    assert np.array_equal(adjust_brightness(img, 1.0), img)
    assert adjust_brightness(img, 0.8).tolist() == [[[80, 160, 200]]]
    assert adjust_brightness(img, 1.2).tolist() == [[[120, 240, 255]]]      # 300 clipped to 255
    assert adjust_brightness(img, 1.2).dtype == np.uint8


def test_same_seed_same_augmentation():
    a, pa = augment_train(image(), np.random.default_rng(7))
    b, pb = augment_train(image(), np.random.default_rng(7))
    assert pa == pb and np.array_equal(a, b)


def test_factor_in_range_and_flip_rate_about_half():
    rng = np.random.default_rng(1)
    params = [augment_train(image(), rng)[1] for _ in range(2000)]
    assert all(BRIGHTNESS_RANGE[0] <= p["brightness"] <= BRIGHTNESS_RANGE[1] for p in params)
    assert 0.45 < np.mean([p["flipped"] for p in params]) < 0.55


def test_shape_and_dtype_preserved():
    out, _ = augment_train(image(), np.random.default_rng(3))
    assert out.shape == image().shape and out.dtype == np.uint8

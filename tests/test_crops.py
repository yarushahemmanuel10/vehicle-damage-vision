import json

import numpy as np
import pytest

from vdv.data.crops import (MAX_ASPECT, MIN_SIDE, check_crop_cache, crop_code_version,
                            crop_window, hash_files, pad_to_square, resize_long_side)

W, H = 1000, 750


def size(win):
    x0, y0, x1, y1 = win
    return x1 - x0, y1 - y0


def contains(win, box):
    x0, y0, x1, y1 = win
    x, y, w, h = box
    return x0 <= x and y0 <= y and x + w <= x1 and y + h <= y1


def test_tiny_box_gets_exactly_min_window():
    box = [500, 400, 8, 8]
    win, flags = crop_window(box, W, H)
    assert size(win) == (MIN_SIDE, MIN_SIDE)
    assert flags["min_window_applied"] and contains(win, box)


def test_tiny_box_in_corner_shifts_inside_image():
    box = [0, 0, 5, 5]
    win, _ = crop_window(box, W, H)
    assert win == (0, 0, MIN_SIDE, MIN_SIDE)


def test_tiny_box_at_far_corner():
    box = [W - 6, H - 6, 6, 6]
    win, _ = crop_window(box, W, H)
    assert win == (W - MIN_SIDE, H - MIN_SIDE, W, H)


def test_elongated_box_is_capped_at_max_aspect():
    box = [100, 300, 458, 18]                         # like scratch #5727
    win, flags = crop_window(box, W, H)
    w, h = size(win)
    assert flags["aspect_capped"] and not flags["aspect_cap_incomplete"]
    assert max(w, h) <= MAX_ASPECT * min(w, h)
    assert contains(win, box)


def test_typical_box_only_gets_margin():
    box = [200, 200, 150, 100]                        # aspect 1.5, large enough
    win, flags = crop_window(box, W, H)
    assert win == (185, 190, 365, 310)                # 10% of 150 = 15, 10% of 100 = 10
    assert not any(flags.values())


def test_aspect_cap_incomplete_when_image_too_small():
    box = [0, 0, 990, 10]
    win, flags = crop_window(box, 1000, 300)
    assert size(win)[1] == 300 and flags["aspect_cap_incomplete"]


def test_window_always_inside_image_and_contains_box():
    rng = np.random.default_rng(0)
    for _ in range(500):
        w, h = rng.uniform(1, 400, size=2)
        x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
        win, _ = crop_window([x, y, w, h], W, H)
        x0, y0, x1, y1 = win
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        assert contains(win, [x, y, w, h])


def test_resize_long_side_keeps_aspect():
    out = resize_long_side(np.zeros((64, 192, 3), np.uint8))
    assert out.shape[:2] == (75, 224)                 # 64 * 224/192 = 74.67 -> 75


def test_pad_to_square_centres_content():
    out = pad_to_square(np.full((75, 224, 3), 255, np.uint8))
    assert out.shape == (224, 224, 3)
    assert out[0, 0, 0] == 128 and out[112, 112, 0] == 255


def test_crop_cache_version_check(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"version": crop_code_version(), "n_crops": 0}))
    check_crop_cache(tmp_path)                        # matching version, complete: accepted
    (tmp_path / "meta.json").write_text(json.dumps({"version": "v0-old", "n_crops": 0}))
    with pytest.raises(RuntimeError):
        check_crop_cache(tmp_path)                    # mismatched version: refused


def test_crop_cache_refuses_missing_pngs(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"version": crop_code_version(), "n_crops": 2}))
    (tmp_path / "train" / "dent").mkdir(parents=True)
    (tmp_path / "train" / "dent" / "1.png").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        check_crop_cache(tmp_path)                    # valid meta, one PNG missing: refused
    (tmp_path / "train" / "dent" / "2.png").write_bytes(b"x")
    assert check_crop_cache(tmp_path)["n_crops"] == 2 # complete: accepted


def test_version_hash_tracks_rules_and_build_script(tmp_path):
    rules, build = tmp_path / "crops.py", tmp_path / "build_crops.py"
    rules.write_text("rules v1")
    without_build = hash_files([rules, build])        # build script missing: skipped, no error
    build.write_text("build v1")
    with_build = hash_files([rules, build])
    assert with_build != without_build
    build.write_text("build v2")
    after_build_edit = hash_files([rules, build])
    assert after_build_edit != with_build             # editing the build script changes the version
    rules.write_text("rules v2")
    assert hash_files([rules, build]) != after_build_edit   # editing the rules changes it too

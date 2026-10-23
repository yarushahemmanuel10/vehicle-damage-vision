from vdv.data.boxes import intersection_area, size_bucket


def test_size_bucket_boundaries():
    assert size_bucket(31 * 31) == "small"
    assert size_bucket(32 * 32) == "medium"          # boundary belongs to the upper bucket
    assert size_bucket(96 * 96 - 1) == "medium"
    assert size_bucket(96 * 96) == "large"
    assert list(size_bucket([10, 2000, 10000])) == ["small", "medium", "large"]


def test_intersection_area():
    assert intersection_area([0, 0, 10, 10], [5, 5, 10, 10]) == 25    # partial overlap
    assert intersection_area([0, 0, 10, 10], [10, 0, 5, 5]) == 0      # edges touch only
    assert intersection_area([0, 0, 10, 10], [2, 2, 3, 3]) == 9       # one inside the other

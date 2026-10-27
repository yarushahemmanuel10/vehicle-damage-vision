import numpy as np

from vdv.review import reliability, select_threshold, threshold_table


def test_routing_counts_and_boundary_is_auto():
    y = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])   # conf .9 .8 .6 .5
    t = threshold_table(y, probs, [0.5, 0.8])
    assert t[0]["n_review"] == 0 and t[0]["auto_accuracy"] == 0.75        # conf exactly 0.5 is auto
    assert t[1]["n_review"] == 2 and t[1]["share_review"] == 0.5 and t[1]["auto_accuracy"] == 1.0


def test_select_lowest_cutoff_meeting_target_else_highest():
    table = [{"cutoff": 0.5, "auto_accuracy": 0.90}, {"cutoff": 0.7, "auto_accuracy": 0.97},
             {"cutoff": 0.9, "auto_accuracy": 0.99}]
    assert select_threshold(table, 0.97) == (0.7, True)
    assert select_threshold(table, 0.995) == (0.9, False)


def test_ece_is_zero_when_calibrated():
    probs = np.tile([0.8, 0.2], (10, 1))
    y = np.array([0] * 8 + [1] * 2)                                      # 80% confident, 80% right
    ece, _ = reliability(y, probs)
    assert abs(ece) < 1e-9


def test_ece_when_overconfident():
    probs = np.tile([1.0, 0.0], (4, 1))
    y = np.array([0, 0, 1, 1])                                           # 100% confident, 50% right
    ece, _ = reliability(y, probs)
    assert abs(ece - 0.5) < 1e-9

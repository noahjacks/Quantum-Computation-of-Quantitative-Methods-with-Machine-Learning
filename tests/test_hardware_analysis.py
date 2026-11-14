"""
CHECK: the statistics used to read the real-hardware result behave sensibly.

* the paired difference between two sets of predictions sees a real gap,
  ignores a fake one, and is tight when the two sets are nearly the same;
* the "can the noise tell early from late?" test finds a planted shift and
  finds nothing in pure noise.
"""
import numpy as np

from qon.hardware_analysis import decodability_test, paired_differences


def test_paired_difference_sees_a_real_gap_and_ignores_a_fake_one():
    rng = np.random.default_rng(0)
    n = 600
    day = np.repeat(np.arange(30), 20)
    y = rng.integers(0, 2, n)
    good = y + 0.8 * rng.normal(size=n)          # carries real information about y
    twin = good + 0.01 * rng.normal(size=n)      # the same predictions again, barely changed
    junk = rng.normal(size=n)                    # pure noise
    d = paired_differences(y, day, {"D": good, "twin": twin, "junk": junk},
                           n_boot=500).set_index("comparison")
    assert not d.loc["D minus twin", "significant"]
    assert abs(d.loc["D minus twin", "difference"]) < 0.01
    assert d.loc["D minus twin", "ci_high"] - d.loc["D minus twin", "ci_low"] < 0.05   # paired, so tight
    assert d.loc["D minus junk", "significant"] and d.loc["D minus junk", "ci_low"] > 0.1


def test_decodability_test_finds_a_planted_shift_and_nothing_in_noise():
    rng = np.random.default_rng(1)
    labels = rng.permutation(np.repeat([0, 1], 100))
    noise = rng.normal(size=(200, 30))
    shifted = noise + 0.5 * labels[:, None]      # every column moves a little for the "late" group
    hit = decodability_test(shifted, labels, n_shuffles=40)
    miss = decodability_test(noise, labels, n_shuffles=40)
    assert hit["auc"] > 0.9 and hit["p_value"] < 0.05
    assert miss["auc"] < 0.65 and miss["p_value"] > 0.05

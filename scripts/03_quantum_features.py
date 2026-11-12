"""
STEP 3 - Run every quote request through the 109-qubit circuit (simulated)
and build the competing feature sets A, B, C, E, F and the drift variants.

    .venv/bin/python scripts/03_quantum_features.py   (about 2 minutes)

Two circuits are simulated: the main one (alpha = 1.0, HSBC's "shorter"
setting) and a small-angle one (alpha = 0.1, the scale of HSBC's "longer"
circuit) that is used as a robustness check.
"""
import _path  # noqa: F401
from qon.feature_sets import build_feature_sets

if __name__ == "__main__":
    build_feature_sets(alpha=1.0)
    build_feature_sets(alpha=0.1)

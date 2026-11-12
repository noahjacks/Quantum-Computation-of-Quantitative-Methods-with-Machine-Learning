"""
config.py  --  EVERY SETTING OF THE EXPERIMENT LIVES HERE
=========================================================

Nothing in the rest of the code contains a "magic number".  If you want to
know why a number has the value it has, or you want to change the experiment,
this is the only file you need to read.

The settings are frozen BEFORE any real quantum hardware is used (this is what
"pre-registration" means: we write the plan down first so nobody can accuse us
of tweaking it afterwards to get a nicer result).  PREREGISTRATION.md in the
project root is generated from this file.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# 0.  Where things live on disk
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # the project folder
DATA_RAW = ROOT / "data" / "raw"                     # downloads, untouched
DATA_PROCESSED = ROOT / "data" / "processed"         # things we compute
RESULTS = ROOT / "results"                           # tables and figures
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

OSBAP_PARQUET = DATA_RAW / "osbap_stage1" / "stage1_osbap_0k_volume_2025.parquet"
OSBAP_URL = ("https://openbondassetpricing.com/wp-content/uploads/2025/12/"
             "stage1_osbap_0k_volume_2025.zip")
FEATURE_LIST_CSV = ROOT / "docs" / "Features_Qiskit.csv"   # the team's 216 features

# --------------------------------------------------------------------------
# 1.  Dates
# --------------------------------------------------------------------------
# HSBC's "active window" was 24 July 2024 - 29 October 2024 (69 trading days).
# We use exactly the same calendar window, on US instead of European bonds.
WINDOW_START = "2024-07-24"
WINDOW_END = "2024-10-29"

# The 40 trading days BEFORE the window are a "calibration period".  Anything
# that has to be tuned once (how to scale the features, how aggressive the
# simulated dealer is, kernel bandwidths) is tuned ONLY on these days, so the
# 69 test days are never used to make a choice.
CALIBRATION_DAYS = 40

# How much history we load so that 3-year look-back features can be computed.
PANEL_START = "2021-01-01"

# --------------------------------------------------------------------------
# 2.  Building the events ("replayed quote requests")
# --------------------------------------------------------------------------
N_EVENTS = 16_000            # HSBC sampled 16,000 events from their window
N_CALIBRATION_EVENTS = 8_000  # events drawn from the calibration period
EVENT_SEED = 20240724        # random seed for which events are sampled

# A "month" and a "week" measured in trading days.
MONTH = 21
WEEK = 5
YEAR = 252

# Eligibility: we only keep liquid, seasoned bonds so that every feature can
# be computed from real numbers (no gaps to fill in).
MAX_STALE_DAYS_REF = 5        # last trade must be within 5 trading days
ASOF_LIMIT_MONTHLY = 10       # a month-end price may be at most 10 days old
MIN_DAILY_RETURNS_63D = 15    # >= 15 day-to-day returns in the last quarter
MIN_TWO_SIDED_DAYS_63D = 5    # >= 5 days with both a customer buy and a sell
N_MONTHS_HISTORY = 36         # 3 years of monthly returns are required

# The simulated dealer's quoting rule (see src/qon/target.py):
#     ask quote = reference price * (1 + KAPPA * half_spread)
#     bid quote = reference price * (1 - KAPPA * half_spread)
# KAPPA is calibrated ONCE on the calibration period so that roughly 37% of
# quotes are filled (the fill rate HSBC reports), then frozen.
TARGET_FILL_RATE = 0.37
MIN_HALF_SPREAD = 0.0002      # 2 basis points floor on the dealer's markup

# --------------------------------------------------------------------------
# 3.  Scaling the features before they go into the circuit (HSBC Sec. 3.2)
# --------------------------------------------------------------------------
#     z = (x - mean) / std          (mean/std from the calibration period)
#     angle = 2*pi * tanh(z / 3)
WINSOR_QUANTILES = (0.005, 0.995)   # clip extreme values before the z-score
TANH_DIVISOR = 3.0

# --------------------------------------------------------------------------
# 4.  The quantum circuit (HSBC's "projected quantum feature map")
# --------------------------------------------------------------------------
N_QUBITS = 109               # same as HSBC
TROTTER_STEPS = 2            # M: same circuit depth as HSBC's "shorter" circuit
ALPHA = 1.0                  # overall rotation scale (HSBC "shorter": 1.0)
INIT_STATE_SEED = 1          # seed of the random single-qubit start states
FEATURE_ORDER_SEED = 7       # seed of the (arbitrary) feature -> qubit-pair map

# --------------------------------------------------------------------------
# 5.  Noise model for the noisy simulator (feature set C)
# --------------------------------------------------------------------------
# The simulated chip copies a real one.  docs/chip_snapshot_fake_fez.json
# holds IBM's published calibration snapshot of ibm_fez (a Heron-class chip
# HSBC also ran on, and one that is on IBM's free Open Plan), restricted to
# the 109-qubit path our circuit would use: the error of every coupler and
# every qubit.  It is written by src/qon/hardware.py::save_snapshot.
SHOTS = 4096                 # measurements per circuit and basis (HSBC: 4096)
NOISE_SEED = 11
CHIP_SNAPSHOT = ROOT / "docs" / "chip_snapshot_fake_fez.json"

# Drift model for the "is it the clock?" check (see src/qon/noise.py).
DRIFT_BATCH_SIZE = 250       # circuits per job: calibration is constant inside
DRIFT_REL_SIGMA = 0.20       # stationary relative std-dev of each error rate
DRIFT_AR1 = 0.9              # job-to-job persistence of the drift
DRIFT_SEED = 13

# --------------------------------------------------------------------------
# 6.  Backtest (how we train and test, HSBC Sec. 3.3)
# --------------------------------------------------------------------------
# HSBC trained on the most recent 500-2000 events.  On calibration data our
# models needed more history to find any signal (see notebook 2), so the
# main run uses 4000; 1000 is run as an "HSBC-sized" robustness check.
TRAIN_SIZE = 4000            # most recent events used for training
TRAIN_SIZE_SMALL = 1000      # robustness check with an HSBC-sized window
BLINDING_DAYS = (0, 1, 4)    # full trading days hidden between train and test
RETUNE_EVERY = 10            # re-run the hyper-parameter search every 10 test days
CV_FOLDS = 3
MODEL_SEED = 0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 99
N_CORES = 3                  # processor cores used by the long runs (gentle on a laptop)

# --------------------------------------------------------------------------
# 7.  Extension: kernels
# --------------------------------------------------------------------------
# Kernel methods compare every training event with every other one, so the
# cost grows with the SQUARE of the number of events.  Part 2 therefore uses
# an evenly thinned sample of the events (the same number from every day).
KERNEL_EVENTS_PER_DAY = 50
KERNEL_TRAIN_SIZE = 800
KERNEL_BLINDING_DAYS = (0, 1, 4)
FQK_ALPHAS = (1.0, 0.3, 0.1, 0.03)   # bandwidths screened on calibration data
KERNEL_SEED = 5


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, TABLES):
        p.mkdir(parents=True, exist_ok=True)

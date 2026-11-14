# Is it quantum, or is it noise?

A controlled public test of HSBC's claim that a quantum computer improved
bond-trading predictions by 34% (Ciceri et al., arXiv:2509.17715), built for the
Qiskit Fall Fest.

**New here? Open [notebooks/00_START_HERE.ipynb](notebooks/00_START_HERE.ipynb).**
It explains everything in plain English and assumes no knowledge of bonds or
quantum computing. Unfamiliar words are in [GLOSSARY.md](GLOSSARY.md).

## The question

HSBC and IBM reported that passing trading data through a circuit on a quantum
chip, and then feeding the measurements to ordinary machine-learning models,
improved the prediction of whether a bond quote gets filled. The catch, from
their own paper: the gain appeared only on real, noisy hardware, not when the
same circuit was simulated perfectly.

* **Part 1.** When a quantum computer seems to help, is it the quantum part,
  the noise, or the clock (hardware drift leaking the date)?
* **Part 2 (extension).** Keep the classical inputs and change the learning
  algorithm instead: can a quantum kernel beat classical machine learning at
  estimating p(fill)?

## Results at a glance

The score is AUC: 0.5 is guessing, 1.0 is perfect. Intervals are 95%.

### Part 1: same models, different inputs (16,000 quote requests, 68 test days)

| Set | What the models are given | Mean daily AUC | Difference to A |
|---|---|---|---|
| A | classical inputs | 0.546 | |
| B | circuit on a perfect quantum computer | 0.529 | −0.017 (−0.023 to −0.010) |
| C | circuit on a noisy simulated chip | 0.528 | −0.018 (−0.023 to −0.012) |
| D | circuit on real IBM hardware | **not run yet** | |
| E | B with classical noise matched to C | 0.525 | −0.021 (−0.027 to −0.014) |
| F | random classical features | 0.538 | −0.008 (−0.013 to −0.002) |

* **Is it quantum?** No. The quantum features score about 3% below the
  classical inputs (HSBC report 34% above, for hardware).
* **Is it noise?** No. A noisy simulated chip scores the same as a perfect one
  and the same as a classical imitation of its noise. With small rotation
  angles (HSBC's best setting) simulated noise makes things worse.
* **Is it the clock?** The clock is real: when events are sent to a drifting
  chip in date order, the drift alone reveals early versus late with 99%
  accuracy. It did not improve predictions here (difference 0.000, −0.005 to
  +0.004), because our classical inputs already contain the date.
* The same picture holds for every model, for blinding of 0, 1 and 4 days, and
  with a smaller training window.

### Part 2: same inputs, different learning algorithms (3,400 test requests)

| Method | Family | Planned test (SVM) | Follow-up (kernel ridge) |
|---|---|---|---|
| Best classical model (random forest) | classical | 0.544 | 0.544 |
| Linear kernel | classical | 0.492 | 0.546 |
| RBF kernel | classical | 0.487 | 0.541 |
| Fidelity quantum kernel | quantum | 0.496 | 0.537 |
| Fidelity quantum kernel, 4096 shots | quantum | 0.509 | 0.534 |
| Projected quantum kernel | quantum | 0.491 | 0.530 |
| Projected quantum kernel, noisy chip | quantum | 0.490 | 0.531 |

* **Planned test.** No quantum kernel advantage. Every SVM scored at guessing
  level, classical kernels included, which shows the SVM itself fails on such
  a weak signal.
* **Follow-up, added after seeing that result and labelled as such.** With
  kernel ridge regression the kernels work, and the quantum kernels score
  about the same as, or slightly below, the classical ones. None beats the
  best classical method.
* The fidelity kernel is exponentially concentrated at the circuit's natural
  setting (typical similarity 10⁻¹⁶) and only works with a small bandwidth.

### What these results do and do not show

They show that everything that can be simulated agrees with HSBC's own
simulations: no gain from the quantum feature map and no gain from simulated
noise. They do not show that HSBC's hardware result is wrong. That needs the
hardware run, which is ready but not yet done, and our data differs from
theirs (see Limitations).

## What is real and what is simulated

| Ingredient | Status |
|---|---|
| Bonds, prices, yields, spreads, trades | Real: US corporate bonds, public TRACE data, HSBC's dates (24 Jul to 29 Oct 2024) |
| Market conditions | Real: FRED, Kenneth French, Federal Reserve |
| The 98 inputs | Real, computed from past information only |
| Quote requests | Derived from real trades: each real customer trade is one request |
| The dealer whose quote is filled or not | Simulated with a simple rule |
| Quantum computer, perfect | Simulated exactly (checked against Qiskit) |
| Quantum computer, noisy | Simulated with a real chip's published error rates |
| Real quantum computer | Not used yet |

## How to run it

Requirements: Python 3.14 (the version this was built and tested with), about
6 GB of disk, and an internet connection for the first step.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then the steps, in order. Times are for a recent laptop.

| Step | Command | Time | What it does |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/01_download_data.py` | 5 min | downloads the free data (1.8 GB) |
| 2 | `.venv/bin/python scripts/02_build_dataset.py` | 1 min | builds the 98 inputs and the fill label |
| 3 | `.venv/bin/python scripts/03_quantum_features.py` | 3 min | simulates the circuit, builds the feature sets |
| 4 | `.venv/bin/python scripts/04_run_part1.py` | about 2 h on 3 cores | Part 1 backtests |
| 5 | `.venv/bin/python scripts/05_run_part2_kernels.py` | about 45 min on 3 cores | Part 2 kernels |
| 6 | `.venv/bin/python scripts/06_hardware_run.py --mode local` | 2 min | dry run of the hardware pipeline |
| 7 | `.venv/bin/python scripts/07_analyse_hardware.py --rehearsal` | 3 min | rehearsal of the hardware analysis |

Steps 4 and 5 use 3 processor cores at low priority by default so that a
laptop stays usable. Add `--cores 8` to go faster (about 40 and 20 minutes);
the machine will run hot. Both steps save their progress and can be stopped and
restarted.

Check everything:

```bash
.venv/bin/python -m pytest tests -q
```

Read the notebooks:

```bash
.venv/bin/jupyter lab notebooks
```

Running on real IBM hardware is described in
[notebooks/05_hardware_run_and_limitations.ipynb](notebooks/05_hardware_run_and_limitations.ipynb).
The free IBM plan gives 10 minutes of quantum time per 28 days. The planned
run is estimated at 6 to 8 minutes, and the script will not spend more than a
set budget (8 minutes by default): it checks what each job cost, stops before
the budget is crossed, saves after every job and continues later from where
it stopped.

## Where things are

| Path | Contents |
|---|---|
| `notebooks/` | six explained notebooks, numbered in reading order |
| `src/qon/` | the code; every file starts with a plain-English explanation |
| `scripts/` | the steps above |
| `tests/` | automated checks |
| `results/` | saved predictions, tables and figures |
| `docs/FEATURE_COVERAGE.md` | what happened to each of the 216 features |
| `docs/Features_Qiskit.csv` | the team's feature list |
| `PREREGISTRATION.md` | the plan, frozen before the hardware run |
| `GLOSSARY.md` | every term in plain English |

Code map (`src/qon/`):

| File | Role |
|---|---|
| `config.py` | every setting of the experiment |
| `download.py`, `panel.py`, `market.py` | get and organise the data |
| `features.py`, `feature_catalog.py` | the 97 features, and why the other 119 are missing |
| `target.py`, `dataset.py` | the replayed-quote fill label and the event table |
| `preprocessing.py` | scaling inputs into gate angles |
| `circuit.py` | the 109-qubit circuit |
| `simulator.py` | exact noiseless and noisy simulation |
| `noise.py` | readout error, shot noise, drift, the classical noise twin, random features |
| `feature_sets.py` | builds sets A, B, C, E, F and the drift variants |
| `models.py`, `backtest.py`, `stats.py` | models, rolling backtest, scores and intervals |
| `kernels.py`, `kernel_backtest.py` | Part 2 |
| `hardware.py`, `hardware_analysis.py` | set D on IBM hardware |
| `analysis.py` | tables and charts |

## How we know it is right

The test-suite (53 tests) checks, among other things, that

* the simulator matches three independent Qiskit simulators, including the
  matrix-product-state simulator HSBC used, on the full 109 qubits;
* no input can see the future: destroying all data from the test day onwards
  changes nothing;
* sampled inputs match a slow, independent recomputation from the raw file;
* every fill label can be re-derived from raw prices;
* training data is always strictly older than the test day;
* the backtest finds a planted signal and finds nothing in noise;
* the fidelity kernel matches Qiskit state overlaps;
* the hardware pipeline assigns each measurement to the right qubit;
* the hardware run cannot overspend: it stops before its budget of quantum
  time, keeps everything already measured, and continues later without
  repeating a circuit (tested with pretend prices on a simulated chip; IBM's
  own usage reporting can only be tested with an account).

## Design in brief

* **Data.** One event is one replayed quote request: a (bond, day, direction)
  on which customers really traded. 16,000 events in the test window, the same
  number each day, plus 8,000 in the 40 days before for calibration.
* **Fill label.** A simulated dealer quotes at the bond's last traded price,
  updated for the market's move since then, plus or minus a multiple of the
  bond's usual half bid-ask spread. The quote is filled if it beats the price
  the customer really got. The multiple is set once so that 37% of quotes
  fill, as HSBC report.
* **Inputs.** 97 of the team's 216 features (all that are free) plus the side
  of the request.
* **Circuit.** 109 qubits in a line, Heisenberg gates on neighbouring pairs,
  two Trotter steps (216 gates), X, Y and Z measured on every qubit (327
  outputs). Same shape as HSBC's "shorter" circuit.
* **Scoring.** Rolling backtest: train on the 4,000 most recent earlier
  events, test on the next day; blinding of 0, 1 and 4 days; logistic
  regression, random forest, XGBoost and a neural network; mean daily AUC with
  paired bootstrap intervals.

## Differences from the HSBC paper

| | HSBC | This project |
|---|---|---|
| Data | 1.7 million private RFQs, European bonds | public US trades replayed as quote requests |
| Fill label | real | stand-in (simulated dealer) |
| Inputs | 216 | 98 |
| Time resolution | microseconds | daily |
| Circuit | 109 qubits, 216 or 432 gates | 109 qubits, 216 gates |
| Hardware | ibm_torino and others | not run yet |
| Feature scaling | statistics of the whole sample | statistics from before the test window only |
| Hyper-parameter grids | large | compact |
| Controls | none of the below | classical noise twin, random features, drift check, random run order, sentinels, frozen plan |

## Limitations

1. The fill label is a stand-in; real quote requests are private.
2. Requests where the customer walked away are invisible in public data.
3. The signal is weak (classical AUC 0.55, against about 0.63 for HSBC), so
   small differences are hard to see. As probabilities, no method here beats
   simply quoting the average fill rate.
4. 97 of 216 features: no ratings, company accounts, share prices, CDS or
   holdings.
5. No real hardware yet. The simulated noise is a simplified model.
6. One circuit and one seed (deliberately), one 68-day window.
7. Part 2's planned SVM test was uninformative; the informative kernel-ridge
   comparison was added afterwards.

The full list is in notebook 5.

## Data sources and credit

* Open Source Bond Asset Pricing, Stage 1 daily panel
  (openbondassetpricing.com), built from FINRA TRACE by Dickerson, Robotti and
  Rossetti. Please cite them if you use the data.
* FRED, Federal Reserve Bank of St. Louis.
* Kenneth R. French data library.
* Gilchrist and Zakrajšek excess bond premium, Federal Reserve Board.
* Calibration snapshot of ibm_fez shipped with qiskit-ibm-runtime.

FRED serves only the most recent three years of the ICE BofA credit-spread
indices. The test window is inside that range today; keep the cached files in
`data/raw/fred/` so that the project stays reproducible later.

Method references: Ciceri et al., arXiv:2509.17715 (the HSBC paper); Huang et
al., Nature Communications 12, 2631 (2021) (projected quantum kernels and the
geometric-difference test).

This is a student research project. Nothing in it is investment advice.

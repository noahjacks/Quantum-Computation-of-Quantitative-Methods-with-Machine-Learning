# Pre-registration: the plan, written down before the hardware run

"Pre-registration" means fixing the experiment before seeing its result, so
that nobody (including us) can adjust it afterwards to get a nicer answer.
This is the central criticism of the HSBC paper: two circuits with different
settings and seeds were reported, and nobody outside knows how many were tried.

**What is frozen here:** every setting of the experiment (they all live in
`src/qon/config.py`), the hardware run protocol, and how the hardware result
will be analysed and interpreted.

**Honest status.** All one-off choices were made using only the calibration
period (the 40 trading days before the test window). No setting was changed
after looking at a test-window score. The simulated feature sets (A, B, C, E, F)
have been computed and scored; the hardware set (D) has not been run. The
hardware plan below is therefore a true pre-registration; the simulated part is
a record of what was done.

## 1. Question

When a quantum computer appears to help a model predict bond fills, is it
(a) the quantum part, (b) the noise, or (c) the clock (time-ordered hardware
drift leaking the date into the features)?

## 2. Data (fixed)

| Item | Value |
|---|---|
| Source | Open Source Bond Asset Pricing Stage 1 daily panel (US TRACE), FRED, Kenneth French library, Federal Reserve EBP |
| Test window | 24 Jul 2024 to 29 Oct 2024 (HSBC's dates), 68 US bond-market trading days |
| Calibration period | the 40 trading days before the window |
| Events | 16,000 in the window (235 or 236 per day), 8,000 in the calibration period; random seed 20240724 |
| Inputs | 97 of the team's 216 features (all that are free) + the side of the request |
| Target | "filled" under the replayed-quote rule in `src/qon/target.py`; dealer aggressiveness kappa set once on the calibration period to a 37% fill rate |
| Eligibility | bond traded within 5 trading days; 36 months of monthly prices; at least 15 day-to-day returns and 5 two-sided days in the last 63 days |

## 3. Circuit (fixed)

| Item | Value | HSBC |
|---|---|---|
| Qubits | 109 in a line | 109 |
| Gate | Heisenberg, exp(-i phi (XX+YY+ZZ)), on neighbouring pairs | same |
| Trotter steps | M = 2 (even pairs then odd pairs, twice): 216 gates | "shorter" circuit: 216 gates |
| Angle | phi = alpha x 2 pi tanh(z/3) / (4M) | same formula |
| alpha | 1.0 (main); 0.1 as a robustness check | 1.0 ("shorter"), 0.1 ("longer") |
| Start state | fixed random single-qubit states, seed 1 | same idea, seed 1 |
| Feature to gate map | shuffled feature list dealt out cyclically, seed 7 | "arbitrarily chosen" |
| Outputs | X, Y, Z average of every qubit: 327 numbers | 327 |

Differences from HSBC that we cannot avoid: we have 98 inputs, not 216, so each
input drives 2 or 3 gates; and we do not simulate their "longer" circuit
(4 Trotter steps), only its small rotation scale.

## 4. Feature sets (fixed)

| Set | Definition |
|---|---|
| A | the 98 inputs, clipped and z-scored with calibration-period statistics |
| B | circuit, exact noiseless simulation |
| C | circuit, simulated chip: depolarizing error after every gate with rates from IBM's calibration snapshot of ibm_fez, readout error with TREX, 4096 shots, jobs of 250 circuits in random order |
| D | circuit on real IBM hardware (section 6) |
| E | B with a per-output shrink factor and Gaussian jitter fitted so that it matches D (C until D exists) |
| F | 327 random Fourier features of A (median-heuristic length scale) |
| Drift check | C with gate errors that wander from job to job (20% log-normal, persistence 0.9), once with events in date order and once in random order |

## 5. Scoring (fixed)

* Models: logistic regression, random forest, gradient boosting (XGBoost),
  neural network. Settings chosen by 3-fold cross-validation inside the
  training window, repeated every 10 test days.
* Rolling backtest: for each test day, train on the 4,000 most recent earlier
  events; blinding of 0, 1 and 4 trading days.
* **Primary outcome:** mean daily AUC averaged over the four models at
  blinding 0, as a paired difference to set A, with a 95% bootstrap interval
  over test days. A difference counts only if that interval excludes zero.
* Everything that is run is reported, including all models, all blindings and
  all robustness checks.

Ways our scoring is smaller than HSBC's: compact hyper-parameter grids, and
re-tuning every 10 days rather than for every event. The neural network uses
early stopping.

## 6. Hardware run protocol (frozen before any quantum time is spent)

| Item | Value |
|---|---|
| Script | `scripts/06_hardware_run.py --mode run` |
| Chip | a 156-qubit Heron r2 on the Open Plan (ibm_fez by default) |
| Qubits | the 109-qubit path found by `find_chain`, avoiding couplers with error above 5% where possible |
| Events | the same number from every test day, chosen with seed 2026 |
| **Run order** | **random (seed 2026), never by date** |
| **Sentinels** | one fixed circuit repeated after every 20 events |
| Shots | 1024 per circuit and basis unless the timing job shows the budget allows more |
| Error handling | Pauli twirling and TREX, as in the HSBC paper; no other mitigation |
| Before the run | a timing job of 5 circuits to size the run to the budget |

## 7. How the hardware result will be read (frozen)

All comparisons use the same events and the same models
(`scripts/07_analyse_hardware.py`).

| Observation | Conclusion |
|---|---|
| D is not better than A | No hardware advantage on this data. |
| D is better than A, and E matches D | **It was the noise.** The effect is reproduced by classical shrink-and-jitter. |
| D is better than A, and D beats E | **Something unexplained.** Hardware adds something the classical copy lacks. Next step: repeat on another day and chip before believing it. |
| Sentinels scatter more than shot noise, or the noise reveals the run position | The chip drifted. Because the order was random this cannot leak the date, but it confirms the risk for any study that runs time-ordered data. |

With a budget of a few hundred hardware events only large effects are
detectable. That is sufficient to test a claim of HSBC's size (AUC rising from
about 0.63 to 0.75 or more).

## 8. Part 2 (the extension)

Inputs are always set A. Methods: the four models above, SVM with linear and
RBF kernels, SVM with the projected quantum kernel (noiseless and noisy), SVM
with the fidelity quantum kernel (exact, and with 4096-shot estimates). The
fidelity kernel's bandwidth alpha is chosen once on calibration data from
{1.0, 0.3, 0.1, 0.03}. Thinned sample of 50 events per day, training window
800, blinding 0, 1 and 4. Same primary outcome, as a paired difference to the
best classical method.

## 9. Additions made after results were seen

One analysis was added after the planned Part 2 results were known. It is
reported separately and labelled as a follow-up wherever it appears.

* **What happened.** In the planned Part 2 comparison every SVM scored at
  guessing level, including the SVMs with ordinary classical kernels, while
  logistic regression on the same inputs did not. That shows the SVM itself
  fails on this weak, unbalanced signal, so the planned test could not
  distinguish a good kernel from a bad one.
* **What was added.** The same kernels, data and backtest, with kernel ridge
  regression in place of the SVM (`run_part2_followup`). The smoothing
  strength is chosen by the same in-window cross-validation.
* **What was not changed.** No setting of Part 1, of the data, of the circuit
  or of the planned Part 2 run was altered after seeing any result. The
  hardware protocol in section 6 is unchanged and still unrun.

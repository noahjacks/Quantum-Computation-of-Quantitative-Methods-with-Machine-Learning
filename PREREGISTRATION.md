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
have been computed and scored. When this plan was frozen the hardware set (D)
had not been run, so the hardware plan below is a true pre-registration; the
simulated part is a record of what was done. The hardware run took place on
14 Nov 2026 and is recorded in section 10.

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
| Spending limit | at most `--budget-seconds` of quantum time per session (default 480 of the free plan's 600); the run stops before crossing it, saves after every job, and can be continued later with the same settings and the same qubits (added 14 Nov 2026, see section 9) |

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
  hardware protocol in section 6 was unchanged and still unrun at that point.

### Change to how the hardware run is executed (14 Nov 2026, before any hardware run)

A teammate asked whether the run could use up the team's free quantum time.
The hardware script was changed so that it cannot:

* **What was added.** A spending limit in quantum seconds, saving of the
  results after every job, and the ability to continue a stopped run.
* **What it does not change.** What is measured: the events, their random
  order, the sentinels, the shots and the error handling are as in section 6.
* **Consequences that will be reported with the result.**
  * If the run stops early, fewer events than planned are measured. Because
    the order is random they are a random subset of the planned events, and
    the analysis uses them as they are. The number actually run is reported.
  * If a stopped run is continued on a later day, the chip may have been
    recalibrated in between. This is reported, and the sentinel circuits show
    how much the chip moved.
* **Cost estimate recorded before the run.** By IBM's published formula, 400
  events at 1024 shots need about 6 to 8 minutes of quantum time.

## 10. The hardware run as executed (14 Nov 2026)

| Item | What was done |
|---|---|
| Chip | `ibm_fez` (156-qubit Heron r2), IBM Open Plan |
| Qubits | a 109-qubit path found by `find_chain` on the live chip; median two-qubit gate error on the path 0.24%, worst 1.1% |
| Events | 400 test-window events, 5 or 6 from every test day, chosen with seed 2026 |
| Run order | random (seed 2026); correlation of run order with event date: 0.007 |
| Sentinels | 21 repeats of one fixed circuit, one after every 20 events |
| Shots | 1024 per circuit and basis |
| Error handling | Pauli twirling and TREX |
| Timing job | 7 circuits, 9 quantum seconds |
| Run | 9 jobs, 421 circuits, 397 quantum seconds, 17:30 to 17:38 UTC, in one session; the spending limit (480) was not reached and nothing was stopped or continued |
| Deviations from section 6 | none |

**Reading the result by section 7.** Two of its rows apply.

* *D is not better than A.* Pooled AUC of the model average: A 0.453, B 0.488,
  C 0.520, D 0.503, E 0.484, F 0.524, on 196 test events. D minus A is +0.050
  (95% interval −0.042 to +0.130). Conclusion as written beforehand: no
  hardware advantage on this data.
* *The noise reveals the run position.* A classifier tells first-half from
  second-half circuits from the noise alone with AUC 0.773. Conclusion as
  written beforehand: the chip drifted; because the order was random this
  cannot leak the date, but it confirms the risk for studies that run
  time-ordered data. The sentinels alone show little drift (their scatter is
  1.08 times what shot noise explains).

**Added after the result was seen.** These do not change the reading above;
they make it precise. All are computed by `src/qon/hardware_analysis.py`.

* *Paired differences* between D and every other set, with their own 95%
  intervals (the planned table gave only each set's interval): D minus B
  +0.015 (−0.023 to +0.054), D minus C −0.017 (−0.060 to +0.024), D minus E
  +0.020 (−0.075 to +0.108), D minus F −0.020 (−0.116 to +0.074). None
  excludes zero.
* *A shuffled-label check* of the run-position result: in 300 repeats with
  shuffled labels the AUC never reached 0.773 (97.5% of repeats were below
  0.574), so p is about 0.003.
* *A warning check.* What is left of the hardware values after the best
  straight-line fit to the perfect values "reveals" early versus late event
  dates (AUC 0.784) although the run order was random. This is not a clock:
  the chip's distortion depends on the inputs, and the inputs contain the
  date (the perfect-chip values alone give 0.984, and the simulated rehearsal
  shows the same effect). It is recorded so that it is not mistaken for one.

**What the run cannot show** is listed in notebook 5, section 6: it was too
small to detect a subtle effect, and no date-ordered batch was sent to the
chip.

## 11. A second session and the combined analysis (added 15 Nov 2026, not pre-registered)

A teammate repeated the hardware run from a second free account. Nothing in
sections 1 to 8 was changed for it, but running it, and combining it with the
first session, were decided after the first result had been seen.

| Item | What was done |
|---|---|
| Chip and qubits | `ibm_fez`, the same 109-qubit path as section 10 |
| Events | 600 requested with the same script and seed; 523 measured (13 of them also in the first session) |
| Run order | random; correlation of run order with event date 0.03 |
| Shots, error handling | as section 10 |
| Run | 11 jobs, 550 circuits, 517 quantum seconds, 23:43 to 23:58 UTC on 14 Nov 2026; stopped by the spending limit (570) |

**Second session alone** (`scripts/07_analyse_hardware.py` on its file): A 0.511, D 0.480, D minus A −0.031 (−0.130 to +0.069); run
position from the noise alone, AUC 0.701. D minus E was +0.043 (+0.005 to
+0.084); it is not seen in the first session or in the combined data and is
treated as chance.

**Both sessions combined** (`scripts/08_combine_runs.py`; 910 different events,
449 test events; repeated events keep their first-session values): A 0.525,
B 0.525, C 0.513, D 0.535, E 0.535, F 0.512. D minus A +0.010 (−0.043 to
+0.064); D minus E 0.000 (−0.036 to +0.038). Reading by section 7 is
unchanged: no hardware advantage on this data.

**Between the sessions.** The 13 shared events agree (correlation 0.990). A
classifier tells which session an event ran in from the noise alone with AUC
0.999 (97.5% of shuffled repeats below 0.547); session is unrelated to event
date (correlation 0.000). The sentinel circuit differs between the sessions,
so sentinel statistics are not pooled.

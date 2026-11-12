"""
CHECK: the noise model, the scoring and the backtest loop behave as described.
"""
import numpy as np
import pandas as pd

from qon import config as C
from qon.backtest import check_no_overlap, run_one, training_rows
from qon.circuit import make_spec
from qon.noise import (NoiseParams, batch_of_events, drifting_gate_errors,
                       fit_classical_surrogate, make_noise_params, measure,
                       random_fourier_features)
from qon.preprocessing import AngleScaler
from qon.stats import bootstrap_mean, daily_auc, summarize


# ----------------------------------------------------------------- noise
def test_shot_noise_has_the_textbook_mean_and_spread():
    """An average of S shots of a +-1 outcome with mean m has std sqrt((1-m^2)/S)."""
    m = np.full((200_000, 3), 0.4)
    params = NoiseParams(lam_slot=np.zeros(1), readout=np.zeros(1), shots=1024)
    est = measure(m, params, np.zeros(len(m), dtype=int), seed=1, trex=False)
    assert abs(est.mean() - 0.4) < 2e-4
    assert abs(est.std() - np.sqrt((1 - 0.4 ** 2) / 1024)) < 2e-4


def test_readout_error_shrinks_and_trex_undoes_it():
    m = np.full((100_000, 3), 0.5)
    params = NoiseParams(lam_slot=np.zeros(1), readout=np.array([0.05]), shots=4096)
    raw = measure(m, params, np.zeros(len(m), dtype=int), seed=2, trex=False)
    assert abs(raw.mean() - 0.5 * (1 - 2 * 0.05)) < 3e-4          # shrunk by 10%
    batch = np.arange(len(m)) // 250                               # many jobs
    fixed = measure(m, params, batch, seed=2, trex=True)
    assert abs(fixed.mean() - 0.5) < 1e-3                          # bias removed
    assert fixed.std() > raw.std()                                 # at the price of more noise


def test_run_order_and_drift():
    order = np.array([3, 0, 2, 1, 4, 5])                           # event 3 runs first ...
    assert list(batch_of_events(order, batch_size=2)) == [0, 1, 1, 0, 2, 2]
    spec = make_spec(98)
    params = make_noise_params(spec)
    assert 0.005 < np.median(params.lam_slot) < 0.05
    batch = np.arange(24000) // C.DRIFT_BATCH_SIZE
    lam = drifting_gate_errors(params, batch, spec.n_qubits - 1)
    assert lam.shape == (24000, spec.n_slots)
    assert np.all(lam[:250] == lam[0])                             # constant inside a job
    ratio = lam / params.lam_slot
    assert abs(ratio.mean() - 1.0) < 0.05                          # drift averages out
    assert 0.1 < np.log(ratio[::250]).std() < 0.3                  # ~20% wobble
    # neighbouring jobs are similar, distant jobs are not (a slow drift)
    z = np.log(ratio[::250, :108])
    near = np.corrcoef(z[:-1].ravel(), z[1:].ravel())[0, 1]
    far = np.corrcoef(z[:-40].ravel(), z[40:].ravel())[0, 1]
    assert near > 0.8 and abs(far) < 0.2
    # a perfect chip has no errors at all
    perfect = make_noise_params(spec, scale=0.0)
    assert perfect.lam_slot.max() == 0 and perfect.readout.max() == 0


def test_classical_surrogate_recovers_shrink_and_jitter():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(50_000, 5))
    target = 0.1 + 0.8 * clean + 0.05 * rng.normal(size=clean.shape)
    s = fit_classical_surrogate(clean, target)
    assert np.allclose(s.slope, 0.8, atol=0.01) and np.allclose(s.sigma, 0.05, atol=0.002)
    fake = s.apply(clean, seed=1)
    assert np.allclose(fake.std(axis=0), target.std(axis=0), rtol=0.02)
    assert not np.allclose(fake, target)                           # same statistics, different noise


def test_random_features_and_scaler():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(500, 20))
    F = random_fourier_features(z, z, 327, seed=0)
    assert F.shape == (500, 327) and np.abs(F).max() <= 1.0
    X = rng.lognormal(size=(2000, 3))
    sc = AngleScaler.fit(X)
    ang = sc.angles(np.vstack([X, [[1e9, -1e9, 0.0]]]))            # absurd outliers
    assert np.abs(ang).max() < 2 * np.pi                           # always a valid angle
    assert abs(sc.zscore(X).mean()) < 1e-9


# ----------------------------------------------------------------- scoring
def test_auc_and_bootstrap():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([.1, .2, .8, .9, .6, .4, .3, .7])
    day = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    auc = daily_auc(y, p, day)
    assert auc[0] == 1.0 and auc[1] == 0.75
    m, lo, hi = bootstrap_mean(np.r_[np.zeros(50), np.ones(50)])
    assert m == 0.5 and 0.38 < lo < 0.45 and 0.55 < hi < 0.62
    rng = np.random.default_rng(0)
    base = 0.55 + 0.03 * rng.normal(size=68)
    table = pd.DataFrame({"A": base, "same": base + 0.001 * rng.normal(size=68), "better": base + 0.05})
    s = summarize(table, "A")
    assert s.loc["better", "significant"] and not s.loc["same", "significant"]
    assert np.isclose(s.loc["better", "diff_vs_A"], 0.05)


# ----------------------------------------------------------------- backtest
def test_training_window_respects_blinding():
    day = np.repeat(np.arange(20), 10)                             # 10 events per day
    for blinding in (0, 1, 4):
        tr = training_rows(day, test_day=12, blinding=blinding, train_size=35)
        assert len(tr) == 35 and day[tr].max() == 12 - 1 - blinding
        check_no_overlap(day, tr, 12, blinding)
    assert len(training_rows(day, 2, 0, 1000)) == 20               # cannot use more than exists


def test_backtest_finds_a_planted_signal_and_nothing_in_noise():
    rng = np.random.default_rng(0)
    n_days, per_day = 30, 120
    day = np.repeat(np.arange(n_days), per_day)
    X = rng.normal(size=(len(day), 6))
    y_signal = (X[:, 0] + 0.5 * rng.normal(size=len(day)) > 0).astype(int)
    y_noise = rng.integers(0, 2, size=len(day))
    test_days = np.arange(20, 30)
    for y, lo, hi in ((y_signal, 0.9, 1.0), (y_noise, 0.42, 0.58)):
        res = run_one(X, y, day, test_days, blinding=1, train_size=1500, models=("LR", "RF"))
        assert set(res["day"]) == set(test_days)
        for model, g in res.groupby("model"):
            auc = daily_auc(y[g["row"].to_numpy()], g["p"].to_numpy(), g["day"].to_numpy()).mean()
            assert lo < auc < hi, (model, auc)

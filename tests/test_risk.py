import numpy as np

from thrift_llm import risk


def test_binomial_pvalue_matches_exact_tail():
    losses = np.array([1] * 3 + [0] * 97, dtype=float)
    # P(Binom(100, 0.1) <= 3) = 0.0078 (exact)
    assert abs(risk.binomial_pvalue(losses, 0.1) - 0.007836) < 1e-4


def test_upper_bound_covers_truth_at_nominal_rate():
    rng = np.random.default_rng(0)
    p, delta, miss = 0.05, 0.1, 0
    for _ in range(400):
        losses = (rng.random(300) < p).astype(float)
        miss += risk.upper_bound(losses, delta) < p
    assert miss / 400 <= delta + 0.03


def test_learn_then_test_controls_risk():
    """Settings with true risk 0.00 .. 0.10; epsilon = 0.05. Chosen setting must have true
    risk <= epsilon in >= 1 - delta of repetitions."""
    rng = np.random.default_rng(1)
    true = np.linspace(0, 0.10, 21)
    bad = 0
    for _ in range(300):
        draws = {i: (rng.random(500) < r).astype(float) for i, r in enumerate(true)}
        cert = risk.learn_then_test(list(range(len(true))), lambda i, d=draws: d[i], 0.05, 0.1)
        if cert.index >= 0 and true[int(cert.setting)] > 0.05:
            bad += 1
    assert bad / 300 <= 0.1


def test_hb_pvalue_continuous_losses():
    rng = np.random.default_rng(2)
    low = rng.random(2000) * 0.04  # mean 0.02
    assert risk.pvalue(low, 0.05) < 0.01
    assert risk.pvalue(low, 0.015) > 0.5

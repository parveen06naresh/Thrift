import numpy as np

from thrift_llm.router import Router


def _data(n, rng):
    texts, qc, qs = [], [], []
    for _ in range(n):
        hard = rng.random() < 0.5
        if hard:
            texts.append(f"Prove that the integral of x^{rng.integers(2, 9)} dx from 0 to 1 equals the stated value step by step")
            qc.append(float(rng.random() < 0.3))
        else:
            texts.append(f"What colour is a {rng.choice(['banana', 'lemon', 'sky', 'leaf'])}?")
            qc.append(float(rng.random() < 0.95))
        qs.append(float(rng.random() < 0.97))
    return texts, np.array(qc), np.array(qs)


def test_router_learns_and_certifies(tmp_path):
    rng = np.random.default_rng(0)
    T, qc, qs = _data(1200, rng)
    r = Router().fit(T[:600], qc[:600], qs[:600], epochs=20)
    cert = r.calibrate(T[600:900], qc[600:900], qs[600:900], epsilon=0.05, delta=0.1)
    s = r.score(T[900:])
    cheap = s >= r.threshold
    easy = np.array(["colour" in t for t in T[900:]])
    assert cheap[easy].mean() > 0.5 and cheap[~easy].mean() < 0.2
    gap = np.clip(qs[900:] - qc[900:], 0, 1)
    assert (cheap * gap).mean() <= 0.07
    assert cert.epsilon == 0.05
    p = tmp_path / "r.json"
    r.save(str(p))
    r2 = Router.load(str(p))
    assert abs(r2.score_one(T[0]) - r.score_one(T[0])) < 1e-2
    assert r2.threshold == r.threshold


def test_uncalibrated_router_never_routes_cheap():
    r = Router()
    assert r.route("anything")[0] == "strong"


def test_bundled_router_loads():
    from thrift_llm.gateway import bundled_router

    r = bundled_router()
    assert r is not None and r.certificate is not None and 0 < r.threshold <= 1.01

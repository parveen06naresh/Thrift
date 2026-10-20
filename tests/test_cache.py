import time

import numpy as np

from thrift_llm.cache import SemanticCache, calibrate_threshold

SCOPE = {"tier": "cheap", "prefix": []}
RESP = {"choices": [{"message": {"content": "Paris"}}]}


def test_exact_hit_ignores_case_and_spacing():
    c = SemanticCache()
    c.put(SCOPE, "What is the capital of France?", RESP)
    h = c.get(SCOPE, "  what is the capital of   france ")
    assert h is not None and h.kind == "exact"


def test_semantic_hit_and_key_term_guard():
    c = SemanticCache(threshold=0.8)
    c.put(SCOPE, "How do I convert 5 km to miles?", RESP)
    assert c.get(SCOPE, "How can I convert 5 km into miles?") is not None
    assert c.get(SCOPE, "How do I convert 50 km to miles?") is None  # different number
    c.put(SCOPE, "Is it safe to eat raw eggs?", RESP)
    assert c.get(SCOPE, "Is it not safe to eat raw eggs?") is None  # negation flips meaning


def test_scope_isolation():
    c = SemanticCache()
    c.put(SCOPE, "Summarise the document", RESP)
    assert c.get({"tier": "cheap", "prefix": [{"role": "system", "content": "other doc"}]}, "Summarise the document") is None
    assert c.get({**SCOPE, "temperature": 0.9}, "Summarise the document") is None


def test_ttl_and_lru():
    c = SemanticCache(ttl_s=0.05, max_entries=2)
    c.put(SCOPE, "q one about apples", RESP)
    c.put(SCOPE, "q two about bananas", RESP)
    c.put(SCOPE, "q three about cherries", RESP)
    assert len(c) == 2 and c.get(SCOPE, "q one about apples") is None
    time.sleep(0.06)
    assert c.get(SCOPE, "q three about cherries") is None


def test_sqlite_persistence(tmp_path):
    p = str(tmp_path / "c.sqlite")
    c = SemanticCache(path=p)
    c.put(SCOPE, "persist me please", RESP)
    c2 = SemanticCache(path=p)
    assert c2.get(SCOPE, "persist me please") is not None


def test_calibrate_threshold_is_conservative():
    rng = np.random.default_rng(0)
    sim = rng.random(5000)
    correct = rng.random(5000) < sim  # higher similarity -> more likely correct
    cert = calibrate_threshold(sim, correct, alpha=0.02, delta=0.1)
    wrong = ((sim >= cert.setting) & ~correct).mean()
    assert wrong <= 0.025

from thrift_llm import tokens


def test_count_messages_positive_and_monotonic():
    a = tokens.count_messages([{"role": "user", "content": "hello"}])
    b = tokens.count_messages([{"role": "user", "content": "hello there, how are you doing today?"}])
    assert 0 < a < b


def test_heuristic_close_to_tiktoken_when_available():
    text = "The quick brown fox jumps over the lazy dog. " * 50 + "Antidisestablishmentarianism and photosynthesis."
    real = tokens.count(text)
    tokens._encoding.cache_clear()
    orig = tokens._encoding
    tokens._encoding = lambda name: None
    try:
        approx = tokens.count(text)
    finally:
        tokens._encoding = orig
    assert abs(approx - real) / real < 0.25

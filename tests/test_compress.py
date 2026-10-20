from thrift_llm.compress import Compressor, normalize_text


def test_normalize_is_lossless_for_words_and_keeps_code():
    text = "Hello    world​!\n\n\n\n\nNext   para\t\twith tabs.\n```python\nx  =  1\n\n\n\ny = 2\n```\n----------------------"
    out = normalize_text(text)
    assert out.split() != []
    assert "Hello world!" in out and "Next para with tabs." in out
    assert "```python\nx  =  1\n\n\n\ny = 2\n```" in out  # code untouched
    assert "\n\n\n" not in out.split("```")[0]
    assert "---" in out and "-----" not in out


def test_dedup_drops_repeated_paragraphs():
    chunk = "The Eiffel Tower is 330 metres tall and was completed in 1889 for the World's Fair."
    msgs = [{"role": "user", "content": f"{chunk}\n\n{chunk}\n\nHow tall is it?"}]
    res = Compressor(select=False).compress(msgs)
    assert res.messages[0]["content"].count("Eiffel") == 1
    assert res.stages["dedup"] > 0


def test_select_keeps_question_and_relevant_evidence():
    filler = " ".join(f"Paragraph {i} talks about gardening tips and soil acidity for tomatoes." for i in range(60))
    evidence = "The Treaty of Westphalia was signed in 1648 in Osnabrueck and Muenster."
    doc = f"{filler}\n\n{evidence}\n\n{filler}\n\nWhen was the Treaty of Westphalia signed?"
    res = Compressor(min_tokens=100, keep_ratio=0.2).compress([{"role": "system", "content": "Be brief."}, {"role": "user", "content": doc}])
    out = res.messages[1]["content"]
    assert "1648" in out
    assert out.rstrip().endswith("When was the Treaty of Westphalia signed?")
    assert res.messages[0]["content"] == "Be brief."
    assert res.tokens_after < 0.5 * res.tokens_before


def test_short_messages_untouched():
    msgs = [{"role": "user", "content": "What is 2 + 2?"}]
    res = Compressor().compress(msgs)
    assert res.messages == msgs
    assert res.saved == 0


def test_multimodal_parts_are_handled():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Hi   there"}, {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    res = Compressor().compress(msgs)
    assert res.messages[0]["content"][0]["text"] == "Hi there"
    assert res.messages[0]["content"][1]["type"] == "image_url"

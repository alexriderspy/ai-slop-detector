from slop.text import clean, sentences, word_count


def test_clean_turns_hn_html_into_plain_text() -> None:
    raw = "I don&#x27;t agree.<p>Here&#x27;s <i>why</i>:  it&#x27;s slow."
    assert clean(raw) == "I don't agree.\n\nHere's why: it's slow."


def test_sentences_and_words() -> None:
    text = "Short one. Then a much longer sentence follows here! Done?"
    assert sentences(text) == ["Short one.", "Then a much longer sentence follows here!", "Done?"]
    assert word_count(text) == 10

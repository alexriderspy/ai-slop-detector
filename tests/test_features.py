from slop.features import STYLE_NAMES, style_features


def test_style_features_line_up_with_their_names() -> None:
    values = style_features("Great point! It is crucial. Truly crucial.")
    named = dict(zip(STYLE_NAMES, values, strict=True))
    assert named["stock_word_per_100_words"] == 2 * 100 / 7
    assert named["exclamation_per_100_words"] == 100 / 7
    assert named["sentence_length"] == 7 / 3


def test_uneven_sentences_are_burstier() -> None:
    even = style_features("One two three. Four five six. Seven eight nine.")[-1]
    uneven = style_features("One. Two three four five six seven eight. Nine.")[-1]
    assert uneven > even == 0

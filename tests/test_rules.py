import pytest

from slop.rules import find_hits


def rules_in(text: str) -> list[str]:
    return [hit.rule for hit in find_hits(text)]


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("Let us delve into the details.", "stock_word"),
        ("This is a robust and seamless solution.", "stock_word"),
        ("It's not just a tool, but a way of thinking.", "not_x_but_y"),
        ("The update — finally — shipped.", "em_dash"),
        ("It is fast, cheap, and reliable.", "triad"),
        ("Great point! I agree.", "stock_opener"),
        ("Overall, it works.", "stock_closer"),
        ("It's important to note that caching helps.", "signposting"),
        ("Love this \U0001f680", "emoji"),
    ],
)
def test_each_rule_fires(text: str, rule: str) -> None:
    assert rule in rules_in(text)


def test_plain_human_text_has_no_hits() -> None:
    text = "I tried this on our build server last year. It broke twice, then we gave up on it."
    assert rules_in(text) == []


def test_hits_keep_their_position_for_highlighting() -> None:
    text = "We should delve deeper."
    (hit,) = find_hits(text)
    assert text[hit.start : hit.end] == "delve" == hit.text

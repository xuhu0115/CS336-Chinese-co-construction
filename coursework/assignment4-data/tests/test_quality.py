import logging

from .adapters import run_classify_quality, run_gopher_quality_filter
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


def test_classify_quality():
    low_quality_cc_path = FIXTURES_PATH / "low_quality_cc.txt"
    with open(low_quality_cc_path) as f:
        low_quality_cc = f.read()
    prediction, score = run_classify_quality(low_quality_cc)
    # TODO: you may have to change this check below, depending on what your
    # quality classifier returns.
    assert prediction == "cc"
    assert isinstance(score, float)
    assert score > 0

    high_quality_wiki_path = FIXTURES_PATH / "high_quality_wiki_reference.txt"
    with open(high_quality_wiki_path) as f:
        high_quality_wiki = f.read()
    prediction, score = run_classify_quality(high_quality_wiki)
    # TODO: you may have to change this check below, depending on what your
    # quality classifier returns.
    assert prediction == "wiki"
    assert isinstance(score, float)
    assert score > 0


def test_gopher_valid_input():
    text = (
        "This should definitely be a valid input text "
        "and of high quality according to Gopher rules. "
    ) * 100
    assert run_gopher_quality_filter(text)


def test_gopher_less_than_50_non_symbol_words():
    text = "The string you are reading is a short snippet of text."
    assert not run_gopher_quality_filter(text)

    text = "The string you are reading is a long snippet of text." * 100
    assert run_gopher_quality_filter(text)


def test_gopher_more_than_100000_non_symbol_words():
    text = "The string you are reading is too long of a text. " * 50000
    assert not run_gopher_quality_filter(text)

    text = "The string you are reading is an okay example of text. " * 5000
    assert run_gopher_quality_filter(text)


def test_gopher_average_word_length_less_than_3():
    text = "the be " * 100
    assert not run_gopher_quality_filter(text)

    text = "the with " * 100
    assert run_gopher_quality_filter(text)


def test_gopher_average_word_length_greater_than_10():
    text = (
        "the and " + "extraordinarily extraordinarily extraordinarily longesest " * 100
    )
    assert not run_gopher_quality_filter(text)

    text = "the and this is fine " * 100
    assert run_gopher_quality_filter(text)


def test_gopher_more_than_30_percent_lines_ending_with_ellipsis():
    lines = [
        "The line here is an example of line ending with an ellipsis..."
        for _ in range(70)
    ]
    lines += ["This is a normal line." for _ in range(30)]
    text = "\n".join(lines)
    assert not run_gopher_quality_filter(text)

    lines = [
        "The line here is an example of ending with ellipsis..." for _ in range(30)
    ]
    lines += ["This is a normal line." for _ in range(230)]
    text = "\n".join(lines)
    assert run_gopher_quality_filter(text)


def test_gopher_less_than_80_percent_words_with_alphabetic_character():
    words = ["123" for _ in range(8)]
    words += ["word" for _ in range(2)]
    text = "the and " + " ".join(words)
    assert not run_gopher_quality_filter(text)

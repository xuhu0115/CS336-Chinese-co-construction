import logging

from .adapters import run_identify_language
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


def test_identify_language_english():
    moby_expected_path = FIXTURES_PATH / "moby_extracted.txt"
    with open(moby_expected_path) as f:
        moby_expected_text = f.read()
    predicted_language, score = run_identify_language(moby_expected_text)
    # TODO: you may have to change this check below, depending on what your
    # language ID system returns.
    assert predicted_language == "en"
    assert isinstance(score, float)
    assert score > 0


def test_identify_language_chinese_simplified():
    predicted_language, score = run_identify_language("欢迎来到我们的网站")
    # TODO: you may have to change this check below, depending on what your
    # language ID system returns.
    assert predicted_language == "zh"
    assert isinstance(score, float)
    assert score > 0

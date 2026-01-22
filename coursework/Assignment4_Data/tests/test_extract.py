import logging

from .adapters import run_extract_text_from_html_bytes
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


def test_extract_text_from_html_bytes():
    moby_path = FIXTURES_PATH / "moby.html"
    with open(moby_path, "rb") as f:
        moby_bytes = f.read()
    moby_expected_path = FIXTURES_PATH / "moby_extracted.txt"
    with open(moby_expected_path) as f:
        moby_expected_text = f.read()
    assert moby_expected_text == run_extract_text_from_html_bytes(moby_bytes)

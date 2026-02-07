import logging

from .adapters import run_mask_emails, run_mask_ips, run_mask_phone_numbers

logger = logging.getLogger(__name__)


def test_mask_emails_single():
    test_string = "Feel free to contact me at test@gmail.com if you have any questions."
    expected_masked_text = (
        "Feel free to contact me at |||EMAIL_ADDRESS||| if you have any questions."
    )
    masked_text, num_masked = run_mask_emails(test_string)
    assert masked_text == expected_masked_text
    assert num_masked == 1


def test_mask_emails_multiple():
    test_string = "The instructors are pl@fakedomain.ai and spl@fakedomain.ai"
    expected_masked_text = (
        "The instructors are |||EMAIL_ADDRESS||| and |||EMAIL_ADDRESS|||"
    )
    masked_text, num_masked = run_mask_emails(test_string)
    assert masked_text == expected_masked_text
    assert num_masked == 2


def test_mask_emails_existing_string():
    test_string = (
        "Some datasets use the string |||EMAIL_ADDRESS||| to represent masked PII. "
        "The instructors are pl@fakedomain.ai and spl@fakedomain.ai"
    )
    expected_masked_text = (
        "Some datasets use the string |||EMAIL_ADDRESS||| to represent masked PII. "
        "The instructors are |||EMAIL_ADDRESS||| and |||EMAIL_ADDRESS|||"
    )
    masked_text, num_masked = run_mask_emails(test_string)
    assert masked_text == expected_masked_text
    assert num_masked == 2


def test_mask_phones_single():
    numbers = ["2831823829", "(283)-182-3829", "(283) 182 3829", "283-182-3829"]
    for number in numbers:
        test_string = f"Feel free to contact me at {number} if you have any questions."
        expected_masked_text = (
            "Feel free to contact me at |||PHONE_NUMBER||| if you have any questions."
        )
        masked_text, num_masked = run_mask_phone_numbers(test_string)
        assert masked_text == expected_masked_text
        assert num_masked == 1


def test_mask_ips():
    test_string = "You can access the server at 192.0.2.146."
    expected_masked_text = "You can access the server at |||IP_ADDRESS|||."
    masked_text, num_masked = run_mask_ips(test_string)
    assert masked_text == expected_masked_text
    assert num_masked == 1

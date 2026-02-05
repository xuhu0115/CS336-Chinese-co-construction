import logging

from .adapters import run_classify_nsfw, run_classify_toxic_speech

logger = logging.getLogger(__name__)


def test_classify_nsfw():
    # This example is taken from the Jigsaw Wikipedia toxic comments training
    # set, so a Jigsaw-trained model should almost certainly classify this as obscene.
    prediction, score = run_classify_nsfw(
        "SUCK MY C*CK WIKIPEDIA EDITORS...F*CKING *SSH*LE DORKS. "
        "JUST TRYING TO MAKE THE SITE BETTER YOU UPTIGHT C*NTS"
    )
    # TODO: you may have to change this check below, depending on what your
    # NSFW classifier system returns.
    assert prediction == "nsfw"
    assert isinstance(score, float)
    assert score > 0

    # This example is taken from the Jigsaw Wikipedia toxic comments training
    # set, so a Jigsaw-trained model should almost certainly classify this as non-obscene.
    prediction, score = run_classify_nsfw(
        "Umm, theres no actual article for prostitution ring.  - Crunch Captain."
    )
    # TODO: you may have to change this check below, depending on what your
    # NSFW classifier system returns.
    assert prediction == "non-nsfw"
    assert isinstance(score, float)
    assert score > 0


def test_classify_toxic_speech():
    # This example is taken from the Jigsaw Wikipedia toxic comments training
    # set, so a Jigsaw-trained model should almost certainly classify this as hateful.
    prediction, score = run_classify_toxic_speech(
        "Why did that idiot revert the reversion I made? "
        "Can that moron not have the decent common manners to post on the talk page? "
        "What a rude fuck. Arrogant twat who doesn't know what he's talking about. "
        "None of you fuckers have any manners."
    )
    # TODO: you may have to change this check below, depending on what your
    # hate-speech classifier system returns.
    assert prediction == "toxic"
    assert isinstance(score, float)
    assert score > 0

    # This example is taken from the Jigsaw Wikipedia toxic comments training
    # set, so a Jigsaw-trained model should almost certainly classify this as non-hateful.
    prediction, score = run_classify_toxic_speech(
        "Why the fc*k should I get a warning for doing nothing?"
    )
    # TODO: you may have to change this check below, depending on what your
    # hate-speech classifier system returns.
    assert prediction == "non-toxic"
    assert isinstance(score, float)
    assert score > 0

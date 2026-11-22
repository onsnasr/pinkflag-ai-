from pinkflag.labeling import drop_reason, weak_label


def test_keeps_real_messages():
    assert drop_reason("That never happened. You're just being paranoid", "Denying") is None
    assert drop_reason("u up?", "Breadcrumbing") is None


def test_drops_noise():
    assert drop_reason("I hear you, but I remember it differently",
                       "How to respond to gaslighting") == "advice_or_reply"
    assert drop_reason("A gaslighter will refuse to admit fault", "") == "article_language"
    assert drop_reason("Breadcrumbing: what it is", "") == "title"
    assert drop_reason("They respect other commitments", "") == "no_you_or_me"


def test_weak_label_agreement_and_conflict():
    assert weak_label("you're too sensitive", "Gaslighting phrases") == ("gaslighting",
                                                                         "rule_agree")
    assert weak_label("if you really loved me you'd stay", "Gaslighting phrases") == (
        None, "rule_conflict")
    assert weak_label("where are you? who are you with?", "Toxic texts") == (
        "controlling", "rule_text")
    assert weak_label("ok see you later", "Examples")[0] is None

from toxic_moderation.segment import split_sentences, summarize


def pred(label, conf):
    rest = (1 - conf) / 2
    return {"label": label, "scores": {lab: (conf if lab == label else rest) for lab in ("clean", "offensive", "hate")}}


def test_split_on_punctuation_and_newlines():
    text = "Video này hay thật sự. Nhưng mà thằng mc nói nhiều quá!\nLần sau làm ngắn hơn nhé"
    assert split_sentences(text) == ["Video này hay thật sự.", "Nhưng mà thằng mc nói nhiều quá!", "Lần sau làm ngắn hơn nhé"]


def test_keeps_numbers_and_abbreviations_together():
    assert split_sentences("Giá 5.5 triệu ở TP.HCM là quá đắt rồi") == ["Giá 5.5 triệu ở TP.HCM là quá đắt rồi"]


def test_short_fragments_are_merged():
    assert split_sentences("Ừ. Mình cũng nghĩ vậy đó bạn. Thế thôi.") == ["Ừ. Mình cũng nghĩ vậy đó bạn. Thế thôi."]
    assert split_sentences("") == []


def test_summary_takes_most_severe_confident_sentence():
    s = summarize([pred("clean", 0.9), pred("offensive", 0.8), pred("hate", 0.95), pred("clean", 0.7)])
    assert s["label"] == "hate" and s["worst_sentence"] == 2 and not s["needs_review"]
    assert s["counts"] == {"clean": 2, "offensive": 1, "hate": 1}


def test_low_confidence_toxic_sentence_only_flags_review():
    s = summarize([pred("clean", 0.9), pred("hate", 0.5)])
    assert s["label"] == "clean" and s["needs_review"]

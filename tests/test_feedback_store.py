import json

import pandas as pd

from toxic_moderation.feedback_store import FIELDS, FeedbackStore, latest, make_record


def rec(item, feedback, correct=None, session="s1"):
    return make_record(session=session, item=item, text="đẹp vl", model_label="offensive",
                       scores={"clean": 0.2, "offensive": 0.7, "hate": 0.1}, feedback=feedback,
                       correct=correct, model_revision="main", model_sha="abc1234")


def test_store_without_token_is_local_only(tmp_path):
    store = FeedbackStore(folder=tmp_path)
    assert not store.enabled
    store.add(rec(0, "sai"))
    store.add(rec(0, "sai", "clean"))
    rows = [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and list(rows[0]) == FIELDS
    assert rows[1]["correct"] == "clean" and rows[1]["text"] == "đẹp vl"
    store.flush()  # no-op without a scheduler


def test_latest_keeps_last_vote_per_session_item():
    df = pd.DataFrame([dict(rec(0, "đúng"), ts="2026-11-11T10:00:00"),
                       dict(rec(0, "sai", "clean"), ts="2026-11-11T10:01:00"),
                       dict(rec(0, "đúng", session="s2"), ts="2026-11-11T10:00:30")])
    out = latest(df)
    assert len(out) == 2
    assert out.set_index("session").loc["s1", "correct"] == "clean"

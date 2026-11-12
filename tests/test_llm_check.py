import pytest

from toxic_moderation.llm_check import LLMError, parse_labels, review


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        return FakeResponse(self.replies.pop(0))


def answer(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_parse_labels_accepts_json_inside_a_code_block():
    assert parse_labels('```json\n["clean", "Hate"]\n```', 2) == ["clean", "hate"]


def test_parse_labels_rejects_wrong_length_or_unknown_label():
    with pytest.raises(LLMError):
        parse_labels('["clean"]', 2)
    with pytest.raises(LLMError):
        parse_labels('["clean", "toxic"]', 2)


def test_review_batches_requests_and_sends_the_key_in_a_header():
    session = FakeSession([answer('["clean", "offensive"]'), answer('["hate"]')])
    labels = review(["a b c", "d e f", "g h i"], api_key="k", batch_size=2, session=session)
    assert labels == ["clean", "offensive", "hate"]
    assert len(session.calls) == 2
    url, headers, body = session.calls[0]
    assert "key=" not in url and headers["x-goog-api-key"] == "k"
    assert "1. a b c" in body["contents"][0]["parts"][0]["text"]


def test_review_raises_on_api_error():
    session = FakeSession([{"error": {"code": 429, "message": "quota"}}])
    with pytest.raises(LLMError, match="429"):
        review(["a b c"], api_key="k", session=session, retries=0)


def test_review_retries_when_the_model_is_busy():
    session = FakeSession([{"error": {"code": 503, "message": "busy"}}, answer('["clean"]')])
    assert review(["a b c"], api_key="k", session=session, backoff=0) == ["clean"]
    assert len(session.calls) == 2

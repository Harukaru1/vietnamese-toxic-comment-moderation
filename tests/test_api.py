from fastapi.testclient import TestClient

from app.main import create_app


class FakePredictor:
    def predict(self, texts):
        return [{"label": "clean", "scores": {"clean": 0.9, "offensive": 0.05, "hate": 0.05}} for _ in texts]

    def preprocess(self, text):
        return text.lower()


client = TestClient(create_app(predictor_factory=lambda: FakePredictor()))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_one_prediction_per_text():
    r = client.post("/predict", json={"texts": ["bài viết hay", "quá tệ"]})
    assert r.status_code == 200
    body = r.json()
    assert [p["text"] for p in body["predictions"]] == ["bài viết hay", "quá tệ"]
    assert body["predictions"][0]["label"] == "clean"
    assert "latency_ms" in body


def test_predict_rejects_empty_list():
    assert client.post("/predict", json={"texts": []}).status_code == 422


def test_demo_page_served():
    assert client.get("/").status_code == 200

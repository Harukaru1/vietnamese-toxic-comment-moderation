"""FastAPI service + Gradio demo for Vietnamese toxic comment moderation.

Run locally:  uvicorn app.main:app --port 7860
  - Demo UI:  http://localhost:7860/
  - API docs: http://localhost:7860/docs
"""

from __future__ import annotations

import time
from functools import lru_cache

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

LABEL_VI = {"clean": "Bình thường", "offensive": "Công kích", "hate": "Thù ghét / xúc phạm"}

EXAMPLES = [
    "Video rất hay và bổ ích, cảm ơn bạn đã chia sẻ",
    "Đá như này thì nghỉ mẹ đi cho rồi",
    "Bọn này ngu như chó, cút hết về nước đi",
]


@lru_cache(maxsize=1)
def get_predictor():
    from toxic_moderation.predictor import Predictor  # heavy import: torch + transformers
    return Predictor()


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64, examples=[EXAMPLES[:2]])


class Prediction(BaseModel):
    text: str
    label: str
    scores: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


def create_app(predictor_factory=get_predictor) -> FastAPI:
    api = FastAPI(
        title="Vietnamese Toxic Comment Moderation API",
        description="Classify Vietnamese comments as clean / offensive / hate (fine-tuned PhoBERT).",
        version="0.1.0",
    )

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @api.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        preds = predictor_factory().predict(req.texts)
        return PredictResponse(
            predictions=[Prediction(text=t, **p) for t, p in zip(req.texts, preds)],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def classify(text: str):
        if not text or not text.strip():
            return {}, ""
        pred = predictor_factory().predict([text])[0]
        scores = {f"{lab} ({LABEL_VI.get(lab, lab)})": s for lab, s in pred["scores"].items()}
        return scores, predictor_factory().preprocess(text)

    with gr.Blocks(title="Vietnamese Toxic Comment Moderation") as demo:
        gr.Markdown(
            "# 🛡️ Vietnamese Toxic Comment Moderation\n"
            "Phân loại bình luận tiếng Việt thành **clean / offensive / hate** bằng PhoBERT fine-tuned. "
            "API: [`/docs`](/docs) · Code: "
            "[GitHub](https://github.com/Harukaru1/vietnamese-toxic-comment-moderation) · "
            "Model: [Hugging Face](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment)"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Bình luận", lines=4, placeholder="Nhập một bình luận tiếng Việt...")
                btn = gr.Button("Phân loại", variant="primary")
                gr.Examples(EXAMPLES, inputs=inp)
            with gr.Column():
                out = gr.Label(label="Kết quả", num_top_classes=3)
                prep = gr.Textbox(label="Văn bản sau tiền xử lý (đầu vào của model)", interactive=False)
        btn.click(classify, inputs=inp, outputs=[out, prep])
        inp.submit(classify, inputs=inp, outputs=[out, prep])
        gr.Markdown(
            "⚠️ Mô hình nghiên cứu, có thể sai, đặc biệt ở ranh giới *offensive* / *hate*. "
            "Không dùng để ra quyết định tự động mà không có người kiểm duyệt."
        )

    return gr.mount_gradio_app(api, demo, path="/")


app = create_app()

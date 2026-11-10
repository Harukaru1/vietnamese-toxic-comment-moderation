"""Streamlit demo for Vietnamese toxic comment moderation.

Run locally:  streamlit run streamlit_app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxic_moderation.predictor import Predictor  # noqa: E402

LABEL_VI = {"clean": "Bình thường", "offensive": "Công kích", "hate": "Thù ghét / xúc phạm"}
LABEL_ICON = {"clean": "🟢", "offensive": "🟠", "hate": "🔴"}
EXAMPLES = [
    "Video rất hay và bổ ích, cảm ơn bạn đã chia sẻ",
    "Đá như này thì nghỉ mẹ đi cho rồi",
    "Bọn này ngu như chó, cút hết về nước đi",
]
MAX_BATCH = 200

st.set_page_config(page_title="Vietnamese Toxic Comment Moderation", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Đang tải mô hình PhoBERT (lần đầu mất khoảng 1 phút)...")
def load_predictor() -> Predictor:
    return Predictor()


st.title("🛡️ Vietnamese Toxic Comment Moderation")
st.markdown(
    "Phân loại bình luận tiếng Việt thành **clean / offensive / hate** bằng PhoBERT fine-tuned.  \n"
    "[Code (GitHub)](https://github.com/Harukaru1/vietnamese-toxic-comment-moderation) · "
    "[Model (Hugging Face)](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment)"
)

predictor = load_predictor()
single, batch = st.tabs(["Một bình luận", "Nhiều bình luận"])

with single:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES[0]
    cols = st.columns(len(EXAMPLES))
    for col, ex, lab in zip(cols, EXAMPLES, ["Ví dụ: clean", "Ví dụ: offensive", "Ví dụ: hate"]):
        if col.button(lab, use_container_width=True):
            st.session_state.text = ex
    text = st.text_area("Bình luận", key="text", height=110)

    if text.strip():
        pred = predictor.predict([text])[0]
        label = pred["label"]
        st.subheader(f"{LABEL_ICON[label]} {label} — {LABEL_VI[label]}")
        for lab, score in sorted(pred["scores"].items(), key=lambda kv: -kv[1]):
            st.progress(score, text=f"{lab}: {score:.1%}")
        with st.expander("Văn bản sau tiền xử lý (đầu vào thực tế của model)"):
            st.code(predictor.preprocess(text), language=None)

with batch:
    st.caption(f"Mỗi dòng một bình luận (tối đa {MAX_BATCH} dòng), hoặc tải lên file CSV có cột `text`.")
    raw = st.text_area("Danh sách bình luận", value="\n".join(EXAMPLES), height=150)
    upload = st.file_uploader("Hoặc tải lên CSV", type="csv")
    if st.button("Phân loại", type="primary"):
        if upload is not None:
            texts = pd.read_csv(upload)["text"].astype(str).tolist()
        else:
            texts = [t for t in raw.splitlines() if t.strip()]
        texts = texts[:MAX_BATCH]
        preds = predictor.predict(texts)
        df = pd.DataFrame({
            "text": texts,
            "label": [p["label"] for p in preds],
            **{f"p_{lab}": [p["scores"][lab] for p in preds] for lab in LABEL_VI},
        })
        counts = df["label"].value_counts()
        for col, lab in zip(st.columns(len(LABEL_VI)), LABEL_VI):
            col.metric(f"{LABEL_ICON[lab]} {lab}", int(counts.get(lab, 0)))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Tải kết quả (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="moderation_results.csv", mime="text/csv")

st.divider()
st.caption(
    "⚠️ Mô hình phục vụ nghiên cứu, có thể sai, đặc biệt ở ranh giới *offensive* / *hate*. "
    "Không nên dùng để tự động xử lý bình luận khi không có người kiểm duyệt."
)

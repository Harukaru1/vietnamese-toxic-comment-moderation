"""Streamlit demo for Vietnamese toxic comment moderation.

Run locally:  streamlit run streamlit_app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxic_moderation.predictor import Predictor  # noqa: E402

LABELS = ["clean", "offensive", "hate"]
INFO = {
    "clean": dict(icon="🟢", name="Bình thường", color="#1a7f37",
                  say="Bình luận này ổn. Không thấy lời lẽ xúc phạm hay thô tục nhắm vào ai.",
                  hint="Khen, chê, góp ý, tranh luận lịch sự. Từ chửi chỉ để nhấn mạnh (\"hay vl\") vẫn tính là bình thường."),
    "offensive": dict(icon="🟠", name="Công kích / thô tục", color="#b35900",
                      say="Có lời lẽ thô tục hoặc bực bội, nhưng không nhắm vào một người hay nhóm người cụ thể.",
                      hint="Chửi thề để xả bực (\"đéo hiểu nổi\", \"mẹ nó lại mất điện\"), nội dung thô tục."),
    "hate": dict(icon="🔴", name="Thù ghét / xúc phạm", color="#c62828",
                 say="Bình luận này đang xúc phạm, hạ thấp hoặc đe doạ một người hay một nhóm người.",
                 hint="Chửi người khác, miệt thị vùng miền, giới tính, dân tộc; đe doạ, trù ẻo."),
}
EXAMPLES = {
    "Khen ngợi": "Video rất hay và bổ ích, cảm ơn bạn đã chia sẻ",
    "Chê nhẹ nhàng": "Tập này hơi dài, mong lần sau làm gọn hơn",
    "Nhấn mạnh (\"vl\")": "Phim hay vl, xem 3 lần rồi vẫn thích",
    "Xả bực": "Đá như này thì nghỉ mẹ đi cho rồi",
    "Xúc phạm": "Bọn này ngu như chó, cút hết về nước đi",
}
MAX_BATCH = 200
REVIEW_BELOW = 0.6  # below this confidence, suggest a human look

st.set_page_config(page_title="Kiểm tra bình luận tiếng Việt", page_icon="🛡️", layout="centered")
st.markdown("""
<style>
.result-card {border-left: 6px solid var(--c); background: color-mix(in srgb, var(--c) 12%, transparent);
              border-radius: 10px; padding: 14px 18px; margin: 8px 0 14px 0;}
.result-card .title {font-size: 1.35rem; font-weight: 700; color: var(--c); margin-bottom: 4px;}
.result-card .sub {opacity: .85;}
.label-chip {display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .85rem;
             background: color-mix(in srgb, var(--c) 15%, transparent); color: var(--c); font-weight: 600;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang tải mô hình PhoBERT (lần đầu mất khoảng 1 phút)...")
def load_predictor() -> tuple[Predictor, str]:
    predictor = Predictor()
    try:  # show which model commit is loaded, so a stale cache is easy to spot
        from huggingface_hub import model_info
        sha = model_info(predictor.model_id, revision=predictor.revision).sha[:7]
    except Exception:  # offline or rate-limited: the demo still works
        sha = "unknown"
    return predictor, sha


def confidence_words(p: float) -> str:
    if p >= 0.9:
        return "Mô hình khá chắc chắn"
    if p >= REVIEW_BELOW:
        return "Mô hình tương đối chắc"
    return "Mô hình còn phân vân, nên để người xem lại"


def result_card(label: str, conf: float) -> None:
    i = INFO[label]
    st.markdown(
        f'<div class="result-card" style="--c:{i["color"]}"><div class="title">{i["icon"]} {i["name"]}</div>'
        f'<div>{i["say"]}</div><div class="sub">{confidence_words(conf)} ({conf:.0%}).</div></div>',
        unsafe_allow_html=True)


ss = st.session_state
ss.setdefault("text", "")
ss.setdefault("history", [])   # every single-comment check in this browser session
ss.setdefault("current", None)

predictor, model_sha = load_predictor()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Ba nhãn nghĩa là gì?")
    for lab in LABELS:
        i = INFO[lab]
        st.markdown(f'<span class="label-chip" style="--c:{i["color"]}">{i["icon"]} {i["name"]}</span>',
                    unsafe_allow_html=True)
        st.caption(i["hint"])
    st.divider()
    st.markdown(
        "Mô hình **PhoBERT** được fine-tune trên khoảng 50 nghìn bình luận tiếng Việt, "
        "và được kiểm tra lại trên bình luận YouTube và diễn đàn năm 2026.\n\n"
        "[Mã nguồn (GitHub)](https://github.com/Harukaru1/vietnamese-toxic-comment-moderation) · "
        "[Mô hình (Hugging Face)](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment)")
    st.caption(f"Model `{predictor.model_id}` @ `{predictor.revision or 'main'}` · commit `{model_sha}`")

# ---------------------------------------------------------------- header
st.title("🛡️ Kiểm tra bình luận tiếng Việt")
st.markdown("Dán một bình luận vào ô bên dưới. Mô hình sẽ cho bạn biết nó **bình thường**, "
            "**công kích** hay **thù ghét**, kèm mức độ chắc chắn.")

single, batch, history = st.tabs(["💬 Một bình luận", "📋 Nhiều bình luận", "🕘 Lịch sử & góp ý"])

# ---------------------------------------------------------------- single comment
with single:
    def use_example() -> None:
        if ss.example:
            ss.text = EXAMPLES[ss.example]
            ss.example = None
            ss.current = None

    st.pills("Chưa biết thử gì? Chọn một ví dụ:", list(EXAMPLES), key="example", on_change=use_example)
    st.text_area("Bình luận", key="text", height=110, max_chars=1000,
                 placeholder="Ví dụ: Đẹp vl, xem mà nổi da gà luôn 😍")

    def clear() -> None:
        ss.text, ss.current = "", None

    c1, c2, _ = st.columns([1.3, 1, 3])
    check = c1.button("🔍 Kiểm tra", type="primary", use_container_width=True)
    c2.button("🧹 Xoá", on_click=clear, use_container_width=True)

    if check and not ss.text.strip():
        st.warning("Bạn nhập một bình luận trước nhé.")
    elif check:
        pred = predictor.predict([ss.text])[0]
        ss.current = dict(id=len(ss.history), time=datetime.now().strftime("%H:%M:%S"), text=ss.text,
                          label=pred["label"], scores=pred["scores"], feedback=None, correct=None)
        ss.history.append(ss.current)

    cur = ss.current
    if cur:
        conf = cur["scores"][cur["label"]]
        result_card(cur["label"], conf)
        for lab in sorted(LABELS, key=lambda lab: -cur["scores"][lab]):
            st.progress(cur["scores"][lab], text=f'{INFO[lab]["icon"]} {INFO[lab]["name"]}: {cur["scores"][lab]:.1%}')
        with st.expander("Mô hình thực sự \"nhìn thấy\" gì? (văn bản sau tiền xử lý)"):
            st.code(predictor.preprocess(cur["text"]), language=None)
            st.caption("Chữ thường, bỏ link/emoji/dấu câu, rút gọn ký tự lặp (\"vlllll\" → \"vl\"), "
                       "rồi tách từ bằng pyvi (từ ghép được nối bằng dấu \"_\").")

        st.markdown("**Kết quả này có đúng không?**")
        fb = st.feedback("thumbs", key=f"fb_{cur['id']}")
        if fb is not None:
            cur["feedback"] = "đúng" if fb == 1 else "sai"
            if fb == 1:
                st.success("Cảm ơn bạn đã xác nhận! 🙌")
            else:
                correct = st.segmented_control(
                    "Theo bạn, nhãn đúng là gì?", [lab for lab in LABELS if lab != cur["label"]],
                    format_func=lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}', key=f"fix_{cur['id']}")
                if correct:
                    cur["correct"] = correct
                    st.success("Đã ghi lại. Bạn có thể tải các góp ý ở tab **Lịch sử & góp ý** "
                               "để dùng khi huấn luyện lại mô hình.")

# ---------------------------------------------------------------- batch
with batch:
    st.markdown("Kiểm tra nhiều bình luận cùng lúc: **mỗi dòng một bình luận** "
                f"(tối đa {MAX_BATCH} dòng), hoặc tải lên file CSV có cột `text`.")
    raw = st.text_area("Danh sách bình luận", value="\n".join(EXAMPLES.values()), height=160)
    upload = st.file_uploader("Hoặc tải lên CSV", type="csv")
    if st.button("🔍 Phân loại tất cả", type="primary"):
        if upload is not None:
            texts = pd.read_csv(upload)["text"].dropna().astype(str).tolist()
        else:
            texts = [t for t in raw.splitlines() if t.strip()]
        if len(texts) > MAX_BATCH:
            st.info(f"Chỉ lấy {MAX_BATCH} dòng đầu tiên.")
        texts = texts[:MAX_BATCH]
        bar = st.progress(0.0, text="Đang phân loại...")
        preds = []
        for k in range(0, len(texts), 32):
            preds += predictor.predict(texts[k:k + 32])
            bar.progress(min(1.0, (k + 32) / max(len(texts), 1)), text=f"Đã xong {len(preds)}/{len(texts)}")
        bar.empty()
        ss.batch = pd.DataFrame({
            "text": texts,
            "label": [p["label"] for p in preds],
            "confidence": [p["scores"][p["label"]] for p in preds],
            **{f"p_{lab}": [p["scores"][lab] for p in preds] for lab in LABELS},
        })

    if "batch" in ss and len(ss.batch):
        df = ss.batch
        counts = df["label"].value_counts()
        cols = st.columns(4)
        for col, lab in zip(cols, LABELS):
            n = int(counts.get(lab, 0))
            col.metric(f'{INFO[lab]["icon"]} {INFO[lab]["name"]}', f"{n} ({n / len(df):.0%})")
        cols[3].metric("🤔 Cần xem lại", int((df["confidence"] < REVIEW_BELOW).sum()),
                       help=f"Độ tin cậy dưới {REVIEW_BELOW:.0%}: nên để người kiểm duyệt xem lại.")

        show = st.segmented_control("Lọc theo nhãn", ["Tất cả"] + LABELS + ["Cần xem lại"], default="Tất cả",
                                    format_func=lambda x: x if x in ("Tất cả", "Cần xem lại") else INFO[x]["name"])
        view = df
        if show in LABELS:
            view = df[df["label"] == show]
        elif show == "Cần xem lại":
            view = df[df["confidence"] < REVIEW_BELOW]
        st.dataframe(
            view.assign(label=view["label"].map(lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}')),
            use_container_width=True, hide_index=True,
            column_order=["text", "label", "confidence"],
            column_config={
                "text": st.column_config.TextColumn("Bình luận", width="large"),
                "label": st.column_config.TextColumn("Nhãn"),
                "confidence": st.column_config.ProgressColumn("Độ tin cậy", min_value=0, max_value=1, format="percent"),
            })
        st.download_button("⬇️ Tải kết quả (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="moderation_results.csv", mime="text/csv")

# ---------------------------------------------------------------- history & feedback
with history:
    if not ss.history:
        st.info("Chưa có gì ở đây. Kiểm tra vài bình luận ở tab **Một bình luận** nhé.")
    else:
        h = pd.DataFrame(ss.history)
        n_fb = h["feedback"].notna().sum()
        n_wrong = (h["feedback"] == "sai").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Đã kiểm tra", len(h))
        c2.metric("Có góp ý", int(n_fb))
        c3.metric("Bạn báo sai", int(n_wrong))
        st.dataframe(
            h.assign(label=h["label"].map(lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}'),
                     correct=h["correct"].map(lambda lab: INFO[lab]["name"] if lab else "")),
            use_container_width=True, hide_index=True,
            column_order=["time", "text", "label", "feedback", "correct"],
            column_config={"time": "Lúc", "text": st.column_config.TextColumn("Bình luận", width="large"),
                           "label": "Mô hình đoán", "feedback": "Góp ý", "correct": "Nhãn đúng (theo bạn)"})
        out = h.assign(model_label=h["label"])[["time", "text", "model_label", "feedback", "correct"]]
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tải lịch sử & góp ý (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name="feedback.csv", mime="text/csv", use_container_width=True)
        if c2.button("🗑️ Xoá lịch sử", use_container_width=True):
            ss.history, ss.current = [], None
            st.rerun()
        st.caption("Lịch sử chỉ lưu trong trình duyệt của bạn ở phiên này; không gửi đi đâu cả.")

st.divider()
st.caption("⚠️ Mô hình phục vụ nghiên cứu và có thể sai, nhất là ranh giới giữa *công kích* và *thù ghét*, "
           "hoặc những câu thù ghét ngầm không có từ chửi. Hãy dùng như một công cụ hỗ trợ, "
           "không thay thế người kiểm duyệt.")

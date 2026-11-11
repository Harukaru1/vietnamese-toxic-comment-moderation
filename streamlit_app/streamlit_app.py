"""Streamlit demo for Vietnamese toxic comment moderation.

Run locally:  streamlit run streamlit_app/streamlit_app.py
"""

from __future__ import annotations

import html
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxic_moderation.feedback_store import FeedbackStore, make_record  # noqa: E402
from toxic_moderation.predictor import Predictor  # noqa: E402
from toxic_moderation.segment import split_sentences, summarize  # noqa: E402

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
MAX_PARAGRAPH_CHARS = 3000
PARAGRAPH_EXAMPLE = ("Hôm qua mình xem trận chung kết, công nhận đội bạn đá hay vl. Hàng thủ chơi rất chắc, "
                     "thủ môn bắt quá đỉnh. Nhưng mà đm trọng tài thổi kiểu gì không hiểu nổi. "
                     "Còn thằng số 7 thì ngu như bò, đá như thế thì cút về quê đi. "
                     "Nói chung trận đấu vẫn đáng xem, mong mùa sau còn hay hơn.")
# Shown in the sidebar; update together with the model on the Hub (see README "Results").
MODEL_CARD = {"version": "PhoBERT v3", "f1_indomain": 0.815, "f1_2026": 0.679, "precision_toxic_2026": 0.82}
GITHUB = "https://github.com/Harukaru1/vietnamese-toxic-comment-moderation"
REVIEW_BELOW = 0.6  # below this confidence, suggest a human look

st.set_page_config(page_title="Kiểm tra bình luận tiếng Việt", page_icon="🛡️", layout="centered")
st.markdown("""
<style>
.result-card {border-left: 6px solid var(--c); background: color-mix(in srgb, var(--c) 12%, transparent);
              border-radius: 10px; padding: 14px 18px; margin: 8px 0 14px 0;}
.result-card .title {font-size: 1.35rem; font-weight: 700; color: var(--c); margin-bottom: 4px;}
.result-card .sub {opacity: .85;}
.sent {padding: 2px 4px; border-radius: 5px; line-height: 2.0;
       background: color-mix(in srgb, var(--c) var(--a), transparent); border-bottom: 2px solid var(--c);}
.sent.unsure {border-bottom-style: dashed;}
.para-box {border: 1px solid color-mix(in srgb, currentColor 15%, transparent); border-radius: 10px; padding: 14px 16px;}
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


def secret(name: str) -> str | None:
    try:  # Streamlit Cloud: App settings -> Secrets; locally: .streamlit/secrets.toml or env var
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:  # no secrets file at all
        return os.environ.get(name)


@st.cache_resource
def load_feedback_store() -> FeedbackStore:
    """One store per app process; disabled (local-only) when no HF token is configured."""
    try:
        return FeedbackStore(repo_id=secret("FEEDBACK_REPO") or "HaruKaru/vi_toxic_feedback", token=secret("HF_TOKEN"))
    except Exception as e:  # bad token / HF down: keep the demo usable
        print("feedback store disabled:", e)
        return FeedbackStore()


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
ss.setdefault("session_id", uuid.uuid4().hex[:12])  # random, only to keep a user's last vote per item
ss.setdefault("share_ok", True)

predictor, model_sha = load_predictor()
store = load_feedback_store()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Ba nhãn nghĩa là gì?")
    for lab in LABELS:
        i = INFO[lab]
        st.markdown(f'<span class="label-chip" style="--c:{i["color"]}">{i["icon"]} {i["name"]}</span>',
                    unsafe_allow_html=True)
        st.caption(i["hint"])
    st.divider()
    st.subheader("Về mô hình")
    st.markdown(f"**{MODEL_CARD['version']}**, fine-tune trên khoảng 52 nghìn bình luận tiếng Việt "
                "và được kiểm tra lại trên 1.000 bình luận YouTube và diễn đàn năm 2026.")
    m1, m2 = st.columns(2)
    m1.metric("F1 tập test", f"{MODEL_CARD['f1_indomain']:.3f}", help="F1-macro trên tập test nội bộ (3 nhãn).")
    m2.metric("F1 thực tế 2026", f"{MODEL_CARD['f1_2026']:.3f}",
              help="F1-macro trên bình luận YouTube/VOZ năm 2026, dữ liệu mô hình chưa từng thấy.")
    st.caption(f"Khi báo một bình luận là độc hại, mô hình đúng khoảng "
               f"{MODEL_CARD['precision_toxic_2026']:.0%} số lần (trên dữ liệu 2026).")
    with st.expander("Mô hình còn hạn chế gì?"):
        st.markdown(
            "- Chỉ đọc **từng bình luận riêng lẻ**, không thấy ngữ cảnh: \"bạn mập quá đấy\" có thể là "
            "trêu đùa hoặc chê bai.\n"
            "- Dễ bỏ sót **thù ghét ngầm** không có từ chửi (mỉa mai, biệt danh miệt thị mới).\n"
            "- Ranh giới *công kích* / *thù ghét* vốn chủ quan; con người cũng không luôn đồng ý.\n\n"
            + ("Góp ý 👍/👎 (kèm câu bình luận) được lưu **ẩn danh** để tác giả rà soát và huấn luyện lại "
               "mô hình; bạn có thể tắt ở mục góp ý." if store.enabled else
               "Góp ý 👍/👎 chỉ lưu trong trình duyệt của bạn; tải về ở tab **Lịch sử & góp ý** "
               "nếu muốn gửi cho tác giả để cải thiện mô hình."))
    st.markdown(f"[Mã nguồn & báo cáo (GitHub)]({GITHUB}) · "
                "[Mô hình (Hugging Face)](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment) · "
                f"[Hướng dẫn gán nhãn]({GITHUB}/blob/main/docs/annotation_guidelines.md)")
    st.caption(f"`{predictor.model_id}` @ `{predictor.revision or 'main'}` · commit `{model_sha}`")

# ---------------------------------------------------------------- header
st.title("🛡️ Kiểm tra bình luận tiếng Việt")
st.markdown("Dán một bình luận vào ô bên dưới. Mô hình sẽ cho bạn biết nó **bình thường**, "
            "**công kích** hay **thù ghét**, kèm mức độ chắc chắn.")

single, paragraph, batch, history = st.tabs(["💬 Một bình luận", "📝 Đoạn văn", "📋 Nhiều bình luận", "🕘 Lịch sử & góp ý"])

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
    check = c1.button("🔍 Kiểm tra", type="primary", width="stretch")
    c2.button("🧹 Xoá", on_click=clear, width="stretch")

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
        cur["feedback"] = None if fb is None else ("đúng" if fb == 1 else "sai")
        if fb != 0:
            cur["correct"] = None  # a thumbs-up (or no vote) clears an earlier correction
        if fb is not None:
            if fb == 1:
                st.success("Cảm ơn bạn đã xác nhận! 🙌")
            else:
                correct = st.segmented_control(
                    "Theo bạn, nhãn đúng là gì?", [lab for lab in LABELS if lab != cur["label"]],
                    format_func=lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}', key=f"fix_{cur['id']}")
                if correct:
                    cur["correct"] = correct
                    st.success("Đã ghi lại, cảm ơn bạn! Góp ý này sẽ giúp mô hình học lại chỗ sai."
                               if store.enabled and ss.share_ok else
                               "Đã ghi lại. Bạn có thể tải các góp ý ở tab **Lịch sử & góp ý** "
                               "để dùng khi huấn luyện lại mô hình.")
        if store.enabled:
            st.toggle("Cho phép lưu góp ý ẩn danh để cải thiện mô hình", key="share_ok",
                      help="Chỉ lưu câu bình luận, nhãn mô hình đoán và góp ý của bạn. "
                           "Không lưu tên, IP hay thông tin cá nhân nào.")
            state = (cur["feedback"], cur["correct"])
            if ss.share_ok and state != cur.get("logged", (None, None)):
                try:
                    store.add(make_record(session=ss.session_id, item=cur["id"], text=cur["text"],
                                          model_label=cur["label"], scores=cur["scores"], feedback=cur["feedback"],
                                          correct=cur["correct"], model_revision=predictor.revision or "main",
                                          model_sha=model_sha))
                    cur["logged"] = state
                except Exception as e:  # never break the demo because of logging
                    print("feedback not saved:", e)

# ---------------------------------------------------------------- short paragraph
with paragraph:
    st.markdown("Dán một **đoạn văn ngắn** (bài đăng, bình luận dài, đánh giá sản phẩm). Mô hình xét **từng câu** "
                "và tô màu những câu có vấn đề, thay vì chỉ cho một nhãn chung.")
    ss.setdefault("para_text", "")

    def use_para_example() -> None:
        ss.para_text = PARAGRAPH_EXAMPLE

    para_text = st.text_area("Đoạn văn", key="para_text", height=170, max_chars=MAX_PARAGRAPH_CHARS,
                             placeholder="Ví dụ: một bài đăng vài câu, mỗi câu kết thúc bằng dấu chấm hoặc xuống dòng...")
    c1, c2 = st.columns([1, 1])
    run_para = c1.button("🔍 Phân tích đoạn văn", type="primary", width="stretch")
    c2.button("✨ Dùng ví dụ mẫu", on_click=use_para_example, width="stretch")

    if run_para and not para_text.strip():
        st.warning("Bạn nhập một đoạn văn trước nhé.")
    elif run_para:
        sents = split_sentences(para_text)
        ss.para_result = dict(sents=sents, preds=predictor.predict(sents), fixed={})
    res = ss.get("para_result")
    if res:
        sents, preds = res["sents"], res["preds"]
        summary = summarize(preds, min_conf=REVIEW_BELOW)
        n_toxic = summary["counts"]["offensive"] + summary["counts"]["hate"]
        i = INFO[summary["label"]]
        detail = (f"{n_toxic} trên {len(sents)} câu có vấn đề." if n_toxic else f"Cả {len(sents)} câu đều ổn.")
        if summary["needs_review"]:
            detail += " Có câu mô hình chưa chắc (gạch chân nét đứt), nên có người xem lại."
        st.markdown(f'<div class="result-card" style="--c:{i["color"]}"><div class="title">{i["icon"]} {i["name"]}</div>'
                    f'<div>Nhãn của đoạn văn lấy theo câu nặng nhất mà mô hình đủ chắc chắn.</div>'
                    f'<div class="sub">{detail}</div></div>', unsafe_allow_html=True)
        spans = []
        for sent, pr in zip(sents, preds):
            conf = pr["scores"][pr["label"]]
            info = INFO[pr["label"]]
            alpha = "0%" if pr["label"] == "clean" else f"{int(10 + 25 * conf)}%"
            cls = "sent unsure" if (pr["label"] != "clean" and conf < REVIEW_BELOW) else "sent"
            style = f'--c:{info["color"]};--a:{alpha};' + ("border-bottom-color:transparent;" if pr["label"] == "clean" else "")
            spans.append(f'<span class="{cls}" style="{style}" title="{info["name"]} ({conf:.0%})">{html.escape(sent)}</span>')
        st.markdown('<div class="para-box">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
        st.caption(" · ".join(f'{INFO[lab]["icon"]} {INFO[lab]["name"]}: {summary["counts"][lab]} câu' for lab in LABELS)
                   + " · Rê chuột vào từng câu để xem độ chắc chắn.")
        with st.expander("Xem chi tiết từng câu"):
            st.dataframe(
                pd.DataFrame({"Câu": sents,
                              "Nhãn": [f'{INFO[pr["label"]]["icon"]} {INFO[pr["label"]]["name"]}' for pr in preds],
                              "Độ chắc chắn": [pr["scores"][pr["label"]] for pr in preds]}),
                width="stretch", hide_index=True,
                column_config={"Câu": st.column_config.TextColumn(width="large"),
                               "Độ chắc chắn": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.caption("Mô hình vẫn đọc từng câu riêng lẻ, nên có thể bỏ sót ý xúc phạm trải qua nhiều câu "
                   "hoặc câu chỉ xúc phạm khi biết ngữ cảnh.")

        st.markdown("**Có câu nào bị gán sai không?**")
        idx = st.selectbox(
            "Chọn câu bị gán sai", range(len(sents)), index=None, placeholder="Chọn câu mô hình gán sai...",
            format_func=lambda k: f'{INFO[preds[k]["label"]]["icon"]} {sents[k][:90]}{"…" if len(sents[k]) > 90 else ""}',
            label_visibility="collapsed", key="para_pick")
        if idx is not None:
            fix = st.segmented_control(
                "Theo bạn, nhãn đúng của câu này là gì?", [lab for lab in LABELS if lab != preds[idx]["label"]],
                format_func=lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}', key=f"para_fix_{idx}")
            if fix and res["fixed"].get(idx) != fix:
                res["fixed"][idx] = fix
                item = dict(id=len(ss.history), time=datetime.now().strftime("%H:%M:%S"), text=sents[idx],
                            label=preds[idx]["label"], scores=preds[idx]["scores"], feedback="sai", correct=fix)
                ss.history.append(item)   # also shows up in the history tab / CSV download
                if store.enabled and ss.share_ok:
                    try:
                        store.add(make_record(session=ss.session_id, item=item["id"], text=item["text"],
                                              model_label=item["label"], scores=item["scores"], feedback="sai",
                                              correct=fix, model_revision=predictor.revision or "main",
                                              model_sha=model_sha))
                    except Exception as e:  # never break the demo because of logging
                        print("feedback not saved:", e)
            if res["fixed"].get(idx):
                st.success(f'Đã ghi lại: câu này nên là **{INFO[res["fixed"][idx]]["name"]}**. Cảm ơn bạn!')
        if store.enabled:
            st.caption("Góp ý được lưu ẩn danh để cải thiện mô hình"
                       + ("." if ss.share_ok else " (bạn đang tắt lưu góp ý ở tab Một bình luận nên sẽ không gửi đi)."))

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
            width="stretch", hide_index=True,
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
            h.assign(feedback=h["feedback"].fillna(""), label=h["label"].map(lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}'),
                     correct=h["correct"].map(lambda lab: INFO[lab]["name"] if lab in INFO else "")),
            width="stretch", hide_index=True,
            column_order=["time", "text", "label", "feedback", "correct"],
            column_config={"time": "Lúc", "text": st.column_config.TextColumn("Bình luận", width="large"),
                           "label": "Mô hình đoán", "feedback": "Góp ý", "correct": "Nhãn đúng (theo bạn)"})
        out = h.assign(model_label=h["label"])[["time", "text", "model_label", "feedback", "correct"]]
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tải lịch sử & góp ý (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name="feedback.csv", mime="text/csv", width="stretch")
        if c2.button("🗑️ Xoá lịch sử", width="stretch"):
            ss.history, ss.current = [], None
            st.rerun()
        st.caption("Lịch sử này chỉ nằm trong trình duyệt của bạn ở phiên này. "
                   + ("Riêng các góp ý 👍/👎 được lưu ẩn danh để cải thiện mô hình (nếu bạn bật cho phép)."
                      if store.enabled else "Không có gì được gửi đi."))

st.divider()
st.caption("⚠️ Mô hình phục vụ nghiên cứu và có thể sai, nhất là ranh giới giữa *công kích* và *thù ghét*, "
           "hoặc những câu thù ghét ngầm không có từ chửi. Hãy dùng như một công cụ hỗ trợ, "
           "không thay thế người kiểm duyệt.")

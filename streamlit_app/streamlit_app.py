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
from toxic_moderation.profanity import find_profanity
from toxic_moderation.segment import split_sentences, summarize  # noqa: E402

LABELS = ["clean", "offensive", "hate"]
INFO = {
    "clean": dict(icon="🟢", name="Bình thường", color="#1a7f37",
                  say="Bình luận này ổn. Không thấy lời lẽ xúc phạm hay thô tục nhắm vào ai.",
                  hint="Khen, chê, góp ý, tranh luận lịch sự. Từ chửi chỉ để nhấn mạnh (\"hay vl\") vẫn tính là bình thường."),
    "offensive": dict(icon="🟠", name="Công kích / thô tục", color="#b35900",
                      say="Có lời lẽ thô tục, mắng một người, hoặc nói xấu một nhóm kiểu \"bọn này\", nhưng chưa tới mức miệt thị nặng.",
                      hint="Chửi thề để xả bực, mắng một người (\"mày ngu thế\"), nói xấu chung một nhóm (\"bọn này hay thượng đẳng\"), nội dung thô tục."),
    "hate": dict(icon="🔴", name="Thù ghét / miệt thị", color="#c62828",
                 say="Bình luận này miệt thị nặng: vì người ta là ai (vùng miền, dân tộc, giới tính, ngoại hình...), coi người như con vật, hoặc đe doạ, trù ẻo.",
                 hint="Miệt thị vùng miền, quốc gia, giới tính; chê ngoại hình; ví người với con vật, lấy khuyết tật ra chửi, chửi gia đình người khác; đe doạ, trù ẻo."),
}
EXAMPLES = {
    "Khen ngợi": "Video rất hay và bổ ích, cảm ơn bạn đã chia sẻ",
    "Chê nhẹ nhàng": "Tập này hơi dài, mong lần sau làm gọn hơn",
    "Nhấn mạnh (\"vl\")": "Phim hay vl, xem 3 lần rồi vẫn thích",
    "Xả bực": "Đá như này thì nghỉ mẹ đi cho rồi",
    "Mắng một người": "Mày ngu thế, nói mãi không hiểu à",
    "Miệt thị": "Bọn này ngu như chó, cút hết về nước đi",
}
MAX_BATCH = 200
MAX_PARAGRAPH_CHARS = 3000
PARAGRAPH_EXAMPLE = ("Hôm qua mình xem trận chung kết, công nhận đội bạn đá hay vl. Hàng thủ chơi rất chắc, "
                     "thủ môn bắt quá đỉnh. Nhưng mà đm trọng tài thổi kiểu gì không hiểu nổi. "
                     "Còn thằng số 7 thì ngu như bò, đá như thế thì cút về quê đi. "
                     "Nói chung trận đấu vẫn đáng xem, mong mùa sau còn hay hơn.")
# Shown in the sidebar; update together with the model on the Hub (see README "Results").
MODEL_CARD = {"version": "PhoBERT v7", "precision_toxic": 0.62, "recall_toxic": 0.49}
MODEL_REVISION = os.getenv("MODEL_REVISION") or "v7"  # Hub branch served by this demo; set to "v6" to roll back
GITHUB = "https://github.com/Harukaru1/vietnamese-toxic-comment-moderation"
REVIEW_BELOW = 0.6  # below this confidence, suggest a human look
# YouTube queues, on P(toxic) = 1 - P(clean). 0.8 was chosen on the validation split; on held-out comments
# it gave precision 0.62 / recall 0.36 for v6 (reports/threshold_v6.json). Between the two a human should decide.
HIDE_ABOVE, REVIEW_ABOVE = 0.8, 0.4
PROFANITY_HELP = ("Bộ lọc riêng, dò theo danh sách từ (vl, đm, đéo...). Mô hình coi từ tục dùng để nhấn mạnh "
                  "(\"hay vcl\") là bình thường; bật mục này nếu kênh của bạn không chấp nhận từ tục dù với ý gì.")

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
    predictor = Predictor(revision=MODEL_REVISION)
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
    st.markdown(f"**{MODEL_CARD['version']}**, fine-tune trên khoảng 52 nghìn bình luận tiếng Việt, rồi tinh chỉnh "
                "thêm trên 1.700 bình luận YouTube năm 2026 gán theo định nghĩa hiện tại (chọn bản tốt nhất trong 3 lần huấn luyện).")
    m1, m2 = st.columns(2)
    m1.metric("Báo đúng", f"{MODEL_CARD['precision_toxic']:.0%}",
              help="Precision: trong các bình luận mô hình báo độc hại, bao nhiêu phần trăm độc hại thật.")
    m2.metric("Bắt được", f"{MODEL_CARD['recall_toxic']:.0%}",
              help="Recall: trong các bình luận độc hại thật, mô hình bắt được bao nhiêu phần trăm.")
    st.caption("Đo trên 968 bình luận YouTube năm 2026 mà mô hình chưa từng thấy, quy đổi về tỉ lệ bình luận "
               "thực tế (khoảng 5% là độc hại). Vì vậy hãy xem kết quả là gợi ý cho người kiểm duyệt.")
    with st.expander("Mô hình còn hạn chế gì?"):
        st.markdown(
            "- Chỉ đọc **từng bình luận riêng lẻ**, không thấy ngữ cảnh: \"bạn mập quá đấy\" có thể là "
            "trêu đùa hoặc chê bai.\n"
            "- Dễ bỏ sót **thù ghét ngầm** không có từ chửi (mỉa mai, biệt danh miệt thị mới).\n"
            "- Ranh giới *công kích* / *thù ghét* vốn chủ quan; con người cũng không luôn đồng ý.\n"
            "- Phân biệt *thù ghét* với *công kích* còn yếu: mô hình mới học định nghĩa hiện tại từ ít ví dụ.\n\n"
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

single, paragraph, batch, video, history = st.tabs(["💬 Một bình luận", "📝 Đoạn văn", "📋 Nhiều bình luận", "📺 Video YouTube", "🕘 Lịch sử & góp ý"])

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
        ss.batch["profanity"] = [", ".join(find_profanity(t)) for t in texts]

    if "batch" in ss and len(ss.batch):
        df = ss.batch
        flag_swear = st.toggle("Đánh dấu cả bình luận có từ tục", key="batch_swear", help=PROFANITY_HELP)
        counts = df["label"].value_counts()
        cols = st.columns(4)
        for col, lab in zip(cols, LABELS):
            n = int(counts.get(lab, 0))
            col.metric(f'{INFO[lab]["icon"]} {INFO[lab]["name"]}', f"{n} ({n / len(df):.0%})")
        cols[3].metric("🤔 Cần xem lại", int((df["confidence"] < REVIEW_BELOW).sum()),
                       help=f"Độ tin cậy dưới {REVIEW_BELOW:.0%}: nên để người kiểm duyệt xem lại.")

        options = ["Tất cả"] + LABELS + ["Cần xem lại"] + (["Có từ tục"] if flag_swear else [])
        show = st.segmented_control("Lọc theo nhãn", options, default="Tất cả",
                                    format_func=lambda x: x if x not in LABELS else INFO[x]["name"])
        view = df
        if show in LABELS:
            view = df[df["label"] == show]
        elif show == "Cần xem lại":
            view = df[df["confidence"] < REVIEW_BELOW]
        elif show == "Có từ tục":
            view = df[df["profanity"] != ""]
        if flag_swear:
            n_swear = int((df["profanity"] != "").sum())
            st.caption(f"🤬 {n_swear} bình luận có từ tục, trong đó {int(((df['profanity'] != '') & (df['label'] == 'clean')).sum())} "
                       "câu mô hình xếp là bình thường.")
        st.dataframe(
            view.assign(label=view["label"].map(lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}')),
            width="stretch", hide_index=True,
            column_order=["text", "label", "confidence"] + (["profanity"] if flag_swear else []),
            column_config={
                "text": st.column_config.TextColumn("Bình luận", width="large"),
                "label": st.column_config.TextColumn("Nhãn"),
                "confidence": st.column_config.ProgressColumn("Độ tin cậy", min_value=0, max_value=1, format="percent"),
                "profanity": st.column_config.TextColumn("Từ tục"),
            })
        st.download_button("⬇️ Tải kết quả (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="moderation_results.csv", mime="text/csv")

# ---------------------------------------------------------------- YouTube video dashboard
@st.cache_data(ttl=900, show_spinner=False)
def fetch_video(video_id: str, max_comments: int, order: str, _key: str) -> tuple[dict, list[dict]]:
    """Video title + public comments. Author names are never requested or stored."""
    from toxic_moderation.youtube import YouTubeClient
    yt = YouTubeClient(api_key=_key)
    info = yt.video_info([video_id])
    if not info:
        raise ValueError("Không tìm thấy video này (có thể đã bị xoá hoặc để riêng tư).")
    return info[0], list(yt.iter_comments(video_id, max_comments=max_comments, order=order))


with video:
    yt_key = secret("YOUTUBE_API_KEY")
    st.markdown("Dán link một video YouTube để xem **bức tranh kiểm duyệt** của phần bình luận: "
                "bao nhiêu phần trăm độc hại, câu nào nên ẩn, câu nào cần người xem lại.")
    if not yt_key:
        st.info("Tính năng này cần khoá YouTube Data API (`YOUTUBE_API_KEY` trong Streamlit secrets). "
                "Bản demo này chưa được cấu hình khoá.")
    else:
        url = st.text_input("Link video", placeholder="https://www.youtube.com/watch?v=...", key="yt_url")
        c1, c2 = st.columns(2)
        n_max = c1.select_slider("Số bình luận tối đa", [50, 100, 200, 300], value=100,
                                 help="Mô hình chạy trên CPU miễn phí nên 100 bình luận mất khoảng 20-30 giây.")
        order = c2.radio("Lấy bình luận", ["relevance", "time"], horizontal=True,
                         format_func=lambda o: "Nổi bật nhất" if o == "relevance" else "Mới nhất")
        if st.button("📺 Phân tích video", type="primary", disabled=not url.strip()):
            try:
                from toxic_moderation.youtube import extract_video_id
                with st.spinner("Đang lấy bình luận từ YouTube..."):
                    info, comments = fetch_video(extract_video_id(url), n_max, order, yt_key)
                from toxic_moderation.preprocess import has_text
                n_fetched = len(comments)
                comments = [c for c in comments if has_text(str(c["text"]))]
                ss.video_skipped = n_fetched - len(comments)
                if not comments:
                    st.warning("Video này không có bình luận công khai (hoặc đã tắt bình luận).")
                else:
                    bar = st.progress(0.0, text="Đang phân loại...")
                    preds = []
                    for k in range(0, len(comments), 32):
                        preds += predictor.predict([c["text"] for c in comments[k:k + 32]])
                        bar.progress(min(1.0, (k + 32) / len(comments)), text=f"Đã xong {len(preds)}/{len(comments)}")
                    bar.empty()
                    ss.video = dict(info=info, df=pd.DataFrame({
                        "text": [c["text"] for c in comments],
                        "likes": [c["like_count"] for c in comments],
                        "reply": [c["parent_id"] is not None for c in comments],
                        "label": [p["label"] for p in preds],
                        "confidence": [p["scores"][p["label"]] for p in preds],
                        "profanity": [", ".join(find_profanity(c["text"])) for c in comments],
                        **{f"p_{lab}": [p["scores"][lab] for p in preds] for lab in LABELS}}))
            except ValueError as e:
                st.error(str(e) if "Không tìm thấy" in str(e) else "Link không hợp lệ. Hãy dán link dạng youtube.com/watch?v=... hoặc youtu.be/...")
            except Exception as e:  # quota exceeded, comments disabled, network
                st.error(f"Không lấy được bình luận: {e}")

        if "video" in ss:
            info, df = ss.video["info"], ss.video["df"]
            st.subheader(info["title"])
            st.caption(f'{info["channel"]} · {info["view_count"]:,} lượt xem · {info["comment_count"]:,} bình luận '
                       f'· đã phân tích {len(df)} bình luận'
                       + (f' · bỏ qua {ss.video_skipped} bình luận chỉ có emoji' if ss.get("video_skipped") else ''))
            toxic = df["label"] != "clean"
            p_toxic = 1 - df["p_clean"]
            hide, review = p_toxic >= HIDE_ABOVE, (p_toxic >= REVIEW_ABOVE) & (p_toxic < HIDE_ABOVE)
            # Optional second stage: an LLM re-reads only the comments the classifier flagged.
            llm_key = secret("GEMINI_API_KEY")
            flagged = p_toxic >= REVIEW_ABOVE
            if llm_key and flagged.any():
                if "llm" not in df and st.button(f"🤖 Nhờ LLM xem lại {int(flagged.sum())} câu bị báo", key="yt_llm",
                                                 help="Chỉ gửi nội dung các câu bị báo (không có tên người viết) tới Gemini. "
                                                      "Trong thử nghiệm trên nhãn do người gán, bước này nâng tỉ lệ báo đúng từ 46% lên 69%."):
                    try:
                        from toxic_moderation.llm_check import review as llm_review
                        with st.spinner("LLM đang đọc lại các câu bị báo..."):
                            verdict = llm_review(df.loc[flagged, "text"].tolist(), api_key=llm_key)
                        df["llm"] = ""
                        df.loc[flagged, "llm"] = verdict
                        ss.video["df"] = df
                    except Exception as e:  # quota, network, malformed answer: keep the classifier-only view
                        st.warning(f"Chưa dùng được LLM lúc này ({e}). Kết quả bên dưới chỉ dựa trên mô hình PhoBERT.")
            if "llm" in df:
                confirmed = df["llm"].isin(["offensive", "hate"])
                hide, review = flagged & confirmed, flagged & ~confirmed & (p_toxic >= HIDE_ABOVE)
                st.success(f"LLM đã xem lại {int(flagged.sum())} câu: xác nhận {int(confirmed.sum())} câu độc hại, "
                           f"gỡ cờ {int((flagged & ~confirmed).sum())} câu. Câu PhoBERT rất chắc mà LLM không đồng ý "
                           "được đưa vào hàng \"Cần xem lại\".")
            cols = st.columns(4)
            cols[0].metric("Tỉ lệ độc hại", f"{toxic.mean():.0%}", help="Công kích + thù ghét, theo dự đoán của mô hình.")
            cols[1].metric(f'{INFO["hate"]["icon"]} {INFO["hate"]["name"]}', int((df["label"] == "hate").sum()))
            cols[2].metric(f'{INFO["offensive"]["icon"]} {INFO["offensive"]["name"]}', int((df["label"] == "offensive").sum()))
            cols[3].metric("🤔 Cần xem lại", int(review.sum()),
                           help=f"Mô hình cho khả năng độc hại từ {REVIEW_ABOVE:.0%} đến {HIDE_ABOVE:.0%}: "
                                "nên để người kiểm duyệt quyết định.")
            st.bar_chart(df["label"].map(lambda lab: INFO[lab]["name"]).value_counts(), horizontal=True, height=160)
            yt_swear = st.toggle("Đánh dấu cả bình luận có từ tục", key="yt_swear", help=PROFANITY_HELP)
            queues = ["hide", "review", "ok"] + (["swear"] if yt_swear else []) + ["all"]
            queue = st.segmented_control(
                "Hàng đợi kiểm duyệt", queues, default="hide", key="yt_queue",
                format_func={"hide": "🚫 Đề xuất ẩn", "review": "🤔 Cần xem lại", "ok": "✅ Để nguyên",
                             "swear": "🤬 Có từ tục", "all": "Tất cả"}.get)
            swear = df["profanity"] != ""
            view = {"hide": df[hide], "review": df[review], "ok": df[~hide & ~review], "swear": df[swear]}.get(queue, df)
            st.caption({"hide": f"Khả năng độc hại từ {HIDE_ABOVE:.0%} trở lên (ở mức này mô hình đúng khoảng 6 trên 10 lần). "
                                "Sắp theo lượt thích: câu nhiều người thấy nhất ở trên.",
                        "review": "Mô hình không chắc. Người kiểm duyệt nên đọc những câu này.",
                        "ok": "Mô hình khá chắc là bình thường.", "all": "Toàn bộ bình luận đã phân tích.",
                        "swear": f"{int(swear.sum())} bình luận có từ tục, trong đó {int((swear & ~hide & ~review).sum())} câu mô hình cho là bình thường. "
                                 "Bộ lọc này chỉ dò theo danh sách từ, không xét ý nghĩa của câu."}.get(queue, ""))
            st.dataframe(
                view.sort_values("likes", ascending=False)
                    .assign(label=lambda v: v["label"].map(lambda lab: f'{INFO[lab]["icon"]} {INFO[lab]["name"]}')),
                width="stretch", hide_index=True,
                column_order=["text", "label", "confidence"] + (["llm"] if "llm" in df else []) + ["likes"]
                + (["profanity"] if yt_swear else []),
                column_config={
                    "text": st.column_config.TextColumn("Bình luận", width="large"),
                    "label": st.column_config.TextColumn("Nhãn"),
                    "confidence": st.column_config.ProgressColumn("Độ tin cậy", min_value=0, max_value=1, format="percent"),
                    "likes": st.column_config.NumberColumn("👍"),
                    "llm": st.column_config.TextColumn("LLM"),
                    "profanity": st.column_config.TextColumn("Từ tục")})
            st.download_button("⬇️ Tải kết quả (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f'youtube_{info["video_id"]}_moderation.csv', mime="text/csv")
            st.caption("Chỉ lấy nội dung bình luận công khai, không lấy tên người viết. Mô hình có thể sai: "
                       "đây là công cụ gợi ý cho người kiểm duyệt, không tự động ẩn gì trên YouTube.")

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

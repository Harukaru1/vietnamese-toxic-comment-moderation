"""Second-stage check: an LLM re-reads the comments the classifier flagged.

PhoBERT is cheap and reads every comment; an LLM is slower and costs money but follows the guideline better.
So only the small share of comments the classifier flags (about 6% of traffic) is sent to the LLM, which
confirms or overrules the flag. On 111 comments flagged by PhoBERT v7 and labeled blind by the author, the
classifier alone was right 46% of the time; where Gemini Flash-Lite agreed, 69% (84% of the truly toxic ones
were kept). See reports/llm_second_stage.json and scripts/eval_llm_stage.py.

Uses the Gemini REST API. The key is read from the argument or the GEMINI_API_KEY environment variable and is
sent in the `x-goog-api-key` header. Comment texts are sent to Google; no author names or ids are included.
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# The lite model answered reliably on the free tier when the larger flash models returned 503.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
LABELS = ("clean", "offensive", "hate")

GUIDELINE = """Bạn là người kiểm duyệt bình luận tiếng Việt trên mạng xã hội. Gán cho mỗi bình luận một nhãn:

clean: không xúc phạm ai. Khen, chê, góp ý, tranh luận, mỉa mai đều là clean. Từ tục trong câu vui, khen,
ngạc nhiên hay đùa ("hay vcl", "đỉnh vãi") là clean. Đòi xử người làm sai theo pháp luật bằng lời thông thường
("quá mất dạy, phải xử thật nặng", "tử hình là xứng đáng") là clean. Chỉ nhắc lại một từ miệt thị là clean.

offensive: mắng một người ("m ngu", "dở hơi à", "ngáo à"); từ tục viết rõ trong câu bực bội, cáu giận
("đéo hiểu nổi", "kêu ca éo gì"); nói xấu một nước, vùng miền, giới tính, dân tộc bằng "bọn/tụi/lũ" với lời
lẽ thường ("bọn Hàn hay thượng đẳng"); dùng từ miệt thị chỉ để gọi tên nhóm trong câu nhẹ ("lũ ricon gào
thét"); chỉ trích người làm sai rồi chồng thêm lăng mạ cá nhân; nói tục về chuyện bài tiết.

hate: miệt thị nặng vì người ta là ai (từ miệt thị kèm công kích, chê cơ thể hay ngoại hình); ví người với
con vật ("sủa", "súc vật", "óc chó"); lấy khuyết tật ra chửi ("thiểu năng", "Down"); chửi gia đình người
khác; hạ nhục tình dục; đe doạ, trù chết, đòi giết ngoài pháp luật.

Xét theo nghĩa của cả câu, không xét theo từng từ riêng lẻ. Khi phân vân giữa hai mức, chọn mức nhẹ hơn."""


class LLMError(RuntimeError):
    pass


def _prompt(texts: list[str]) -> str:
    items = "\n".join(f"{i + 1}. {' '.join(str(t).split())[:500]}" for i, t in enumerate(texts))
    return (f"{GUIDELINE}\n\nCác bình luận:\n{items}\n\n"
            f'Trả lời bằng một mảng JSON gồm đúng {len(texts)} phần tử theo thứ tự, mỗi phần tử là '
            '"clean", "offensive" hoặc "hate". Không viết gì khác.')


def parse_labels(reply: str, n: int) -> list[str]:
    """Extract n labels from the model's reply; raise LLMError if the reply does not contain exactly n."""
    m = re.search(r"\[.*\]", reply, flags=re.S)
    try:
        labels = [str(x).strip().lower() for x in json.loads(m.group(0))] if m else []
    except json.JSONDecodeError:
        labels = []
    if len(labels) != n or any(lab not in LABELS for lab in labels):
        raise LLMError(f"Unexpected reply for {n} comments: {reply[:200]!r}")
    return labels


def review(texts: list[str], api_key: str | None = None, model: str = DEFAULT_MODEL, batch_size: int = 20,
           timeout: float = 60, session: requests.Session | None = None, retries: int = 3,
           backoff: float = 2.0) -> list[str]:
    """Label each comment with the LLM. Raises LLMError on an API error or a malformed reply."""
    http = session or requests.Session()
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key or os.getenv("GEMINI_API_KEY", "")}
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body = {"contents": [{"parts": [{"text": _prompt(batch)}]}], "generationConfig": {"temperature": 0}}
        for attempt in range(retries + 1):
            r = http.post(API.format(model=model), headers=headers, json=body, timeout=timeout)
            try:
                data = r.json()
            except ValueError:  # a gateway in between answered with plain text (502/504)
                data = {"error": {"code": getattr(r, "status_code", 502), "message": str(getattr(r, "text", ""))[:200]}}
            busy = "error" in data and data["error"].get("code") in (429, 500, 502, 503, 504)
            if not busy or attempt == retries:
                break
            time.sleep(backoff * 2 ** attempt)  # the free tier is often briefly overloaded
        if "error" in data:
            raise LLMError(f"{data['error'].get('code')}: {data['error'].get('message')}")
        try:
            reply = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:  # blocked or empty answer
            raise LLMError(f"No answer from the model: {json.dumps(data)[:200]}") from e
        out += parse_labels(reply, len(batch))
    return out

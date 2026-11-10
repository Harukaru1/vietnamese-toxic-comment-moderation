"""Minimal YouTube Data API v3 client (search videos, fetch comments).

The API key is read from the YOUTUBE_API_KEY environment variable (or Streamlit secrets in the
app). If it is not set, requests are sent without a key, which works when a proxy injects the
`X-Goog-Api-Key` header.

Quota (default 10,000 units/day): search = 100 units, videos/commentThreads = 1 unit per call.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

import requests

API = "https://www.googleapis.com/youtube/v3"
_VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")


class YouTubeError(RuntimeError):
    pass


def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL (watch, youtu.be, shorts, live) or a bare 11-char video id."""
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID.search(s)
    if not m:
        raise ValueError(f"Not a YouTube video URL or id: {url_or_id!r}")
    return m.group(1)


class YouTubeClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20):
        self.session = requests.Session()
        key = api_key or os.getenv("YOUTUBE_API_KEY")
        if key:
            self.session.headers["X-Goog-Api-Key"] = key
        self.timeout = timeout

    def _get(self, endpoint: str, **params) -> dict:
        r = self.session.get(f"{API}/{endpoint}", params=params, timeout=self.timeout)
        data = r.json()
        if "error" in data:
            err = data["error"]
            raise YouTubeError(f"{err.get('code')}: {err.get('message')}")
        return data

    def search_videos(self, query: str, max_results: int = 10, published_after: str | None = None,
                      order: str = "relevance") -> list[str]:
        params = dict(part="id", q=query, type="video", maxResults=max_results, order=order,
                      regionCode="VN", relevanceLanguage="vi")
        if published_after:
            params["publishedAfter"] = published_after
        return [it["id"]["videoId"] for it in self._get("search", **params)["items"]]

    def video_info(self, video_ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(video_ids), 50):
            data = self._get("videos", part="snippet,statistics", id=",".join(video_ids[i:i + 50]))
            for v in data["items"]:
                out.append({
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel": v["snippet"]["channelTitle"],
                    "published_at": v["snippet"]["publishedAt"],
                    "view_count": int(v["statistics"].get("viewCount", 0)),
                    "comment_count": int(v["statistics"].get("commentCount", 0)),
                })
        return out

    def iter_comments(self, video_id: str, max_comments: int = 500, order: str = "relevance",
                      include_replies: bool = True) -> Iterator[dict]:
        """Yield top-level comments (and the replies returned with them), newest API fields only."""
        n, token = 0, None
        while n < max_comments:
            params = dict(part="snippet,replies", videoId=video_id, maxResults=100, order=order,
                          textFormat="plainText")
            if token:
                params["pageToken"] = token
            data = self._get("commentThreads", **params)
            for th in data.get("items", []):
                top = th["snippet"]["topLevelComment"]
                items = [(top, None)]
                if include_replies:
                    items += [(r, top["id"]) for r in th.get("replies", {}).get("comments", [])]
                for c, parent in items:
                    s = c["snippet"]
                    yield {
                        "comment_id": c["id"],
                        "parent_id": parent,
                        "video_id": video_id,
                        "text": s.get("textOriginal") or s.get("textDisplay", ""),
                        "like_count": s.get("likeCount", 0),
                        "published_at": s.get("publishedAt"),
                    }
                    n += 1
                    if n >= max_comments:
                        return
            token = data.get("nextPageToken")
            if not token:
                return

"""Shared, anonymous store for demo feedback (👍/👎 + corrected label).

Each feedback event is appended as one JSON line to a local file; a background
`huggingface_hub.CommitScheduler` pushes the folder to a **private** HF dataset every few minutes.
The log is append-only: if a user changes their vote, a new line is written with the same
(session, item) key, and readers keep the last line per key (see `latest`).

No IP address, user name or other identifier is stored, only a random per-browser-session id.
Without a token (`HF_TOKEN`) the store is disabled and the demo works exactly as before.
"""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FIELDS = ["ts", "session", "item", "text", "model_label", "p_clean", "p_offensive", "p_hate",
          "feedback", "correct", "model_revision", "model_sha"]


def make_record(*, session: str, item: int, text: str, model_label: str, scores: dict[str, float],
                feedback: str, correct: str | None, model_revision: str, model_sha: str) -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session": session, "item": item,
            "text": text, "model_label": model_label,
            **{f"p_{lab}": round(float(scores.get(lab, 0.0)), 4) for lab in ("clean", "offensive", "hate")},
            "feedback": feedback, "correct": correct, "model_revision": model_revision, "model_sha": model_sha}


class FeedbackStore:
    """Append feedback lines locally; optionally sync them to a private HF dataset."""

    def __init__(self, repo_id: str | None = None, token: str | None = None, every_minutes: float = 2,
                 folder: Path | None = None) -> None:
        self.folder = Path(folder or tempfile.mkdtemp(prefix="feedback_"))
        self.folder.mkdir(parents=True, exist_ok=True)
        # one file per app process, so two replicas never write the same file in the repo
        self.path = self.folder / f"feedback-{uuid.uuid4().hex[:12]}.jsonl"
        self.repo_id = repo_id
        self._lock = threading.Lock()
        self.scheduler = None
        if repo_id and token:
            from huggingface_hub import CommitScheduler
            self.scheduler = CommitScheduler(repo_id=repo_id, repo_type="dataset", folder_path=self.folder,
                                             path_in_repo="data", every=every_minutes, token=token,
                                             private=True, allow_patterns=["*.jsonl"])
            self._lock = self.scheduler.lock  # don't append while a commit is reading the file

    @property
    def enabled(self) -> bool:
        return self.scheduler is not None

    def add(self, record: dict) -> None:
        line = json.dumps({k: record.get(k) for k in FIELDS}, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def flush(self) -> None:
        """Push now instead of waiting for the next scheduled commit."""
        if self.scheduler is not None:
            self.scheduler.trigger().result()


def latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each user's final vote per item (the log is append-only)."""
    return df.sort_values("ts").drop_duplicates(["session", "item"], keep="last").reset_index(drop=True)


def load_feedback(repo_id: str, token: str | None = None) -> pd.DataFrame:
    """Download every feedback file from the dataset and return the final votes."""
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(repo_id, repo_type="dataset", token=token, allow_patterns=["data/*.jsonl"]))
    files = sorted(local.glob("data/*.jsonl"))
    if not files:
        return pd.DataFrame(columns=FIELDS)
    df = pd.concat([pd.read_json(p, lines=True, dtype={"session": str}) for p in files], ignore_index=True)
    return latest(df)

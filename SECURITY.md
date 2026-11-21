# Security

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's
[private vulnerability reporting](https://github.com/Harukaru1/vietnamese-toxic-comment-moderation/security/advisories/new)
instead. Expect a reply within a week.

## What the demo does to stay safe

| Risk | Measure | Code |
|---|---|---|
| HTML/script injection through comments | Comment text is escaped before it is placed in custom HTML; everything else goes through Streamlit widgets | `streamlit_app/streamlit_app.py` |
| Formula injection in downloaded CSV files | Cells and headers starting with `=`, `+`, `-`, `@`, tab or CR get a leading `'` | `src/toxic_moderation/safety.py` |
| Prompt injection against the LLM stage | Each comment is sent as a JSON string, the prompt says comment content is data only, and the answer must be exactly one valid label per comment | `src/toxic_moderation/llm_check.py` |
| Abuse of the owner's API quotas | Per-session caps: 10 video analyses, 5 LLM re-checks, 100 feedback records | `streamlit_app/streamlit_app.py` |
| Oversized input | Text boxes are length-limited, inputs are cut at 2,000 characters, uploads at 2 MB and 200 rows | `.streamlit/config.toml`, `safety.py` |
| Leaking keys or internals | API keys live in Streamlit secrets and are sent in request headers, never in URLs; visitors see generic errors, not tracebacks; `secrets.toml` is git-ignored | `youtube.py`, `llm_check.py`, `.gitignore` |
| Privacy of feedback | Feedback is anonymous (random session id, no name or IP), opt-out, stored in a private dataset and reviewed before any use | `src/toxic_moderation/feedback_store.py` |
| Vulnerable dependencies | `pip-audit` runs in CI on every push; Dependabot proposes updates weekly | `.github/workflows/ci.yml`, `.github/dependabot.yml` |

Owners of a deployment should also restrict each Google API key to the one API it serves and set a daily quota
in Google Cloud, so a leaked key cannot be used for anything else or run up costs.

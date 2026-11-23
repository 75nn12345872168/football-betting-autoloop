"""
x_publisher.py — Automated Weekly Transparency Publisher for X (Twitter) API v2.

Security & Reliability Guarantees:
  * Reads OAuth 1.0a credentials EXCLUSIVELY from environment variables:
      X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
    (never hardcoded, never echoed to logs).
  * Uses requests-oauthlib (`OAuth1Session`) -> POST https://api.twitter.com/2/tweets.
  * Fail-Safe: missing credentials, missing package, or HTTP/network errors log a
    clean notice and skip publishing without interrupting autoloop_master.py.
  * Strict Idempotency: persists published matchweeks (`matchweek_id`, `tweet_id`,
    `posted_at`) in `data/archive/x_posted_weeks.json` so no matchweek is ever posted twice.
  * Publishes ONLY completed matchweeks marked `archive_status == "CLOSED_ARCHIVED"`
    in `data/archive/weekly_archive.json`, strictly <= 280 characters.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from requests_oauthlib import OAuth1Session as _DefaultOAuth1Session
except Exception:
    _DefaultOAuth1Session = None

REPO_DIR = Path(__file__).resolve().parent
WEEKLY_ARCHIVE_PATH = REPO_DIR / "data" / "archive" / "weekly_archive.json"
X_POSTED_STATE_PATH = REPO_DIR / "data" / "archive" / "x_posted_weeks.json"
X_TWEETS_ENDPOINT = "https://api.twitter.com/2/tweets"
TERMINAL_URL = "https://alpha-pitch-terminal.vercel.app"
MAX_TWEET_CHARS = 280


def load_posted_weeks(state_path: Path = X_POSTED_STATE_PATH) -> Dict[str, Any]:
    """Load idempotency state tracking already-published matchweeks on X."""
    default: Dict[str, Any] = {"updated_at": None, "posted": {}}
    if not state_path or not state_path.exists():
        return default
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("posted"), dict):
            return raw
    except Exception:
        pass
    return default


def save_posted_weeks(state: Dict[str, Any], state_path: Path = X_POSTED_STATE_PATH) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def find_next_unposted_closed_week(
    archive_path: Path = WEEKLY_ARCHIVE_PATH,
    state_path: Path = X_POSTED_STATE_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Return the latest matchweek in `weekly_archive.json` whose status is
    `CLOSED_ARCHIVED` and whose `matchweek_id` is not yet in `x_posted_weeks.json`.
    """
    if not archive_path or not archive_path.exists():
        return None
    try:
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    weeks = archive.get("weeks")
    if not isinstance(weeks, list) or not weeks:
        return None

    state = load_posted_weeks(state_path)
    posted_ids = set(state.get("posted", {}).keys())

    closed = [
        wk
        for wk in weeks
        if isinstance(wk, dict)
        and str(wk.get("archive_status", "")).upper() == "CLOSED_ARCHIVED"
        and wk.get("matchweek_id")
        and str(wk["matchweek_id"]) not in posted_ids
    ]
    if not closed:
        return None

    # Sort newest-first by window_start and pick the latest unposted closed matchweek
    closed.sort(key=lambda w: str(w.get("window_start", "")), reverse=True)
    return closed[0]


def _fmt_signed_pct(val: Any) -> str:
    if val is None or val == "":
        return "N/A"
    try:
        f = float(val)
        return f"{f:+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_signed_units(val: Any) -> str:
    if val is None or val == "":
        return "0.00u"
    try:
        f = float(val)
        return f"{f:+.2f}u"
    except (TypeError, ValueError):
        return "0.00u"


def _compact_matchweek_label(week: Dict[str, Any]) -> str:
    ws = str(week.get("window_start", "")).strip()
    we = str(week.get("window_end", "")).strip()
    if len(ws) == 10 and len(we) == 10:
        if ws[:4] == we[:4]:
            return f"{ws}→{we[5:]}"
        return f"{ws}→{we}"
    return str(week.get("label") or week.get("matchweek_id") or "Matchweek").strip()


def format_weekly_audit_tweet(
    week: Dict[str, Any],
    terminal_url: str = TERMINAL_URL,
) -> str:
    """
    Build the institutional weekly transparency audit tweet (strictly <= 280 chars).
    """
    label_full = str(week.get("label") or _compact_matchweek_label(week)).strip()
    label_short = _compact_matchweek_label(week)
    wins = int(week.get("wins", 0) or 0)
    half_wins = int(week.get("half_wins", 0) or 0)
    losses = int(week.get("losses", 0) or 0)
    min_odds_roi = _fmt_signed_pct(week.get("conservative_min_odds_roi_pct"))
    flat_1u = _fmt_signed_units(week.get("flat_net_units"))
    closing_roi = _fmt_signed_pct(week.get("closing_odds_roi_pct"))

    def _build(lbl: str, cta: str, tags: str) -> str:
        return (
            f"📊 AlphaPitch v4 · Weekly Audit ({lbl})\n"
            f"Status: CLOSED_ARCHIVED 🔒\n\n"
            f"Record: {wins}W - {half_wins}HW - {losses}L\n"
            f"Min-Odds ROI: {min_odds_roi}\n"
            f"Flat 1u P&L: {flat_1u}\n"
            f"Closing CLV: {closing_roi}\n\n"
            f"{cta}\n"
            f"{terminal_url}\n\n"
            f"{tags}"
        )

    candidates = [
        _build(
            label_full,
            "Full match-by-match ledger & live +EV signals 👇",
            "#SportsQuant #PlusEV #FootballAnalytics",
        ),
        _build(
            label_short,
            "Full ledger & live +EV signals 👇",
            "#SportsQuant #PlusEV #FootballAnalytics",
        ),
        _build(
            label_short,
            "Full ledger & +EV signals 👇",
            "#SportsQuant #PlusEV #FootballAnalytics",
        ),
        _build(
            label_short,
            "Ledger & +EV signals 👇",
            "#SportsQuant #PlusEV",
        ),
    ]
    for tweet in candidates:
        if len(tweet) <= MAX_TWEET_CHARS:
            return tweet
    return candidates[-1][:MAX_TWEET_CHARS]


def _read_x_credentials() -> Optional[Dict[str, str]]:
    api_key = os.getenv("X_API_KEY", "").strip()
    api_secret = os.getenv("X_API_SECRET", "").strip()
    access_token = os.getenv("X_ACCESS_TOKEN", "").strip()
    access_secret = os.getenv("X_ACCESS_TOKEN_SECRET", "").strip()
    if not all([api_key, api_secret, access_token, access_secret]):
        return None
    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "access_token": access_token,
        "access_secret": access_secret,
    }


def post_tweet(
    text: str,
    oauth_session_cls: Any = None,
    timeout: int = 20,
) -> Optional[str]:
    """
    Publish `text` to X API v2 (`POST https://api.twitter.com/2/tweets`) using OAuth 1.0a.
    Fail-Safe: returns `tweet_id` on success (200/201), or `None` on missing credentials/errors.
    """
    creds = _read_x_credentials()
    if creds is None:
        print("[X PUBLISHER] X API credentials not configured in environment; skipping safely.")
        return None

    session_cls = oauth_session_cls or _DefaultOAuth1Session
    if session_cls is None:
        print("[X PUBLISHER WARN] requests-oauthlib is not installed; skipping safely.")
        return None

    try:
        oauth = session_cls(
            creds["api_key"],
            client_secret=creds["api_secret"],
            resource_owner_key=creds["access_token"],
            resource_owner_secret=creds["access_secret"],
        )
        resp = oauth.post(
            X_TWEETS_ENDPOINT,
            json={"text": text},
            timeout=timeout,
        )
        if resp.status_code in (200, 201):
            payload = resp.json() if hasattr(resp, "json") else {}
            tweet_id = str((payload.get("data") or {}).get("id") or "").strip()
            if tweet_id:
                print(f"[X PUBLISHER] Published weekly audit tweet successfully (tweet_id={tweet_id}).")
                return tweet_id
            print("[X PUBLISHER WARN] HTTP 201 received but missing data.id in response.")
            return None

        err_preview = str(getattr(resp, "text", ""))[:200]
        for secret_val in creds.values():
            if secret_val:
                err_preview = err_preview.replace(secret_val, "***")
        print(f"[X PUBLISHER WARN] HTTP {resp.status_code} from X API: {err_preview} (fail-safe skip)")
        return None
    except Exception as exc:
        safe_err = str(exc)
        for secret_val in creds.values():
            if secret_val:
                safe_err = safe_err.replace(secret_val, "***")
        print(f"[X PUBLISHER WARN] Exception while posting to X ({safe_err[:150]}) — continuing fail-safe.")
        return None


def publish_latest_closed_week(
    archive_path: Path = WEEKLY_ARCHIVE_PATH,
    state_path: Path = X_POSTED_STATE_PATH,
    oauth_session_cls: Any = None,
) -> Dict[str, Any]:
    """
    Check `weekly_archive.json` for the latest unposted `CLOSED_ARCHIVED` matchweek,
    format the <= 280 char audit tweet, publish to X, and record `tweet_id` in
    `x_posted_weeks.json` for strict idempotency.
    """
    week = find_next_unposted_closed_week(archive_path=archive_path, state_path=state_path)
    if not week:
        print("[X PUBLISHER] No new CLOSED_ARCHIVED matchweek pending X publication.")
        return {
            "posted": False,
            "reason": "no_unposted_closed_week",
            "matchweek_id": None,
            "tweet_id": None,
            "text": None,
        }

    mw_id = str(week["matchweek_id"])
    tweet_text = format_weekly_audit_tweet(week)
    tweet_id = post_tweet(tweet_text, oauth_session_cls=oauth_session_cls)
    if not tweet_id:
        return {
            "posted": False,
            "reason": "post_skipped_or_failed",
            "matchweek_id": mw_id,
            "tweet_id": None,
            "text": tweet_text,
        }

    state = load_posted_weeks(state_path)
    state.setdefault("posted", {})[mw_id] = {
        "matchweek_id": mw_id,
        "label": str(week.get("label", "")),
        "tweet_id": tweet_id,
        "posted_at": datetime.now(timezone.utc).isoformat(),
        "text": tweet_text,
    }
    save_posted_weeks(state, state_path)
    return {
        "posted": True,
        "reason": "published",
        "matchweek_id": mw_id,
        "tweet_id": tweet_id,
        "text": tweet_text,
    }


if __name__ == "__main__":
    result = publish_latest_closed_week()
    print(json.dumps(result, ensure_ascii=False, indent=2))

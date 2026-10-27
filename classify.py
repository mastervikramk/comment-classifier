#!/usr/bin/env python3
"""
Forum Comment Classifier & HTML Report Generator
------------------------------------------------
Production-grade moderation pipeline using Cloudflare Workers AI (Clef model).

Features:
- Academic forum-tailored prompt criteria (separating work criticism from personal attacks)
- Concurrent worker pool with exponential backoff & jitter for rate-limiting
- Content-addressable cache (auto-invalidates when prompt or model changes)
- Fast-fail on authentication errors (HTTP 401/403)
- Modern responsive HTML dashboard with Light/Dark mode, KPI metrics, live search,
  sorting, and a 1-click 'Copy for MISTAKES.md' tool.

Usage:
    python classify.py                      # uses `clef`, caches results
    python classify.py --model clef-flash   # uses `clef-flash`
    python classify.py --limit 5            # quick smoke test on 5 comments
    python classify.py --force              # ignore cache, re-classify all
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("comment-classifier")

# ==============================================================================
# Constants & Moderation Prompt
# ==============================================================================
LABELS = ["acceptable", "toxic", "spam"]
DEFAULT_WORKERS = 4
MAX_RETRIES = 5
LOW_CONFIDENCE_THRESHOLD = 0.55

# High-precision prompt tuned for academic course forums
QUESTION_CONFIG = {
    "type": "choice",
    "instructions": (
        "`comment` is a post from an online forum for a college course on "
        "using AI coding tools (Claude, Codex, etc.), written by a student, TA "
        "or instructor. Classify it into exactly one category. Judge the "
        "content of the comment itself, not the topic: a comment can be about "
        "an uncomfortable subject (cheating, grading, a classmate's bug) and "
        "still be acceptable."
    ),
    "criteria": {
        "acceptable": (
            "A normal forum post: a question, answer, tip, feedback, "
            "complaint, opinion or disagreement about the course, the "
            "assignment, grading, AI tools or someone's project. Blunt or "
            "critical feedback about WORK (a bug, a design, a policy) is "
            "acceptable. Thanks, jokes among peers, and sharing one's own "
            "project are acceptable."
        ),
        "toxic": (
            "Abusive or hostile toward a PERSON or GROUP: insults, name "
            "calling, mockery, personal attacks, threats, harassment, "
            "demeaning stereotypes about gender, region, branch of study, "
            "etc., or telling someone to get lost or quit. Pick this even if "
            "the comment also contains a valid point."
        ),
        "spam": (
            "Unsolicited promotion or scam content unrelated to genuine "
            "discussion: ads, assignment/solution/notes selling, leaked "
            "papers, referral or giveaway links, get-rich-quick or job "
            "offers, follower/crypto schemes, requests to DM, WhatsApp, "
            "Telegram or pay for something."
        ),
    },
}


# ==============================================================================
# Domain Models
# ==============================================================================
@dataclasses.dataclass
class Comment:
    id: int
    author: str
    thread: str
    posted_at: str
    comment: str


@dataclasses.dataclass
class Classification:
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    raw_response: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ClassifiedComment:
    comment: Comment
    classification: Classification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.comment.id,
            "author": self.comment.author,
            "thread": self.comment.thread,
            "posted_at": self.comment.posted_at,
            "comment": self.comment.comment,
            "classification": {
                "choice": self.classification.choice,
                "confidence": self.classification.confidence,
                "probabilities": self.classification.probabilities,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassifiedComment":
        comment = Comment(
            id=data["id"],
            author=data["author"],
            thread=data["thread"],
            posted_at=data["posted_at"],
            comment=data["comment"],
        )
        cls_data = data["classification"]
        classification = Classification(
            choice=cls_data["choice"],
            confidence=cls_data["confidence"],
            probabilities=cls_data["probabilities"],
        )
        return cls(comment=comment, classification=classification)


# ==============================================================================
# Exceptions
# ==============================================================================
class FatalAPIError(RuntimeError):
    """Errors where retrying won't help (bad token, invalid account, 401/403)."""


# ==============================================================================
# Credentials Management
# ==============================================================================
def load_credentials(creds_path: Path) -> Tuple[str, str]:
    """
    Loads Cloudflare credentials from environment variables or a local shell file.
    Does not expose sensitive values in logs.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        if creds_path.is_file():
            try:
                content = creds_path.read_text(encoding="utf-8")
                matches = dict(
                    re.findall(r'export\s+(\w+)=["\']?([^"\'\n]*)["\']?', content)
                )
                account_id = account_id or matches.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
                api_token = api_token or matches.get("CLOUDFLARE_API_TOKEN", "").strip()
            except Exception as e:
                logger.error(f"Failed to read credentials from {creds_path}: {e}")
        else:
            logger.warning(f"Credentials file not found at {creds_path}")

    if not account_id or not api_token:
        raise FatalAPIError(
            "Missing credentials. Copy cloud-flare.sh to cloud-flare.local.sh and fill in "
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN."
        )

    return account_id, api_token


# ==============================================================================
# Cloudflare Clef API Client
# ==============================================================================
class ClefClassifierClient:
    """Client for Cloudflare Workers AI decision model with retry & backoff."""

    BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/{model}"

    def __init__(self, account_id: str, api_token: str, model: str = "clef"):
        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self.endpoint = self.BASE_URL.format(account_id=account_id, model=model)

    def classify_comment(self, comment_text: str, max_retries: int = MAX_RETRIES) -> Classification:
        payload = {
            "state": {"comment": comment_text},
            "questions": {"moderation": QUESTION_CONFIG},
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(self.endpoint, data=body, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=45) as response:
                    raw_body = response.read().decode("utf-8")
                    data = json.loads(raw_body)

                if not data.get("success", False):
                    errors = data.get("errors", [])
                    raise FatalAPIError(f"Cloudflare API failure: {errors}")

                answer = data["result"]["answers"]["moderation"]
                choice = answer["choice"]
                if choice not in LABELS:
                    raise FatalAPIError(f"Unexpected label returned from model: {choice!r}")

                return Classification(
                    choice=choice,
                    confidence=float(answer.get("confidence", 0.0)),
                    probabilities={k: float(v) for k, v in answer.get("probabilities", {}).items()},
                    raw_response=data,
                )

            except urllib.error.HTTPError as err:
                if err.code in (401, 403):
                    raise FatalAPIError(
                        f"HTTP {err.code}: Authentication failed. Verify CLOUDFLARE_ACCOUNT_ID and API token permissions."
                    ) from err

                if err.code == 429 or err.code >= 500:
                    last_error = err
                    sleep_duration = (2 ** attempt) + random.uniform(0.2, 0.8)
                    logger.warning(f"HTTP {err.code} on attempt {attempt}/{max_retries}. Backing off for {sleep_duration:.2f}s...")
                    time.sleep(sleep_duration)
                    continue

                error_msg = err.read().decode("utf-8", errors="replace")[:300]
                raise FatalAPIError(f"HTTP {err.code} Error: {error_msg}") from err

            except (urllib.error.URLError, TimeoutError) as err:
                last_error = err
                if attempt < max_retries:
                    sleep_duration = (2 ** attempt) + random.uniform(0.2, 0.8)
                    time.sleep(sleep_duration)
                    continue
                break

        raise RuntimeError(f"Gave up after {max_retries} attempts: {last_error}")


# ==============================================================================
# Fingerprint & Cache Persistence
# ==============================================================================
def compute_prompt_fingerprint(model: str) -> str:
    """Computes a SHA256 fingerprint of the prompt criteria and model."""
    blob = json.dumps({"q": QUESTION_CONFIG, "m": model}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def load_cached_results(cache_path: Path, expected_fingerprint: str) -> Dict[int, ClassifiedComment]:
    """Loads previously classified comments if the fingerprint matches."""
    if not cache_path.is_file():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("fingerprint") != expected_fingerprint:
                logger.info("Prompt or model has changed; invalidating old cache.")
                return {}
            items = data.get("results", [])
        elif isinstance(data, list):
            items = data
        else:
            return {}

        results = {}
        for item in items:
            classified = ClassifiedComment.from_dict(item)
            results[classified.comment.id] = classified
        logger.info(f"Loaded {len(results)} cached classifications (fingerprint: {expected_fingerprint})")
        return results
    except Exception as e:
        logger.warning(f"Could not load cache from {cache_path}: {e}")
        return {}


def save_cached_results(
    cache_path: Path,
    results: List[ClassifiedComment],
    model: str,
    fingerprint: str,
) -> None:
    """Persists current classifications to JSON cache with fingerprint."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": fingerprint,
            "model": model,
            "total": len(results),
            "results": [item.to_dict() for item in results],
        }
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to persist cache: {e}")


# ==============================================================================
# HTML Dashboard & Report Generator
# ==============================================================================
def generate_html_report(
    classified_comments: List[ClassifiedComment],
    output_path: Path,
    model_name: str,
) -> None:
    """
    Renders a responsive, clean, and interactive HTML moderation dashboard.
    Supports light and dark modes, KPI summary metrics, live search, and sorting.
    """
    total = len(classified_comments)
    counts = Counter(c.classification.choice for c in classified_comments)
    acceptable_count = counts.get("acceptable", 0)
    toxic_count = counts.get("toxic", 0)
    spam_count = counts.get("spam", 0)

    # Low confidence or borderline entries
    borderline_count = 0
    for c in classified_comments:
        probs = sorted(c.classification.probabilities.values(), reverse=True)
        margin = probs[0] - probs[1] if len(probs) > 1 else 1.0
        if c.classification.confidence < LOW_CONFIDENCE_THRESHOLD or margin < 0.25:
            borderline_count += 1

    items_json = json.dumps([item.to_dict() for item in classified_comments])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Forum Moderation Dashboard | Cloudflare Clef</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-canvas: #ffffff;
      --bg-surface: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-subtle: #94a3b8;
      --border-color: #e2e8f0;
      --border-focus: #3b82f6;

      --accent-acceptable: #10b981;
      --accent-acceptable-bg: #ecfdf5;
      --accent-acceptable-border: #a7f3d0;
      --accent-acceptable-text: #065f46;

      --accent-toxic: #ef4444;
      --accent-toxic-bg: #fef2f2;
      --accent-toxic-border: #fecaca;
      --accent-toxic-text: #991b1b;

      --accent-spam: #f59e0b;
      --accent-spam-bg: #fffbeb;
      --accent-spam-border: #fde68a;
      --accent-spam-text: #92400e;

      --accent-borderline: #8b5cf6;
      --accent-borderline-bg: #f5f3ff;
      --accent-borderline-border: #ddd6fe;
      --accent-borderline-text: #5b21b6;

      --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
      --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.07);
      --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.08);
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 16px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background-color: var(--bg-canvas);
      color: var(--text-main);
      line-height: 1.5;
      padding-bottom: 80px;
    }}

    header {{
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }}

    .header-container {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 24px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}

    .brand-title h1 {{
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .badge-model {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      background: #e0e7ff;
      color: #3730a3;
      padding: 3px 8px;
      border-radius: 9999px;
      font-weight: 600;
    }}

    .main-container {{
      max-width: 1280px;
      margin: 28px auto;
      padding: 0 24px;
    }}

    /* Stats Grid */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}

    .stat-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px 20px;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .stat-card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }}

    .stat-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
    }}

    .stat-card.total::before {{ background: #64748b; }}
    .stat-card.acceptable::before {{ background: var(--accent-acceptable); }}
    .stat-card.toxic::before {{ background: var(--accent-toxic); }}
    .stat-card.spam::before {{ background: var(--accent-spam); }}
    .stat-card.borderline::before {{ background: var(--accent-borderline); }}

    .stat-label {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .stat-value {{
      font-size: 1.85rem;
      font-weight: 800;
      margin-top: 4px;
      color: var(--text-main);
      display: flex;
      align-items: baseline;
      gap: 6px;
    }}

    .stat-percent {{
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--text-muted);
    }}

    /* Distribution Progress Bar */
    .distribution-bar-wrapper {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 14px 20px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
    }}

    .distribution-title {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}

    .bar-container {{
      height: 12px;
      width: 100%;
      background: var(--border-color);
      border-radius: 9999px;
      overflow: hidden;
      display: flex;
    }}

    .bar-segment {{
      height: 100%;
      transition: width 0.3s ease;
    }}
    .bar-segment.acceptable {{ background: var(--accent-acceptable); }}
    .bar-segment.toxic {{ background: var(--accent-toxic); }}
    .bar-segment.spam {{ background: var(--accent-spam); }}

    /* Controls Bar */
    .controls-panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px 20px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      justify-content: space-between;
      align-items: center;
    }}

    .filter-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .tab-btn {{
      background: var(--bg-canvas);
      color: var(--text-muted);
      border: 1px solid var(--border-color);
      padding: 8px 14px;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .tab-btn:hover {{
      background: var(--border-color);
      color: var(--text-main);
    }}

    .tab-btn.active {{
      background: var(--text-main);
      color: var(--bg-surface);
      border-color: var(--text-main);
    }}

    .tab-btn .tab-count {{
      font-size: 0.75rem;
      padding: 2px 6px;
      border-radius: 9999px;
      background: rgba(125, 125, 125, 0.15);
    }}

    .search-sort-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      flex: 1;
      justify-content: flex-end;
      min-width: 280px;
    }}

    .search-box {{
      position: relative;
      flex: 1;
      max-width: 320px;
    }}

    .search-box input {{
      width: 100%;
      padding: 8px 14px 8px 36px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
      background: var(--bg-surface);
      color: var(--text-main);
      font-size: 0.875rem;
      outline: none;
      transition: border-color 0.15s;
    }}

    .search-box input:focus {{
      border-color: var(--border-focus);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }}

    .search-icon {{
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-subtle);
      pointer-events: none;
    }}

    .sort-select {{
      padding: 8px 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
      font-size: 0.875rem;
      background: var(--bg-surface);
      color: var(--text-main);
      cursor: pointer;
      outline: none;
    }}

    /* Comments Cards Grid */
    .comments-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}

    .comment-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 18px 22px;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}

    .comment-card:hover {{
      border-color: var(--border-focus);
      box-shadow: var(--shadow-md);
    }}

    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .author-info {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .avatar {{
      width: 34px;
      height: 34px;
      border-radius: 9999px;
      background: var(--border-color);
      color: var(--text-main);
      font-weight: 700;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      justify-content: center;
      text-transform: uppercase;
    }}

    .author-name {{
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-main);
    }}

    .thread-tag {{
      font-size: 0.75rem;
      background: var(--bg-canvas);
      color: var(--text-muted);
      border: 1px solid var(--border-color);
      padding: 2px 8px;
      border-radius: 6px;
      font-weight: 600;
    }}

    .meta-details {{
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .id-badge {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      background: var(--bg-canvas);
      border: 1px solid var(--border-color);
      padding: 2px 7px;
      border-radius: 6px;
      font-weight: 600;
      color: var(--text-muted);
    }}

    .comment-text {{
      font-size: 0.95rem;
      color: var(--text-main);
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fafc;
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      border-left: 3px solid #cbd5e1;
    }}

    .card-bottom {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      padding-top: 8px;
      border-top: 1px solid var(--border-color);
    }}

    .prediction-group {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .decision-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .decision-pill.acceptable {{
      background: var(--accent-acceptable-bg);
      color: var(--accent-acceptable-text);
      border: 1px solid var(--accent-acceptable-border);
    }}

    .decision-pill.toxic {{
      background: var(--accent-toxic-bg);
      color: var(--accent-toxic-text);
      border: 1px solid var(--accent-toxic-border);
    }}

    .decision-pill.spam {{
      background: var(--accent-spam-bg);
      color: var(--accent-spam-text);
      border: 1px solid var(--accent-spam-border);
    }}

    .confidence-tag {{
      font-size: 0.78rem;
      color: var(--text-muted);
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 4px;
    }}

    .borderline-indicator {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: var(--accent-borderline-bg);
      color: var(--accent-borderline-text);
      border: 1px solid var(--accent-borderline-border);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
    }}

    .probs-bar {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      color: var(--text-muted);
    }}

    .bar-mini-wrapper {{
      width: 70px;
      height: 6px;
      background: var(--border-color);
      border-radius: 9999px;
      overflow: hidden;
      display: flex;
    }}

    .actions-group {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .btn-mistake {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 5px 10px;
      border-radius: var(--radius-sm);
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}

    .btn-mistake:hover {{
      background: var(--bg-canvas);
      border-color: var(--text-muted);
      color: var(--text-main);
    }}

    .copied-toast {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #0f172a;
      color: #ffffff;
      padding: 10px 18px;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      font-weight: 600;
      box-shadow: var(--shadow-lg);
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.25s ease;
      z-index: 100;
    }}

    .copied-toast.show {{
      transform: translateY(0);
      opacity: 1;
    }}

    .empty-state {{
      text-align: center;
      padding: 48px 20px;
      background: var(--bg-surface);
      border: 1px dashed var(--border-color);
      border-radius: var(--radius-md);
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-container">
      <div class="brand-title">
        <h1>
          Forum Comment Moderation
          <span class="badge-model">Cloudflare {model_name}</span>
        </h1>
      </div>
      <div>
        <span class="id-badge">Total: {total} items</span>
      </div>
    </div>
  </header>

  <main class="main-container">

    <!-- KPI Summary Cards -->
    <section class="stats-grid">
      <div class="stat-card total">
        <span class="stat-label">Total Comments</span>
        <div class="stat-value">{total}</div>
      </div>
      <div class="stat-card acceptable">
        <span class="stat-label">Acceptable</span>
        <div class="stat-value">
          {acceptable_count}
          <span class="stat-percent">({(acceptable_count / max(1, total) * 100):.1f}%)</span>
        </div>
      </div>
      <div class="stat-card toxic">
        <span class="stat-label">Toxic</span>
        <div class="stat-value">
          {toxic_count}
          <span class="stat-percent">({(toxic_count / max(1, total) * 100):.1f}%)</span>
        </div>
      </div>
      <div class="stat-card spam">
        <span class="stat-label">Spam</span>
        <div class="stat-value">
          {spam_count}
          <span class="stat-percent">({(spam_count / max(1, total) * 100):.1f}%)</span>
        </div>
      </div>
      <div class="stat-card borderline">
        <span class="stat-label">Borderline / Low Conf</span>
        <div class="stat-value">
          {borderline_count}
          <span class="stat-percent">({(borderline_count / max(1, total) * 100):.1f}%)</span>
        </div>
      </div>
    </section>

    <!-- Visual Distribution Bar -->
    <div class="distribution-bar-wrapper">
      <div class="distribution-title">
        <span>Moderation Distribution</span>
        <span>{acceptable_count} Acceptable / {toxic_count} Toxic / {spam_count} Spam</span>
      </div>
      <div class="bar-container">
        <div class="bar-segment acceptable" style="width: {(acceptable_count / max(1, total) * 100):.2f}%" title="Acceptable: {acceptable_count}"></div>
        <div class="bar-segment toxic" style="width: {(toxic_count / max(1, total) * 100):.2f}%" title="Toxic: {toxic_count}"></div>
        <div class="bar-segment spam" style="width: {(spam_count / max(1, total) * 100):.2f}%" title="Spam: {spam_count}"></div>
      </div>
    </div>

    <!-- Controls & Filter Toolbar -->
    <section class="controls-panel">
      <div class="filter-tabs">
        <button class="tab-btn active" data-filter="all">All <span class="tab-count">{total}</span></button>
        <button class="tab-btn" data-filter="acceptable">Acceptable <span class="tab-count">{acceptable_count}</span></button>
        <button class="tab-btn" data-filter="toxic">Toxic <span class="tab-count">{toxic_count}</span></button>
        <button class="tab-btn" data-filter="spam">Spam <span class="tab-count">{spam_count}</span></button>
        <button class="tab-btn" data-filter="borderline">&#9873; Borderline <span class="tab-count">{borderline_count}</span></button>
      </div>

      <div class="search-sort-group">
        <div class="search-box">
          <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input type="text" id="searchInput" placeholder="Search comments, author, ID...">
        </div>

        <select id="sortSelect" class="sort-select">
          <option value="id-asc">Sort: ID (1 &rarr; 100)</option>
          <option value="id-desc">Sort: ID (100 &rarr; 1)</option>
          <option value="conf-asc">Sort: Lowest Confidence First</option>
          <option value="conf-desc">Sort: Highest Confidence First</option>
        </select>
      </div>
    </section>

    <!-- Comments Container -->
    <div id="commentsContainer" class="comments-grid">
      <!-- Injected by JavaScript -->
    </div>

  </main>

  <div id="toast" class="copied-toast">Template copied for MISTAKES.md!</div>

  <script>
    const commentsData = {items_json};

    let activeFilter = 'all';
    let searchQuery = '';
    let activeSort = 'id-asc';

    const container = document.getElementById('commentsContainer');
    const searchInput = document.getElementById('searchInput');
    const sortSelect = document.getElementById('sortSelect');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const toast = document.getElementById('toast');

    function isBorderline(item) {{
      const c = item.classification;
      if (c.confidence < {LOW_CONFIDENCE_THRESHOLD}) return true;
      const probs = Object.values(c.probabilities || {{}}).sort((a, b) => b - a);
      if (probs.length > 1 && (probs[0] - probs[1]) < 0.25) return true;
      return false;
    }}

    function copyMistakeTemplate(id, choice, text) {{
      const cleaned = (text || '').replace(/"/g, '').slice(0, 100);
      const snippet = [
        "- **Comment #" + id + "**:",
        "  - Clef's label: `" + choice + "` | Correct label: `[ENTER_LABEL]`",
        "  - Reason: [1-2 sentences explaining why]",
        '  - Excerpt: "' + cleaned + '..."'
      ].join('\\n') + '\\n';
      navigator.clipboard.writeText(snippet).then(() => {{
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2500);
      }});
    }}

    function render() {{
      let filtered = commentsData.filter(item => {{
        if (activeFilter === 'borderline') {{
          if (!isBorderline(item)) return false;
        }} else if (activeFilter !== 'all') {{
          if (item.classification.choice !== activeFilter) return false;
        }}

        if (searchQuery.trim() !== '') {{
          const q = searchQuery.toLowerCase();
          const matchText = (item.comment || '').toLowerCase().includes(q);
          const matchAuthor = (item.author || '').toLowerCase().includes(q);
          const matchThread = (item.thread || '').toLowerCase().includes(q);
          const matchId = String(item.id).includes(q);
          return matchText || matchAuthor || matchThread || matchId;
        }}

        return true;
      }});

      filtered.sort((a, b) => {{
        if (activeSort === 'id-asc') return a.id - b.id;
        if (activeSort === 'id-desc') return b.id - a.id;
        if (activeSort === 'conf-asc') return a.classification.confidence - b.classification.confidence;
        if (activeSort === 'conf-desc') return b.classification.confidence - a.classification.confidence;
        return 0;
      }});

      if (filtered.length === 0) {{
        container.innerHTML = `
          <div class="empty-state">
            <h3>No comments match your filter</h3>
            <p>Try selecting a different filter tab or clearing the search query.</p>
          </div>
        `;
        return;
      }}

      container.innerHTML = filtered.map(item => {{
        const c = item.classification;
        const choice = c.choice || 'unknown';
        const confPercent = Math.round((c.confidence || 0) * 100);
        const borderline = isBorderline(item);

        const initials = (item.author || 'User')
          .split(' ')
          .map(p => p[0])
          .slice(0, 2)
          .join('');

        const probs = c.probabilities || {{}};
        const pAcc = Math.round((probs.acceptable || 0) * 100);
        const pTox = Math.round((probs.toxic || 0) * 100);
        const pSpam = Math.round((probs.spam || 0) * 100);

        return `
          <article class="comment-card" id="card-${{item.id}}">
            <div class="card-top">
              <div class="author-info">
                <div class="avatar">${{initials}}</div>
                <div>
                  <div class="author-name">${{escapeHtml(item.author)}}</div>
                  <div class="thread-tag">${{escapeHtml(item.thread)}}</div>
                </div>
              </div>
              <div class="meta-details">
                <span>${{escapeHtml(item.posted_at)}}</span>
                <span class="id-badge">ID #${{item.id}}</span>
              </div>
            </div>

            <div class="comment-text">${{escapeHtml(item.comment)}}</div>

            <div class="card-bottom">
              <div class="prediction-group">
                <span class="decision-pill ${{choice}}">
                  ${{choice}}
                </span>
                <span class="confidence-tag">
                  Confidence: <strong>${{confPercent}}%</strong>
                </span>
                ${{borderline ? '<span class="borderline-indicator">&#9873; Borderline</span>' : ''}}
                <div class="probs-bar" title="Acceptable: ${{pAcc}}% | Toxic: ${{pTox}}% | Spam: ${{pSpam}}%">
                  <div class="bar-mini-wrapper">
                    <div style="width: ${{pAcc}}%; background: var(--accent-acceptable);"></div>
                    <div style="width: ${{pTox}}%; background: var(--accent-toxic);"></div>
                    <div style="width: ${{pSpam}}%; background: var(--accent-spam);"></div>
                  </div>
                  <span>A:${{pAcc}}% T:${{pTox}}% S:${{pSpam}}%</span>
                </div>
              </div>

              <div class="actions-group">
                <button class="btn-mistake" onclick="copyMistakeTemplate(${{item.id}}, '${{choice}}', '${{escapeAttr(item.comment)}}')">
                  📋 Copy for MISTAKES.md
                </button>
              </div>
            </div>
          </article>
        `;
      }}).join('');
    }}

    function escapeHtml(str) {{
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function escapeAttr(str) {{
      if (!str) return '';
      return String(str)
        .replace(/'/g, "\\\\'")
        .replace(/"/g, '&quot;')
        .replace(/\\n/g, ' ');
    }}

    tabButtons.forEach(btn => {{
      btn.addEventListener('click', () => {{
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.filter;
        render();
      }});
    }});

    searchInput.addEventListener('input', (e) => {{
      searchQuery = e.target.value;
      render();
    }});

    sortSelect.addEventListener('change', (e) => {{
      activeSort = e.target.value;
      render();
    }});

    render();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated dashboard HTML: {output_path.resolve()}")


# ==============================================================================
# CLI Arguments
# ==============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify forum comments using Cloudflare Clef AI model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("comments.json"),
        help="Path to input comments JSON file (default: comments.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("index.html"),
        help="Path to output HTML report (default: index.html)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("classification_results.json"),
        help="Path to classification cache JSON (default: classification_results.json)",
    )
    parser.add_argument(
        "--creds",
        type=Path,
        default=Path("cloud-flare.local.sh"),
        help="Path to Cloudflare credentials file (default: cloud-flare.local.sh)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clef",
        choices=["clef", "clef-flash"],
        help="Cloudflare Workers AI model variant (default: clef)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent worker threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to first N comments (for smoke testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running classifications, ignoring existing cache.",
    )
    return parser.parse_args()


# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    args = parse_arguments()

    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # 1. Load dataset
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    comments_raw = raw_data.get("comments", [])
    if args.limit:
        comments_raw = comments_raw[: args.limit]

    comments = [
        Comment(
            id=item["id"],
            author=item.get("author", "Unknown"),
            thread=item.get("thread", "General"),
            posted_at=item.get("posted_at", ""),
            comment=item.get("comment", ""),
        )
        for item in comments_raw
    ]
    logger.info(f"Loaded {len(comments)} comments from {args.input}")

    # 2. Check Cache with Fingerprint
    fingerprint = compute_prompt_fingerprint(args.model)
    cached_results = {} if args.force else load_cached_results(args.cache, fingerprint)
    pending_comments = [c for c in comments if c.id not in cached_results]

    logger.info(
        f"Execution plan: {len(comments)} total | "
        f"{len(comments) - len(pending_comments)} cached | "
        f"{len(pending_comments)} to classify with {args.model}"
    )

    # 3. Classify pending comments
    results_map: Dict[int, ClassifiedComment] = dict(cached_results)

    if pending_comments:
        account_id, api_token = load_credentials(args.creds)
        client = ClefClassifierClient(account_id, api_token, model=args.model)

        def worker_task(comment: Comment) -> Tuple[Comment, Classification]:
            classification = client.classify_comment(comment.comment)
            return comment, classification

        failures = []
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_to_comment = {
                    pool.submit(worker_task, comment): comment
                    for comment in pending_comments
                }

                completed_count = 0
                for future in as_completed(future_to_comment):
                    comment = future_to_comment[future]
                    completed_count += 1
                    try:
                        _, classification = future.result()
                        classified = ClassifiedComment(comment, classification)
                        results_map[comment.id] = classified
                        logger.info(
                            f"[{completed_count}/{len(pending_comments)}] "
                            f"#{comment.id:<3} {classification.choice:<10} "
                            f"conf={classification.confidence:.2f}"
                        )
                    except FatalAPIError as e:
                        logger.error(f"Fatal API error on #{comment.id}: {e}")
                        # Persist what we got so far
                        save_cached_results(args.cache, list(results_map.values()), args.model, fingerprint)
                        sys.exit(1)
                    except Exception as e:
                        failures.append(comment.id)
                        logger.error(f"Failed #{comment.id}: {e}")

        finally:
            save_cached_results(
                args.cache,
                [results_map[c.id] for c in comments if c.id in results_map],
                args.model,
                fingerprint,
            )

        if failures:
            logger.warning(f"{len(failures)} comment(s) failed: {failures}. Rerun to retry them.")

    # 4. Final Ordered Results & HTML Generation
    final_classified = [results_map[c.id] for c in comments if c.id in results_map]
    if not final_classified:
        logger.error("No results available to render.")
        sys.exit(1)

    generate_html_report(
        classified_comments=final_classified,
        output_path=args.output,
        model_name=args.model,
    )

    counts = Counter(c.classification.choice for c in final_classified)
    logger.info(
        f"Complete! HTML saved to {args.output.name}: "
        + ", ".join(f"{counts.get(l, 0)} {l}" for l in LABELS)
    )


if __name__ == "__main__":
    main()

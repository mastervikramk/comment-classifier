# Project Notes

## How to Run the Program

### 1. Prerequisites
- Python 3.10+
- A Cloudflare account with Workers AI enabled.

### 2. Configure Credentials
Create a file named `cloud-flare.local.sh` in the project root directory (this file is ignored by `.gitignore` to prevent leaking secrets):
```bash
export CLOUDFLARE_ACCOUNT_ID="your_account_id_here"
export CLOUDFLARE_API_TOKEN="your_workers_ai_api_token_here"
```

### 3. Run the Classifier
Run the classification script using Python standard libraries (no external pip dependencies required):
```bash
python classify.py
```

#### Optional CLI Flags:
- `--input <path>`: Specify input comments JSON file (default: `comments.json`).
- `--output <path>`: Specify output HTML dashboard path (default: `index.html`).
- `--model <name>`: Model variant to use: `clef` (default) or `clef-flash`.
- `--force`: Ignore local cache and re-run all classifications.
- `--rate-limit-delay <seconds>`: Pacing between requests (default: `1.2` seconds).

Classification results are cached locally in `classification_results.json` so interrupted runs can be resumed immediately without re-calling the API.

---

## Tools and Models Used

1. **Cloudflare Workers AI (`@cf/cloudflare/clef`)**:
   - Used Cloudflare's decision model for zero-shot text classification into `acceptable`, `toxic`, and `spam`.
   - Tuned prompt criteria specifically for academic forum moderation (accommodating student discussions, project feedback, venting vs. personal attacks, and homework-for-hire spam).
2. **Python 3 Standard Library**:
   - Built a modular, object-oriented pipeline using `dataclasses`, `urllib.request`, `json`, `argparse`, and exponential backoff retry handling for rate limits.
3. **HTML5 / CSS / Vanilla JavaScript**:
   - Designed a responsive, modern card dashboard (`index.html`) with KPI summary metrics, category filtering, search, and sort functionality.
4. **Antigravity AI Assistant**:
   - Paired with AI for architecture design, API client implementation, and interactive UI styling.

---

## Reflection & Takeaways

- **Performance & Accuracy:** The Clef model demonstrated strong baseline performance on clear-cut cases (e.g., promotional spam and blunt insults). However, nuance and context (such as sarcastic peer banter, self-deprecating student humor, and apologies) often result in lower confidence scores or borderline classifications.
- **The Human-in-the-Loop Value:** Automated AI moderation dramatically reduces TA workload by filtering 80-90% of obvious cases, but human oversight remains indispensable for borderline, empathetic, or culturally nuanced forum dynamics.


import os
import json
import time
import subprocess
import urllib.parse
from pathlib import Path
from dotenv import dotenv_values

TARGET_REPO = "scikit-learn/scikit-learn"
RAW_DATA_PATH = Path("data/raw/github_issues_raw.json")

# target 4-class taxonomy mapped to repo labels
LABEL_QUERIES = [
    ("bug", "Bug", 3),
    ("feature", "New Feature", 2),
    ("feature", "Enhancement", 1),
    ("docs", "Documentation", 3),
    ("question", "Question", 2),
]


def load_github_token() -> str:
    env_vars = dotenv_values(".env")
    token = env_vars.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    return token.strip().strip("\"'")


def fetch_search_page(query: str, page: int, token: str, retries: int = 3):
    encoded_q = urllib.parse.quote(query)
    url = f"https://api.github.com/search/issues?q={encoded_q}&per_page=100&page={page}"
    cmd = [
        "curl",
        "-s",
        "-D",
        "-",
        "-m",
        "30",
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Accept: application/vnd.github.v3+json",
        "-H",
        "User-Agent: bug-triage-pipeline",
        url,
    ]

    for attempt in range(1, retries + 1):
        try:
            raw = subprocess.check_output(cmd).decode("utf-8", errors="replace")
            sep = "\r\n\r\n" if "\r\n\r\n" in raw else "\n\n"
            parts = raw.split(sep, 1)
            if len(parts) < 2:
                return {}, []

            header_text, body = parts[0], parts[1]
            headers = {}
            for line in header_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            data = json.loads(body)
            items = data.get("items", [])
            return headers, items
        except Exception as e:
            # retry on temporary timeout or network glitch
            if attempt == retries:
                print(f"    failed after {retries} attempts: {e}")
                return {}, []
            time.sleep(2.0 * attempt)


def fetch_issues(repo: str = TARGET_REPO, force: bool = False):
    all_issues = {}
    if RAW_DATA_PATH.exists() and not force:
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        for item in cached:
            all_issues[item["id"]] = item
        if len(all_issues) >= 1000:
            print(f"loaded {len(all_issues)} cached issues from {RAW_DATA_PATH}")
            return list(all_issues.values())

    token = load_github_token()
    if not token:
        raise ValueError("GITHUB_TOKEN is missing in .env")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    for target_class, label_name, max_pages in LABEL_QUERIES:
        print(f"fetching class '{target_class}' with label '{label_name}' (up to {max_pages} pages)...")
        for page in range(1, max_pages + 1):
            query = f'repo:{repo} is:issue is:closed label:"{label_name}"'
            headers, items = fetch_search_page(query, page, token)
            remaining = headers.get("x-ratelimit-remaining", "unknown")

            if not items:
                break

            added_count = 0
            for item in items:
                # exclude pull requests if any returned
                if "pull_request" in item:
                    continue

                issue_id = item["id"]
                if issue_id not in all_issues:
                    item["_target_class"] = target_class
                    all_issues[issue_id] = item
                    added_count += 1

            print(f"  page {page}: added {added_count} issues | total cached: {len(all_issues)} | remaining quota: {remaining}")

            # persist incrementally so progress is preserved across any interruption
            with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(list(all_issues.values()), f, indent=2)

            # rate-limit cooldown between requests
            time.sleep(1.0)

    issue_list = list(all_issues.values())
    print(f"successfully saved {len(issue_list)} raw issues to {RAW_DATA_PATH}")
    return issue_list


if __name__ == "__main__":
    fetch_issues()

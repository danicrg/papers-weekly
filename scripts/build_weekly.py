#!/usr/bin/env python3
import os, re, json, time, math, datetime, urllib.parse, threading
from typing import List, Dict, Any, Optional, Tuple
import requests
import feedparser
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import HfApi


load_dotenv()

# -------- Config --------
CATEGORIES = ["cs.LG", "cs.AI", "stat.ML"]
DAYS_BACK = int(os.getenv("DAYS_BACK", "7"))
TOP_N = int(os.getenv("TOP_N", "50"))
OUT_PATH = os.getenv("OUT_PATH", "public/generated/weekly.json")

# Scoring weights
W_REDDIT, W_HN, W_HF = 3.0, 2.0, 1.5

# Concurrency knobs (tune as needed)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "24"))   # total concurrent papers scored
REDDIT_CONCURRENCY = int(os.getenv("REDDIT_CONCURRENCY", "5"))
HN_CONCURRENCY = int(os.getenv("HN_CONCURRENCY", "10"))
BSKY_CONCURRENCY = int(os.getenv("BSKY_CONCURRENCY", "10"))

# Backoff/safety
SLEEP_SHORT = 0.4
UA = "papers-weekly/0.5 (https://github.com/you/papers-weekly)"
ARXIV_PAGE = 200

# -------- Shared HTTP session (keep-alive) --------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

api = HfApi()

# --- add near top ---
import threading, time
from collections import deque

class RateLimiter:
    """Thread-safe sliding-window limiter: at most max_calls per period seconds."""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._times = deque()
        self._lock = threading.Lock()

    def wait(self):
        while True:
            with self._lock:
                now = time.time()
                # drop timestamps outside the window
                while self._times and now - self._times[0] >= self.period:
                    self._times.popleft()
                if len(self._times) < self.max_calls:
                    self._times.append(now)
                    return
                sleep_for = self.period - (now - self._times[0])
            time.sleep(max(0.01, sleep_for))

# Limit Reddit search to ~90/min to be safe
REDDIT_MAX_PER_MIN = int(os.getenv("REDDIT_MAX_PER_MIN", "90"))
_reddit_rl = RateLimiter(REDDIT_MAX_PER_MIN, 60.0)

# -------- Helpers --------
def log1p(x: int) -> float:
    return math.log1p(max(0, x))

def utc_today() -> datetime.datetime:
    return datetime.datetime.utcnow()

def days_ago(n: int) -> datetime.datetime:
    return utc_today() - datetime.timedelta(days=n)

def ymd(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _fetch_category_since(category: str, since_dt: datetime.datetime) -> list:
    start = 0
    out = []
    query = f"cat:{category}"

    while True:
        url = (
            "https://export.arxiv.org/api/query?"
            + urllib.parse.urlencode({
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": ARXIV_PAGE,
            })
        )
        r = SESSION.get(url, timeout=30)
        feed = feedparser.parse(r.text)
        if not feed.entries:
            break

        stop = False
        for e in feed.entries:
            pub = e.get("published") or e.get("updated")
            dt = datetime.datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
            if dt < since_dt:
                stop = True
                break
            out.append(e)

        if stop:
            break
        start += ARXIV_PAGE
        time.sleep(3.2)  # respect arXiv throttle

    return out

def fetch_arxiv_since(categories: list, since_dt: datetime.datetime) -> list:
    seen_ids = set()
    combined = []

    for cat in categories:
        entries = _fetch_category_since(cat, since_dt)
        for e in entries:
            # Extract stable arXiv id without version suffix
            raw_id = e.get("id", "")
            short_id = raw_id.split("/")[-1].split("v")[0]
            if short_id in seen_ids:
                continue
            seen_ids.add(short_id)
            combined.append(e)

    n = len(combined)

    # Just in case, add the ones in the json already that are in the window
    try:
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            old = json.load(f)
            for p in old.get("papers", []):
                pub = p.get("published")
                if not pub:
                    continue
                dt = datetime.datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
                if dt >= since_dt and p["id"] not in seen_ids:
                    combined.append({
                        "id": p["abs_url"],
                        "title": p["title"],
                        "authors": [{"name": a} for a in p.get("authors", [])],
                        "abs_url": p["abs_url"],
                        "pdf_url": p["pdf_url"],
                        "published": pub,
                    })
                    seen_ids.add(p["id"])
        print(f"[info] re-added {len(combined) - n} papers from existing json")
    except Exception:
        pass

    return combined

# -------- Reddit (free OAuth client credentials) --------
def reddit_token(client_id: str, client_secret: str) -> Optional[str]:
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    headers = {"User-Agent": UA}
    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth,
        data={"grant_type": "client_credentials"},
        headers=headers,
        timeout=20,
    )
    if not r.ok:
        print("[reddit] token failed", r.text)
        return None
    return r.json().get("access_token")

# Provider semaphores to cap concurrency
_sema_reddit = threading.Semaphore(REDDIT_CONCURRENCY)
_sema_hn = threading.Semaphore(HN_CONCURRENCY)
_sema_bsky = threading.Semaphore(BSKY_CONCURRENCY)

def _reddit_get(url, params, token, retries=5):
    headers = {"Authorization": f"Bearer {token}", "User-Agent": UA}
    for attempt in range(1, retries + 1):
        # global rate-limit gate
        _reddit_rl.wait()
        with _sema_reddit:  # keep a few requests in flight to hide latency
            r = SESSION.get(url, params=params, headers=headers, timeout=25)

        if r.status_code == 429:
            # Try to honor server hint if present
            retry_after = r.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else 60.0
            print(f"[reddit] 429 hit, sleeping {wait_s:.0f}s (attempt {attempt}/{retries})")
            time.sleep(wait_s)
            continue

        if r.ok:
            return r.json()

        # transient errors
        if 500 <= r.status_code < 600:
            backoff = min(2 ** attempt, 8)
            print(f"[reddit] {r.status_code}, backing off {backoff}s")
            time.sleep(backoff)
            continue

        # client error other than 429
        r.raise_for_status()

    raise RuntimeError("Reddit request failed after retries")

def count_reddit_mentions(abs_url: str, token: Optional[str]) -> int:
    if not token:
        return 0
    start_time = time.time()
    pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"
    queries = [
        {"q": f"url:{abs_url}", "type": "link", "limit": "100", "sort": "new", "restrict_sr": "false"},
        {"q": f"url:{pdf_url}", "type": "link", "limit": "100", "sort": "new", "restrict_sr": "false"},
    ]
    m = re.search(r'(\d{4}\.\d{4,5})', abs_url)
    short_id = m.group(1) if m else abs_url
    queries.append({"q": f"\"{short_id}\" arxiv.org", "limit": "100", "sort": "new", "restrict_sr": "false"})

    seen = set()
    total = 0
    for q in queries:
        js = _reddit_get("https://oauth.reddit.com/search", q, token)
        children = js.get("data", {}).get("children", [])
        for c in children:
            perm = c.get("data", {}).get("permalink")
            if perm and perm not in seen:
                seen.add(perm)
                total += 1
    print(f"[reddit] {short_id}: {total} in {time.time() - start_time:.1f}s")
    return total

# -------- Hacker News via Algolia (free) --------
def count_hn_mentions(abs_url: str) -> int:
    start_time = time.time()
    params = {"query": f"\"{abs_url}\"", "tags": "story"}
    with _sema_hn:
        r = SESSION.get("https://hn.algolia.com/api/v1/search", params=params, timeout=20)
        time.sleep(SLEEP_SHORT)
    if r.status_code != 200:
        return 0
    js = r.json()
    n = int(js.get("nbHits", 0))
    print(f"[hn] {abs_url.split('/')[-1]}: {n} in {time.time() - start_time:.1f}s")
    return n

# --- replace your hf_signals(...) with this ---

def hf_signals_precise(abs_url: str, max_models: int | None = None, max_datasets: int | None = None):
    """
    Count HF repos that explicitly tag this paper: `arxiv:<id>`.
    Streams all results; optionally cap with max_models/max_datasets to bound runtime.
    """
    start_time = time.time()
    arxiv_id = abs_url.split("/")[-1]  # extract id if full URL given
    # Models citing the paper
    m_count = 0
    m_downloads = 0
    with _sema_bsky:
        for i, m in enumerate(api.list_models(filter=[f"arxiv:{arxiv_id}"], sort="downloads", direction=-1)):
            m_count += 1
            m_downloads += int(getattr(m, "downloads", 0) or 0)
            if max_models and i + 1 >= max_models:
                break

        # Datasets citing the paper
        d_count = 0
        d_downloads = 0
        for i, d in enumerate(api.list_datasets(filter=[f"arxiv:{arxiv_id}"], sort="downloads", direction=-1)):
            d_count += 1
            d_downloads += int(getattr(d, "downloads", 0) or 0)
            if max_datasets and i + 1 >= max_datasets:
                break
    print(f"[hf] {arxiv_id}: {m_count} models (+{m_downloads} dl), {d_count} datasets (+{d_downloads} dl) in {time.time() - start_time:.1f}s")
    return {
        "hf_model_count": m_count,
        "hf_model_downloads": m_downloads,
        "hf_dataset_count": d_count,
        "hf_dataset_downloads": d_downloads,
    }


def hf_paper_github_stars(arxiv_id: str) -> dict:
    # 1) Try the Papers API
    with _sema_bsky:
        r = SESSION.get(f"https://huggingface.co/api/papers/{arxiv_id}", timeout=20)
        time.sleep(SLEEP_SHORT)
        if r.status_code == 200:
            try:
                js = r.json()
                print(f"[hf] paper {arxiv_id}: {js.get('githubStars', 0) or 0} stars")
                return {
                    "github_stars": js.get("githubStars", 0) or 0
                }
            except Exception as ex:
                print(f"[hf] paper {arxiv_id} parse error: {ex}")
    return {
        "github_stars": 0
    }

# -------- Per-paper scorer (runs provider calls in parallel) --------
def score_one(paper: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
    abs_url = paper["abs_url"]

    # run the three providers concurrently for this paper
    with ThreadPoolExecutor(max_workers=3) as local_pool:
        fut_reddit = local_pool.submit(count_reddit_mentions, abs_url, token)
        fut_hn     = local_pool.submit(count_hn_mentions, abs_url)
        fut_hf   = local_pool.submit(hf_signals_precise, abs_url)
        fut_stars = local_pool.submit(hf_paper_github_stars, paper["id"])

        reddit_mentions = fut_reddit.result()
        hn_mentions     = fut_hn.result()
        hf_signals   = fut_hf.result()
        gh_stars       = fut_stars.result().get("github_stars", 0)

    score = (
        W_REDDIT * log1p(reddit_mentions) +
        W_HN     * log1p(hn_mentions) +
        W_HF   * log1p(hf_signals["hf_model_count"] + hf_signals["hf_dataset_count"])
        + 0.1 * math.sqrt(gh_stars)
    )

    return {
        "id": paper["id"],
        "title": paper["title"],
        "authors": paper["authors"],
        "abs_url": abs_url,
        "pdf_url": paper["pdf_url"],
        "published": paper["published"],
        "signals": {
            "reddit_mentions": reddit_mentions,
            "hn_mentions": hn_mentions,
            "hf_count": hf_signals["hf_model_count"] + hf_signals["hf_dataset_count"],
            "gh_stars": gh_stars,
        },
        "score": round(score, 4),
    }




# -------- Main build --------
def build():
    since = days_ago(DAYS_BACK)
    since_str = ymd(since)

    print(f"[info] fetching arXiv since {since_str} for {CATEGORIES}")
    entries = fetch_arxiv_since(CATEGORIES, since)
    print(f"[info] fetched {len(entries)} total entries from arXiv")

    filtered = []
    for e in entries:
        abs_url = e.get("id", "")
        m = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", abs_url)
        if not m:
            continue
        short_id = m.group(1)
        authors = [a.get("name") for a in e.get("authors", [])]
        published = e.get("published") or e.get("updated")
        filtered.append({
            "id": short_id,
            "title": (e.get("title", "") or "").replace("\n", " ").strip(),
            "authors": authors,
            "abs_url": f"https://arxiv.org/abs/{short_id}",
            "pdf_url": f"https://arxiv.org/pdf/{short_id}.pdf",
            "published": published,
        })

    print(f"[info] candidates in window: {len(filtered)}")

    reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_secret = os.getenv("REDDIT_SECRET", "")
    token = reddit_token(reddit_client_id, reddit_secret) if (reddit_client_id and reddit_secret) else None
    print("[info] Reddit token acquired" if token else "[warn] No Reddit token; reddit_mentions will be 0")

    # Parallelize across papers too, with capped total threads
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(score_one, p, token): p for p in filtered}
        with tqdm(total=len(filtered), desc="Scoring papers", unit="paper") as pbar:
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    row = fut.result()
                    rows.append(row)
                except Exception as ex:
                    tqdm.write(f"[error] scoring {p['id']}: {ex}")
                finally:
                    pbar.update(1)

    rows.sort(key=lambda x: x["score"], reverse=True)
    top = rows[:TOP_N]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_utc": utc_today().isoformat() + "Z",
            "window_days": DAYS_BACK,
            "categories": CATEGORIES,
            "top_n": TOP_N,
            "papers": top
        }, f, indent=2)

    print(f"[done] wrote {OUT_PATH} with {len(top)} papers")

if __name__ == "__main__":
    build()
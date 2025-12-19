Web Security Testing with MinHash

This project gathers DOM states from web pages (static and dynamic) and
creates shingle-based representations for similarity estimation using
MinHash-style hashing.

Contents
- Overview
- Requirements
- Quick start (backend + Streamlit frontend)
- Module guide
- Notes on crawling and hashing

Requirements
- Python 3.9+ (tested on macOS)
- Google Chrome and matching ChromeDriver in PATH (for Selenium)
- See `requirements.txt` for Python packages

Quick start
1) Create and activate a virtual environment (optional but recommended):
   python3 -m venv .venv && source .venv/bin/activate

2) Install dependencies:
   pip install -r requirements.txt

3) Run the Streamlit UI:
   streamlit run frontend/app.py

4) Use the UI to:
   - Static mode: enter one or more URLs (comma-separated) and fetch.
   - Dynamic mode: enter a start URL, set max states, fetch via Selenium
     (requires ChromeDriver).

Backend (programmatic) usage
from backend.fetcher import fetch_static_html, fetch_dynamic_states
from backend.core import extract_tags, generate_shingles, universal_hash, MinHashLSH

states = fetch_static_html(["https://example.com"])
tags = extract_tags(states[0])
shingles = generate_shingles(tags, k=3)
hashed = [universal_hash(seed=0, x=sh) for sh in shingles]

detector = MinHashLSH(k=12, ell=200, tau=0.85)
for state in states:
    detector.add_state(state)
unique_states = detector.get_unique_states()

Module guide
[backend/fetcher.py]
- fetch_static_html(urls): Fetches static pages with requests, uses a
  desktop user-agent, parses via BeautifulSoup, returns list of <body>
  HTML (or full HTML when <body> is missing).
- fetch_dynamic_states(start_url, max_states=10, actions=None): Uses
  Selenium headless Chrome to load the start page, capture the DOM, and
  then simulate clicks (actions like ("tag name", "button") or ("id",
  "submit-btn")). Also performs BFS over same-domain links found in <a>
  tags until max_states is reached, avoiding duplicates with a visited
  set.

[backend/core.py]
- class TagExtractor(html.parser.HTMLParser): Parses HTML and records
  ordered opening tags (html, body, p, a, ...), ignoring attributes and
  text.
- extract_tags(html_content): Runs TagExtractor on a string of HTML.
- generate_shingles(tags, k): Builds k-mer shingles from an ordered tag
  list to represent DOM structure.
- universal_hash(seed, x): MD5-based hash; seed simulates multiple
  independent hash functions for MinHash-style signatures.
- MinHashLSH: Maintains sketches and performs duplicate detection /
  deduplication via approximate Jaccard similarity.

Notes
- Dynamic crawling depends on ChromeDriver compatibility; keep Chrome and
  driver versions aligned.
- Restrict BFS to your target domain to avoid unintentional wide crawls.
- Deduplication is basic; further canonicalization may be needed for
  noisy sites.


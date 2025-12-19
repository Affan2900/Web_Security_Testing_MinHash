import os
import sys
from typing import List

import streamlit as st

# Allow importing backend package when running `streamlit run frontend/app.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.fetcher import fetch_dynamic_states, fetch_static_html  # noqa: E402
from backend.core import MinHashLSH  # noqa: E402


def deduplicate(states: List[str], k: int, ell: int, tau: float):
    """Return unique states using MinHashLSH."""
    detector = MinHashLSH(k=k, ell=ell, tau=tau)
    for state in states:
        detector.add_state(state)
    return detector.get_unique_states()


st.set_page_config(page_title="Web State Collector", page_icon="🔍", layout="wide")
st.title("Web State Collector (MinHash-based)")
st.write(
    "Enter a URL and choose static or dynamic crawling. "
    "Results are deduplicated using MinHash + LSH."
)

mode = st.radio("Crawl type", ["Static (requests)", "Dynamic (Selenium)"])

if mode.startswith("Static"):
    st.subheader("Static input")
    single_url = st.text_input(
        "Single URL (static)",
        placeholder="https://example.com",
    )
else:
    st.subheader("Dynamic input")
    single_url = st.text_input(
        "Start URL (dynamic/Selenium)",
        placeholder="https://example.com",
    )

multi_urls = st.text_area(
    "Multiple URLs (comma or newline separated)",
    placeholder="https://example1.com, https://example2.com",
    height=120,
)

if mode.startswith("Dynamic"):
    max_states = st.slider("Max states per URL (dynamic)", min_value=1, max_value=25, value=8)
else:
    max_states = None

advanced = st.expander("Advanced MinHash parameters")
with advanced:
    k = st.slider("Shingle size (k)", min_value=3, max_value=20, value=12)
    ell = st.slider("Hash functions (ell)", min_value=10, max_value=400, value=200, step=10)
    tau = st.slider("Similarity threshold (tau)", min_value=0.5, max_value=0.95, value=0.85)

if st.button("Fetch and Deduplicate"):
    try:
        urls = []
        if single_url.strip():
            urls.append(single_url.strip())
        if multi_urls.strip():
            raw = multi_urls.replace("\n", ",").split(",")
            urls.extend([u.strip() for u in raw if u.strip()])

        if not urls:
            st.error("Provide at least one valid URL (single or multiple).")
            st.stop()

        states = []
        if mode.startswith("Static"):
            states = fetch_static_html(urls)
        else:
            for start_url in urls:
                states.extend(fetch_dynamic_states(start_url=start_url, max_states=max_states))

        if not states:
            st.warning("No states were collected.")
        else:
            unique_states = deduplicate(states, k=k, ell=ell, tau=tau)
            st.success(f"Collected {len(states)} states; {len(unique_states)} are unique.")

            with st.expander("Preview first unique state"):
                st.code(unique_states[0][:2000], language="html")

    except Exception as exc:
        st.error(f"An error occurred: {exc}")
        st.stop()

st.caption(
    "Dynamic crawling requires ChromeDriver compatible with your installed Chrome. "
    "Use responsibly and stay within your target domain."
)



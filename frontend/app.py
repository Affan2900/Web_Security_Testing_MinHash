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


def deduplicate(states: List[str], k: int, ell: int, tau: float, 
                dynamic_threshold: bool = False, context_aware: bool = False,
                use_ml_for_threshold: bool = True, use_ml_for_shingling: bool = True):
    """Return unique states using MinHashLSH with ML-enhanced novelties."""
    detector = MinHashLSH(k=k, ell=ell, tau=tau, 
                         dynamic_threshold=dynamic_threshold,
                         context_aware=context_aware,
                         use_ml_for_threshold=use_ml_for_threshold,
                         use_ml_for_shingling=use_ml_for_shingling)
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
    
    st.markdown("**Novel Features (ML-Enhanced):**")
    
    col1, col2 = st.columns(2)
    
    # Check if ML model exists to set defaults
    ml_model_exists = False
    try:
        import os
        model_paths = [
            'models/bayesian_optimizer.pkl',
            os.path.join(ROOT, 'models', 'bayesian_optimizer.pkl')
        ]
        for path in model_paths:
            if os.path.exists(path):
                ml_model_exists = True
                break
    except Exception:
        pass
    
    with col1:
        dynamic_threshold = st.checkbox(
            "🔧 Dynamic Threshold Adaptation", 
            value=ml_model_exists,  # Auto-enable if model exists
            help="ML-enhanced: Automatically adjusts tau based on RIA complexity using trained Bayesian model"
        )
        use_ml_for_threshold = st.checkbox(
            "  → Use ML Model for Threshold",
            value=ml_model_exists,  # Auto-enable if model exists
            disabled=not dynamic_threshold,
            help="Use trained Bayesian model to predict optimal threshold (falls back to heuristics if model unavailable)"
        )
    
    with col2:
        context_aware = st.checkbox(
            "🔒 Context-Aware Shingling ", 
            value=ml_model_exists,  # Auto-enable if model exists
            help="ML-enhanced: Security-weighted element prioritization using trained Bayesian model"
        )
        use_ml_for_shingling = st.checkbox(
            "  → Use ML Model for Shingling",
            value=ml_model_exists,  # Auto-enable if model exists
            disabled=not context_aware,
            help="Use trained Bayesian model to predict security relevance weights (falls back to heuristics if model unavailable)"
        )
    
   
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
            # Show processing status
            with st.spinner("Processing states with ML-enhanced novelties..."):
                unique_states = deduplicate(states, k=k, ell=ell, tau=tau,
                                          dynamic_threshold=dynamic_threshold,
                                          context_aware=context_aware,
                                          use_ml_for_threshold=use_ml_for_threshold if dynamic_threshold else False,
                                          use_ml_for_shingling=use_ml_for_shingling if context_aware else False)
            
            # Build status message
            feature_status = []
            ml_status = []
            
            if dynamic_threshold:
                if use_ml_for_threshold:
                    feature_status.append("🔧 Dynamic Threshold (ML)")
                    ml_status.append("ML")
                else:
                    feature_status.append("🔧 Dynamic Threshold (Heuristic)")
            
            if context_aware:
                if use_ml_for_shingling:
                    feature_status.append("🔒 Context-Aware (ML)")
                    ml_status.append("ML")
                else:
                    feature_status.append("🔒 Context-Aware (Heuristic)")
            
            status_text = f"✅ Collected {len(states)} states; {len(unique_states)} are unique."
            if feature_status:
                status_text += f"\n\n**Active Features:** {', '.join(feature_status)}"
            if ml_status:
                status_text += f"\n\n🤖 **ML-Enhanced:** Using trained Bayesian model for optimal performance"
            
            st.success(status_text)

            with st.expander("Preview first unique state"):
                st.code(unique_states[0][:2000], language="html")

    except Exception as exc:
        st.error(f"An error occurred: {exc}")
        st.stop()

st.caption(
    "Dynamic crawling requires ChromeDriver compatible with your installed Chrome. "
    "Use responsibly and stay within your target domain."
)

# Sidebar with ML model info
with st.sidebar:
    st.header("📊 ML Model Status")
    
    try:
        import os
        model_path = 'models/bayesian_optimizer.pkl'
        if os.path.exists(model_path):
            st.success("✅ ML Model Available")
            st.caption("ML-enhanced novelties will use trained Bayesian model for optimal performance.")
            
            # Show model info
            try:
                import os
                import time
                model_size = os.path.getsize(model_path) / 1024  # KB
                model_mtime = os.path.getmtime(model_path)
                model_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(model_mtime))
                
                st.info(f"**Model Size:** {model_size:.1f} KB\n**Last Trained:** {model_date}")
            except Exception:
                pass
            
            st.markdown("---")
            st.markdown("**To retrain model:**")
            st.code("""
1. python scrape_websites.py
2. python collect_training_data.py
3. python train_bayesian_model.py
            """, language="bash")
        else:
            st.warning("⚠️ ML Model Not Found")
            st.caption("Novelties will use heuristic fallback. Train model for better performance.")
            
            st.markdown("---")
            st.markdown("**To train ML model:**")
            st.code("""
1. python scrape_websites.py
2. python collect_training_data.py
3. python train_bayesian_model.py
            """, language="bash")
    except Exception:
        st.info("ℹ️ ML model status unknown")
    
    st.markdown("---")
    st.markdown("### 📚 About Novelties")
    st.markdown("""
    ** ML-enhanced dynamic threshold adaptation
    
    ** ML-enhanced context-aware shingling
    
    Both use trained Bayesian model to optimize performance.
    """)



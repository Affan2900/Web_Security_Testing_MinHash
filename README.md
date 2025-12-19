# Web Security Testing with MinHash

Collect DOM states (static or dynamic), turn them into shingles, and deduplicate using a MinHash+LSH sketch. A Streamlit UI is included, and an optional ML pipeline can adapt `k`, `ell`, and `tau` to each page type.

## Features
- Static fetching via `requests` + BeautifulSoup; dynamic crawling with headless Chrome/Selenium and same-domain BFS.
- MinHash-LSH deduplication with tunable shingle size (`k`), sketch size (`ell`), and similarity threshold (`tau`).
- Streamlit UI (`frontend/app.py`) for interactive collection and deduplication.
- Adaptive mode that predicts hyperparameters with a trained Bayesian (GP) model; pipeline scripts live in the repo.
- **Novelty 1: Dynamic Threshold Adaptation** - Automatically adjusts similarity threshold (`tau`) based on RIA complexity.
- **Novelty 2: Context-Aware Shingling** - Security-weighted element prioritization for vulnerability-prone components.

## Project layout (high level)
- `backend/core.py`: Tag extraction, shingling, MinHashLSH implementation (supports `adaptive=True`, dynamic threshold, context-aware shingling).
- `backend/fetcher.py`: Static and dynamic state collection helpers.
- `frontend/app.py`: Streamlit UI for collecting and deduplicating states.
- `security_relevance.py`: Security relevance scorer for context-aware element weighting (Novelty 2).
- `hyperparameter_optimizer.py`: Loads the trained GP models and predicts `k`, `ell`, `tau`.
- `feature_extractor.py`: Extracts complexity features from HTML for ML-based adaptations.
- `README_ML_PIPELINE.md`: Detailed docs for the adaptive training pipeline.

## Requirements
- Python 3.9+
- Google Chrome and matching ChromeDriver on your `PATH` for Selenium-based crawling and training.
- Python deps: `pip install -r requirements.txt` (BeautifulSoup4, lxml, requests, selenium, numpy, scikit-learn).

## Quick start (Streamlit UI)
1. (Optional) create a venv: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Ensure ChromeDriver matches your Chrome version (`chromedriver --version`) and is on `PATH`.
4. Launch UI: `streamlit run frontend/app.py`
5. In the UI:
   - Static mode fetches the provided URLs via `requests`.
   - Dynamic mode opens the start URL in headless Chrome, follows same-domain links, and captures up to `max_states`.
   - Adjust `k`, `ell`, and `tau` under **Advanced MinHash parameters**, then run. The UI reports total vs unique states and lets you preview the first unique DOM.

## Programmatic usage
```python
from backend.fetcher import fetch_static_html, fetch_dynamic_states
from backend.core import MinHashLSH

# Collect states (static example)
states = fetch_static_html(["https://example.com"])
# Or dynamic: states = fetch_dynamic_states("https://quotes.toscrape.com/js/", max_states=5)

detector = MinHashLSH(k=12, ell=200, tau=0.85)  # or adaptive=True with a trained model
for html in states:
    detector.add_state(html)

unique_states = detector.get_unique_states()
print(f"Unique states: {len(unique_states)}")
```

### Using Novel Features

**Novelty 1: Dynamic Threshold Adaptation**
```python
detector = MinHashLSH(k=12, ell=200, tau=0.85, dynamic_threshold=True)

```

**Novelty 2: Context-Aware Shingling**
```python
# Enable context-aware shingling with security-weighted elements
detector = MinHashLSH(k=12, ell=200, tau=0.85, context_aware=True)
# Elements are weighted by security relevance (forms, scripts, event handlers)
# Vulnerability-prone components are prioritized in similarity detection
```

**Combined Usage (ML-Enhanced)**
```python
# Use both ML-enhanced novelties together
detector = MinHashLSH(
    k=12, ell=200, tau=0.85,
    dynamic_threshold=True,        # Novelty 1
    context_aware=True,            # Novelty 2
    use_ml_for_threshold=True,     # Use ML for Novelty 1 (default)
    use_ml_for_shingling=True     # Use ML for Novelty 2 (default)
)
```

**Heuristic-Only Mode** (if ML model unavailable):
```python
# Falls back to heuristic rules automatically
detector = MinHashLSH(
    k=12, ell=200, tau=0.85,
    dynamic_threshold=True,
    context_aware=True
    # ML flags default to True but will fallback if model missing
)
```

- Adaptive mode: `MinHashLSH(adaptive=True, html_sample=states[0])` will load `models/bayesian_optimizer.pkl` via `hyperparameter_optimizer.py`. Without the trained model, it raises a clear error—run the training pipeline first.
- Run scripts from the repo root so relative imports (e.g., `hyperparameter_optimizer`) resolve correctly.

## Adaptive ML pipeline (summary)
Run from the repo root; see `README_ML_PIPELINE.md` for details and caveats.
1. `python scrape_websites.py` – scrape diverse sites (static + Selenium); saves HTML under `training_data/raw_html/` and features to `training_data/features.json`.
2. `python collect_training_data.py` – grid-search optimal `k`, `ell`, `tau` per site type; writes `training_data/training_dataset.json`.
3. `python train_bayesian_model.py` – train Gaussian Process models; saves to `models/bayesian_optimizer.pkl`.
4. (Optional) `python hyperparameter_optimizer.py` – sanity-check model loading/predictions.

## Novel Features Explained

### Novelty 1: Dynamic Threshold Adaptation (ML-Enhanced)

**Problem**: Fixed similarity threshold (`tau`) may not optimally balance coverage and redundancy across diverse RIAs. Simple static pages need stricter thresholds, while complex dynamic apps need more lenient ones.

**Solution**: The system uses **trained Bayesian model** to predict optimal `tau` per state, then blends it with base threshold for adaptive optimization. Falls back to heuristic rules if ML model unavailable.

**ML-Based Approach**:
- Uses `HyperparameterOptimizer` to predict optimal `tau` from HTML features
- Blends ML prediction (70%) with base threshold (30%) for stability
- Automatically adapts based on learned patterns from training data

**Heuristic Fallback** (when ML unavailable):
- DOM size and depth (larger/deeper = lower threshold)
- JavaScript presence (JS-heavy apps = lower threshold)
- Form elements count (many forms = lower threshold)
- AJAX indicators (dynamic content = lower threshold)

**Benefits**:
- Better coverage for complex RIAs without excessive false positives
- Reduced redundancy for simple static pages
- ML-learned optimal thresholds per application type
- Automatic optimization based on training data patterns

**Implementation**: `compute_adaptive_threshold()` in `backend/core.py` uses ML model when `use_ml_for_threshold=True` (default).

### Novelty 2: Context-Aware Shingling (ML-Enhanced)

**Problem**: Standard k-mer shingling treats all DOM elements equally, missing security-relevant patterns. Vulnerability-prone components (forms, scripts, event handlers) should have higher weight in similarity detection.

**Solution**: Security-weighted shingling using **trained Bayesian model** to predict optimal security relevance weights. Falls back to heuristic rules if ML model unavailable.

**ML-Based Approach**:
- Uses `HyperparameterOptimizer` predictions (`k`, `ell`, `tau`) to infer security relevance
- Higher predicted `tau` = more security-critical = higher weight
- Higher predicted `k`/`ell` = more complexity = higher security relevance
- Combines ML predictions with security indicators (forms, JS, AJAX)
- Tag-specific security boosts based on learned patterns

**Heuristic Fallback** (when ML unavailable):
- Base weights: forms (3.0x), scripts (3.5x), inputs (2.5x)
- Attribute boosts: event handlers (+0.5x), security attributes (+0.3x)
- Feature-based complexity adjustments

**Benefits**:
- Prioritizes vulnerability-prone components using ML-learned patterns
- Reduces false positives by focusing on security-relevant DOM structure
- Better detection of security-critical state changes
- ML-optimized weights based on training data

**Implementation**: `SecurityRelevanceScorer` in `security_relevance.py` uses ML model when `use_ml_model=True` (default).

## Notes and troubleshooting
- Chrome/ChromeDriver versions must match; reinstall/update ChromeDriver if Selenium cannot start.
- Dynamic crawling restricts BFS to links starting with the start URL to avoid runaway crawls; still stay within your target domain and respect `robots.txt`.
- If Selenium is unavailable, static fetching still works; adaptive mode and training also require `scikit-learn`.
- Novel features (`dynamic_threshold`, `context_aware`) are enabled by default when using `adaptive=True`, but can be toggled independently.
- **ML Integration**: Both novelties use trained Bayesian model by default (`use_ml_for_threshold=True`, `use_ml_for_shingling=True`). They automatically fallback to heuristic rules if model unavailable.
- Context-aware shingling requires `security_relevance.py` module; falls back to standard shingling if unavailable.
- **Testing ML Novelties**: Run `python test_ml_novelties.py` to test ML-enhanced novelties (requires trained model).
- Tests: `python -m unittest test_adaptive_hyperparams.py` (requires a trained model for adaptive tests).

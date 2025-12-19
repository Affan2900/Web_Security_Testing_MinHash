import html.parser
import hashlib
import random
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TagExtractor(html.parser.HTMLParser):
    """Extracts sequence of opening tag names from HTML in document order."""

    def __init__(self):
        super().__init__()
        self.tags = []  # List to store opening tags

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag.lower())  # Normalize to lowercase

    def handle_startendtag(self, tag, attrs):
        self.tags.append(tag.lower())  # Treat self-closing as start


def extract_tags(html_content):
    """Parse HTML and return list of opening tags."""
    parser = TagExtractor()
    parser.feed(html_content)
    return parser.tags


def generate_shingles(tags, k):
    """Generate set of k-shingles (tuples of k consecutive tags)."""
    if len(tags) < k:
        return set()
    shingles = set()
    for i in range(len(tags) - k + 1):
        shingle = tuple(tags[i : i + k])
        shingles.add(shingle)
    return shingles


def compute_adaptive_threshold(base_tau, html_content, use_ml=True):
    """
    Compute adaptive Jaccard similarity threshold based on RIA complexity.
    
    Novelty 1: Dynamic Threshold Adaptation (ML-Enhanced)
    Uses trained Bayesian model to predict optimal tau per state, then blends
    with base_tau for adaptive threshold that optimizes coverage vs redundancy.
    
    Args:
        base_tau (float): Base threshold value
        html_content (str): HTML content to analyze
        use_ml (bool): If True, use ML model; otherwise use heuristic fallback
        
    Returns:
        float: Adapted threshold value (typically 0.60-0.95)
    """
    try:
        if use_ml:
            # ML-based approach: Use Bayesian model to predict optimal tau
            try:
                from hyperparameter_optimizer import HyperparameterOptimizer
                # Use None to auto-detect model path
                optimizer = HyperparameterOptimizer(model_path=None)
                _, _, ml_tau = optimizer.predict_hyperparameters(html_content)
                
                # Blend ML prediction with base_tau (weighted average)
                # ML prediction is more accurate, but base_tau provides stability
                # Weight: 70% ML prediction, 30% base_tau
                adaptive_tau = 0.7 * ml_tau + 0.3 * base_tau
                
                # Clamp to valid range
                adaptive_tau = max(0.60, min(0.95, adaptive_tau))
                
                return adaptive_tau
                
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
                # Fallback to heuristic if ML model unavailable or has compatibility issues
                # ValueError can occur with numpy version mismatches in pickle
                print(f"[WARNING] ML model not available for dynamic threshold: {e}")
                print("[INFO] Falling back to heuristic-based adaptation")
                use_ml = False
            except Exception as e:
                # Catch any other unexpected errors
                print(f"[WARNING] Unexpected error loading ML model for threshold: {e}")
                print("[INFO] Falling back to heuristic-based adaptation")
                use_ml = False
        
        if not use_ml:

            from feature_extractor import extract_features
            features = extract_features(html_content)
            
            # Base adjustment factors
            adjustment = 0.0
            
            # Low complexity = higher threshold (more strict) to reduce false positives
            dom_size = features.get('dom_size', 0)
            dom_depth = features.get('dom_depth', 0)
            form_elements = features.get('form_elements', 0)
            js_detected = features.get('js_detected', 0)
            ajax_indicators = features.get('ajax_indicators', 0)
            
            # Complexity-based adjustments
            if dom_size > 2000:
                adjustment -= 0.08  # Large DOMs need lower threshold
            elif dom_size < 100:
                adjustment += 0.05  # Small DOMs can use higher threshold
            
            if dom_depth > 20:
                adjustment -= 0.05  # Deep DOMs are more variable
            
            # Dynamic content indicators
            if js_detected == 1:
                adjustment -= 0.06  # JS-heavy apps need lower threshold
            if ajax_indicators > 2:
                adjustment -= 0.04  # AJAX-heavy apps are more dynamic
            
            # Form-heavy pages need careful handling
            if form_elements > 10:
                adjustment -= 0.05  # Many forms = more state variations
            elif form_elements == 0:
                adjustment += 0.03  # No forms = more static
            
            # Clamp adjustment to reasonable range
            adjustment = max(-0.15, min(0.10, adjustment))
            
            # Compute adaptive threshold
            adaptive_tau = base_tau + adjustment
            
            # Clamp to valid range [0.60, 0.95]
            adaptive_tau = max(0.60, min(0.95, adaptive_tau))
            
            return adaptive_tau
        
    except Exception:
        # Final fallback to base threshold if everything fails
        return base_tau


def universal_hash(seed, x):
    """Simple universal hash using MD5 with seed."""
    m = hashlib.md5()
    m.update(str(seed).encode("utf-8") + str(x).encode("utf-8"))
    return int(m.hexdigest(), 16)


class MinHashLSH:
    def __init__(self, k=12, ell=200, tau=0.85, adaptive=False, html_sample=None,
                 dynamic_threshold=None, context_aware=None, use_ml_for_threshold=True,
                 use_ml_for_shingling=True):
        """
        Initialize MinHash LSH detector.
        
        Args:
            k (int): Shingle size
            ell (int): Number of hash functions (sketch size)
            tau (float): Base Jaccard similarity threshold
            adaptive (bool): If True, use ML model to select hyperparameters
            html_sample (str): Sample HTML for adaptive parameter selection
            dynamic_threshold (bool): If True, use dynamic threshold adaptation (Novelty 1).
                                     If None, auto-detects based on ML model availability.
            context_aware (bool): If True, use context-aware shingling (Novelty 2).
                                  If None, auto-detects based on ML model availability.
            use_ml_for_threshold (bool): If True, use ML model for dynamic threshold (Novelty 1)
            use_ml_for_shingling (bool): If True, use ML model for context-aware shingling (Novelty 2)
        """
        # Auto-detect ML model availability
        ml_model_available = False
        try:
            # Try multiple possible paths for the model
            possible_paths = [
                'models/bayesian_optimizer.pkl',  # From project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'models', 'bayesian_optimizer.pkl'),  # Relative to backend/core.py
            ]
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    ml_model_available = True
                    break
        except Exception:
            # If detection fails, assume model not available
            pass
        
        # Auto-enable novelties if model available and not explicitly set
        if dynamic_threshold is None:
            dynamic_threshold = ml_model_available  # Enable if model exists
        if context_aware is None:
            context_aware = ml_model_available  # Enable if model exists
        
        # Log auto-detection result
        if ml_model_available and (dynamic_threshold or context_aware):
            print(f"[Auto-Detection] ML model found - Novelties enabled: "
                  f"dynamic_threshold={dynamic_threshold}, context_aware={context_aware}")
        if adaptive and html_sample:
            try:
                from hyperparameter_optimizer import predict_hyperparameters
                k, ell, tau = predict_hyperparameters(html_sample)
                print(f"[Adaptive Mode] ML-predicted hyperparameters: k={k}, ell={ell}, tau={tau:.2f}")
            except FileNotFoundError as e:
                print(f"\n[ERROR] Adaptive mode requires trained ML model!")
                print(f"Please run the training pipeline:")
                print(f"  1. python scrape_websites.py")
                print(f"  2. python collect_training_data.py")
                print(f"  3. python train_bayesian_model.py\n")
                raise
            except Exception as e:
                print(f"[ERROR] Adaptive mode failed: {e}")
                raise
        
        self.k = k  # Shingle size
        self.ell = ell  # Number of hash functions (sketch size)
        self.base_tau = tau  # Base Jaccard similarity threshold
        self.dynamic_threshold = dynamic_threshold  # Novelty 1: Dynamic threshold adaptation
        self.context_aware = context_aware  # Novelty 2: Context-aware shingling
        self.use_ml_for_threshold = use_ml_for_threshold  # Use ML model for dynamic threshold
        self.use_ml_for_shingling = use_ml_for_shingling  # Use ML model for context-aware shingling
        self.hash_functions = [
            random.randint(0, 2**32 - 1) for _ in range(ell)
        ]  # Seeds for hash funcs
        self.hash_tables = [
            defaultdict(set) for _ in range(ell)
        ]  # LSH tables: list of dict(value -> set(ids))
        self.unique_states = {}  # id -> original html_content
        self.next_id = 0
        
        # Initialize security relevance scorer if context-aware mode enabled
        if self.context_aware:
            try:
                from security_relevance import SecurityRelevanceScorer, generate_context_aware_shingles
                self.security_scorer = SecurityRelevanceScorer(
                    use_ml_weights=True,
                    use_ml_model=self.use_ml_for_shingling
                )
                self.generate_context_aware_shingles = generate_context_aware_shingles
            except ImportError:
                print("[WARNING] Context-aware shingling requested but security_relevance module not found.")
                print("[WARNING] Falling back to standard shingling.")
                self.context_aware = False

    def compute_sketch(self, shingles):
        """Compute MinHash sketch: list of ell min-hash values."""
        sketch = []
        for i in range(self.ell):
            min_val = float("inf")
            for shingle in shingles:
                # Handle both standard shingles (tuples) and context-aware shingles (tuples with weights)
                if isinstance(shingle, tuple) and len(shingle) == 2 and isinstance(shingle[1], float):
                    # Context-aware shingle: (tag_tuple, weight)
                    shingle_to_hash = shingle[0]
                else:
                    # Standard shingle: tuple of tags
                    shingle_to_hash = shingle
                
                h_val = universal_hash(self.hash_functions[i], shingle_to_hash)
                if h_val < min_val:
                    min_val = h_val
            sketch.append(min_val)
        return sketch
    
    def _extract_shingles(self, html_content):
        """
        Extract shingles using standard or context-aware method.
        
        Novelty 2: Context-Aware Shingling
        Uses security-weighted elements to prioritize vulnerability-prone components.
        
        Args:
            html_content (str): HTML content to process
            
        Returns:
            set: Set of shingles (standard or context-aware)
        """
        if self.context_aware:
            try:
                weighted_tags = self.security_scorer.extract_weighted_tags(html_content)
                shingles = self.generate_context_aware_shingles(weighted_tags, self.k, min_weight=0.5)
                return shingles
            except Exception as e:
                # Fallback to standard shingling if context-aware fails
                print(f"[WARNING] Context-aware shingling failed: {e}. Using standard shingling.")
                tags = extract_tags(html_content)
                return generate_shingles(tags, self.k)
        else:
            # Standard shingling
            tags = extract_tags(html_content)
            return generate_shingles(tags, self.k)

    def is_duplicate(self, html_content):
        """
        Check if state is duplicate based on max estimated Jaccard similarity.
        
        Novelty 1: Uses dynamic threshold adaptation if enabled.
        Novelty 2: Uses context-aware shingling if enabled.
        """
        # Extract shingles (context-aware or standard)
        shingles = self._extract_shingles(html_content)
        if not shingles:
            return True  # Empty/invalid -> treat as duplicate

        sketch = self.compute_sketch(shingles)

        # Count collisions per existing doc_id
        collision_counts = defaultdict(int)
        for i in range(self.ell):
            v = sketch[i]
            bucket = self.hash_tables[i][v]
            for doc_id in bucket:
                collision_counts[doc_id] += 1

        if not collision_counts:
            return False  # No collisions -> unique

        max_count = max(collision_counts.values())
        max_sim = max_count / self.ell
        
        # Novelty 1: Dynamic threshold adaptation (ML-enhanced)
        if self.dynamic_threshold:
            adaptive_tau = compute_adaptive_threshold(
                self.base_tau, html_content, use_ml=self.use_ml_for_threshold
            )
            return max_sim >= adaptive_tau
        else:
            return max_sim >= self.base_tau

    def add_state(self, html_content):
        """Add state if not duplicate, update LSH tables."""
        if self.is_duplicate(html_content):
            return False  # Duplicate, not added
        
        # Extract shingles (context-aware or standard)
        shingles = self._extract_shingles(html_content)
        sketch = self.compute_sketch(shingles)
        state_id = self.next_id
        self.unique_states[state_id] = html_content
        for i in range(self.ell):
            v = sketch[i]
            self.hash_tables[i][v].add(state_id)
        self.next_id += 1
        return True  # Added as unique

    def get_unique_states(self):
        """Return list of unique HTML contents."""
        return list(self.unique_states.values())


if __name__ == "__main__":
    # Sample HTML states (replace with your real pages)
    sample_htmls = [
        "<html><head><title>Test</title></head><body><p>Hello</p><a href='#'>Link</a></body></html>",
        "<html><head><title>Test</title></head><body><p>Hello World</p><a href='#'>Link</a></body></html>",  # Similar
        "<html><body><div><ul><li>Item1</li><li>Item2</li></ul></div></body></html>",  # Different
    ]

    # Test with default (hardcoded) parameters
    print("\n=== Test 1: Default Parameters ===")
    detector = MinHashLSH(k=3, ell=10, tau=0.8)  # Small params for demo
    for html in sample_htmls:
        added = detector.add_state(html)
        print(f"Added as unique: {added}")
    
    # Test with adaptive parameters
    print("\n=== Test 2: Adaptive Parameters ===")
    detector_adaptive = MinHashLSH(adaptive=True, html_sample=sample_htmls[0])
    for html in sample_htmls:
        added = detector_adaptive.add_state(html)
        print(f"Added as unique: {added}")

    # Test with novelties enabled
    print("\n=== Test 3: With Novelties (Dynamic Threshold + Context-Aware) ===")
    detector_novel = MinHashLSH(k=3, ell=10, tau=0.8, dynamic_threshold=True, context_aware=True)
    for html in sample_htmls:
        added = detector_novel.add_state(html)
        print(f"Added as unique: {added}")

    uniques = detector.get_unique_states()
    print(f"Unique states found: {len(uniques)}")


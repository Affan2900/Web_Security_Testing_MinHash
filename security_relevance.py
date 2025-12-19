"""
Security Relevance Scorer for Context-Aware Shingling
Weights DOM elements based on their security relevance for vulnerability detection.
Uses ML-based scoring to prioritize vulnerability-prone components.
"""

import re
from bs4 import BeautifulSoup
from collections import Counter
from feature_extractor import extract_features


# Security-relevant tag weights (base weights, can be enhanced with ML)
SECURITY_TAG_WEIGHTS = {
    # Form elements - high security relevance
    'form': 3.0,
    'input': 2.5,
    'textarea': 2.0,
    'select': 2.0,
    'button': 2.0,
    
    # Script and dynamic content - high relevance
    'script': 3.5,
    'iframe': 3.0,
    'embed': 2.5,
    'object': 2.5,
    
    # Event handlers and interactive elements
    'a': 1.5,  # Links can have onclick handlers
    'div': 1.2,  # Often used with event handlers
    'span': 1.1,
    
    # Security-related attributes
    'meta': 1.5,  # CSP, security headers
    
    # Default weight for other tags
    '_default': 1.0
}


# Security-relevant attributes
SECURITY_ATTRIBUTES = {
    'onclick', 'onload', 'onerror', 'onfocus', 'onblur',
    'onchange', 'onsubmit', 'onmouseover', 'onmouseout',
    'data-ajax', 'data-action', 'ng-click', 'v-on:click',
    'action', 'method', 'type', 'name', 'id', 'class',
    'src', 'href', 'data-src', 'data-href'
}


class SecurityRelevanceScorer:
    """
    Scores DOM elements based on security relevance for context-aware shingling.
    Uses ML model (Bayesian optimizer) to predict optimal security weights.
    """
    
    def __init__(self, use_ml_weights=True, use_ml_model=True):
        """
        Initialize the security relevance scorer.
        
        Args:
            use_ml_weights (bool): If True, use ML-based feature analysis
                                   to adjust weights dynamically
            use_ml_model (bool): If True, use trained Bayesian model for predictions
        """
        self.use_ml_weights = use_ml_weights
        self.use_ml_model = use_ml_model
        self.base_weights = SECURITY_TAG_WEIGHTS.copy()
        self.ml_optimizer = None
        
        # Initialize ML optimizer lazily (only when needed, not during __init__)
        # This prevents errors during training when model doesn't exist yet
        self._ml_optimizer_initialized = False
    
    def _ensure_ml_optimizer(self):
        """Lazily initialize ML optimizer only when needed."""
        if not self.use_ml_model:
            return False
        
        if self._ml_optimizer_initialized:
            return self.ml_optimizer is not None
        
        self._ml_optimizer_initialized = True
        
        try:
            from hyperparameter_optimizer import HyperparameterOptimizer
            self.ml_optimizer = HyperparameterOptimizer()
            print("[ML Model] Loaded Bayesian optimizer for context-aware shingling")
            return True
        except FileNotFoundError as e:
            print(f"[WARNING] ML model not found for context-aware shingling: {e}")
            print("[INFO] Falling back to heuristic-based weights")
            self.use_ml_model = False
            self.ml_optimizer = None
            return False
        except (ImportError, RuntimeError, ValueError) as e:
            # Handle numpy compatibility issues and other errors gracefully
            print(f"[WARNING] ML model loading failed (may be version incompatibility): {e}")
            print("[INFO] Falling back to heuristic-based weights")
            self.use_ml_model = False
            self.ml_optimizer = None
            return False
        except Exception as e:
            print(f"[WARNING] Unexpected error loading ML model: {e}")
            print("[INFO] Falling back to heuristic-based weights")
            self.use_ml_model = False
            self.ml_optimizer = None
            return False
    
    def get_element_weight(self, tag_name, attrs=None, html_context=None):
        """
        Get security relevance weight for a DOM element.
        
        Args:
            tag_name (str): HTML tag name
            attrs (list): List of (attr_name, attr_value) tuples
            html_context (str): Full HTML content for context analysis
            
        Returns:
            float: Security relevance weight (higher = more security-relevant)
        """
        tag_name = tag_name.lower()
        
        # Base weight from tag type
        base_weight = self.base_weights.get(tag_name, self.base_weights['_default'])
        
        # Boost weight if element has security-relevant attributes
        attr_boost = 1.0
        if attrs:
            for attr_name, attr_value in attrs:
                attr_name_lower = attr_name.lower()
                if attr_name_lower in SECURITY_ATTRIBUTES:
                    attr_boost += 0.3
                # Check for event handlers
                if attr_name_lower.startswith('on') or 'click' in attr_name_lower:
                    attr_boost += 0.5
                # Check for form-related attributes
                if attr_name_lower in ['action', 'method', 'name', 'type']:
                    attr_boost += 0.4
        
        # ML-based adjustment if enabled
        ml_adjustment = 1.0
        if self.use_ml_weights and html_context:
            ml_adjustment = self._compute_ml_adjustment(tag_name, html_context)
        
        return base_weight * attr_boost * ml_adjustment
    
    def _compute_ml_adjustment(self, tag_name, html_content):
        """
        Compute ML-based weight adjustment using trained Bayesian model.
        
        Novelty 2: Context-Aware Shingling (ML-Enhanced)
        Uses ML model predictions to determine security relevance weights,
        prioritizing vulnerability-prone components based on learned patterns.
        
        Args:
            tag_name (str): Tag name being scored
            html_content (str): Full HTML content
            
        Returns:
            float: ML adjustment factor (typically 0.8-2.0)
        """
        try:
            # Lazy load ML optimizer only when needed
            if self.use_ml_model and self._ensure_ml_optimizer() and self.ml_optimizer:
                # Use trained Bayesian model to predict optimal hyperparameters
                # These predictions indicate complexity and security relevance
                k_pred, ell_pred, tau_pred = self.ml_optimizer.predict_hyperparameters(html_content)
                
                # Extract features for additional analysis
                features = extract_features(html_content)
                
                # ML-based security relevance calculation
                # Higher predicted tau = more strict matching needed = higher security relevance
                # Higher predicted k/ell = more complexity = higher security relevance
                
                # Base adjustment from ML predictions
                # tau_pred indicates how strict matching should be (higher = more security-critical)
                tau_factor = (tau_pred - 0.60) / (0.95 - 0.60)  # Normalize to [0, 1]
                
                # Complexity factor from k and ell predictions
                k_factor = (k_pred - 5) / (25 - 5)  # Normalize k to [0, 1]
                ell_factor = (ell_pred - 50) / (500 - 50)  # Normalize ell to [0, 1]
                complexity_factor = (k_factor + ell_factor) / 2
                
                # Security relevance indicators from features
                security_indicators = 0.0
                if features.get('form_elements', 0) > 5:
                    security_indicators += 0.3
                if features.get('js_detected', 0) == 1:
                    security_indicators += 0.2
                if features.get('ajax_indicators', 0) > 0:
                    security_indicators += 0.2
                
                # Tag-specific security boost
                tag_security_boost = 0.0
                if tag_name in ['form', 'input', 'script', 'iframe']:
                    tag_security_boost = 0.4
                elif tag_name in ['button', 'textarea', 'select']:
                    tag_security_boost = 0.2
                
                # Combine ML predictions with security indicators
                ml_adjustment = 1.0 + (tau_factor * 0.3) + (complexity_factor * 0.2) + \
                               security_indicators + tag_security_boost
                
                # Clamp to reasonable range [0.8, 2.0]
                ml_adjustment = max(0.8, min(2.0, ml_adjustment))
                
                return ml_adjustment
                
            elif self.use_ml_weights:
                # Heuristic fallback: Use feature-based analysis
                features = extract_features(html_content)
                
                # Higher complexity = more security concerns = higher weight
                complexity_factor = 0.0
                
                # DOM complexity indicators
                if features.get('dom_size', 0) > 1000:
                    complexity_factor += 0.1
                if features.get('dom_depth', 0) > 15:
                    complexity_factor += 0.1
                if features.get('form_elements', 0) > 5:
                    complexity_factor += 0.15
                if features.get('js_detected', 0) == 1:
                    complexity_factor += 0.1
                if features.get('ajax_indicators', 0) > 0:
                    complexity_factor += 0.1
                
                # Tag-specific adjustments
                if tag_name in ['form', 'input', 'script']:
                    complexity_factor += 0.2
                
                # Normalize to reasonable range [0.8, 1.5]
                ml_adjustment = 1.0 + min(complexity_factor, 0.5)
                return ml_adjustment
            
            else:
                # No ML adjustment
                return 1.0
            
        except Exception as e:
            # Fallback to no adjustment if ML fails
            if self.use_ml_model:
                print(f"[WARNING] ML adjustment failed: {e}. Using heuristic fallback.")
            return 1.0
    
    def extract_weighted_tags(self, html_content):
        """
        Extract tags with security relevance weights from HTML.
        
        Args:
            html_content (str): HTML content to analyze
            
        Returns:
            list: List of (tag_name, weight) tuples in document order
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            weighted_tags = []
            
            # Extract all elements in document order
            for element in soup.find_all():
                tag_name = element.name.lower() if element.name else 'unknown'
                attrs = element.attrs.items() if hasattr(element, 'attrs') else []
                weight = self.get_element_weight(tag_name, list(attrs), html_content)
                
                # Include tag multiple times based on weight (rounded)
                # Higher weight = more representation in shingles
                count = max(1, int(round(weight)))
                for _ in range(count):
                    weighted_tags.append((tag_name, weight))
            
            return weighted_tags
            
        except Exception:
            # Fallback to simple tag extraction
            try:
                from backend.core import extract_tags
            except ImportError:
                # Alternative import path
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from backend.core import extract_tags
            tags = extract_tags(html_content)
            return [(tag, 1.0) for tag in tags]


def generate_context_aware_shingles(weighted_tags, k, min_weight=0.5):
    """
    Generate context-aware shingles with security-weighted elements.
    Filters and weights shingles based on security relevance.
    
    Args:
        weighted_tags (list): List of (tag_name, weight) tuples
        k (int): Shingle size
        min_weight (float): Minimum weight threshold for shingle inclusion
        
    Returns:
        set: Set of weighted shingles (tuples)
    """
    if len(weighted_tags) < k:
        return set()
    
    shingles = set()
    for i in range(len(weighted_tags) - k + 1):
        shingle_tags = weighted_tags[i:i + k]
        
        # Extract tag names and compute average weight
        tag_names = tuple(tag for tag, _ in shingle_tags)
        avg_weight = sum(weight for _, weight in shingle_tags) / k
        
        # Only include shingles above weight threshold
        if avg_weight >= min_weight:
            # Create weighted shingle representation
            # Include weight in shingle for hashing
            weighted_shingle = (tag_names, round(avg_weight, 2))
            shingles.add(weighted_shingle)
    
    return shingles


# Example usage
if __name__ == "__main__":
    sample_html = """
    <html>
        <head>
            <script src="app.js"></script>
        </head>
        <body>
            <form action="/submit" method="post">
                <input type="text" name="username" onclick="validate()">
                <input type="password" name="password">
                <button type="submit">Login</button>
            </form>
            <div>
                <p>Regular content</p>
            </div>
        </body>
    </html>
    """
    
    scorer = SecurityRelevanceScorer(use_ml_weights=True)
    weighted_tags = scorer.extract_weighted_tags(sample_html)
    
    print("Weighted Tags:")
    for tag, weight in weighted_tags[:10]:  # Show first 10
        print(f"  {tag}: {weight:.2f}")
    
    shingles = generate_context_aware_shingles(weighted_tags, k=3)
    print(f"\nGenerated {len(shingles)} context-aware shingles (k=3)")


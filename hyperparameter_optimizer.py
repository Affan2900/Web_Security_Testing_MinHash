"""
Bayesian Optimization-based Hyperparameter Optimizer for MinHash LSH
Uses trained Gaussian Process Regression models to predict optimal hyperparameters.

PURE ML APPROACH - No heuristic fallback.
"""

import os
import pickle
import numpy as np
from feature_extractor import extract_features

# Try to import ML libraries
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ERROR: scikit-learn not installed!")
    print("Install with: pip install scikit-learn")


class HyperparameterOptimizer:
    """
    Bayesian Optimization-based hyperparameter optimizer.
    Requires trained models - no fallback.
    """
    
    def __init__(self, model_path='models/bayesian_optimizer.pkl'):
        """
        Initialize the optimizer and load trained models.
        
        Args:
            model_path (str): Path to saved trained models
        """
        self.model_path = model_path
        self.models = {'k': None, 'ell': None, 'tau': None}
        self.is_trained = False
        
        # Load models (required)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please run the training pipeline:\n"
                f"  1. python scrape_websites.py\n"
                f"  2. python collect_training_data.py\n"
                f"  3. python train_bayesian_model.py"
            )
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required but not installed.\n"
                "Install with: pip install scikit-learn"
            )
        
        self._load_models()
    
    def predict_hyperparameters(self, html_content):
        """
        Predict optimal hyperparameters for given HTML content.
        
        Args:
            html_content (str): HTML string to analyze
            
        Returns:
            tuple: (k, ell, tau) predicted hyperparameter values
        """
        if not self.is_trained:
            raise RuntimeError("Models not loaded. Check model path.")
        
        # Extract features
        features = extract_features(html_content)
        feature_vector = self._features_to_vector(features)
        
        # Predict with each model
        k_pred = self.models['k'].predict([feature_vector])[0]
        ell_pred = self.models['ell'].predict([feature_vector])[0]
        tau_pred = self.models['tau'].predict([feature_vector])[0]
        
        # Round and clamp to reasonable ranges
        k_pred = int(round(max(5, min(25, k_pred))))
        ell_pred = int(round(max(50, min(500, ell_pred))))
        tau_pred = float(max(0.60, min(0.95, tau_pred)))
        
        return k_pred, ell_pred, tau_pred
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                self.models = pickle.load(f)
            self.is_trained = True
            print(f"[ML Model] Loaded trained Bayesian models from: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
    
    def _features_to_vector(self, features):
        """
        Convert feature dictionary to numpy array.
        
        Args:
            features (dict): Feature dictionary from extract_features()
            
        Returns:
            np.array: Feature vector for ML model
        """
        # Order must match training!
        feature_order = [
            'dom_size', 'dom_depth', 'branching_factor', 'unique_tags',
            'tag_diversity', 'tag_entropy', 'js_detected', 'form_elements',
            'ajax_indicators', 'html_size', 'external_resources', 'div_span_ratio'
        ]
        
        vector = [features.get(key, 0) for key in feature_order]
        return np.array(vector)


# Convenience function for quick usage
def predict_hyperparameters(html_content, model_path='models/bayesian_optimizer.pkl'):
    """
    Convenience function to predict hyperparameters.
    
    Args:
        html_content (str): HTML string to analyze
        model_path (str): Path to trained model
        
    Returns:
        tuple: (k, ell, tau) predicted hyperparameter values
    """
    optimizer = HyperparameterOptimizer(model_path)
    return optimizer.predict_hyperparameters(html_content)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Bayesian Hyperparameter Optimizer (Pure ML)")
    print("=" * 60)
    
    # Test with sample HTML
    sample_html = "<html><body>" + "<div>" * 1000 + "Content" + "</div>" * 1000 + "</body></html>"
    
    try:
        k, ell, tau = predict_hyperparameters(sample_html)
        print(f"\n✓ Predicted hyperparameters:")
        print(f"  k (shingle size): {k}")
        print(f"  ell (sketch size): {ell}")
        print(f"  tau (similarity threshold): {tau:.2f}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

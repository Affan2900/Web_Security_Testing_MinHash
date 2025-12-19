"""
Unit Tests for ML-Based Adaptive Hyperparameter Selection
Tests feature extraction and MinHashLSH integration with ML models.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import extract_features, get_default_features
from core import MinHashLSH


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction from HTML."""
    
    def test_simple_html(self):
        """Test feature extraction on simple HTML."""
        simple_html = "<html><body><p>Hello World</p></body></html>"
        features = extract_features(simple_html)
        
        # Check all features exist
        self.assertIn('dom_size', features)
        self.assertIn('dom_depth', features)
        self.assertIn('unique_tags', features)
        
        # Check reasonable values
        self.assertGreater(features['dom_size'], 0)
        self.assertGreaterEqual(features['dom_depth'], 0)
    
    def test_complex_html(self):
        """Test feature extraction on complex HTML."""
        complex_html = "<html><head><script src='app.js'></script></head><body>" + \
                       "<div>" * 100 + "Content" + "</div>" * 100 + "</body></html>"
        features = extract_features(complex_html)
        
        # Complex HTML should have larger DOM size
        self.assertGreater(features['dom_size'], 50)
        self.assertEqual(features['js_detected'], 1)
    
    def test_invalid_html(self):
        """Test feature extraction on invalid input."""
        features = extract_features("")
        self.assertEqual(features, get_default_features())
        
        features = extract_features(None)
        self.assertEqual(features, get_default_features())
    
    def test_js_detection(self):
        """Test JavaScript detection."""
        html_with_js = "<html><body><script>alert('test');</script></body></html>"
        features = extract_features(html_with_js)
        self.assertEqual(features['js_detected'], 1)
        
        html_without_js = "<html><body><p>No JS here</p></body></html>"
        features = extract_features(html_without_js)
        self.assertEqual(features['js_detected'], 0)



class TestMinHashLSHIntegration(unittest.TestCase):
    """Test integration of adaptive mode with MinHashLSH."""
    
    def test_default_mode(self):
        """Test MinHashLSH with default parameters."""
        detector = MinHashLSH(k=10, ell=100, tau=0.8)
        
        self.assertEqual(detector.k, 10)
        self.assertEqual(detector.ell, 100)
        self.assertEqual(detector.tau, 0.8)
    
    def test_adaptive_mode_with_sample(self):
        """Test MinHashLSH with adaptive parameter selection."""
        sample_html = "<html><body>" + "<div>" * 500 + "Content" + "</div>" * 500 + "</body></html>"
        detector = MinHashLSH(adaptive=True, html_sample=sample_html)
        
        # Should have selected params (different from default likely)
        self.assertIsNotNone(detector.k)
        self.assertIsNotNone(detector.ell)
        self.assertIsNotNone(detector.tau)
        
        # Params should be in reasonable ranges
        self.assertGreater(detector.k, 5)
        self.assertLess(detector.k, 25)
        self.assertGreater(detector.ell, 50)
        self.assertLess(detector.ell, 500)
        self.assertGreater(detector.tau, 0.6)
        self.assertLess(detector.tau, 0.95)
    
    def test_adaptive_mode_without_sample(self):
        """Test adaptive mode without HTML sample (should use defaults)."""
        detector = MinHashLSH(k=12, ell=200, tau=0.85, adaptive=True, html_sample=None)
        
        # Without sample, should keep provided defaults
        self.assertEqual(detector.k, 12)
        self.assertEqual(detector.ell, 200)
        self.assertEqual(detector.tau, 0.85)
    
    def test_duplicate_detection_works(self):
        """Test that duplicate detection still works with appropriate parameters."""
        # Use HTML with enough tags to generate shingles (need at least k tags)
        html1 = "<html><head><title>Test</title></head><body><div><ul><li><p>Test page 1</p></li><li><span>Item</span></li></ul></div></body></html>"
        html2 = "<html><head><title>Test</title></head><body><div><ul><li><p>Test page 1</p></li><li><span>Item</span></li></ul></div></body></html>"  # Identical
        html3 = "<html><head></head><body><table><tr><td><h1>Heading</h1></td></tr><tr><td><form><input/><button>Go</button></form></td></tr></table></body></html>"  # Very different structure
        
        # Use smaller k to ensure shingles can be generated (need at least k tags)
        detector = MinHashLSH(k=5, ell=100, tau=0.8)
        
        # First should be added
        self.assertTrue(detector.add_state(html1))
        
        # Identical should be detected as duplicate
        self.assertFalse(detector.add_state(html2))
        
        # Different should be added
        self.assertTrue(detector.add_state(html3))
        
        # Should have 2 unique states
        self.assertEqual(len(detector.get_unique_states()), 2)





if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

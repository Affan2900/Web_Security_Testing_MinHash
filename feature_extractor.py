"""
Feature Extractor for RIA Complexity Analysis
Extracts structural and complexity metrics from HTML content to inform
adaptive hyperparameter selection for MinHash LSH.
"""

from bs4 import BeautifulSoup
import re
from collections import Counter
import math


def extract_features(html_content):
    """
    Extract complexity features from HTML content.
    
    Args:
        html_content (str): HTML string to analyze
        
    Returns:
        dict: Feature vector with RIA complexity metrics
    """
    if not html_content or not isinstance(html_content, str):
        return get_default_features()
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception:
        return get_default_features()
    
    features = {}
    
    # DOM Complexity Metrics
    features['dom_size'] = _calculate_dom_size(soup)
    features['dom_depth'] = _calculate_dom_depth(soup)
    features['branching_factor'] = _calculate_branching_factor(soup)
    features['unique_tags'] = _count_unique_tags(soup)
    
    # Content Diversity Metrics
    features['tag_diversity'] = _calculate_tag_diversity(soup)
    features['tag_entropy'] = _calculate_tag_entropy(soup)
    
    # Dynamism Indicators
    features['js_detected'] = _detect_javascript(html_content, soup)
    features['form_elements'] = _count_form_elements(soup)
    features['ajax_indicators'] = _detect_ajax_indicators(html_content)
    
    # Size Metrics
    features['html_size'] = len(html_content)
    features['external_resources'] = _count_external_resources(soup)
    features['div_span_ratio'] = _calculate_div_span_ratio(soup)
    
    return features


def get_default_features():
    """Return default feature vector for invalid/empty HTML."""
    return {
        'dom_size': 0,
        'dom_depth': 0,
        'branching_factor': 0,
        'unique_tags': 0,
        'tag_diversity': 0,
        'tag_entropy': 0,
        'js_detected': 0,
        'form_elements': 0,
        'ajax_indicators': 0,
        'html_size': 0,
        'external_resources': 0,
        'div_span_ratio': 0
    }


def _calculate_dom_size(soup):
    """Count total number of HTML elements."""
    return len(soup.find_all())


def _calculate_dom_depth(soup, element=None, current_depth=0):
    """Calculate maximum depth of DOM tree using iterative approach."""
    if element is None:
        element = soup
    
    max_depth = 0
    # Use iterative approach with stack to avoid recursion limit
    stack = [(element, 0)]
    
    while stack:
        current_elem, depth = stack.pop()
        max_depth = max(max_depth, depth)
        
        if hasattr(current_elem, 'children'):
            for child in current_elem.children:
                if hasattr(child, 'name') and child.name:
                    stack.append((child, depth + 1))
    
    return max_depth


def _calculate_branching_factor(soup):
    """Calculate average branching factor (children per parent node)."""
    all_elements = soup.find_all()
    
    if not all_elements:
        return 0
    
    total_children = 0
    parent_count = 0
    
    for element in all_elements:
        children = [child for child in element.children if hasattr(child, 'name') and child.name]
        if children:
            total_children += len(children)
            parent_count += 1
    
    return total_children / parent_count if parent_count > 0 else 0


def _count_unique_tags(soup):
    """Count number of unique HTML tag types."""
    all_tags = [elem.name for elem in soup.find_all() if hasattr(elem, 'name')]
    return len(set(all_tags))


def _calculate_tag_diversity(soup):
    """Calculate Shannon diversity index for tags."""
    all_tags = [elem.name for elem in soup.find_all() if hasattr(elem, 'name')]
    
    if not all_tags:
        return 0
    
    tag_counts = Counter(all_tags)
    total = len(all_tags)
    
    # Shannon diversity: count of unique species (tags)
    return len(tag_counts)


def _calculate_tag_entropy(soup):
    """Calculate Shannon entropy of tag distribution."""
    all_tags = [elem.name for elem in soup.find_all() if hasattr(elem, 'name')]
    
    if not all_tags:
        return 0
    
    tag_counts = Counter(all_tags)
    total = len(all_tags)
    
    entropy = 0
    for count in tag_counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def _detect_javascript(html_content, soup):
    """Detect presence of JavaScript (0 or 1)."""
    # Check for script tags
    if soup.find('script'):
        return 1
    
    # Check for common JS frameworks in content
    js_patterns = [
        r'react', r'angular', r'vue', r'jquery',
        r'onclick', r'onload', r'addEventListener'
    ]
    
    for pattern in js_patterns:
        if re.search(pattern, html_content, re.IGNORECASE):
            return 1
    
    return 0


def _count_form_elements(soup):
    """Count number of form-related elements."""
    form_tags = ['form', 'input', 'textarea', 'select', 'button']
    count = 0
    
    for tag in form_tags:
        count += len(soup.find_all(tag))
    
    return count


def _detect_ajax_indicators(html_content):
    """Detect AJAX/dynamic content indicators."""
    ajax_patterns = [
        r'XMLHttpRequest', r'fetch\(', r'axios',
        r'\.ajax\(', r'data-ajax'
    ]
    
    count = 0
    for pattern in ajax_patterns:
        if re.search(pattern, html_content, re.IGNORECASE):
            count += 1
    
    return count


def _count_external_resources(soup):
    """Count links to external resources (scripts, styles, images)."""
    count = 0
    
    # Scripts
    count += len(soup.find_all('script', src=True))
    
    # Stylesheets
    count += len(soup.find_all('link', rel='stylesheet'))
    
    # Images
    count += len(soup.find_all('img', src=True))
    
    return count


def _calculate_div_span_ratio(soup):
    """Calculate ratio of structural (div) to inline (span) elements."""
    div_count = len(soup.find_all('div'))
    span_count = len(soup.find_all('span'))
    
    if span_count == 0:
        return div_count  # Return div count if no spans
    
    return div_count / span_count


# Example usage
if __name__ == "__main__":
    # Test with sample HTML
    sample_html = """
    <html>
        <head>
            <script src="app.js"></script>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <div class="container">
                <h1>Test Page</h1>
                <div class="content">
                    <p>Paragraph 1</p>
                    <p>Paragraph 2</p>
                    <form>
                        <input type="text" name="username">
                        <button>Submit</button>
                    </form>
                </div>
                <div class="sidebar">
                    <span>Sidebar content</span>
                </div>
            </div>
        </body>
    </html>
    """
    
    features = extract_features(sample_html)
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

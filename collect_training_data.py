"""
Training Data Collection via Grid Search
Finds optimal hyperparameters for each scraped website using grid search.
"""

import json
import os
from core import MinHashLSH, generate_shingles, extract_tags
from feature_extractor import extract_features
import numpy as np
from itertools import product


# Grid search parameter ranges
PARAM_GRID = {
    'k': [8, 10, 12, 15, 18],
    'ell': [100, 150, 200, 250, 300],
    'tau': [0.70, 0.75, 0.80, 0.85, 0.90]
}


def evaluate_parameters(html_pages, k, ell, tau):
    """
    Evaluate a specific set of hyperparameters on HTML pages.
    
    Args:
        html_pages (list): List of HTML strings from same website
        k (int): Shingle size
        ell (int): Number of hash functions
        tau (float): Similarity threshold
        
    Returns:
        dict: Metrics including coverage, redundancy, unique_count
    """
    detector = MinHashLSH(k=k, ell=ell, tau=tau)
    
    unique_count = 0
    duplicate_count = 0
    
    for html in html_pages:
        if detector.add_state(html):
            unique_count += 1
        else:
            duplicate_count += 1
    
    total_pages = len(html_pages)
    coverage = unique_count / total_pages if total_pages > 0 else 0
    redundancy = duplicate_count / total_pages if total_pages > 0 else 0
    
    # Penalize extremely high or low values
    # Ideal: high coverage (find unique states) but some redundancy detection
    # Score = coverage - (redundancy penalty if too strict or too lenient)
    if redundancy < 0.05:  # Too lenient, not catching duplicates
        score = coverage * 0.7
    elif redundancy > 0.5:  # Too strict, treating everything as duplicate
        score = coverage * 0.5
    else:
        score = coverage * (1 + (redundancy * 0.2))  # Slight bonus for catching some dups
    
    return {
        'unique_count': unique_count,
        'duplicate_count': duplicate_count,
        'coverage': coverage,
        'redundancy': redundancy,
        'score': score
    }


def grid_search_optimal_params(html_pages, site_type='unknown'):
    """
    Run grid search to find optimal hyperparameters for a set of pages.
    
    Args:
        html_pages (list): List of HTML strings
        site_type (str): Type of website
        
    Returns:
        dict: Optimal parameters and metrics
    """
    print(f"\n  Running grid search for {site_type} ({len(html_pages)} pages)...")
    
    best_score = -1
    best_params = None
    best_metrics = None
    
    total_combinations = len(PARAM_GRID['k']) * len(PARAM_GRID['ell']) * len(PARAM_GRID['tau'])
    current = 0
    
    for k, ell, tau in product(PARAM_GRID['k'], PARAM_GRID['ell'], PARAM_GRID['tau']):
        current += 1
        
        if current % 10 == 0:
            print(f"    Progress: {current}/{total_combinations}")
        
        metrics = evaluate_parameters(html_pages, k, ell, tau)
        
        if metrics['score'] > best_score:
            best_score = metrics['score']
            best_params = {'k': k, 'ell': ell, 'tau': tau}
            best_metrics = metrics
    
    print(f"  ✓ Best params: k={best_params['k']}, ell={best_params['ell']}, tau={best_params['tau']:.2f}")
    print(f"    Coverage: {best_metrics['coverage']:.2%}, Redundancy: {best_metrics['redundancy']:.2%}")
    
    return {
        'site_type': site_type,
        'optimal_params': best_params,
        'metrics': best_metrics,
        'num_pages': len(html_pages)
    }


def collect_training_data(raw_html_dir='training_data/raw_html',
                          features_file='training_data/features.json',
                          output_file='training_data/training_dataset.json'):
    """
    Collect training data by running grid search on all scraped websites.
    
    Args:
        raw_html_dir (str): Directory with raw HTML files
        features_file (str): JSON file with extracted features
        output_file (str): Output file for training dataset
        
    Returns:
        list: Training dataset
    """
    print("=" * 60)
    print("Training Data Collection via Grid Search")
    print("=" * 60)
    
    # Load extracted features
    print(f"\nLoading features from: {features_file}")
    with open(features_file, 'r') as f:
        all_features = json.load(f)
    
    # Group by site type
    site_groups = {}
    html_cache = {}
    
    for feature_data in all_features:
        site_type = feature_data['site_type']
        filename = feature_data['filename']
        
        if site_type not in site_groups:
            site_groups[site_type] = []
        
        # Load HTML
        filepath = os.path.join(raw_html_dir, site_type, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            html_cache[filename] = html_content
            site_groups[site_type].append({
                'filename': filename,
                'features': feature_data,
                'html': html_content
            })
    
    print(f"Loaded {len(all_features)} pages from {len(site_groups)} site types")
    
    # Run grid search for each site type
    training_data = []
    
    for i, (site_type, pages) in enumerate(site_groups.items(), 1):
        print(f"\n[{i}/{len(site_groups)}] Processing {site_type}...")
        
        html_pages = [p['html'] for p in pages]
        
        # Grid search
        result = grid_search_optimal_params(html_pages, site_type)
        
        # Create training samples (one per page with site-level optimal params)
        for page_data in pages:
            training_sample = {
                'features': {k: v for k, v in page_data['features'].items() 
                           if k not in ['filename', 'site_type']},
                'optimal_params': result['optimal_params'],
                'site_type': site_type,
                'filename': page_data['filename']
            }
            training_data.append(training_sample)
    
    # Save training dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✓ Training dataset created: {len(training_data)} samples")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Next step: Run 'python train_bayesian_model.py'")
    
    return training_data


if __name__ == "__main__":
    # Check if features exist
    features_file = 'training_data/features.json'
    
    if not os.path.exists(features_file):
        print("ERROR: features.json not found!")
        print("Please run 'python scrape_websites.py' first")
        exit(1)
    
    # Collect training data
    training_data = collect_training_data()
    
    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("Training Dataset Summary")
    print(f"{'=' * 60}")
    
    k_values = [sample['optimal_params']['k'] for sample in training_data]
    ell_values = [sample['optimal_params']['ell'] for sample in training_data]
    tau_values = [sample['optimal_params']['tau'] for sample in training_data]
    
    print(f"\nOptimal Parameters Distribution:")
    print(f"  k:   min={min(k_values)}, max={max(k_values)}, mean={np.mean(k_values):.1f}")
    print(f"  ell: min={min(ell_values)}, max={max(ell_values)}, mean={np.mean(ell_values):.1f}")
    print(f"  tau: min={min(tau_values):.2f}, max={max(tau_values):.2f}, mean={np.mean(tau_values):.2f}")

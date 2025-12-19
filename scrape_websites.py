"""
Website Scraper for Training Data Collection
Scrapes 10-15 diverse websites to build training dataset for ML model.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import os
from feature_extractor import extract_features

# List of diverse websites to scrape
WEBSITES = [
    # E-commerce
    {'url': 'https://www.amazon.com', 'type': 'ecommerce', 'max_pages': 5},
    {'url': 'https://www.etsy.com', 'type': 'ecommerce', 'max_pages': 5},
    
    # News/Media
    {'url': 'https://www.bbc.com/news', 'type': 'news', 'max_pages': 5},
    {'url': 'https://www.cnn.com', 'type': 'news', 'max_pages': 5},
    
    # Blogs/Content
    {'url': 'https://medium.com', 'type': 'blog', 'max_pages': 5},
    {'url': 'https://wordpress.com', 'type': 'blog', 'max_pages': 5},
    
    # Forums/Social
    {'url': 'https://www.reddit.com', 'type': 'forum', 'max_pages': 5},
    {'url': 'https://stackoverflow.com', 'type': 'forum', 'max_pages': 5},
    
    # SPAs/Dynamic
    {'url': 'https://quotes.toscrape.com/js/', 'type': 'spa', 'max_pages': 5},
    {'url': 'https://www.airbnb.com', 'type': 'spa', 'max_pages': 5},
    
    # Documentation/Static
    {'url': 'https://docs.python.org', 'type': 'documentation', 'max_pages': 5},
    {'url': 'https://developer.mozilla.org', 'type': 'documentation', 'max_pages': 5},
    
    # Web Apps
    {'url': 'https://www.github.com', 'type': 'webapp', 'max_pages': 5},
    {'url': 'https://www.youtube.com', 'type': 'webapp', 'max_pages': 5},
]


def scrape_static_website(url, max_pages=5):
    """
    Scrape static website using requests.
    
    Args:
        url (str): Base URL to scrape
        max_pages (int): Maximum number of pages to scrape
        
    Returns:
        list: List of HTML strings
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    html_pages = []
    visited = set()
    to_visit = [url]
    
    while len(html_pages) < max_pages and to_visit:
        current_url = to_visit.pop(0)
        
        if current_url in visited:
            continue
            
        try:
            print(f"  Fetching: {current_url}")
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            html_pages.append(response.text)
            visited.add(current_url)
            
            # Extract more links (basic breadth-first)
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True)[:5]:  # Limit to 5 links per page
                href = link['href']
                if href.startswith('/'):
                    full_url = url.rstrip('/') + href
                elif href.startswith(url):
                    full_url = href
                else:
                    continue
                    
                if full_url not in visited:
                    to_visit.append(full_url)
            
            time.sleep(1)  # Be polite
            
        except Exception as e:
            print(f"  Error fetching {current_url}: {e}")
            continue
    
    return html_pages


def scrape_dynamic_website(url, max_pages=5):
    """
    Scrape dynamic website using Selenium.
    
    Args:
        url (str): URL to scrape
        max_pages (int): Maximum number of pages to scrape
        
    Returns:
        list: List of HTML strings
    """
    options = Options()
    options.headless = True
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"  Chrome driver error: {e}")
        return []
    
    html_pages = []
    
    try:
        print(f"  Fetching dynamic: {url}")
        driver.get(url)
        time.sleep(3)  # Wait for JS
        
        # Get initial state
        html_pages.append(driver.page_source)
        
        # Try to navigate and get more states
        for i in range(max_pages - 1):
            try:
                # Try clicking buttons, links, etc.
                clickable = driver.find_elements(By.TAG_NAME, 'a')[:3]
                if clickable:
                    clickable[i % len(clickable)].click()
                    time.sleep(2)
                    html_pages.append(driver.page_source)
            except:
                break
                
    except Exception as e:
        print(f"  Error with Selenium: {e}")
    finally:
        driver.quit()
    
    return html_pages


def scrape_all_websites(output_dir='training_data/raw_html'):
    """
    Scrape all websites in WEBSITES list.
    
    Args:
        output_dir (str): Directory to save raw HTML
        
    Returns:
        dict: Mapping of website type to list of HTML pages
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = {}
    
    for i, site_info in enumerate(WEBSITES, 1):
        url = site_info['url']
        site_type = site_info['type']
        max_pages = site_info['max_pages']
        
        print(f"\n[{i}/{len(WEBSITES)}] Scraping {site_type}: {url}")
        
        # Try dynamic first (handles both static and dynamic)
        html_pages = scrape_dynamic_website(url, max_pages)
        
        # Fallback to static if Selenium fails
        if not html_pages:
            print("  Falling back to static scraping...")
            html_pages = scrape_static_website(url, max_pages)
        
        print(f"  Collected {len(html_pages)} pages")
        
        # Store by type
        if site_type not in all_data:
            all_data[site_type] = []
        all_data[site_type].extend(html_pages)
        
        # Save to disk
        site_dir = os.path.join(output_dir, site_type)
        os.makedirs(site_dir, exist_ok=True)
        
        for j, html in enumerate(html_pages):
            filename = f"{site_type}_{i}_{j}.html"
            filepath = os.path.join(site_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
    
    return all_data


def extract_features_from_scraped_data(raw_html_dir='training_data/raw_html', 
                                       output_file='training_data/features.json'):
    """
    Extract features from all scraped HTML files.
    
    Args:
        raw_html_dir (str): Directory containing raw HTML files
        output_file (str): Output JSON file for features
        
    Returns:
        list: List of feature dictionaries
    """
    all_features = []
    
    print("\nExtracting features from scraped data...")
    
    for root, dirs, files in os.walk(raw_html_dir):
        for filename in files:
            if filename.endswith('.html'):
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    features = extract_features(html_content)
                    
                    # Add metadata
                    features['filename'] = filename
                    features['site_type'] = os.path.basename(root)
                    
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
    
    # Save features
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    print(f"Extracted features from {len(all_features)} pages")
    print(f"Saved to: {output_file}")
    
    return all_features


if __name__ == "__main__":
    print("=" * 60)
    print("Website Scraper for Training Data Collection")
    print("=" * 60)
    
    # Step 1: Scrape websites
    print("\nStep 1: Scraping 10-15 diverse websites...")
    scraped_data = scrape_all_websites()
    
    total_pages = sum(len(pages) for pages in scraped_data.values())
    print(f"\n✓ Total pages scraped: {total_pages}")
    print(f"✓ Website types: {list(scraped_data.keys())}")
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
    features = extract_features_from_scraped_data()
    
    print("\n✓ Scraping complete!")
    print(f"✓ Next step: Run 'python train_model.py' to find optimal hyperparameters")

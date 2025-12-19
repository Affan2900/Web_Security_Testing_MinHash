import requests
from bs4 import BeautifulSoup  # For cleaning/extracting body; pip install beautifulsoup4
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from core import MinHashLSH

def fetch_static_html(urls):
    """Fetch HTML from static URLs using requests."""
    html_states = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract body content as string (paper focuses on DOM structure)
            body_html = str(soup.body) if soup.body else response.text
            html_states.append(body_html)
            print(f"Fetched static: {url}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return html_states

def fetch_dynamic_states(start_url, max_states=10, actions=None):
    """Use Selenium for dynamic RIA crawling: simulate actions to get states."""
    options = Options()
    options.headless = True  # Run without browser UI
    driver = webdriver.Chrome(options=options)  # Assumes chromedriver in PATH
    html_states = []
    
    driver.get(start_url)
    time.sleep(2)  # Wait for JS to load
    
    # Collect initial state
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    body_html = str(soup.body) if soup.body else html
    html_states.append(body_html)
    print(f"Fetched initial: {start_url}")
    
    # Simulate actions (e.g., clicks) to generate new states
    if actions is None:
        actions = []  # Add custom actions, e.g., [(By.TAG_NAME, 'a'), (By.ID, 'button')]
    visited = set()
    state_queue = [start_url]
    
    while len(html_states) < max_states and state_queue:
        current_url = state_queue.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)
        driver.get(current_url)
        time.sleep(2)
        
        # Perform actions to trigger state changes (paper's UI events)
        for by, value in actions:
            try:
                element = driver.find_element(by, value)
                element.click()
                time.sleep(1)  # Wait for DOM update
                new_html = driver.page_source
                new_body = str(BeautifulSoup(new_html, 'html.parser').body)
                if new_body not in html_states:
                    html_states.append(new_body)
                    print(f"Fetched dynamic state from action at {current_url}")
            except:
                pass
        
        # Extract links for further crawling (basic BFS)
        links = driver.find_elements(By.TAG_NAME, 'a')
        for link in links:
            href = link.get_attribute('href')
            if href and href not in visited and href.startswith(start_url):
                state_queue.append(href)
    
    driver.quit()
    return html_states[:max_states]

# Example usage: Integrate with MinHashLSH
if __name__ == "__main__":
    # Static example
    # urls = ['https://example.com/page1', 'https://example.com/page2']
    # states = fetch_static_html(urls)
    
    # Dynamic RIA example (e.g., start at a site with JS)
    start_url = 'https://quotes.toscrape.com/js/'  
    actions = [(By.TAG_NAME, 'button')] 
    states = fetch_dynamic_states(start_url, max_states=5, actions=actions)
    
    if not states:
        print("No states fetched. Exiting.")
        exit(0)
    
    # Test 1: With fixed hyperparameters
    print("\n=== Test 1: Fixed Hyperparameters ===")
    detector_fixed = MinHashLSH(k=12, ell=200, tau=0.85)
    for state in states:
        added = detector_fixed.add_state(state)
        print(f"Added as unique: {added}")
    uniques_fixed = detector_fixed.get_unique_states()
    print(f"Unique states found (fixed params): {len(uniques_fixed)}")
    
    # Test 2: With adaptive hyperparameters
    print("\n=== Test 2: Adaptive Hyperparameters ===")
    detector_adaptive = MinHashLSH(adaptive=True, html_sample=states[0])
    for state in states:
        added = detector_adaptive.add_state(state)
        print(f"Added as unique: {added}")
    uniques_adaptive = detector_adaptive.get_unique_states()
    print(f"Unique states found (adaptive params): {len(uniques_adaptive)}")
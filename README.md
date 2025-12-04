-----------------------------------------------------------------------
 [fetcher.py]

****fetch_static_html(urls)****

Purpose: Collect HTML states from static web pages.

Sends HTTP GET requests using requests.

Uses a desktop browser User-Agent to avoid blocking.

Parses each page with BeautifulSoup.

Extracts only the <body> section (or full HTML if missing).

Appends the extracted HTML to a list of states.

Prints success or error messages for each URL.

Returns a list of captured static HTML states.

-----------------------------------------------------------------------

****fetch_dynamic_states(start_url, max_states=10, actions=None)****

Purpose: Crawl dynamic, JavaScript-driven pages using Selenium and extract DOM states.

Browser Setup

Initializes a headless Chrome instance via Selenium.

Opens the starting URL.

Waits for JS execution to complete before capturing the DOM.

Initial State Capture

Extracts <body> HTML of the landing page.

Stores the initial DOM state in the state list.

Simulated User Interactions

Accepts a list of actions like:

(By.TAG_NAME, 'button')

(By.ID, 'submit-btn')

For each action:

Locates the matching element.

Clicks it to trigger UI changes.

Waits for DOM updates.

Extracts and stores the resulting DOM state (if new).

Automatic Crawling (BFS)

Finds all <a> elements on each loaded page.

Extracts their href attributes.

Adds internal links (same domain prefix) to a queue.

Performs breadth-first traversal until max_states is reached.

State Management

Maintains a visited set to avoid reprocessing URLs.

Ensures no duplicate DOM states are added.

Returns up to max_states unique dynamic states.

-----------------------------------------------------------------------

[core.py]

****class TagExtractor(html.parser.HTMLParser)****

Purpose: Parses HTML content and extracts an ordered list of all opening tag names (e.g., html, body, p, a), ignoring text content and attributes, to represent the DOM structure.

-----------------------------------------------------------------------

****def extract_tags(html_content)****

Purpose: Initializes and runs TagExtractor on the provided HTML string.

-----------------------------------------------------------------------

****def generate_shingles(tags, k)****

Purpose: Creates the set representation of the web page by generating $k$-mers (shingles), which are sequences of k consecutive DOM element tags19.

Parameter: k   -> (self.k), the shingle size.

-----------------------------------------------------------------------

****def universal_hash(seed, x)****

Purpose: A basic, parameterized hash function (using MD5) to map shingle representations to numerical values. The seed is used to simulate different, independent hash functions.


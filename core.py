import html.parser
import hashlib
import random
from collections import defaultdict

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
        shingle = tuple(tags[i:i + k])
        shingles.add(shingle)
    return shingles

def universal_hash(seed, x):
    """Simple universal hash using MD5 with seed."""
    m = hashlib.md5()
    m.update(str(seed).encode('utf-8') + str(x).encode('utf-8'))
    return int(m.hexdigest(), 16)

class MinHashLSH:
    def __init__(self, k=12, ell=200, tau=0.85):
        self.k = k  # Shingle size
        self.ell = ell  # Number of hash functions (sketch size)
        self.tau = tau  # Jaccard similarity threshold
        self.hash_functions = [random.randint(0, 2**32 - 1) for _ in range(ell)]  # Seeds for hash funcs
        self.hash_tables = [defaultdict(set) for _ in range(ell)]  # LSH tables: list of dict(value -> set(ids))
        self.unique_states = {}  # id -> original html_content
        self.next_id = 0

    def compute_sketch(self, shingles):
        """Compute MinHash sketch: list of ell min-hash values."""
        sketch = []
        for i in range(self.ell):
            min_val = float('inf')
            for shingle in shingles:
                h_val = universal_hash(self.hash_functions[i], shingle)
                if h_val < min_val:
                    min_val = h_val
            sketch.append(min_val)
        return sketch

    def is_duplicate(self, html_content):
        """Check if state is duplicate based on max estimated Jaccard similarity."""
        tags = extract_tags(html_content)
        shingles = generate_shingles(tags, self.k)
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
        return max_sim >= self.tau

    def add_state(self, html_content):
        """Add state if not duplicate, update LSH tables."""
        if self.is_duplicate(html_content):
            return False  # Duplicate, not added
        tags = extract_tags(html_content)
        shingles = generate_shingles(tags, self.k)
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

# Example usage
if __name__ == "__main__":
    # Sample HTML states (replace with your real pages)
    sample_htmls = [
        "<html><head><title>Test</title></head><body><p>Hello</p><a href='#'>Link</a></body></html>",
        "<html><head><title>Test</title></head><body><p>Hello World</p><a href='#'>Link</a></body></html>",  # Similar
        "<html><body><div><ul><li>Item1</li><li>Item2</li></ul></div></body></html>",  # Different
    ]

    detector = MinHashLSH(k=3, ell=10, tau=0.8)  # Small params for demo; use paper's in production
    for html in sample_htmls:
        added = detector.add_state(html)
        print(f"Added as unique: {added}")

    uniques = detector.get_unique_states()
    print(f"Unique states found: {len(uniques)}")
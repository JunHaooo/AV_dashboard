import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json
import os
import time
import re
import concurrent.futures
from threading import Lock

# ------------------------------- 
# Configuration - MODIFY THESE TO SUIT YOUR NEEDS
# ------------------------------- 
seed_urls = [
    "https://techcrunch.com/tag/autonomous-vehicles/",
    "https://www.wired.com/tag/self-driving-cars/"
]

# SPEED CONTROLS - Adjust these for faster/slower crawling
max_depth = 1  # REDUCED from 2 for faster testing - increase for deeper crawling
crawl_delay = 0.2  # REDUCED from 1 second for faster crawling - increase to be more polite
max_concurrent_requests = 3  # Number of simultaneous requests - increase for speed, decrease to be polite
max_pages_per_run = 10  # LIMIT pages crawled per run for testing - set to None for unlimited

# KEYWORD DISCOVERY SETTINGS
auto_discover_keywords = True  # Set to False to disable automatic keyword discovery
min_keyword_frequency = 2  # Minimum times a keyword must appear to be added
keyword_discovery_patterns = [
    r'\b(electric\s+vehicle[s]?)\b',
    r'\b(autonomous\s+driving)\b',
    r'\b(self\s*[-]?\s*driving)\b',
    r'\b(artificial\s+intelligence)\b',
    r'\b(machine\s+learning)\b',
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:car|vehicle|auto))\b'
]

KEYWORDS_FILE = "data/keywords.json"

# Thread-safe lock for updating keyword counts
keyword_lock = Lock()

def load_keywords():
    """Loads keywords from the JSON file as a simple list.
    Returns a list of keywords and initializes internal counters.
    """
    default_keywords = ["autonomous", "self-driving", "av", "lidar", "tesla", "waymo", "evs", 
                       "electric vehicle", "driverless", "autopilot", "robotaxi", "cruise"]
    
    if not os.path.exists(KEYWORDS_FILE) or os.path.getsize(KEYWORDS_FILE) == 0:
        keywords_list = default_keywords
        save_keywords(keywords_list)
    else:
        with open(KEYWORDS_FILE, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            keywords_list = data
        else:
            # If it's an old format with counts, extract just the keywords
            keywords_list = list(data.keys())
            save_keywords(keywords_list)  # Convert to new format
    
    # Ensure default keywords are present
    for kw in default_keywords:
        if kw not in keywords_list:
            keywords_list.append(kw)
    
    return keywords_list

def save_keywords(keywords_list):
    """Saves the keywords list to the JSON file."""
    os.makedirs(os.path.dirname(KEYWORDS_FILE), exist_ok=True)
    with open(KEYWORDS_FILE, 'w') as f:
        json.dump(keywords_list, f, indent=2)

def discover_new_keywords(text, keywords_list, keyword_counts):
    """Discovers new keywords from page content using pattern matching.
    Returns updated keywords_list and keyword_counts with new keywords added.
    """
    if not auto_discover_keywords:
        return keywords_list, keyword_counts
        
    discovered = {}
    text_lower = text.lower()
    
    # Use predefined patterns to find potential keywords
    for pattern in keyword_discovery_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            keyword = match.group(1).lower().strip()
            if keyword and len(keyword) > 2:  # Skip very short keywords
                discovered[keyword] = discovered.get(keyword, 0) + 1
    
    # Add discovered keywords that meet minimum frequency and aren't already tracked
    with keyword_lock:
        for keyword, freq in discovered.items():
            if freq >= min_keyword_frequency and keyword not in keywords_list:
                keywords_list.append(keyword)
                keyword_counts[keyword] = freq
                print(f"🔍 Discovered new keyword: '{keyword}' (frequency: {freq})")
                # Save updated keywords list to file
                save_keywords(keywords_list)
    
    return keywords_list, keyword_counts

def crawl_single_page(url, depth, keywords_list, keyword_counts, visited):
    """Crawls a single page and returns found links and updated keyword counts.
    Thread-safe function for concurrent crawling.
    """
    if url in visited:
        return [], keywords_list, keyword_counts
    
    try:
        print(f"🕷️  Crawling ({depth}) -> {url}")
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            print(f"❌ Failed to crawl {url}: HTTP {response.status_code}")
            return [], keywords_list, keyword_counts

        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text()
        
        # Discover new keywords from page content
        keywords_list, keyword_counts = discover_new_keywords(page_text, keywords_list, keyword_counts)
        
        found_links = []
        
        # Collect and filter links
        for link in soup.find_all("a", href=True):
            href = urljoin(url, link["href"])
            if urlparse(href).scheme in ["http", "https"]:
                # Check if link contains any keywords (in URL or link text)
                link_text = link.get_text().lower()
                url_and_text = f"{href.lower()} {link_text}"
                
                for keyword in keywords_list:
                    if keyword.lower() in url_and_text:
                        with keyword_lock:
                            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                        found_links.append(href)
                        print(f"✅ Found relevant link: {href[:80]}..." if len(href) > 80 else href)
                        break  # Avoid multiple counts for the same link

        return found_links, keywords_list, keyword_counts

    except Exception as e:
        print(f"❌ Error crawling {url}: {e}")
        return [], keywords_list, keyword_counts


def crawl(seed_urls, keywords_list, max_depth=2):
    """
    Main crawling function using BFS with keyword filtering.
    Returns found links and keyword counts.
    """
    keyword_counts = {kw: 0 for kw in keywords_list}  # Initialize counts
    queue = deque([(url, 0) for url in seed_urls])  # (url, depth)
    visited = set()
    found_links = []
    pages_crawled = 0

    while queue and (max_pages_per_run is None or pages_crawled < max_pages_per_run):
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue

        print(f"🕷️ Crawling ({depth}) -> {url} (Queue size: {len(queue)})")
        visited.add(url)
        found_links.append(url)
        pages_crawled += 1

        time.sleep(crawl_delay)

        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code != 200:
                print(f"❌ Failed to crawl {url}: HTTP {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text()
            
            # Discover new keywords from page content
            keywords_list, keyword_counts = discover_new_keywords(page_text, keywords_list, keyword_counts)

            # Collect and filter links for next depth level
            if depth < max_depth:
                for link in soup.find_all("a", href=True):
                    href = urljoin(url, link["href"])
                    if urlparse(href).scheme in ["http", "https"] and href not in visited:
                        # Check if link contains any keywords (in URL or link text)
                        link_text = link.get_text().lower()
                        url_and_text = f"{href.lower()} {link_text}"
                        
                        for keyword in keywords_list:
                            if keyword.lower() in url_and_text:
                                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                                queue.append((href, depth + 1))
                                print(f"➕ Added to queue: {href[:60]}..." if len(href) > 60 else href)
                                break  # Avoid multiple counts for the same link

        except Exception as e:
            print(f"❌ Error crawling {url}: {e}")

    return found_links, keywords_list, keyword_counts


# ------------------------------- 
# Run
# ------------------------------- 
keywords_list = load_keywords()
collected_links, updated_keywords, keyword_counts = crawl(seed_urls, keywords_list, max_depth=max_depth)

# Save updated keywords list (only the list, not counts)
save_keywords(updated_keywords)

print("\n" + "="*60)
print("📊 CRAWLING RESULTS")
print("="*60)

print(f"\n🔗 Visited Links ({len(collected_links)}):")
for i, link in enumerate(collected_links, 1):
    print(f"{i:2d}. {link}")

print(f"\n🔍 Keywords Used ({len(updated_keywords)}):")
for keyword in sorted(updated_keywords):
    print(f"   • {keyword}")

print(f"\n📈 Keyword Frequency (this session):")
sorted_counts = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
for keyword, count in sorted_counts:
    if count > 0:
        print(f"   {keyword}: {count}")

print(f"\n⚙️  Crawl Settings:")
print(f"   • Max depth: {max_depth}")
print(f"   • Crawl delay: {crawl_delay}s")
print(f"   • Max pages per run: {max_pages_per_run}")
print(f"   • Auto-discover keywords: {auto_discover_keywords}")
print("="*60)
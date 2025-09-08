import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json
import os
import time
import re
from threading import Lock
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
seed_urls_file = "data/seeds.json"
keywords_file = "data/keywords.json"
summaries_file = "data/summaries.json"

max_depth = 2
crawl_delay = 0.5
max_pages_per_run = 20
max_concurrent_requests = 3  # for future threading if needed

# AI settings
semantic_threshold = 0.4  # similarity threshold for following links
topic_text = "autonomous vehicles, self-driving cars, electric vehicles"  # main topic

# Thread lock
lock = Lock()

# -------------------------------
# Load / Save JSON utilities
# -------------------------------
def load_json(filepath, default):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return default
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------------
# AI models initialization
# -------------------------------
print("🔧 Loading AI models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_embedding = embedding_model.encode(topic_text)
kw_model = KeyBERT()

# -------------------------------
# Helper functions
# -------------------------------
def extract_keywords(text, top_n=10):
    """Extracts keywords from text using KeyBERT"""
    return [kw[0].lower() for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=top_n)]

def semantic_similarity(text):
    """Computes similarity between text and main topic"""
    emb = embedding_model.encode(text)
    sim = util.cos_sim(emb, topic_embedding)
    return sim.item()

def summarize_text(text, n_sentences=3):
    """Simple extractive summary using top TF-IDF-like sentences"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= n_sentences:
        return text
    # Compute simple score: sentence similarity to topic
    scores = [semantic_similarity(s) for s in sentences]
    top_idx = np.argsort(scores)[-n_sentences:]
    summary = ' '.join([sentences[i] for i in sorted(top_idx)])
    return summary

# -------------------------------
# Crawler
# -------------------------------
def crawl(seed_urls, max_depth=2, max_pages=None):
    queue = deque([(url, 0) for url in seed_urls])
    visited = set()
    discovered_domains = set(urlparse(u).netloc for u in seed_urls)
    all_keywords = load_json(keywords_file, [])
    summaries = load_json(summaries_file, {})

    pages_crawled = 0

    while queue and (max_pages is None or pages_crawled < max_pages):
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        try:
            print(f"🕷️ Crawling ({depth}) -> {url}")
            r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                print(f"❌ Failed: {r.status_code}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            page_text = soup.get_text(separator=' ', strip=True)

            # Semantic relevance check
            sim = semantic_similarity(page_text)
            if sim < semantic_threshold:
                print(f"⚠️ Page not relevant (sim={sim:.2f})")
                continue

            # Extract keywords and update global list
            kws = extract_keywords(page_text)
            with lock:
                new_kws = [k for k in kws if k not in all_keywords]
                if new_kws:
                    all_keywords.extend(new_kws)
                    save_json(keywords_file, all_keywords)
                    print(f"🔍 New keywords: {new_kws}")

            # Generate summary
            summary = summarize_text(page_text)
            with lock:
                summaries[url] = summary
                save_json(summaries_file, summaries)
                print(f"📝 Summary saved ({len(summary)} chars)")

            # Extract links and decide which to follow
            for link_tag in soup.find_all("a", href=True):
                href = urljoin(url, link_tag["href"])
                parsed = urlparse(href)
                if parsed.scheme not in ["http", "https"]:
                    continue
                domain = parsed.netloc
                link_text = link_tag.get_text(" ", strip=True)
                combined_text = f"{link_text} {href}"

                # Semantic check on link text
                if semantic_similarity(combined_text) >= semantic_threshold:
                    if domain not in discovered_domains:
                        discovered_domains.add(domain)
                        seed_urls.append(href)  # update seed list dynamically
                    queue.append((href, depth + 1))

            pages_crawled += 1
            time.sleep(crawl_delay)

        except Exception as e:
            print(f"❌ Error crawling {url}: {e}")
            continue

    # Save updated seeds for next run
    save_json(seed_urls_file, seed_urls)
    print(f"🏁 Crawl finished. Pages crawled: {pages_crawled}, Keywords tracked: {len(all_keywords)}")
    return visited, all_keywords, summaries

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    print("🔧 Loading seeds...")
    seed_urls = load_json(seed_urls_file, [
        "https://techcrunch.com/tag/autonomous-vehicles/",
        "https://www.wired.com/tag/self-driving-cars/"
    ])
    visited, keywords, summaries = crawl(seed_urls, max_depth=max_depth, max_pages=max_pages_per_run)
    print("\n✅ Crawl completed.")
    print(f"Pages visited: {len(visited)}")
    print(f"Keywords discovered: {len(keywords)}")
    print(f"Summaries saved for {len(summaries)} pages")

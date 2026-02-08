import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.icy-veins.com"
LIST_PAGE = "/genshin-impact/characters"

def get_soup(url):
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {url}")
        return None
    return BeautifulSoup(r.text, "html.parser")

def extract_character_links():
    soup = get_soup(urljoin(BASE_URL, LIST_PAGE))
    if not soup:
        return []

    # find all <a> tags that look like character links
    # often these have hrefs containing "-guide-" for build pages
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/genshin-impact/" in href and "guide" in href:
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)
    return links

def save_text_from_url(url):
    soup = get_soup(url)
    if not soup:
        return

    # strip scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    filename = url.split("/")[-1] + ".txt"

    os.makedirs("data/raw", exist_ok=True)
    with open(os.path.join("data/raw", filename), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {filename}")

def main():
    links = extract_character_links()
    print(f"Found {len(links)} pages")

    for link in links:
        save_text_from_url(link)

if __name__ == "__main__":
    main()
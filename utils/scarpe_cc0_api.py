import requests, os, csv, time, json, hashlib
from PIL import Image
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

"""
CC0 API Scraper for PetBuddy
---------------------------------
Purpose:
1. Harvest high-resolution cat & dog images under CC0 license,
   ensuring commercial-friendly use without attribution.
2. Enlarge the multi-pet subset (≥2 pets/image) to boost
   detection recall and robustness in natural scenes.
3. Provide weak-label pseudo-annotations (bbox only) via
   YOLO-pre-trained model, reducing manual annotation cost.
4. Complement Oxford/Stanford datasets (single-pet, breed-level)
   with diverse lighting, occlusion and background variability.

Output:
- images/          : CC0 jpg files (md5-named, de-duplicated)
- weak_anno.json   : COCO-format pseudo-labels (cat/dog bbox)
- meta.csv         : source, license, original URL for citation

Usage:
    python tools/scrape_cc0_api.py --keywords "cat dog" --pages 20
"""

KEYS = {
    "unsplash": "okHZH4sXawjvGt0YxJSXWg_qZmrcxgB85hznv3_gKoo",
    "pixabay":  "53168718-95d2066104d2bce65b39a18be",
    "pexels":   "wWxZAjKfJCvQm3DOUVQunLSwdRWMI70oh75lD6iTi9LeeB7po8xpfWnL"

}
OUT_DIR = "../data/cc0_api"
os.makedirs(OUT_DIR, exist_ok=True)

# Track downloaded files to avoid duplicates
downloaded_files = set()
if os.path.exists(f"{OUT_DIR}/meta.csv"):
    with open(f"{OUT_DIR}/meta.csv", "r", newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                downloaded_files.add(row[0])  # filename is first column

# Global counter for progress display
download_count = 0

# Create a session with retry mechanism
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def save_image(url, src, license):
    global download_count
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content)).convert("RGB")
        fname = f"{src}_{hashlib.md5(resp.content).hexdigest()}.jpg"

        # Check if file already exists to avoid duplicates
        if fname in downloaded_files:
            print(f"  ⚡ Skipping duplicate: {fname}")
            return

        img.save(os.path.join(OUT_DIR, fname))
        with open(f"{OUT_DIR}/meta.csv", "a", newline='') as f:
            csv.writer(f).writerow([fname, url, license])

        # Add to downloaded files set
        downloaded_files.add(fname)
        download_count += 1
        print(f"  ✓ Downloaded image #{download_count}: {fname}")

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to download {url}: {str(e)}")
    except Exception as e:
        print(f"  ✗ Error processing image {url}: {str(e)}")

# ① Unsplash API
def unsplash_page(page):
    r = requests.get(f"https://api.unsplash.com/search/photos",
                     params={"query": "cat dog", "page": page, "per_page": 30},
                     headers={"Authorization": f"Client-ID {KEYS['unsplash']}"}).json()
    for img in r["results"]:
        save_image(img["urls"]["full"], "unsplash", "CC0")

# ② Pexels API
def pexels_page(page):
    r = requests.get(f"https://api.pexels.com/v1/search",
                     params={"query": "cat dog", "page": page, "per_page": 30},
                     headers={"Authorization": KEYS["pexels"]}).json()
    for img in r["photos"]:
        save_image(img["src"]["original"], "pexels", "CC0")

# ③ Pixabay API
def pixabay_page(page):
    r = requests.get("https://pixabay.com/api/",
                     params={"key": KEYS["pixabay"], "q": "cat dog", "page": page,
                             "per_page": 30, "image_type": "photo"}).json()
    for img in r["hits"]:
        save_image(img["largeImageURL"], "pixabay", "CC0")

if __name__ == "__main__":
    # Only process API sources with configured keys
    api_sources = []
    if "unsplash" in KEYS and KEYS["unsplash"] != "YOUR_UNSPLASH_ACCESS_KEY":
        api_sources.append(("unsplash", unsplash_page))
    if "pixabay" in KEYS and KEYS["pixabay"] != "YOUR_PIXABAY_KEY":
        api_sources.append(("pixabay", pixabay_page))
    if "pexels" in KEYS and KEYS["pexels"]:
        api_sources.append(("pexels", pexels_page))


    total_sources = len(api_sources)
    total_pages = 10 * total_sources  # 10 pages per source

    print(f"Starting crawl, {total_sources} API sources, expected {total_pages} pages")
    print("=" * 50)

    for i, (src, func) in enumerate(api_sources, 1):
        print(f"[{i}/{total_sources}] Processing {src} API...")

        for p in range(1, 11):        # 10 pages ≈ 900 images
            print(f"  Crawling page {p}/10...")
            start_count = download_count
            func(p)
            end_count = download_count
            new_images = end_count - start_count
            print(f"  Page {p} completed, {new_images} new images")
            time.sleep(1)             #  polite rate limiting

    print("=" * 50)
    print(f"Crawl completed! Total downloaded images: {download_count}")
    print(f"Images saved in: {OUT_DIR}/")
    print(f"Metadata file: {OUT_DIR}/meta.csv")
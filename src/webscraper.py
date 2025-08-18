from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time, os, requests
from pathlib import Path

# Target URL to scrape from
TARGET_URL = "https://www.thereformation.com/new"


# If this file is inside `src/`, this resolves the project root (one level up)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
SAVE_DIR = IMAGES_DIR / "original_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_products():
    """
    Uses Playwright to open the target page, scroll to the bottom
    until all products are loaded, and return the final HTML content.
    """
    with sync_playwright() as p:
        # Launch Chromium browser (set headless=True to hide browser window)
        browser = p.chromium.launch(headless=False)
        page = browser.new_context().new_page()
        page.goto(TARGET_URL)
        time.sleep(5)  # wait for initial content to load

        # Keep scrolling down until the page stops loading new content
        prev_height = 0
        while True:
            curr_height = page.evaluate("document.body.scrollHeight")
            page.evaluate(f"window.scrollTo(0, {curr_height})")
            time.sleep(2)  # wait for new content to load
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:  # no more content
                break
            prev_height = new_height

        # Get full HTML and close the browser
        html = page.content()
        browser.close()
        return html


def extract_products_and_download_images(html, max_images=None):
    """
    Parses the HTML content with BeautifulSoup, finds product tiles,
    extracts image URLs, and downloads up to max_images images (if specified).
    """
    soup = BeautifulSoup(html, "html.parser")
    tiles = soup.find_all("div", class_="product-tile")

    count = 0  # track how many images we’ve saved
    for idx, tile in enumerate(tiles):
        # Stop if we’ve reached the limit
        if max_images is not None and count >= max_images:
            break
        try:
            # Find the image inside each product tile
            img_tag = tile.find("img")
            image_url = img_tag.get("src") if img_tag else None
            if not image_url:
                continue

            # Save file with a unique name
            filename = f"product_{idx}_{idx}.jpg"
            save_path = SAVE_DIR / filename

            # Download and write the image to disk
            response = requests.get(image_url, timeout=10)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Saved: {save_path}")
            count += 1
        except Exception as e:
            print(f"Failed to save product #{idx}: {e}")


# --- Main execution ---
# Step 1: scrape the page and collect the HTML
html_content = load_all_products()

# Step 2: extract product images and download them (limit 40 for testing)
extract_products_and_download_images(html_content, max_images=40)

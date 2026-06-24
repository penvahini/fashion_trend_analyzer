# src/webscraper.py
"""
Scrapes product images from one retailer's "new arrivals" page for a single
run. Single-site by design for now -- see README's "Future Enhancements"
section for the multi-retailer tradeoffs (selector maintenance burden per
site, normalizing categories/sizes across retailers) that made this an
explicit scope cut rather than an oversight.
"""
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time, os, json, argparse, requests
from pathlib import Path
from datetime import date, datetime, timezone

from aws_clients import get_s3_client, get_sqs_client, get_queue_url, S3_BUCKET

# Target URL to scrape from
TARGET_URL = "https://www.thereformation.com/new"


# If this file is inside `src/`, this resolves the project root (one level up)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
ORIGINAL_IMAGES_DIR = IMAGES_DIR / "original_images"


def load_all_products(headless: bool = True):
    """
    Uses Playwright to open the target page, scroll to the bottom
    until all products are loaded, and return the final HTML content.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
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


def upload_and_enqueue(local_path: Path, filename: str, run_id: str, source_url: str, s3_client, sqs_client, queue_url):
    """Uploads one image to S3 and pushes an ingestion job message to SQS."""
    s3_key = f"raw/{run_id}/{filename}"
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
    sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps({
            "run_id": run_id,
            "filename": filename,
            "s3_key": s3_key,
            "source_url": source_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }),
    )

def extract_products_and_download_images(html, save_dir: Path, run_id: str, max_images=None, use_queue=False):
    """
    Parses the HTML content with BeautifulSoup, finds product tiles,
    extracts image URLs, and downloads up to max_images images (if specified).
    Writes a manifest.json recording the scrape timestamp for each image,
    so later pipeline stages can analyze trends over time.

    If use_queue is True, each image is also uploaded to S3 and a job message
    is pushed to SQS, simulating a distributed scrape -> queue -> worker pipeline.
    """
    soup = BeautifulSoup(html, "html.parser")
    tiles = soup.find_all("div", class_="product-tile")

    s3_client = sqs_client = queue_url = None
    if use_queue:
        s3_client = get_s3_client()
        sqs_client = get_sqs_client()
        queue_url = get_queue_url(sqs_client)

    manifest = []
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
            save_path = save_dir / filename

            # Download and write the image to disk
            response = requests.get(image_url, timeout=10)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Saved: {save_path}")

            if use_queue:
                upload_and_enqueue(save_path, filename, run_id, image_url, s3_client, sqs_client, queue_url)
                print(f"Enqueued: {filename}")

            manifest.append({
                "filename": filename,
                "source_url": image_url,
                "run_id": run_id,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            })
            count += 1
        except Exception as e:
            print(f"Failed to save product #{idx}: {e}")

    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=date.today().isoformat(),
                     help="Identifier for this scrape run, defaults to today's date (YYYY-MM-DD)")
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--use-queue", action="store_true",
                     help="Upload each image to S3 and enqueue an ingestion job on SQS (requires LocalStack/AWS configured)")
    ap.add_argument("--headed", action="store_true",
                     help="Show the browser window (useful for debugging selectors); default is headless")
    args = ap.parse_args()

    save_dir = ORIGINAL_IMAGES_DIR / args.run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    html_content = load_all_products(headless=not args.headed)
    extract_products_and_download_images(
        html_content, save_dir, args.run_id, max_images=args.max_images, use_queue=args.use_queue,
    )

if __name__ == "__main__":
    main()

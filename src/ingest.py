import os
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

# 1. Setup Logging - This creates a file in logs to track what happens
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/ingest.log"), logging.StreamHandler()]
)

class CSSFScraper:
    SITE_ORIGIN = "https://www.cssf.lu"
    RSS_URL = f"{SITE_ORIGIN}/en/publications/"
    CIRCULAR_CATEGORY = "CSSF circular"

    # Default requests User-Agent looks like a script; many sites (incl. regulators) block it.
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, download_dir="data"):
        self.download_dir = download_dir
        self.headers = self.DEFAULT_HEADERS
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def _absolute_url(self, href: str) -> str:
        if href.startswith("http"):
            return href
        return f"{self.SITE_ORIGIN}{href}" if href.startswith("/") else f"{self.SITE_ORIGIN}/{href}"

    def _get_circular_pages_from_rss(self) -> list[str]:
        """Fetch RSS and return document page URLs tagged as CSSF circular."""
        response = requests.get(self.RSS_URL, headers=self.headers, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        pages: list[str] = []
        for item in root.findall(".//item"):
            category = item.find("category")
            if category is None or (category.text or "").strip() != self.CIRCULAR_CATEGORY:
                continue
            link = item.find("link")
            if link is not None and link.text:
                pages.append(link.text.strip())

        logging.info(f"Found {len(pages)} CSSF circular(s) in RSS feed")
        return pages

    def _get_pdf_links_from_page(self, page_url: str) -> list[str]:
        """Scrape PDF download links from a single publication document page."""
        response = requests.get(page_url, headers=self.headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        pdfs: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].split("?")[0]
            if href.lower().endswith(".pdf"):
                pdfs.append(self._absolute_url(href))
        return pdfs

    def get_pdf_links(self) -> list[str]:
        """
        Discover PDFs via RSS (CSSF circular only) → visit each document page → collect .pdf URLs.
        """
        logging.info("Fetching CSSF circulars from publications RSS...")
        try:
            circular_pages = self._get_circular_pages_from_rss()
            if not circular_pages:
                logging.warning("No CSSF circular items in RSS feed")
                return []

            pdf_links: list[str] = []
            for page_url in circular_pages:
                try:
                    page_pdfs = self._get_pdf_links_from_page(page_url)
                    logging.info(f"  {page_url} → {len(page_pdfs)} PDF link(s)")
                    pdf_links.extend(page_pdfs)
                except Exception as e:
                    logging.error(f"Failed to scrape PDFs from {page_url}: {e}")

            unique = list(dict.fromkeys(pdf_links))
            logging.info(f"Total unique PDF URLs: {len(unique)}")
            return unique
        except Exception as e:
            logging.error(f"Failed to fetch links: {e}")
            return []

    def _basename_from_url(self, url: str) -> str:
        """Last path segment only, e.g. circular-12-2024.pdf — same name = same doc for CSSF."""
        base = url.split("/")[-1].split("?")[0]
        if not base.lower().endswith(".pdf"):
            base = "document.pdf" if not base else f"{base}.pdf"
        return base

    def download_pdfs(self, links, limit=5):
        logging.info(f"Starting download of up to {limit} PDFs...")
        seen_basenames: set[str] = set()
        downloaded = 0
        skipped = 0

        for link in tqdm(links):
            if downloaded >= limit:
                break
            file_path = None
            try:
                file_name = self._basename_from_url(link)
                file_path = os.path.join(self.download_dir, file_name)

                if file_name in seen_basenames:
                    logging.info(
                        f"Skipping duplicate URL (same PDF name '{file_name}'): {link}"
                    )
                    skipped += 1
                    continue

                if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                    logging.info(f"Skipping (already on disk): {file_name}")
                    seen_basenames.add(file_name)
                    skipped += 1
                    continue

                seen_basenames.add(file_name)

                response = requests.get(
                    link, headers=self.headers, stream=True, timeout=30
                )
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and not link.lower().endswith(".pdf"):
                    logging.warning(
                        f"Unexpected Content-Type '{content_type}' for {link}; skipping"
                    )
                    continue

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    logging.error(f"Empty file after download, removed: {link}")
                    continue

                downloaded += 1
                logging.info(f"Saved: {file_name}")
            except requests.HTTPError as e:
                logging.error(f"HTTP error for {link}: {e}")
            except Exception as e:
                logging.error(f"Error downloading {link}: {e}")
                if file_path and os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                    os.remove(file_path)

        logging.info(f"Done. Downloaded: {downloaded}, skipped (existing): {skipped}")

if __name__ == "__main__":
    # This part runs when you execute the script
    scraper = CSSFScraper()
    pdf_urls = scraper.get_pdf_links()
    
    if pdf_urls:
        scraper.download_pdfs(pdf_urls, limit=5)
        print("\nSuccess! Check your 'data' folder.")
    else:
        print("No PDFs found. Check the logs/ingest.log file.")
"""
scraper/scrapers/pmindia_scraper.py
=====================================
Secondary scraper for pmindia.gov.in — PM India official scheme announcements.

Uses Playwright to bypass WAFs and Cloudflare blocks.

Coverage:
  - PM-level flagship schemes (PMJAY, PM Kisan, PM Awas, etc.)
  - Latest announcements and updates from the Prime Minister's Office
"""

import time
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

from scraper.scrapers.base_scraper import BaseScraper

logger = logging.getLogger("sugamgov_scraper.pmindia")

PMINDIA_URLS = [
    "https://www.pmindia.gov.in/en/government_tr_rec/schemes/",
    "https://www.pmindia.gov.in/en/major-initiatives/",
]
MAX_SCHEMES = 50


class PmIndiaScraper(BaseScraper):
    """
    Scraper for pmindia.gov.in official PM scheme pages using Playwright.
    """

    SOURCE_NAME = "pmindia_gov"
    BASE_URL = PMINDIA_URLS[0]

    def __init__(self):
        super().__init__(timeout=30, max_retries=3)

    def _parse_listing_page(self, base_url: str, html: str) -> List[Dict[str, str]]:
        """Parses the PM India schemes listing page to extract detail URLs."""
        if not _BS4_AVAILABLE:
            return []
        soup = BeautifulSoup(html, "lxml")
        scheme_links = []

        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            text = a.get_text(strip=True)

            if len(text) > 8 and not any(skip in href.lower() for skip in ["#", "javascript", "facebook", "twitter", "instagram", "youtube", "feed", "privacy"]):
                if any(kw in href.lower() for kw in ["/government_tr_rec/", "/major-initiatives/", "/scheme", "/initiative"]):
                    full_url = urljoin(base_url, href)
                    scheme_links.append({"scheme_name": text, "url": full_url})

        # Deduplicate
        seen = set()
        unique = []
        for s in scheme_links:
            if s["url"] not in seen and s["url"] not in PMINDIA_URLS:
                seen.add(s["url"])
                unique.append(s)

        return unique[:MAX_SCHEMES]

    def _parse_scheme_page(self, scheme_name: str, url: str, html: str) -> Optional[Dict[str, Any]]:
        """Parses an individual PM India scheme detail page."""
        if not _BS4_AVAILABLE:
            return None
        soup = BeautifulSoup(html, "lxml")
        content_div = (
            soup.find("div", class_="entry-content")
            or soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="post-content")
        )
        if not content_div:
            return None

        paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
        full_text = "\n\n".join(paragraphs)

        if len(full_text) < 50:
            return None

        details = full_text
        benefits = ""
        eligibility = ""
        application = ""

        headings = content_div.find_all(["h2", "h3", "h4", "strong", "b"])
        for h in headings:
            heading_text = h.get_text(strip=True).lower()
            next_sibling = h.find_next_sibling()
            sibling_text = next_sibling.get_text(strip=True) if next_sibling else ""

            if any(kw in heading_text for kw in ["benefit", "advantage", "features"]):
                benefits = sibling_text or benefits
            elif any(kw in heading_text for kw in ["eligible", "eligibility", "criteria", "who can"]):
                eligibility = sibling_text or eligibility
            elif any(kw in heading_text for kw in ["apply", "application", "how to", "process"]):
                application = sibling_text or application

        slug = re.sub(r"[^a-z0-9]+", "-", scheme_name.lower()).strip("-")[:30]
        hash_suffix = hashlib.md5(url.encode()).hexdigest()[:8]
        scheme_id = f"LPM_{hash_suffix}"

        return {
            "scheme_id":   scheme_id,
            "scheme_name": scheme_name.strip(),
            "slug":        slug,
            "details":     details[:5000],
            "benefits":    benefits[:2000],
            "eligibility": eligibility[:2000],
            "application": application[:2000],
            "documents":   "",
            "level":       "Central",
            "state":       None,
            "states":      [],
            "categories":  ["Central Government"],
            "tags":        "pm india flagship central scheme",
            "source_url":  url,
            "data_source": self.SOURCE_NAME,
        }

    def fetch_all(self) -> List[Dict[str, Any]]:
        if not _BS4_AVAILABLE or not _PLAYWRIGHT_AVAILABLE:
            logger.warning("Missing dependencies for pmindia scraper.")
            return []

        logger.info("Starting pmindia.gov.in scrape via Playwright...")
        results: List[Dict[str, Any]] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )
            page = context.new_page()

            try:
                scheme_links = []
                for test_url in PMINDIA_URLS:
                    try:
                        logger.info("Checking pmindia URL: %s", test_url)
                        page.goto(test_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                        listing_html = page.content()
                        links = self._parse_listing_page(test_url, listing_html)
                        if links:
                            scheme_links.extend(links)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", test_url, e)

                # Deduplicate
                seen_urls = set()
                deduped_links = []
                for l in scheme_links:
                    if l["url"] not in seen_urls:
                        seen_urls.add(l["url"])
                        deduped_links.append(l)

                logger.info("Found %d scheme links on pmindia", len(deduped_links))

                for i, link in enumerate(deduped_links[:MAX_SCHEMES], 1):
                    try:
                        page.goto(link["url"], wait_until="domcontentloaded", timeout=self.timeout * 1000)
                        time.sleep(0.5)
                        scheme_html = page.content()
                        parsed = self._parse_scheme_page(link["scheme_name"], link["url"], scheme_html)
                        if parsed:
                            results.append(parsed)
                    except Exception as e:
                        logger.debug("Error loading PMIndia scheme %s: %s", link["url"], e)

            except Exception as e:
                logger.error("Failed during pmindia scrape: %s", e)
            finally:
                browser.close()

        logger.info("pmindia.gov.in scrape complete: %d schemes fetched", len(results))
        return results

"""
scraper/scrapers/myscheme_scraper.py
=====================================
Primary scraper for myscheme.gov.in — India's official national scheme portal.

Uses Playwright to interact directly with the internal API endpoints:
  - Search Catalog: /api/apisetu/search/schemes?lang=en&q=[]&keyword=&sort=&from={offset}&size={size}
  - Scheme Details: /api/apisetu/schemes?slug={slug}&lang=en

This bypasses WAF / Cloudflare protections by executing within the authenticated browser session.
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

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

logger = logging.getLogger("sugamgov_scraper.myscheme")

MYSCHEME_BASE = "https://www.myscheme.gov.in"
MYSCHEME_SEARCH_PAGE = "https://www.myscheme.gov.in/search"
MAX_SCHEMES_DEFAULT = 100
BATCH_SIZE = 50


def _clean_text(val: Any) -> str:
    """Recursively formats and strips HTML/markdown/lists to clean plain text."""
    if not val:
        return ""
    if isinstance(val, list):
        return "\n".join(_clean_text(item) for item in val if item)
    if isinstance(val, dict):
        return "\n".join(f"{k}: {_clean_text(v)}" for k, v in val.items() if v)
    text_str = str(val).strip()
    if "<" in text_str and ">" in text_str and _BS4_AVAILABLE:
        try:
            soup = BeautifulSoup(text_str, "lxml")
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            pass
    return text_str


class MySchemeScraper(BaseScraper):
    """
    High-performance scraper for myscheme.gov.in using Playwright + Internal APIs.
    """

    SOURCE_NAME = "myscheme_web"
    BASE_URL = MYSCHEME_BASE

    def __init__(
        self,
        max_schemes: int = MAX_SCHEMES_DEFAULT,
        offset: int = 0,
        delay: float = 0.1,
        skip_existing: bool = True,
        existing_ids: Optional[Any] = None,
    ):
        super().__init__(timeout=30, max_retries=3)
        self.max_schemes = max_schemes
        self.offset = max(0, offset)
        self.delay = delay
        self.skip_existing = skip_existing
        self.existing_ids = set(existing_ids) if existing_ids else set()

    def fetch_all(self) -> List[Dict[str, Any]]:
        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not installed. Please run: pip install playwright && playwright install chromium")
            return []

        logger.info(
            "Starting myscheme.gov.in scrape via Playwright (Target: up to %d new schemes, Start offset: %d)...",
            self.max_schemes,
            self.offset,
        )
        results: List[Dict[str, Any]] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )
            page = context.new_page()

            try:
                logger.info("Connecting to %s to establish session...", MYSCHEME_SEARCH_PAGE)
                page.goto(MYSCHEME_SEARCH_PAGE, wait_until="networkidle", timeout=self.timeout * 1000)

                offset = self.offset
                total_schemes_available = 0
                skipped_known_count = 0

                while len(results) < self.max_schemes:
                    fetch_size = BATCH_SIZE
                    search_data = page.evaluate("""
                        async (params) => {
                            const { from, size } = params;
                            try {
                                const res = await fetch(`https://www.myscheme.gov.in/api/apisetu/search/schemes?lang=en&q=[]&keyword=&sort=&from=${from}&size=${size}`);
                                if (!res.ok) return null;
                                return await res.json();
                            } catch (e) {
                                return null;
                            }
                        }
                    """, {"from": offset, "size": fetch_size})

                    if not search_data or not isinstance(search_data, dict):
                        logger.warning("Empty or invalid response from myscheme search API at offset %d", offset)
                        break

                    data_obj = search_data.get("data", {})
                    if not total_schemes_available:
                        total_schemes_available = data_obj.get("summary", {}).get("total", 0)
                        logger.info("Total schemes registered on myscheme.gov.in: %s", total_schemes_available)

                    hits = data_obj.get("hits", {})
                    items = hits.get("items", [])
                    if not items:
                        logger.info("No more scheme items returned by search API.")
                        break

                    items_processed = 0
                    for item in items:
                        if len(results) >= self.max_schemes:
                            break
                        items_processed += 1

                        try:
                            fields = item.get("fields", {})
                            slug = fields.get("slug", "").strip()
                            scheme_name = fields.get("schemeName", "").strip()
                            if not scheme_name:
                                continue

                            # Generate unique scheme_id
                            if slug:
                                scheme_id = f"LMS_{slug[:16]}"
                            else:
                                scheme_id = f"LMS_{hashlib.md5(scheme_name.encode()).hexdigest()[:10]}"

                            # Skip if already in database
                            if self.skip_existing and self.existing_ids and scheme_id in self.existing_ids:
                                skipped_known_count += 1
                                continue

                            # Fetch full detail for the scheme via slug API
                            detail_data = None
                            if slug:
                                try:
                                    detail_data = page.evaluate("""
                                        async (s) => {
                                            try {
                                                const res = await fetch(`https://www.myscheme.gov.in/api/apisetu/schemes?slug=${s}&lang=en`);
                                                if (!res.ok) return null;
                                                return await res.json();
                                            } catch (e) {
                                                return null;
                                            }
                                        }
                                    """, slug)
                                except Exception as e:
                                    logger.debug("Detail fetch error for %s: %s", slug, e)

                            en_content = {}
                            if detail_data and isinstance(detail_data, dict):
                                data_block = detail_data.get("data")
                                if isinstance(data_block, dict):
                                    en_content = data_block.get("en") or {}
                                    if not isinstance(en_content, dict):
                                        en_content = {}

                            # Extract rich text attributes
                            brief = _clean_text(en_content.get("briefDescription") or fields.get("briefDescription", ""))
                            details_text = _clean_text(en_content.get("details"))
                            if not details_text:
                                details_text = brief
                            elif brief and brief not in details_text:
                                details_text = f"{brief}\n\n{details_text}"

                            benefits_text = _clean_text(en_content.get("benefits"))
                            eligibility_text = _clean_text(en_content.get("eligibilityCriteria"))
                            application_text = _clean_text(en_content.get("applicationProcess"))
                            documents_text = _clean_text(en_content.get("documentsRequired"))

                            level = en_content.get("level") or fields.get("level") or "Central"
                            states_list = en_content.get("beneficiaryState") or fields.get("beneficiaryState") or []
                            if isinstance(states_list, str):
                                states_list = [states_list]

                            primary_state = None
                            if states_list and "All" not in states_list:
                                primary_state = states_list[0]
                                level = "State"

                            categories = en_content.get("schemeCategory") or fields.get("schemeCategory") or []
                            if isinstance(categories, str):
                                categories = [categories]

                            raw_tags = en_content.get("tags") or fields.get("tags") or []
                            if isinstance(raw_tags, list):
                                tags_str = ", ".join(str(t) for t in raw_tags if t)
                            else:
                                tags_str = str(raw_tags)

                            scheme_url = f"{MYSCHEME_BASE}/schemes/{slug}" if slug else MYSCHEME_BASE

                            scheme_dict = {
                                "scheme_id":   scheme_id,
                                "scheme_name": scheme_name,
                                "slug":        slug,
                                "details":     details_text[:5000],
                                "benefits":    benefits_text[:3000],
                                "eligibility": eligibility_text[:3000],
                                "application": application_text[:2000],
                                "documents":   documents_text[:2000],
                                "level":       level,
                                "state":       primary_state,
                                "states":      states_list if "All" not in states_list else [],
                                "categories":  categories[:10],
                                "tags":        tags_str[:500],
                                "source_url":  scheme_url,
                                "data_source": self.SOURCE_NAME,
                            }

                            results.append(scheme_dict)
                            logger.info("[%d/%d] Scraped: %s (%s)", len(results), self.max_schemes, scheme_name, level)
                            time.sleep(self.delay)

                        except Exception as item_err:
                            logger.warning("Error processing scheme item: %s", item_err)
                            continue

                    offset += items_processed

            except Exception as e:
                logger.error("Error during myscheme Playwright scrape: %s", e, exc_info=True)
            finally:
                browser.close()

        logger.info("myscheme.gov.in scrape complete: %d schemes fetched successfully", len(results))
        return results

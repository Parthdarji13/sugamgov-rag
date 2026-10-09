"""
scraper/scrapers/base_scraper.py
=================================
Abstract base class for all SugamGov scheme scrapers.

Every scraper must:
  - Implement `fetch_all()` returning a list of raw scheme dicts.
  - Populate the REQUIRED_FIELDS in each returned dict.
  - Handle its own network errors gracefully (return [] on failure).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger("sugamgov_scraper")

# Fields every scraper must populate (maps to 'schemes' table columns)
REQUIRED_FIELDS = [
    "scheme_id",     # Unique identifier e.g. "LIVE_myscheme_12345"
    "scheme_name",   # Full official name
    "source_url",    # URL the data was fetched from
    "data_source",   # Scraper name e.g. "myscheme_api"
]

# Optional but highly recommended fields
OPTIONAL_FIELDS = [
    "details",       # Description / overview
    "benefits",      # What beneficiaries receive
    "eligibility",   # Who can apply
    "application",   # How to apply
    "documents",     # Required documents
    "level",         # "Central" or "State"
    "state",         # Primary state (None for central)
    "states",        # List of states (for multi-state schemes)
    "categories",    # List of category strings
    "tags",          # Comma-separated tags string
    "slug",          # URL slug
]


class BaseScraper(ABC):
    """
    Abstract base class for all SugamGov live scheme scrapers.
    """

    SOURCE_NAME: str = "base"   # Override in subclasses
    BASE_URL: str = ""           # Override in subclasses

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"sugamgov_scraper.{self.SOURCE_NAME}")

    @abstractmethod
    def fetch_all(self) -> List[Dict[str, Any]]:
        """
        Fetch all available schemes from the source.

        Returns:
            List of raw scheme dicts. Each dict must contain REQUIRED_FIELDS.
            Returns [] if the source is unreachable or returns an error.
        """
        raise NotImplementedError

    def validate(self, scheme: Dict[str, Any]) -> bool:
        """
        Validates that a scraped scheme dict has all required fields
        and a non-empty scheme_name.
        """
        for field in REQUIRED_FIELDS:
            if field not in scheme or not scheme[field]:
                self.logger.warning(
                    "Scheme missing required field '%s': %s",
                    field,
                    str(scheme)[:120],
                )
                return False
        if not str(scheme.get("scheme_name", "")).strip():
            return False
        return True

    def fetch_and_validate(self) -> List[Dict[str, Any]]:
        """
        Calls fetch_all() and filters out invalid schemes.
        Safe to call — never raises. Returns [] on any error.
        """
        try:
            raw = self.fetch_all()
        except Exception as exc:
            self.logger.error(
                "Unhandled error in %s.fetch_all(): %s",
                self.__class__.__name__,
                exc,
                exc_info=True,
            )
            return []

        valid = [s for s in raw if self.validate(s)]
        self.logger.info(
            "%s: fetched %d schemes, %d valid after validation",
            self.SOURCE_NAME,
            len(raw),
            len(valid),
        )
        return valid

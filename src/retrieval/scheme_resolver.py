"""
src/retrieval/scheme_resolver.py
================================
Dynamic In-Memory Scheme Resolution & Entity Linking Engine for SugamGov AI RAG.

Solves the core retrieval challenge across ALL 5,954+ government schemes:
  1. Acronyms & Short Names: "pm kisan", "pmay", "pmjay", "pm svanidhi", "kcc", "apy", "mgnrega".
  2. Typos & Transliterations: "pm kishan", "ayushman baharat", "fasal beema", "swach bharat".
  3. Parenthetical & Sub-titles: "KSY", "PM-DAKSH", "PM-KISAN", "PMFBY".
  4. Conversational Queries: "tell me about pm kisan", "eligibility for ayushman bharat".

Architecture:
  - Loads all 5,954+ schemes into an in-memory catalog once on startup (cached, thread-safe, <2MB RAM).
  - Automatically generates bidirectional aliases:
      * "Pradhan Mantri" <-> "PM"
      * "Chief Minister" <-> "CM"
      * Extracted parenthetical codes: e.g. "(PM SVANidhi)" -> "PM SVANidhi", "SVANidhi"
      * Initialisms / Acronyms: e.g. "Pradhan Mantri Awas Yojana" -> "PMAY", "PM Awas"
      * Suffix-stripped stems: removes generic suffixes ("yojana", "scheme", "mission", etc.)
  - High-speed matching using RapidFuzz (token_set_ratio & partial_ratio) and normalized substring matching.
  - Returns ranked SchemeResolution candidates and directly retrieves authoritative chunks.
"""

import os
import re
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

from src.retrieval.models import RetrievalResult

logger = logging.getLogger("sugamgov_resolver")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

try:
    from rapidfuzz import fuzz, process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not available. Fuzzy matching will use basic token overlap.")


# Common conversational prefixes and suffixes to strip before scheme name matching
CONVERSATIONAL_PREFIXES = [
    r"^tell\s+me\s+about\s+",
    r"^can\s+you\s+tell\s+me\s+about\s+",
    r"^what\s+is\s+(the\s+)?",
    r"^what\s+are\s+(the\s+)?",
    r"^give\s+(me\s+)?(details|information|info)\s+(about|of|on)\s+",
    r"^how\s+to\s+apply\s+(for\s+)?",
    r"^how\s+can\s+i\s+apply\s+(for\s+)?",
    r"^who\s+is\s+eligible\s+(for\s+)?",
    r"^eligibility\s+(criteria\s+)?(for|of)\s+",
    r"^documents\s+(required\s+)?(for|of)\s+",
    r"^benefits\s+(of|for)\s+",
    r"^details\s+(of|about|for)\s+",
    r"^apply\s+(for|process\s+for)\s+",
    r"^information\s+(about|on|for)\s+",
    r"^explain\s+(about\s+)?",
    r"^i\s+want\s+to\s+know\s+about\s+",
    r"^please\s+tell\s+me\s+about\s+",
    # Hindi romanized
    r"^ke\s+bare\s+me\s+batao\s+",
    r"^kya\s+hai\s+",
    r"^batao\s+",
    r"^jaankari\s+do\s+",
    # Gujarati romanized
    r"^vishe\s+mahiti\s+aapo\s+",
    r"^su\s+chhe\s+",
]

GENERIC_SCHEME_TERMS: Set[str] = {
    "yojana", "yojna", "scheme", "mission", "pariyojana", "pariyojna",
    "programme", "program", "portal", "abhiyan", "initiative", "project",
    "board", "nigam", "kosh", "nidhi", "vibhag", "department",
}


def normalize_query_text(text_val: str) -> str:
    """Cleans whitespace, lowercases, and strips surrounding punctuation."""
    if not text_val:
        return ""
    cleaned = text_val.lower().strip()
    # Normalize hyphens and dashes to spaces
    cleaned = re.sub(r"[\-_/]+", " ", cleaned)
    # Remove special characters except alphanumeric and spaces
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def strip_conversational_fillers(query: str) -> str:
    """Strips common conversational introductory phrasing."""
    cleaned = query.strip()
    for pattern in CONVERSATIONAL_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    # Also strip common trailing phrasing
    cleaned = re.sub(r"\s+(ke\s+bare\s+me\s+batao|kya\s+hai|details|information|scheme)$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


@dataclass
class SchemeCatalogEntry:
    """Indexed catalog item representing one government scheme."""
    scheme_id: str
    scheme_name: str
    slug: str
    level: Optional[str]
    state: Optional[str]
    categories: List[str] = field(default_factory=list)
    # Generated search aliases
    aliases: Set[str] = field(default_factory=set)
    # Normalized search tokens
    tokens: Set[str] = field(default_factory=set)


@dataclass
class SchemeMatch:
    """Result of matching a user query to a scheme."""
    scheme_id: str
    scheme_name: str
    score: float
    matched_alias: str
    match_type: str  # 'exact_alias', 'acronym', 'substring', 'fuzzy'
    level: Optional[str] = None
    state: Optional[str] = None
    categories: List[str] = field(default_factory=list)


class SchemeResolver:
    """
    In-memory scheme resolution engine with fuzzy matching, alias generation,
    and authoritative chunk retrieval.
    """

    def __init__(self, engine_or_url: Optional[Union[Engine, str]] = None):
        if isinstance(engine_or_url, Engine):
            self.engine = engine_or_url
        elif isinstance(engine_or_url, str):
            self.engine = create_engine(engine_or_url)
        else:
            if ENV_PATH.exists():
                load_dotenv(ENV_PATH)
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL environment variable is required.")
            self.engine = create_engine(db_url)

        self._catalog: List[SchemeCatalogEntry] = []
        self._id_map: Dict[str, SchemeCatalogEntry] = {}
        self._alias_index: Dict[str, str] = {}  # alias_norm -> scheme_id
        self._lock = threading.Lock()
        self._loaded = False

    def _ensure_catalog_loaded(self) -> None:
        """Loads and indexes all schemes from the database on first call."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            logger.info("Initializing SchemeResolver in-memory catalog from PostgreSQL...")
            sql = text("""
                SELECT 
                    scheme_id,
                    scheme_name,
                    slug,
                    level,
                    state,
                    categories
                FROM schemes
                ORDER BY id ASC;
            """)

            with self.engine.connect() as conn:
                rows = conn.execute(sql).fetchall()

            catalog: List[SchemeCatalogEntry] = []
            id_map: Dict[str, SchemeCatalogEntry] = {}
            alias_index: Dict[str, str] = {}

            for r in rows:
                sid = str(r.scheme_id).strip()
                sname = str(r.scheme_name).strip()
                sslug = str(r.slug or "").strip()
                slevel = str(r.level).strip() if r.level else None
                sstate = str(r.state).strip() if r.state else None
                scats = r.categories if isinstance(r.categories, list) else []

                entry = SchemeCatalogEntry(
                    scheme_id=sid,
                    scheme_name=sname,
                    slug=sslug,
                    level=slevel,
                    state=sstate,
                    categories=scats,
                )

                # Generate comprehensive aliases for this scheme
                aliases = self._generate_aliases(sname, sslug)
                entry.aliases = aliases

                # Token set of significant words
                norm_name = normalize_query_text(sname)
                entry.tokens = {w for w in norm_name.split() if len(w) > 1}

                catalog.append(entry)
                id_map[sid] = entry

                # Register exact aliases in fast map (preferring earlier / central schemes if conflict)
                for a in aliases:
                    norm_a = normalize_query_text(a)
                    if norm_a and norm_a not in alias_index:
                        alias_index[norm_a] = sid

            self._catalog = catalog
            self._id_map = id_map
            self._alias_index = alias_index
            self._loaded = True
            logger.info(
                "SchemeResolver initialized: %d schemes indexed with %d fast-match aliases.",
                len(catalog), len(alias_index)
            )

    def _generate_aliases(self, scheme_name: str, slug: str) -> Set[str]:
        """
        Generates common abbreviations, acronyms, and variations for a scheme name.
        """
        aliases: Set[str] = set()
        clean = scheme_name.strip()
        aliases.add(clean)

        # 1. Extract parenthetical abbreviations, e.g. "PM Street Vendor's ... (PM SVANidhi)"
        parens = re.findall(r"\(([^)]+)\)", clean)
        for p in parens:
            p_strip = p.strip()
            if len(p_strip) >= 2:
                aliases.add(p_strip)
                # Without hyphens / spaces: e.g. "PM-KISAN" -> "PM KISAN", "PMKISAN"
                aliases.add(re.sub(r"[\s\-]+", "", p_strip))
                aliases.add(re.sub(r"[\-_]+", " ", p_strip))

        # 2. Main title without parentheses
        main_title = re.sub(r"\([^)]*\)", "", clean).strip()
        if main_title and main_title != clean:
            aliases.add(main_title)

        # 3. Handle 'Pradhan Mantri' <-> 'PM' in all variations
        pm_patterns = [
            r"\bpradhan\s+mantri\b",
            r"\bprime\s+minister\b",
            r"\bpradhān\s+mantrī\b",
        ]
        has_pm = any(re.search(pat, clean, re.IGNORECASE) for pat in pm_patterns)
        if has_pm:
            for pat in pm_patterns:
                pm_short = re.sub(pat, "PM", clean, flags=re.IGNORECASE).strip()
                aliases.add(pm_short)
                pm_main = re.sub(pat, "PM", main_title, flags=re.IGNORECASE).strip()
                aliases.add(pm_main)

        # Handle 'Chief Minister' <-> 'CM'
        cm_patterns = [r"\bchief\s+minister\b", r"\bmukhyamantri\b", r"\bmukhya\s+mantri\b"]
        has_cm = any(re.search(pat, clean, re.IGNORECASE) for pat in cm_patterns)
        if has_cm:
            for pat in cm_patterns:
                cm_short = re.sub(pat, "CM", clean, flags=re.IGNORECASE).strip()
                aliases.add(cm_short)

        # 4. Generate initialism acronym from capitalized words
        # e.g., "Pradhan Mantri Awas Yojana" -> "PMAY", "PM Awas"
        words = [w for w in re.split(r"[\s\-_]+", main_title) if w and w[0].isalnum()]
        if len(words) >= 3:
            initials = "".join(w[0] for w in words).upper()
            if 3 <= len(initials) <= 8:
                aliases.add(initials)

        # 5. Extract short core name by removing generic terms (Yojana, Scheme, etc.)
        meaningful_words = [w for w in words if w.lower() not in GENERIC_SCHEME_TERMS]
        if meaningful_words and len(meaningful_words) < len(words):
            core_name = " ".join(meaningful_words)
            if len(core_name) >= 3:
                aliases.add(core_name)
                # If it had PM, also add PM + core name
                if has_pm:
                    aliases.add(f"PM {core_name}")

        # 6. From slug: "pradhan-mantri-kisan-samman-nidhi" -> "pradhan mantri kisan samman nidhi"
        if slug:
            slug_spaced = slug.replace("-", " ").strip()
            aliases.add(slug_spaced)
            if has_pm:
                aliases.add(re.sub(r"\bpradhan\s+mantri\b", "pm", slug_spaced, flags=re.IGNORECASE).strip())

        # Clean all aliases
        clean_aliases = set()
        for a in aliases:
            norm = re.sub(r"\s+", " ", a).strip()
            if len(norm) >= 2:
                clean_aliases.add(norm)

        return clean_aliases

    def resolve(
        self,
        query: str,
        state: Optional[str] = None,
        top_k: int = 3,
        min_confidence: float = 75.0,
    ) -> List[SchemeMatch]:
        """
        Resolves a user search query against the scheme catalog.

        Strategy:
          1. Exact match on normalized aliases (confidence: 100).
          2. Substring match: query contains alias OR alias contains query (confidence: 90-95).
          3. Fuzzy matching with RapidFuzz token_set_ratio & partial_ratio (confidence: 75-95).
          4. Ranks candidates deterministically, prioritizing exact alias matches and state alignment.
        """
        self._ensure_catalog_loaded()

        if not query or not query.strip():
            return []

        raw_query = query.strip()
        core_query = strip_conversational_fillers(raw_query)
        norm_core = normalize_query_text(core_query)
        norm_raw = normalize_query_text(raw_query)

        candidates: Dict[str, SchemeMatch] = {}

        # ── Step 1: Direct Exact Alias Match ─────────────────────────────────
        for candidate_text in [norm_core, norm_raw]:
            if candidate_text in self._alias_index:
                matched_sid = self._alias_index[candidate_text]
                entry = self._id_map[matched_sid]
                candidates[matched_sid] = SchemeMatch(
                    scheme_id=matched_sid,
                    scheme_name=entry.scheme_name,
                    score=100.0,
                    matched_alias=candidate_text,
                    match_type="exact_alias",
                    level=entry.level,
                    state=entry.state,
                    categories=entry.categories,
                )

        # ── Step 2: Substring Alias & Acronym Match ──────────────────────────
        # Check if the query contains a distinctive scheme alias (e.g. "tell me about pm kisan")
        # or if an alias matches the query words.
        clean_state = state.strip().lower() if state and state.strip() else None

        for entry in self._catalog:
            # If state filter given and scheme is from a different state, skip
            if clean_state and entry.state and entry.state.lower() != clean_state and entry.level != "Central":
                continue

            for alias in entry.aliases:
                norm_a = normalize_query_text(alias)
                if len(norm_a) < 3:
                    continue

                # Exact match against alias
                if norm_core == norm_a or norm_raw == norm_a:
                    if entry.scheme_id not in candidates or candidates[entry.scheme_id].score < 100.0:
                        candidates[entry.scheme_id] = SchemeMatch(
                            scheme_id=entry.scheme_id,
                            scheme_name=entry.scheme_name,
                            score=100.0,
                            matched_alias=alias,
                            match_type="exact_alias",
                            level=entry.level,
                            state=entry.state,
                            categories=entry.categories,
                        )
                    continue

                # Substring match: alias is a full word boundary match inside query
                # e.g., "tell me about pm kisan eligibility" -> "pm kisan"
                alias_word_pattern = r"\b" + re.escape(norm_a) + r"\b"
                if re.search(alias_word_pattern, norm_core) or re.search(alias_word_pattern, norm_raw):
                    # Longer distinctive aliases get higher score
                    alias_len = len(norm_a.split())
                    match_score = 96.0 if alias_len >= 2 else 90.0
                    if entry.scheme_id not in candidates or candidates[entry.scheme_id].score < match_score:
                        candidates[entry.scheme_id] = SchemeMatch(
                            scheme_id=entry.scheme_id,
                            scheme_name=entry.scheme_name,
                            score=match_score,
                            matched_alias=alias,
                            match_type="substring",
                            level=entry.level,
                            state=entry.state,
                            categories=entry.categories,
                        )

        # If high-confidence exact/substring matches were found, return top results immediately
        if candidates and any(c.score >= 95.0 for c in candidates.values()):
            sorted_candidates = sorted(
                candidates.values(),
                key=lambda m: (-m.score, 0 if m.level == "Central" else 1, m.scheme_id),
            )
            return sorted_candidates[:top_k]

        # ── Step 3: Fuzzy Match using RapidFuzz (Typos & Transliterations) ───
        if _RAPIDFUZZ_AVAILABLE and len(norm_core) >= 3:
            query_tokens = [w for w in norm_core.split() if len(w) >= 3 and w not in GENERIC_SCHEME_TERMS]
            for entry in self._catalog:
                if clean_state and entry.state and entry.state.lower() != clean_state and entry.level != "Central":
                    continue

                best_entry_score = 0.0
                best_alias = ""

                for alias in entry.aliases:
                    norm_a = normalize_query_text(alias)
                    if len(norm_a) < 3:
                        continue

                    # token_set_ratio is robust against word order and typos
                    ratio = fuzz.token_set_ratio(norm_core, norm_a)
                    if ratio > best_entry_score:
                        best_entry_score = ratio
                        best_alias = alias

                    # Also test WRatio for typos in short phrases
                    wratio = fuzz.WRatio(norm_core, norm_a)
                    if wratio > best_entry_score:
                        best_entry_score = wratio
                        best_alias = alias

                if best_entry_score >= min_confidence:
                    if entry.scheme_id not in candidates or candidates[entry.scheme_id].score < best_entry_score:
                        candidates[entry.scheme_id] = SchemeMatch(
                            scheme_id=entry.scheme_id,
                            scheme_name=entry.scheme_name,
                            score=float(best_entry_score),
                            matched_alias=best_alias,
                            match_type="fuzzy",
                            level=entry.level,
                            state=entry.state,
                            categories=entry.categories,
                        )

        # Sort all matched candidates descending by score
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda m: (-m.score, 0 if m.level == "Central" else 1, m.scheme_id),
        )
        return sorted_candidates[:top_k]

    def get_chunks_for_scheme(
        self,
        scheme_id: str,
        query: str = "",
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """
        Directly retrieves chunks from `scheme_chunks` for a resolved scheme_id.
        Orders chunks to prioritize the field matching the user's intent:
          - If query mentions 'eligibility', eligibility chunks rank highest.
          - If query mentions 'documents', documents chunks rank highest.
          - If query mentions 'apply' or 'process', application chunks rank highest.
          - Default order: scheme_name -> benefits -> eligibility -> details -> application.
        """
        clean_sid = scheme_id.strip()
        lower_q = query.lower() if query else ""

        # Identify aspect priority from user query
        priority_field = "scheme_name"
        if any(term in lower_q for term in ["document", "documents", "certificate", "id proof", "कागजात", "દસ્તાવેજ"]):
            priority_field = "documents"
        elif any(term in lower_q for term in ["eligibility", "eligible", "criteria", "who can", "पात्रता"]):
            priority_field = "eligibility_criteria"
        elif any(term in lower_q for term in ["benefit", "benefits", "amount", "money", "how much", "लाभ"]):
            priority_field = "benefits"
        elif any(term in lower_q for term in ["apply", "application", "process", "procedure", "how to apply", "आवेदन", "અરજી"]):
            priority_field = "application"

        sql = text("""
            SELECT 
                c.chunk_id,
                c.scheme_id,
                s.scheme_name,
                c.field_name,
                c.chunk_text,
                c.metadata,
                s.state,
                s.level,
                s.categories
            FROM scheme_chunks c
            JOIN schemes s ON c.scheme_id = s.scheme_id
            WHERE c.scheme_id = :scheme_id
            ORDER BY
                CASE 
                    WHEN c.field_name = :priority_field THEN 0
                    WHEN c.field_name = 'scheme_name' THEN 1
                    WHEN c.field_name = 'benefits' THEN 2
                    WHEN c.field_name = 'eligibility_criteria' OR c.field_name = 'eligibility' THEN 3
                    WHEN c.field_name = 'details' THEN 4
                    WHEN c.field_name = 'application' THEN 5
                    WHEN c.field_name = 'documents' THEN 6
                    ELSE 7
                END,
                c.id ASC
            LIMIT :limit;
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql, {
                "scheme_id": clean_sid,
                "priority_field": priority_field,
                "limit": limit,
            }).fetchall()

        results: List[RetrievalResult] = []
        for i, row in enumerate(rows):
            cats = row.categories if isinstance(row.categories, list) else []
            meta = row.metadata if isinstance(row.metadata, dict) else {}
            # Assign authoritative high scores (0.95 down to 0.85) to guaranteed scheme chunks
            rank_score = max(0.85, 0.98 - (i * 0.02))

            results.append(
                RetrievalResult(
                    chunk_id=str(row.chunk_id),
                    scheme_id=str(row.scheme_id),
                    scheme_name=str(row.scheme_name),
                    field_name=str(row.field_name),
                    chunk_text=str(row.chunk_text),
                    score=rank_score,
                    metadata=meta,
                    state=row.state,
                    level=row.level,
                    categories=cats,
                )
            )

        return results


# Global singleton instance for high-speed reuse across requests
_global_resolver: Optional[SchemeResolver] = None
_resolver_lock = threading.Lock()


def get_scheme_resolver(engine: Optional[Engine] = None) -> SchemeResolver:
    """Returns the shared SchemeResolver singleton."""
    global _global_resolver
    if _global_resolver is None:
        with _resolver_lock:
            if _global_resolver is None:
                _global_resolver = SchemeResolver(engine_or_url=engine)
    return _global_resolver

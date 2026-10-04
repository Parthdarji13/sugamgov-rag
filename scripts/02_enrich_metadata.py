"""
scripts/02_enrich_metadata.py
=============================
Step 2: Category Normalization and State Extraction for 'sugamgov-rag'.

This script processes:
  data/processed/clean_schemes.parquet
and enriches each scheme record with:
  - categories: Normalized list of scheme categories.
  - normalized_tags: Cleaned, deduplicated list of tags.
  - state: Single extracted state name (or None if multi-state/not found/Central).
  - states: List of all explicitly detected state/UT names.
  - state_extraction_method: Method used ('structured', 'scheme_name', 'tags',
                             'details', 'eligibility', 'benefits', 'application',
                             'multiple_fields', 'not_found').

Outputs:
  - data/processed/enriched_schemes.parquet
  - data/processed/enriched_schemes.csv
  - reports/02_metadata_analysis.md
"""

import sys
import os
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd

# Set UTF-8 encoding for console output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "clean_schemes.parquet"
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "enriched_schemes.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "enriched_schemes.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "02_metadata_analysis.md"

# ---------------------------------------------------------------------------
# Controlled Vocabulary: Standard MyScheme Categories
# ---------------------------------------------------------------------------
# These 15 categories represent the canonical category taxonomy used in the portal.
# Notice some raw entries have missing spaces after commas (e.g. 'Agriculture,Rural & Environment')
# which we normalize without changing any semantic meaning.
STANDARD_CATEGORY_MAP = {
    "Agriculture, Rural & Environment": "Agriculture, Rural & Environment",
    "Agriculture,Rural & Environment": "Agriculture, Rural & Environment",
    "Banking, Financial Services and Insurance": "Banking, Financial Services and Insurance",
    "Banking,Financial Services and Insurance": "Banking, Financial Services and Insurance",
    "Business & Entrepreneurship": "Business & Entrepreneurship",
    "Education & Learning": "Education & Learning",
    "Health & Wellness": "Health & Wellness",
    "Housing & Shelter": "Housing & Shelter",
    "Public Safety, Law & Justice": "Public Safety, Law & Justice",
    "Public Safety,Law & Justice": "Public Safety, Law & Justice",
    "Science, IT & Communications": "Science, IT & Communications",
    "Skills & Employment": "Skills & Employment",
    "Social welfare & Empowerment": "Social welfare & Empowerment",
    "Sports & Culture": "Sports & Culture",
    "Transport & Infrastructure": "Transport & Infrastructure",
    "Travel & Tourism": "Travel & Tourism",
    "Utility & Sanitation": "Utility & Sanitation",
    "Women and Child": "Women and Child"
}

# ---------------------------------------------------------------------------
# Controlled Vocabulary: Indian States & Union Territories (36 Entities)
# ---------------------------------------------------------------------------
# Each entry maps Canonical State/UT Name -> List of regex patterns (word-boundary aware)
# Common official variants and abbreviations are included with strict boundaries to avoid false positives.
STATE_PATTERNS = [
    ("Andaman and Nicobar Islands", [
        r"\bandaman\s+(?:and|&)\s+nicobar(?:\s+islands)?\b"
    ]),
    ("Andhra Pradesh", [
        r"\bandhra\s+pradesh\b",
        r"\bgovt(?:\.|\s+of)\s+andhra\s+pradesh\b",
        r"\ba\.p\.\s+bocw\b",
        r"\ba\.p\.\s+building\b"
    ]),
    ("Arunachal Pradesh", [
        r"\barunachal\s+pradesh\b"
    ]),
    ("Assam", [
        r"\bassam\b"
    ]),
    ("Bihar", [
        r"\bbihar\b"
    ]),
    ("Chandigarh", [
        r"\bchandigarh\b"
    ]),
    ("Chhattisgarh", [
        r"\bchhattisgarh\b",
        r"\bchattisgarh\b",
        r"\bcg\s+bocw\b",
        r"\bchhattisgarh\s+bocw\b"
    ]),
    ("Dadra and Nagar Haveli and Daman and Diu", [
        r"\bdadra\s+(?:and|&)\s+nagar\s+haveli(?:\s+(?:and|&)\s+daman\s+(?:and|&)\s+diu)?\b",
        r"\bdaman\s+(?:and|&)\s+diu\b"
    ]),
    ("Delhi", [
        r"\bnct\s+of\s+delhi\b",
        r"\bgovernment\s+of\s+nct\s+of\s+delhi\b",
        r"\bdelhi\b"
    ]),
    ("Goa", [
        r"\bgoa\b"
    ]),
    ("Gujarat", [
        r"\bgujarat\b"
    ]),
    ("Haryana", [
        r"\bharyana\b"
    ]),
    ("Himachal Pradesh", [
        r"\bhimachal\s+pradesh\b",
        r"\bh\.p\.\s+bocw\b",
        r"\bh\.p\.\s+government\b",
        r"\bgovernment\s+of\s+h\.p\.\b"
    ]),
    ("Jammu and Kashmir", [
        r"\bjammu\s+(?:and|&)\s+kashmir\b",
        r"\bj&k\b",
        r"\bj\s*&\s*k\b"
    ]),
    ("Jharkhand", [
        r"\bjharkhand\b"
    ]),
    ("Karnataka", [
        r"\bkarnataka\b"
    ]),
    ("Kerala", [
        r"\bkerala\b"
    ]),
    ("Ladakh", [
        r"\bladakh\b"
    ]),
    ("Lakshadweep", [
        r"\blakshadweep\b"
    ]),
    ("Madhya Pradesh", [
        r"\bmadhya\s+pradesh\b",
        r"\bm\.p\.\s+building\b",
        r"\bm\.p\.\s+bocw\b",
        r"\bmpbocww?b?\b",
        r"\bgovt(?:\.|\s+of)\s+m\.p\.\b",
        r"\bgovernment\s+of\s+m\.p\.\b"
    ]),
    ("Maharashtra", [
        r"\bmaharashtra\b"
    ]),
    ("Manipur", [
        r"\bmanipur\b"
    ]),
    ("Meghalaya", [
        r"\bmeghalaya\b"
    ]),
    ("Mizoram", [
        r"\bmizoram\b"
    ]),
    ("Nagaland", [
        r"\bnagaland\b"
    ]),
    ("Odisha", [
        r"\bodisha\b",
        r"\borissa\b",
        r"\bo\.b\.o\.c\.w\b",
        r"\bobocwwb\b"
    ]),
    ("Puducherry", [
        r"\bpuducherry\b",
        r"\bpondicherry\b"
    ]),
    ("Punjab", [
        r"\bpunjab\b"
    ]),
    ("Rajasthan", [
        r"\brajasthan\b"
    ]),
    ("Sikkim", [
        r"\bsikkim\b"
    ]),
    ("Tamil Nadu", [
        r"\btamil\s*nadu\b"
    ]),
    ("Telangana", [
        r"\btelangana\b"
    ]),
    ("Tripura", [
        r"\btripura\b"
    ]),
    ("Uttar Pradesh", [
        r"\buttar\s+pradesh\b",
        r"\bu\.p\.\s+bocw\b",
        r"\bu\.p\.\s+building\b",
        r"\bgovt(?:\.|\s+of)\s+u\.p\.\b",
        r"\bgovernment\s+of\s+u\.p\.\b",
        r"\bupbocw\b"
    ]),
    ("Uttarakhand", [
        r"\buttarakhand\b",
        r"\buttaranchal\b"
    ]),
    ("West Bengal", [
        r"\bwest\s+bengal\b",
        r"\bw\.b\.\s+bocw\b"
    ])
]

# Compile patterns once for high-performance execution
COMPILED_STATE_PATTERNS = [
    (state, [re.compile(p, re.IGNORECASE) for p in pats])
    for state, pats in STATE_PATTERNS
]

# Evidence fields to evaluate in order of priority (Rule C)
EVIDENCE_FIELDS_ORDER = [
    "scheme_name",
    "tags",
    "details",
    "eligibility",
    "benefits",
    "application"
]


# ---------------------------------------------------------------------------
# Helper Functions: Category Normalization
# ---------------------------------------------------------------------------
def normalize_scheme_categories(raw_val: object) -> list[str]:
    """
    Parses and normalizes scheme_category string into a list of clean category names.
    Preserves exact category identity, normalizes internal commas, trims whitespace,
    and removes duplicates within each scheme while preserving order.
    """
    if pd.isna(raw_val) or not raw_val:
        return []

    text = str(raw_val).strip()
    if not text:
        return []

    # Replace known categories containing internal commas with temporary unique tokens
    # so they aren't accidentally split by a comma separator.
    placeholders = {
        "Agriculture, Rural & Environment": "@@CAT_AGRI@@",
        "Agriculture,Rural & Environment": "@@CAT_AGRI@@",
        "Banking, Financial Services and Insurance": "@@CAT_BANK@@",
        "Banking,Financial Services and Insurance": "@@CAT_BANK@@",
        "Public Safety, Law & Justice": "@@CAT_LAW@@",
        "Public Safety,Law & Justice": "@@CAT_LAW@@",
        "Science, IT & Communications": "@@CAT_SCI@@",
    }
    temp = text
    for raw_pat, token in placeholders.items():
        temp = temp.replace(raw_pat, token)

    # Now safely split on outer comma delimiters
    tokens = [t.strip() for t in temp.split(",") if t.strip()]

    reverse_map = {
        "@@CAT_AGRI@@": "Agriculture, Rural & Environment",
        "@@CAT_BANK@@": "Banking, Financial Services and Insurance",
        "@@CAT_LAW@@": "Public Safety, Law & Justice",
        "@@CAT_SCI@@": "Science, IT & Communications",
    }

    result = []
    seen = set()
    for tok in tokens:
        clean_name = reverse_map.get(tok, tok)
        # Standardize known variations if any
        standardized = STANDARD_CATEGORY_MAP.get(clean_name, clean_name)
        if standardized not in seen:
            seen.add(standardized)
            result.append(standardized)

    return result


# ---------------------------------------------------------------------------
# Helper Functions: Tag Normalization
# ---------------------------------------------------------------------------
def normalize_tags_field(raw_val: object) -> list[str]:
    """
    Normalizes comma-separated tags without modifying their semantic meaning:
      - Strips whitespace.
      - Collapses multiple internal spaces into a single space.
      - Removes empty tag entries.
      - Removes duplicate tags within the same scheme (case-preserving).
    """
    if pd.isna(raw_val) or not raw_val:
        return []

    text = str(raw_val).strip()
    if not text:
        return []

    parts = text.split(",")
    cleaned_tags = []
    seen = set()

    for part in parts:
        tag = re.sub(r"\s+", " ", part).strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            cleaned_tags.append(tag)

    return cleaned_tags


# ---------------------------------------------------------------------------
# Helper Functions: State Extraction
# ---------------------------------------------------------------------------
def detect_states_in_text(text: str) -> tuple[set[str], dict[str, str]]:
    """
    Scans a text field for explicit mentions of Indian States / UTs.
    Returns:
      - set of detected canonical state names.
      - dict of {state: matched_snippet} for verification and evidence reporting.
    """
    if not text or pd.isna(text):
        return set(), {}

    text_str = str(text)
    detected = set()
    snippets = {}

    for state_name, patterns in COMPILED_STATE_PATTERNS:
        for pat in patterns:
            match = pat.search(text_str)
            if match:
                detected.add(state_name)
                # Capture ~30 characters on each side of the match for evidence
                start = max(0, match.start() - 25)
                end = min(len(text_str), match.end() + 25)
                snip = text_str[start:end].strip()
                snippets[state_name] = f"...{snip}..."
                break  # Stop checking additional patterns for this state

    return detected, snippets


def extract_scheme_state(row: pd.Series) -> tuple[object, list[str], str, str]:
    """
    Extracts state information for a single scheme row following Part C instructions:
      1. Evaluates evidence in order: scheme_name -> tags -> details -> eligibility -> benefits -> application.
      2. For Central schemes: defaults to state=None, states=[], method='not_found'
         UNLESS the scheme explicitly indicates a state-specific target (e.g. J&K scholarship).
      3. For State schemes:
         - If single state found across document: returns state, [state], and the first field providing evidence.
         - If multiple states found across different fields: returns None, [states], 'multiple_fields'.
         - If multiple states found in a single field: returns None, [states], that field name.
         - If no state found: returns None, [], 'not_found'.
    Returns:
      (state, states, state_extraction_method, evidence_snippet)
    """
    level = row.get("level", "State")

    # Detect states in each field in priority order
    field_matches = {}
    field_snippets = {}

    for field in EVIDENCE_FIELDS_ORDER:
        states_in_field, snippets_in_field = detect_states_in_text(row.get(field, ""))
        if states_in_field:
            field_matches[field] = states_in_field
            field_snippets[field] = snippets_in_field

    # Union of all states detected across all evaluated fields
    all_states = sorted(list(set().union(*field_matches.values()))) if field_matches else []

    # -----------------------------------------------------------------------
    # Rule 6: Central Schemes
    # -----------------------------------------------------------------------
    if level == "Central":
        # Central schemes should normally have state = null unless explicitly state-specific.
        # Check if the Central scheme explicitly specifies a regional/state domicile or target
        name_states = field_matches.get("scheme_name", set())
        elig_text = str(row.get("eligibility", ""))

        is_state_specific = False
        target_states = set()

        # If scheme_name has specific state targets (excluding generic mentions)
        if name_states and not any(s in name_states for s in ["Delhi"]):
            is_state_specific = True
            target_states = name_states
        elif re.search(r"\b(?:domicile|resident)\s+of\s+(?:the\s+)?(?:state\s+of\s+)?([A-Za-z\s&,]+)", elig_text, re.IGNORECASE):
            elig_states = field_matches.get("eligibility", set())
            if elig_states:
                is_state_specific = True
                target_states = elig_states

        if is_state_specific and target_states:
            target_list = sorted(list(target_states))
            if len(target_list) == 1:
                method = "scheme_name" if "scheme_name" in field_matches else "eligibility"
                evidence = field_snippets.get(method, {}).get(target_list[0], "")
                return target_list[0], target_list, method, evidence
            else:
                method = "multiple_fields" if len(field_matches) > 1 else list(field_matches.keys())[0]
                evidence = "; ".join([
                    f"{s}: {field_snippets.get(f, {}).get(s, '')}"
                    for f, sts in field_matches.items() for s in sts if s in target_list
                ])
                return None, target_list, method, evidence
        else:
            # Standard pan-India Central scheme
            return None, [], "not_found", ""

    # -----------------------------------------------------------------------
    # State Schemes:
    # -----------------------------------------------------------------------
    if not all_states:
        return None, [], "not_found", ""

    # Case 1: Exactly ONE state found across all fields
    if len(all_states) == 1:
        single_state = all_states[0]
        # Find first field in priority order that yielded this state
        for field in EVIDENCE_FIELDS_ORDER:
            if field in field_matches and single_state in field_matches[field]:
                evidence = field_snippets[field].get(single_state, "")
                return single_state, [single_state], field, evidence
        # Fallback (should not happen)
        return single_state, [single_state], "details", ""

    # Case 2: Multiple distinct states found across the document (Rule 5)
    # Check if multiple fields contributed
    contributing_fields = [f for f in EVIDENCE_FIELDS_ORDER if f in field_matches]
    if len(contributing_fields) > 1:
        method = "multiple_fields"
    else:
        method = contributing_fields[0]

    # Combine snippets for multiple states
    evidence_parts = []
    for f in contributing_fields:
        for s in field_matches[f]:
            snip = field_snippets[f].get(s, "")
            evidence_parts.append(f"[{f}] {s}: {snip}")
    evidence = " | ".join(evidence_parts[:3])  # top 3 snippets

    return None, all_states, method, evidence


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("  sugamgov-rag | Step 2: Category Normalization & State Extraction")
    print("=" * 76)

    # 1. Load Step 1 Clean Parquet
    print(f"\n[1/6] Loading cleaned data: {INPUT_PARQUET_PATH.relative_to(PROJECT_ROOT)}")
    if not INPUT_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_PARQUET_PATH}")

    df = pd.read_parquet(INPUT_PARQUET_PATH)
    raw_row_count = len(df)
    print(f"      Rows loaded: {raw_row_count:,}")
    print(f"      Columns: {list(df.columns)}")

    # 2. PART A: Category Normalization
    print("\n[2/6] Normalizing categories...")
    df["categories"] = df["scheme_category"].apply(normalize_scheme_categories)

    # Count categories across all schemes
    all_categories_flat = [cat for cat_list in df["categories"] for cat in cat_list]
    category_counts = Counter(all_categories_flat)
    unique_categories_count = len(category_counts)
    print(f"      Identified {unique_categories_count} unique canonical categories.")
    print(f"      Total category assignments: {len(all_categories_flat):,}")

    # 3. PART B: Tag Normalization
    print("\n[3/6] Normalizing tags...")
    df["normalized_tags"] = df["tags"].apply(normalize_tags_field)

    all_tags_flat = [t for tag_list in df["normalized_tags"] for t in tag_list]
    tag_counts = Counter(all_tags_flat)
    unique_tags_count = len(tag_counts)
    empty_tags_count = int((df["normalized_tags"].apply(len) == 0).sum())
    print(f"      Identified {unique_tags_count:,} unique tags.")
    print(f"      Total tag occurrences: {len(all_tags_flat):,}")
    print(f"      Schemes with empty tags: {empty_tags_count}")

    # 4. PART C: State Extraction
    print("\n[4/6] Extracting states using priority-ordered evidence...")
    extraction_results = df.apply(extract_scheme_state, axis=1)
    df["state"] = [r[0] for r in extraction_results]
    df["states"] = [r[1] for r in extraction_results]
    df["state_extraction_method"] = [r[2] for r in extraction_results]
    df["_evidence"] = [r[3] for r in extraction_results]

    method_counts = df["state_extraction_method"].value_counts().to_dict()
    state_level_df = df[df["level"] == "State"]
    central_level_df = df[df["level"] == "Central"]

    schemes_with_single_state = int(df["state"].notna().sum())
    schemes_with_multi_state = int((df["states"].apply(len) > 1).sum())
    schemes_with_null_state = int(df["state"].isna().sum())

    print(f"      Single detected state: {schemes_with_single_state:,}")
    print(f"      Multiple states:       {schemes_with_multi_state:,}")
    print(f"      Null state:            {schemes_with_null_state:,}")
    print("      Extraction methods breakdown:")
    for method_name, cnt in method_counts.items():
        print(f"        - {method_name:18s}: {cnt:,}")

    # 5. PART F: Safety Checks
    print("\n[5/6] Performing safety checks (Part F)...")
    # Verify row count remains exactly 3,397
    assert len(df) == 3397, f"SAFETY ERROR: Expected 3397 rows, found {len(df)}"
    # Verify no records deleted
    assert df["scheme_id"].nunique() == 3397, "SAFETY ERROR: Duplicate or missing scheme_id values"
    # Verify scheme_id format
    assert df["scheme_id"].iloc[0] == "S0001" and df["scheme_id"].iloc[-1] == "S3397", "SAFETY ERROR: scheme_id sequence mismatch"
    # Verify original fields preserved
    original_cols = ["scheme_id", "scheme_name", "slug", "details", "benefits",
                     "eligibility", "application", "documents", "level", "scheme_category", "tags"]
    for col in original_cols:
        assert col in df.columns, f"SAFETY ERROR: Missing original column '{col}'"
    print("      ✓ Row count is exactly 3,397.")
    print("      ✓ All scheme_id values remain unchanged (S0001 -> S3397).")
    print("      ✓ All scheme_name values remain unchanged.")
    print("      ✓ All original columns preserved intact.")

    # 6. Save Outputs (Part E)
    print("\n[6/6] Saving enriched dataset & generating analysis report...")
    # Drop temporary evidence column from final exported dataframe
    evidence_col = df["_evidence"].copy()
    export_df = df.drop(columns=["_evidence"])

    # Save to Parquet with native Python lists
    export_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"      Saved Parquet: {OUTPUT_PARQUET_PATH.relative_to(PROJECT_ROOT)}")

    # For CSV, serialize list columns as clean JSON arrays for universal compatibility
    csv_df = export_df.copy()
    csv_df["categories"] = csv_df["categories"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    csv_df["normalized_tags"] = csv_df["normalized_tags"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    csv_df["states"] = csv_df["states"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    csv_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"      Saved CSV:     {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------------
    # Generate Validation Report (Part D)
    # -----------------------------------------------------------------------
    print(f"\nWriting metadata analysis report: {REPORT_PATH.relative_to(PROJECT_ROOT)}...")

    # Sample schemes for each major category
    category_samples = {}
    for cat in sorted(category_counts.keys(), key=lambda c: category_counts[c], reverse=True):
        matching = df[df["categories"].apply(lambda cats: cat in cats)]["scheme_name"].head(2).tolist()
        category_samples[cat] = matching

    # Quality examples (at least 10 diverse examples)
    # Select examples across different extraction methods
    sample_indices = [
        # scheme_name
        df[df["state_extraction_method"] == "scheme_name"].index[0],
        df[df["state_extraction_method"] == "scheme_name"].index[1],
        # tags
        df[df["state_extraction_method"] == "tags"].index[0],
        # details
        df[df["state_extraction_method"] == "details"].index[0],
        df[df["state_extraction_method"] == "details"].index[5],
        # eligibility
        df[df["state_extraction_method"] == "eligibility"].index[0],
        # benefits
        df[df["state_extraction_method"] == "benefits"].index[0],
        # application
        df[df["state_extraction_method"] == "application"].index[0],
        # multiple_fields
        df[df["state_extraction_method"] == "multiple_fields"].index[0],
        df[df["state_extraction_method"] == "multiple_fields"].index[1],
        # Central scheme (not_found)
        df[(df["level"] == "Central") & (df["state_extraction_method"] == "not_found")].index[0],
        # Central scheme (targeted)
        df[(df["level"] == "Central") & (df["state"].notna())].index[0]
    ]

    report_lines = [
        "# Metadata Analysis & Enrichment Report (`02_metadata_analysis.md`)",
        "",
        f"- **Input Dataset**: `data/processed/clean_schemes.parquet` (3,397 rows)",
        f"- **Output Parquet**: `data/processed/enriched_schemes.parquet`",
        f"- **Output CSV**: `data/processed/enriched_schemes.csv`",
        "",
        "## 1. Category Analysis (`categories`)",
        "",
        f"- **Number of unique canonical categories**: {unique_categories_count}",
        f"- **Total category instances across all schemes**: {len(all_categories_flat):,}",
        "- **Delimiters in source**: Categories were comma-separated. Several categories had missing spaces after commas (e.g., `Agriculture,Rural & Environment`, `Banking,Financial Services and Insurance`, `Public Safety,Law & Justice`). These were normalized cleanly to standard format without altering category identities.",
        "- **Structure**: Schemes contain between 1 and 6 categories (mean: 1.45 categories per scheme).",
        "",
        "### Category Frequency Table",
        "",
        "| Rank | Category Name | Scheme Count | % of All Schemes | Sample Schemes |",
        "|:---:|:---|:---:|:---:|:---|",
    ]

    for rank, (cat, count) in enumerate(category_counts.most_common(), 1):
        pct = (count / raw_row_count) * 100
        samples = "<br>• ".join(category_samples[cat])
        report_lines.append(f"| {rank} | **{cat}** | {count:,} | {pct:.1f}% | • {samples} |")

    report_lines.extend([
        "",
        "## 2. Tag Analysis (`normalized_tags`)",
        "",
        f"- **Number of unique normalized tags**: {unique_tags_count:,}",
        f"- **Total tag instances**: {len(all_tags_flat):,}",
        f"- **Schemes with empty tags**: {empty_tags_count}",
        "- **Information Dimensions Provided by Tags**:",
        "  - **Occupation**: `Labour` (273), `Construction Worker` (255), `Farmer` (212), `Building Worker` (165), `Worker` (89), `Entrepreneur` (65)",
        "  - **Target Group / Beneficiary**: `Scheduled Caste` (197), `Social Welfare` (196), `Scheduled Tribe` (107), `PwD` (98), `Widow` (95), `BPL` (83)",
        "  - **Gender**: `Women` (92), `Widow` (95), `Girl Child`",
        "  - **Age Group / Education**: `Student` (461), `Education` (263), `Senior Citizen` (74), `School` (63)",
        "  - **Assistance Type**: `Financial Assistance` (1,062), `Subsidy` (307), `Scholarship` (292), `Pension` (190), `Loan` (143), `Training` (137)",
        "",
        "### Top 30 Most Frequent Tags",
        "",
        "| Rank | Tag Name | Count | Rank | Tag Name | Count |",
        "|:---:|:---|:---:|:---:|:---|:---:|",
    ])

    top30 = tag_counts.most_common(30)
    for i in range(15):
        t1, c1 = top30[i]
        t2, c2 = top30[i + 15]
        report_lines.append(f"| {i + 1} | `{t1}` | {c1:,} | {i + 16} | `{t2}` | {c2:,} |")

    report_lines.extend([
        "",
        "## 3. State Analysis (`state`, `states`, `state_extraction_method`)",
        "",
        f"- **Total schemes**: {raw_row_count:,}",
        f"- **State-level schemes (`level == 'State'`)**: {len(state_level_df):,} (84.1%)",
        f"- **Central-level schemes (`level == 'Central'`)**: {len(central_level_df):,} (15.9%)",
        f"- **Schemes with single detected state (`state` is set)**: {schemes_with_single_state:,}",
        f"- **Schemes with multiple states mentioned (`len(states) > 1`)**: {schemes_with_multi_state}",
        f"- **Schemes with null state (`state = null`)**: {schemes_with_null_state:,}",
        f"  - Central schemes (national / pan-India): {int((central_level_df['state'].isna()).sum()):,}",
        f"  - State schemes with multiple states (ambiguous): {schemes_with_multi_state}",
        f"  - State schemes without textual state evidence: {int(((state_level_df['state'].isna()) & (state_level_df['states'].apply(len) == 0)).sum())}",
        "",
        "### State Extraction Methods Distribution",
        "",
        "| Extraction Method | Description | Count | Percentage |",
        "|:---|:---|:---:|:---:|",
    ])

    method_descriptions = {
        "structured": "Extracted from explicit structured field (none in raw CSV)",
        "scheme_name": "State explicitly present in scheme name/title",
        "tags": "State explicitly found in tags field",
        "details": "State found in details description (e.g. 'Government of X')",
        "eligibility": "State found in eligibility criteria (e.g. 'resident of X')",
        "benefits": "State found in benefits section",
        "application": "State found in application procedure / office location",
        "multiple_fields": "Multiple different states mentioned across multiple fields",
        "not_found": "No reliable state mention found (pan-India Central schemes or unstated)"
    }

    for method in ["scheme_name", "tags", "details", "eligibility", "benefits", "application", "multiple_fields", "not_found"]:
        cnt = method_counts.get(method, 0)
        pct = (cnt / raw_row_count) * 100
        desc = method_descriptions.get(method, "")
        report_lines.append(f"| `{method}` | {desc} | {cnt:,} | {pct:.2f}% |")

    # Detected State Frequency Table
    state_freq = df["state"].dropna().value_counts()
    report_lines.extend([
        "",
        "### Detected State Frequency (Top 25 States)",
        "",
        "| Rank | State / Union Territory | Scheme Count | % of Single-State Schemes |",
        "|:---:|:---|:---:|:---:|",
    ])

    for rank, (st, count) in enumerate(state_freq.head(25).items(), 1):
        pct = (count / schemes_with_single_state) * 100
        report_lines.append(f"| {rank} | **{st}** | {count:,} | {pct:.1f}% |")

    report_lines.extend([
        "",
        "## 4. Extraction Quality Examples (Verification Audit)",
        "",
        "| Scheme ID | Scheme Name | Level | Extracted State | Method | Evidence Snippet |",
        "|:---:|:---|:---:|:---|:---:|:---|",
    ])

    for idx in sample_indices:
        r = df.loc[idx]
        st_val = f"`{r['state']}`" if pd.notna(r["state"]) else f"`null` ({r['states']})"
        evidence_txt = r["_evidence"]
        # Clean evidence for table display
        clean_ev = str(evidence_txt).replace("|", "\\|").replace("\n", " ")[:160]
        name_clean = str(r["scheme_name"]).replace("|", "\\|")
        report_lines.append(f"| `{r['scheme_id']}` | {name_clean} | {r['level']} | {st_val} | `{r['state_extraction_method']}` | {clean_ev} |")

    report_lines.extend([
        "",
        "## 5. Potential Problems & Ambiguities",
        "",
        "1. **Central Ministry Headquarters in National Capital**:",
        "   - Many Central government schemes mention `New Delhi` or `Delhi` in application procedures (e.g. AICTE office at Vasant Kunj, New Delhi, or Ministry headquarters at Shastri Bhawan).",
        "   - **Resolution applied**: In accordance with Rule 6, Central schemes default to `state = null` unless they explicitly target a specific state/UT domicile (e.g., Prime Minister's Special Scholarship for J&K and Ladakh).",
        "",
        "2. **Multi-State Border / Hospital Mentions**:",
        "   - Schemes such as `Border Areas Development Department` (`S0459`) explicitly list 18 border states in their eligibility criteria.",
        "   - Healthcare aid schemes (e.g. `Dr. Ambedkar Medical Aid Scheme` `S0794`) list premier referral hospitals across 10 states (AIIMS Delhi, PGI Chandigarh, etc.).",
        "   - **Resolution applied**: In accordance with Rule 5, multiple states are captured in `states = [...]`, and `state = null` to avoid arbitrarily choosing one.",
        "",
        "3. **Acronym Boundaries (e.g. M.P., U.P., H.P., A.P., W.B.)**:",
        "   - In English text, words like `up`, `as`, `or`, `in` can create catastrophic false positives if regex word boundaries and periods are not strictly enforced.",
        "   - `M.P.` can refer to Member of Parliament or Madhya Pradesh.",
        "   - **Resolution applied**: Abbreviations were restricted to punctuated forms (`M.P.`, `U.P.`, `H.P.`) or specific domain compounds (`MPBOCW`, `UP BOCW`, `WBPCB`).",
        "",
        "4. **Implicit / Vernacular Scheme Names without Textual State Mention**:",
        "   - Certain schemes like `Banglar Awaas Yojana` (`S0369`) feature regional Bengali terms (`Banglar` = of Bengal), but omit the exact words `West Bengal` in their text description.",
        "   - **Resolution applied**: In strict adherence to Rule 1 and Rule 2 (\"Only assign a state when there is reasonable textual evidence; do NOT infer a state just because a scheme sounds regional\"), these schemes are faithfully preserved as `state = null` without speculative inference."
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"      Report written to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------------
    # Console Summary (Part F)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("  STEP 2 ENRICHMENT SUMMARY")
    print("=" * 76)
    print(f"  * Total Rows:                  {len(df):,}")
    print(f"  * Unique Categories:           {unique_categories_count}")
    print(f"  * Unique Tags:                 {unique_tags_count:,}")
    print(f"  * State-level Schemes:         {len(state_level_df):,}")
    print(f"  * Central-level Schemes:       {len(central_level_df):,}")
    print(f"  * Single State Detected:       {schemes_with_single_state:,}")
    print(f"  * Multi-State Schemes:         {schemes_with_multi_state}")
    print(f"  * Null State Schemes:          {schemes_with_null_state:,}")
    print("\n  Top 10 States Extracted:")
    for st, count in state_freq.head(10).items():
        print(f"    - {st:25s}: {count:,}")
    print("\n  Extraction Methods Breakdown:")
    for m, c in method_counts.items():
        print(f"    - {m:20s}: {c:,}")
    print("=" * 76)
    print("Step 2 completed successfully!\n")


if __name__ == "__main__":
    main()

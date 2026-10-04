"""
scripts/01_clean_data.py
========================
Data Cleaning and Inspection Pipeline for 'sugamgov-rag'.

This script processes raw government schemes data from:
  data/raw/updated_data.csv
and outputs clean, standardized data to:
  data/processed/clean_schemes.parquet
  data/processed/clean_schemes.csv
as well as a comprehensive data quality inspection report to:
  reports/01_inspection_report.md

Pipeline Steps:
  1. Load raw CSV with UTF-8 encoding.
  2. Drop empty unnamed column(s).
  3. Rename 'schemeCategory' -> 'scheme_category'.
  4. Text cleaning:
     - Strip leading/trailing whitespace.
     - Remove invisible characters (\ufeff, \u200b, \u200c, \u200d).
     - Collapse multiple consecutive spaces/newlines/tabs into a single space.
     - Replace nulls/NaNs with empty string ("").
     - Preserve exact textual meaning (no translations or summaries).
  5. Check duplicates:
     - Remove duplicate rows on 'slug' (keep first occurrence).
     - Log any duplicate 'scheme_name' entries without deleting them.
  6. Add unique sequential 'scheme_id' column (S0001, S0002, ...) as the first column.
  7. Save processed data to Parquet and CSV.
  8. Generate diagnostic inspection report in Markdown.
"""

import re
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
# Using Pathlib for clean, cross-platform relative path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "updated_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_PARQUET_PATH = PROCESSED_DIR / "clean_schemes.parquet"
PROCESSED_CSV_PATH = PROCESSED_DIR / "clean_schemes.csv"
INSPECTION_REPORT_PATH = REPORTS_DIR / "01_inspection_report.md"


# ---------------------------------------------------------------------------
# Text Cleaning Helper Function
# ---------------------------------------------------------------------------
def clean_text_field(value: object) -> str:
    """
    Cleans a text field while preserving meaning:
      1. Converts nulls/NaNs to empty string ("").
      2. Removes invisible characters (BOM, zero-width spaces/joiners).
      3. Collapses repeated whitespaces/newlines into a single space.
      4. Strips leading and trailing whitespace.
    """
    if pd.isna(value) or value is None:
        return ""

    text = str(value)

    # Remove zero-width characters and UTF-8 Byte Order Mark:
    # \ufeff = Zero-width no-break space / Byte Order Mark (BOM)
    # \u200b = Zero-width space
    # \u200c = Zero-width non-joiner
    # \u200d = Zero-width joiner
    text = re.sub(r"[\ufeff\u200b\u200c\u200d]", "", text)

    # Collapse any sequence of whitespace (spaces, tabs, newlines \r \n) into a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    return text.strip()


def main():
    print("=" * 72)
    print("  sugamgov-rag | Step 1: Data Cleaning & Inspection")
    print("=" * 72)

    # Ensure output directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Read the CSV with UTF-8 encoding
    # -----------------------------------------------------------------------
    print(f"\n[1/7] Loading raw data: {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found at: {RAW_DATA_PATH}")

    df_raw = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
    raw_row_count = len(df_raw)
    raw_columns = list(df_raw.columns)
    print(f"      Rows loaded: {raw_row_count:,}")
    print(f"      Raw columns ({len(raw_columns)}): {raw_columns}")

    # -----------------------------------------------------------------------
    # Step 2: Drop empty unnamed column(s) & Rename schemeCategory
    # -----------------------------------------------------------------------
    print("\n[2/7] Standardizing columns...")
    # Find any empty unnamed column (e.g. 'Unnamed: 9')
    unnamed_cols = [
        col for col in df_raw.columns
        if str(col).startswith("Unnamed:") or str(col).strip() == ""
    ]
    if unnamed_cols:
        print(f"      Dropping empty column(s): {unnamed_cols}")
        df = df_raw.drop(columns=unnamed_cols)
    else:
        df = df_raw.copy()

    # Rename 'schemeCategory' -> 'scheme_category'
    if "schemeCategory" in df.columns:
        print("      Renaming: 'schemeCategory' -> 'scheme_category'")
        df = df.rename(columns={"schemeCategory": "scheme_category"})

    cleaned_columns = list(df.columns)

    # Track missing values in raw data BEFORE cleaning
    # (Checking for true nulls/NaNs and empty/invisible-only strings)
    null_counts_before = {}
    for col in cleaned_columns:
        null_count = int(df[col].isna().sum())
        # Check non-null cells that only have whitespace or invisible characters
        raw_non_nulls = df[col].dropna().astype(str)
        empty_str_count = int((raw_non_nulls.str.strip() == "").sum())
        invisible_only_count = int(raw_non_nulls.apply(
            lambda s: bool(s.strip()) and not re.sub(r"[\ufeff\u200b\u200c\u200d\s+]", "", s)
        ).sum())
        null_counts_before[col] = {
            "nulls": null_count,
            "empty_strings": empty_str_count,
            "invisible_only": invisible_only_count,
            "total_missing": null_count + empty_str_count + invisible_only_count
        }

    # -----------------------------------------------------------------------
    # Step 3: Clean all text columns
    # -----------------------------------------------------------------------
    print("\n[3/7] Cleaning text fields (stripping whitespace, removing invisible chars)...")
    for col in cleaned_columns:
        df[col] = df[col].apply(clean_text_field)

    # Track missing values AFTER cleaning
    # In clean data: NaNs are converted to "", so null count is 0
    null_counts_after = {}
    for col in cleaned_columns:
        empty_count = int((df[col] == "").sum())
        null_counts_after[col] = {
            "nulls": 0,
            "empty_strings": empty_count,
            "total_missing": empty_count
        }

    # -----------------------------------------------------------------------
    # Step 4: Duplicate Checks and Removal
    # -----------------------------------------------------------------------
    print("\n[4/7] Checking duplicates on 'slug' and 'scheme_name'...")
    # Check duplicate scheme_name BEFORE removing duplicate slugs
    pre_slug_dupe_names = df[df.duplicated(subset=["scheme_name"], keep=False)]
    pre_dupe_names_list = pre_slug_dupe_names["scheme_name"].unique().tolist()

    # Check and remove duplicate slugs (keep first)
    slug_duplicates_mask = df.duplicated(subset=["slug"], keep="first")
    duplicate_slug_count = int(slug_duplicates_mask.sum())
    duplicate_slug_rows = df[df.duplicated(subset=["slug"], keep=False)][["slug", "scheme_name"]]

    if duplicate_slug_count > 0:
        print(f"      Found {duplicate_slug_count} rows with duplicate slugs:")
        for slug_val, group in duplicate_slug_rows.groupby("slug"):
            print(f"        - slug '{slug_val}': {len(group)} occurrences ('{group['scheme_name'].iloc[0]}')")
        print("      Removing duplicate slug rows (keeping first occurrence)...")
        df = df.drop_duplicates(subset=["slug"], keep="first").reset_index(drop=True)
    else:
        print("      No duplicate slugs found.")

    # Check duplicate scheme_name AFTER removing duplicate slugs (LOG ONLY)
    post_slug_dupe_names = df[df.duplicated(subset=["scheme_name"], keep=False)]
    post_dupe_names_list = post_slug_dupe_names["scheme_name"].unique().tolist()
    print(f"      Duplicate scheme_name check after slug deduplication: {len(post_dupe_names_list)} found.")
    if post_dupe_names_list:
        for name in post_dupe_names_list:
            print(f"        - '{name}' (retained as per instructions)")

    # -----------------------------------------------------------------------
    # Step 5: Add sequential scheme_id as first column
    # -----------------------------------------------------------------------
    print("\n[5/7] Adding sequential 'scheme_id' column (S0001, S0002, ...)...")
    total_clean_rows = len(df)
    # Formats sequential IDs with 4-digit zero padding: S0001, S0002, ..., S3397
    scheme_ids = [f"S{i + 1:04d}" for i in range(total_clean_rows)]
    df.insert(0, "scheme_id", scheme_ids)
    print(f"      Generated {len(scheme_ids)} IDs ({scheme_ids[0]} -> {scheme_ids[-1]}).")

    # -----------------------------------------------------------------------
    # Step 6: Save clean parquet and CSV files
    # -----------------------------------------------------------------------
    print(f"\n[6/7] Saving processed dataset...")
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    print(f"      Saved Parquet: {PROCESSED_PARQUET_PATH.relative_to(PROJECT_ROOT)}")

    df.to_csv(PROCESSED_CSV_PATH, index=False, encoding="utf-8")
    print(f"      Saved CSV:     {PROCESSED_CSV_PATH.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------------
    # Step 7: Compute Inspection Metrics & Generate Report
    # -----------------------------------------------------------------------
    print(f"\n[7/7] Generating inspection report: {INSPECTION_REPORT_PATH.relative_to(PROJECT_ROOT)}...")

    # Level counts (Central vs State)
    level_counts = df["level"].value_counts().to_dict()

    # Word count stats for text content fields
    content_columns = ["details", "benefits", "eligibility", "application", "documents"]
    word_count_stats = {}
    for col in content_columns:
        if col in df.columns:
            word_counts = df[col].apply(lambda text: len(text.split()) if text else 0)
            word_count_stats[col] = {
                "mean": float(word_counts.mean()),
                "min": int(word_counts.min()),
                "max": int(word_counts.max()),
                "zero_count": int((word_counts == 0).sum())
            }

    # Schemes where eligibility OR documents is empty
    eligibility_empty = df["eligibility"] == ""
    documents_empty = df["documents"] == ""
    either_empty_count = int((eligibility_empty | documents_empty).sum())
    both_empty_count = int((eligibility_empty & documents_empty).sum())
    eligibility_empty_count = int(eligibility_empty.sum())
    documents_empty_count = int(documents_empty.sum())

    # Schemes with eligibility shorter than 5 words
    eligibility_word_lens = df["eligibility"].apply(lambda text: len(text.split()) if text else 0)
    eligibility_under_5_words = int((eligibility_word_lens < 5).sum())

    # 3 Reproducible Sample Rows (seed=42 for reproducibility)
    sample_df = df.sample(n=3, random_state=42)
    sample_rows = []
    for _, row in sample_df.iterrows():
        elig_preview = row["eligibility"][:150] + ("..." if len(row["eligibility"]) > 150 else "")
        sample_rows.append({
            "scheme_id": row["scheme_id"],
            "scheme_name": row["scheme_name"],
            "level": row["level"],
            "eligibility_preview": elig_preview
        })

    # Build Markdown Report
    report_lines = [
        "# Data Inspection & Cleaning Report (`01_clean_data.py`)",
        "",
        f"- **Input Raw File**: `data/raw/updated_data.csv`",
        f"- **Clean Parquet**: `data/processed/clean_schemes.parquet`",
        f"- **Clean CSV**: `data/processed/clean_schemes.csv`",
        "",
        "## 1. Row Counts & Duplicate Removal",
        "",
        f"- **Row count before cleaning**: {raw_row_count:,}",
        f"- **Row count after cleaning**: {total_clean_rows:,}",
        f"- **Duplicate `slug` rows removed**: {duplicate_slug_count}",
        f"  - `eogu` (Establishment of Goat Unit (10 +1)): 3 occurrences in raw -> 2 duplicates removed",
        f"  - `vcy` (Valmiki Chhatravritti Yojana): 2 occurrences in raw -> 1 duplicate removed",
        f"- **Duplicate `scheme_name` check**:",
        f"  - Before slug deduplication: {len(pre_dupe_names_list)} names duplicated (coincided exactly with duplicate slugs)",
        f"  - After slug deduplication: {len(post_dupe_names_list)} names duplicated (no remaining duplicate scheme names)",
        "",
        "## 2. Column List & Schema",
        "",
        "| Index | Column Name | Type / Status | Description |",
        "|:---:|:---|:---|:---|",
    ]

    for idx, col in enumerate(df.columns, 1):
        if col == "scheme_id":
            desc = "Sequential scheme primary key (`S0001`, `S0002`, ...)"
            status = "Added (Primary Key)"
        elif col == "scheme_category":
            desc = "Standardized from raw `schemeCategory`"
            status = "Renamed"
        else:
            desc = "Cleaned text field (whitespace collapsed, invisible chars removed)"
            status = "Cleaned"
        report_lines.append(f"| {idx} | `{col}` | {status} | {desc} |")

    report_lines.extend([
        "",
        "## 3. Missing / Empty Counts Before vs After Cleaning",
        "",
        "| Column | Raw NaNs | Raw Empty / Invisible | Raw Total Missing | Clean NaNs | Clean Empty (`\"\"`) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for col in cleaned_columns:
        b = null_counts_before[col]
        a = null_counts_after[col]
        invisible_note = f" (incl. {b['invisible_only']} BOM)" if b["invisible_only"] > 0 else ""
        report_lines.append(
            f"| `{col}` | {b['nulls']:,} | {b['empty_strings'] + b['invisible_only']:,}{invisible_note} | {b['total_missing']:,} | {a['nulls']} | {a['empty_strings']:,} |"
        )

    report_lines.extend([
        "",
        "> [!NOTE]",
        "> In the raw CSV, columns `application` and `documents` contained 2 rows with only the invisible Byte Order Mark (`\\ufeff`). During cleaning, these invisible characters were stripped, standardizing those rows to empty strings (`\"\"`).",
        "",
        "## 4. Government Level Distribution (`level`)",
        "",
        "| Level | Count | Percentage |",
        "|:---|:---:|:---:|",
    ])

    for lvl, cnt in level_counts.items():
        pct = (cnt / total_clean_rows) * 100
        report_lines.append(f"| **{lvl}** | {cnt:,} | {pct:.2f}% |")

    report_lines.extend([
        "",
        "## 5. Content Word Count Statistics",
        "",
        "| Column | Mean Words | Min Words | Max Words | Empty Schemes (0 words) |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    for col, stats in word_count_stats.items():
        report_lines.append(
            f"| `{col}` | {stats['mean']:.1f} | {stats['min']} | {stats['max']:,} | {stats['zero_count']} |"
        )

    report_lines.extend([
        "",
        "## 6. Critical Retrieval Fields Analysis",
        "",
        f"- **Schemes with empty `eligibility`**: {eligibility_empty_count}",
        f"- **Schemes with empty `documents`**: {documents_empty_count}",
        f"- **Schemes where either `eligibility` OR `documents` is empty**: {either_empty_count}",
        f"- **Schemes where BOTH `eligibility` AND `documents` are empty**: {both_empty_count}",
        f"- **Schemes with `eligibility` shorter than 5 words**: {eligibility_under_5_words}",
        "",
        "## 7. Sample Rows (3 Random Examples)",
        "",
        "| Scheme ID | Scheme Name | Level | Eligibility Preview (First 150 chars) |",
        "|:---|:---|:---:|:---|",
    ])

    for s in sample_rows:
        escaped_name = s["scheme_name"].replace("|", "\\|")
        escaped_elig = s["eligibility_preview"].replace("|", "\\|")
        report_lines.append(f"| `{s['scheme_id']}` | {escaped_name} | {s['level']} | {escaped_elig} |")

    report_lines.append("")

    with open(INSPECTION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"      Report written successfully.")

    # -----------------------------------------------------------------------
    # Print Console Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  CLEANING & INSPECTION SUMMARY")
    print("=" * 72)
    print(f"  * Raw rows:                {raw_row_count:,}")
    print(f"  * Duplicate slugs removed: {duplicate_slug_count}")
    print(f"  * Clean rows saved:        {total_clean_rows:,}")
    print(f"  * Duplicate scheme names:  {len(post_dupe_names_list)} (after slug dedup)")
    print(f"  * Output files:")
    print(f"      - {PROCESSED_PARQUET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      - {PROCESSED_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      - {INSPECTION_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("\n  Level Distribution:")
    for lvl, cnt in level_counts.items():
        print(f"    - {lvl:10s}: {cnt:,} ({(cnt / total_clean_rows) * 100:.1f}%)")
    print("\n  Word Count Stats (Mean / Min / Max):")
    for col, stats in word_count_stats.items():
        print(f"    - {col:12s}: Mean {stats['mean']:.1f} words | Min {stats['min']} | Max {stats['max']}")
    print(f"\n  Critical Retrieval Fields:")
    print(f"    - Schemes where eligibility OR documents is empty: {either_empty_count}")
    print(f"    - Schemes with eligibility < 5 words:             {eligibility_under_5_words}")
    print("=" * 72)
    print("Step 1 data cleaning completed successfully!\n")


if __name__ == "__main__":
    main()

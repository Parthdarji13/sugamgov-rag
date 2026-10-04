# sugamgov-rag

A Python RAG (Retrieval-Augmented Generation) backend for an Indian government schemes assistant.

## Project Structure

```text
sugamgov-rag/
├── data/
│   ├── raw/                 # Original data (e.g., updated_data.csv)
│   └── processed/           # Cleaned data (clean_schemes.parquet, clean_schemes.csv)
├── scripts/                 # Data processing and pipeline scripts
│   └── 01_clean_data.py     # Initial cleaning, standardization, and inspection
├── reports/                 # Pipeline inspection and diagnostic reports
│   └── 01_inspection_report.md
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation
```

## Setup & Getting Started

1. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Data Cleaning**:
   ```bash
   python scripts/01_clean_data.py
   ```

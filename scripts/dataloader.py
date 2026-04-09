"""
dataloader.py — Dataset Loading and Validation Module
=====================================================

Shark Tank India Decision Modeling Project

Responsibilities:
    1. Download the dataset from Kaggle using the Kaggle API
    2. Validate schema: check expected columns, data types, and shape
    3. Report missing values, duplicates, and basic consistency issues
    4. Return a clean, validated DataFrame for downstream preprocessing

Prerequisites:
    - Install the kaggle package:  pip install kaggle
    - Set up Kaggle API credentials:
        • Option A: Place kaggle.json in ~/.kaggle/kaggle.json (Linux/Mac)
                     or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)
        • Option B: Set environment variables KAGGLE_USERNAME and KAGGLE_KEY

Usage:
    from dataloader import load_dataset

    # Download from Kaggle and load (default dataset)
    df = load_dataset()

    # Specify a different Kaggle dataset slug
    df = load_dataset(kaggle_dataset="thirumani/shark-tank-india")

    # Load from a local CSV file (legacy fallback)
    df = load_dataset(path="data/Shark Tank India Dataset.csv")
"""

import os
import sys
import logging
import zipfile
import glob
import pandas as pd
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "dataloader.log"), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────
DEFAULT_KAGGLE_DATASET = "thirumani/shark-tank-india"
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# ─────────────────────────────────────────────────────────────────
# Expected Schema Definition
# ─────────────────────────────────────────────────────────────────
# Complete column list from the Kaggle Shark Tank India dataset
# (Satya Thirumani, 80 columns, Seasons 1–5)
# Each entry: column_name → expected pandas dtype category

EXPECTED_SCHEMA = {
    # ── Metadata ──
    "Season Number":                  "numeric",
    "Startup Name":                   "string",
    "Episode Number":                 "numeric",
    "Pitch Number":                   "numeric",
    "Season Start":                   "date",
    "Season End":                     "date",
    "Original Air Date":              "date",
    "Episode Title":                  "string",
    "Anchor":                         "string",

    # ── Business Information ──
    "Industry":                       "string",
    "Business Description":           "string",
    "Company Website":                "string",
    "Started in":                     "numeric",
    "Number of Presenters":           "numeric",
    "Male Presenters":                "numeric",
    "Female Presenters":              "numeric",
    "Transgender Presenters":         "numeric",
    "Couple Presenters":              "numeric",
    "Pitchers Average Age":           "string",
    "Pitchers City":                  "string",
    "Pitchers State":                 "string",

    # ── Financial Information ──
    "Yearly Revenue":                 "numeric",
    "Monthly Sales":                  "numeric",
    "Gross Margin":                   "numeric",
    "Net Margin":                     "numeric",
    "EBITDA":                         "numeric",
    "Cash Burn":                      "string",
    "SKUs":                           "numeric",
    "Has Patents":                    "string",
    "Bootstrapped":                   "string",
    "Part of Match off":              "string",

    # ── Ask / Deal Details ──
    "Original Ask Amount":            "numeric",
    "Original Offered Equity":        "numeric",
    "Valuation Requested":            "numeric",
    "Received Offer":                 "numeric",
    "Accepted Offer":                 "numeric",
    "Total Deal Amount":              "numeric",
    "Total Deal Equity":              "numeric",
    "Total Deal Debt":                "numeric",
    "Debt Interest":                  "numeric",
    "Deal Valuation":                 "numeric",
    "Number of sharks in deal":       "numeric",
    "Deal has conditions":            "string",
    "Royalty Percentage":             "numeric",
    "Royalty Recouped Amount":        "numeric",
    "Advisory Shares Equity":         "numeric",

    # ── Shark-wise Investment: Namita ──
    "Namita Investment Amount":       "numeric",
    "Namita Investment Equity":       "numeric",
    "Namita Debt Amount":             "numeric",

    # ── Shark-wise Investment: Vineeta ──
    "Vineeta Investment Amount":      "numeric",
    "Vineeta Investment Equity":      "numeric",
    "Vineeta Debt Amount":            "numeric",

    # ── Shark-wise Investment: Anupam ──
    "Anupam Investment Amount":       "numeric",
    "Anupam Investment Equity":       "numeric",
    "Anupam Debt Amount":             "numeric",

    # ── Shark-wise Investment: Aman ──
    "Aman Investment Amount":         "numeric",
    "Aman Investment Equity":         "numeric",
    "Aman Debt Amount":               "numeric",

    # ── Shark-wise Investment: Peyush ──
    "Peyush Investment Amount":       "numeric",
    "Peyush Investment Equity":       "numeric",
    "Peyush Debt Amount":             "numeric",

    # ── Shark-wise Investment: Ritesh ──
    "Ritesh Investment Amount":       "numeric",
    "Ritesh Investment Equity":       "numeric",
    "Ritesh Debt Amount":             "numeric",

    # ── Shark-wise Investment: Amit ──
    "Amit Investment Amount":         "numeric",
    "Amit Investment Equity":         "numeric",
    "Amit Debt Amount":               "numeric",

    # ── Shark-wise Investment: Guest ──
    "Guest Investment Amount":        "numeric",
    "Guest Investment Equity":        "numeric",
    "Guest Debt Amount":              "numeric",
    "Invested Guest Name":            "string",
    "All Guest Names":                "string",

    # ── Shark Presence ──
    "Namita Present":                 "numeric",
    "Vineeta Present":                "numeric",
    "Anupam Present":                 "numeric",
    "Aman Present":                   "numeric",
    "Peyush Present":                 "numeric",
    "Ritesh Present":                 "numeric",
    "Amit Present":                   "numeric",
}

# Columns critical to the three prediction tasks
CRITICAL_COLUMNS = [
    "Total Deal Amount",       # Target for funding amount prediction (regression)
    "Received Offer",          # Target for deal/no-deal prediction (classification)
    "Accepted Offer",          # Also relevant for deal prediction
    "Original Ask Amount",     # Key feature
    "Original Offered Equity", # Key feature
    "Valuation Requested",     # Key feature
    "Industry",                # Key feature
]

# Shark names for multi-label prediction task
SHARKS = ["Namita", "Vineeta", "Anupam", "Aman", "Peyush", "Ritesh", "Amit"]


# ─────────────────────────────────────────────────────────────────
# Kaggle API Download
# ─────────────────────────────────────────────────────────────────

def _download_from_kaggle(
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    download_dir: str = DEFAULT_DATA_DIR,
    force: bool = False,
) -> str:
    """
    Download a dataset from Kaggle using the Kaggle API.

    Parameters
    ----------
    dataset_slug : str
        The Kaggle dataset identifier in 'owner/dataset-name' format.
        Default: 'thirumani/shark-tank-india'
    download_dir : str
        Local directory to download and extract files into.
        Default: '<project_root>/data'
    force : bool
        If True, re-download even if files already exist locally.

    Returns
    -------
    str
        Path to the extracted CSV file.

    Raises
    ------
    ImportError
        If the kaggle package is not installed.
    RuntimeError
        If Kaggle API authentication fails or download errors occur.
    FileNotFoundError
        If no CSV file is found after extraction.
    """
    # Check if data already exists locally (skip download if not forced)
    if not force:
        existing_csvs = glob.glob(os.path.join(download_dir, "*.csv"))
        if existing_csvs:
            logger.info(
                f"Dataset already exists locally ({len(existing_csvs)} CSV file(s) found). "
                f"Use force=True to re-download."
            )
            if len(existing_csvs) == 1:
                return existing_csvs[0]
            else:
                # Return the largest CSV (most likely the main dataset)
                largest = max(existing_csvs, key=os.path.getsize)
                logger.info(f"Multiple CSVs found — using largest: {os.path.basename(largest)}")
                return largest

    # ── Import and authenticate Kaggle API ──
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required.\n"
            "Install it with:  pip install kaggle\n\n"
            "You also need Kaggle API credentials:\n"
            "  1. Go to https://www.kaggle.com/settings → 'Create New Token'\n"
            "  2. Save the downloaded kaggle.json to:\n"
            "     • Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json\n"
            "     • Linux/Mac: ~/.kaggle/kaggle.json\n"
            "  OR set environment variables: KAGGLE_USERNAME and KAGGLE_KEY"
        )

    logger.info("Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API authentication successful ✓")
    except Exception as e:
        raise RuntimeError(
            f"Kaggle API authentication failed: {e}\n\n"
            "Please ensure your credentials are set up correctly:\n"
            "  • kaggle.json in ~/.kaggle/ (or C:\\Users\\<user>\\.kaggle\\ on Windows)\n"
            "  • OR environment variables KAGGLE_USERNAME and KAGGLE_KEY"
        )

    # ── Download the dataset ──
    os.makedirs(download_dir, exist_ok=True)
    logger.info(f"Downloading dataset '{dataset_slug}' to '{download_dir}'...")

    try:
        api.dataset_download_files(
            dataset_slug,
            path=download_dir,
            unzip=True,
        )
        logger.info(f"Dataset downloaded and extracted successfully ✓")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset '{dataset_slug}': {e}\n"
            "Please check:\n"
            "  • Dataset slug is correct (format: 'owner/dataset-name')\n"
            "  • You have internet connectivity\n"
            "  • You have accepted the dataset's terms on Kaggle (if required)"
        )

    # ── Locate the extracted CSV ──
    csv_files = glob.glob(os.path.join(download_dir, "*.csv"))

    if not csv_files:
        # Check for nested directories (some datasets extract into subdirs)
        csv_files = glob.glob(os.path.join(download_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found after extracting dataset '{dataset_slug}' "
            f"into '{download_dir}'. Check the dataset contents on Kaggle."
        )

    if len(csv_files) == 1:
        filepath = csv_files[0]
    else:
        # Multiple CSVs — pick the largest one (likely the main dataset)
        filepath = max(csv_files, key=os.path.getsize)
        logger.info(
            f"Multiple CSV files found after extraction: "
            f"{[os.path.basename(f) for f in csv_files]}"
        )
        logger.info(f"Using largest CSV: {os.path.basename(filepath)}")

    logger.info(f"Dataset file: {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────────
# CSV Reading (Local File)
# ─────────────────────────────────────────────────────────────────

def _find_csv(path: str) -> str:
    """
    Resolve the CSV file path. If `path` is a directory,
    look for a CSV file inside it (expects exactly one).
    """
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if len(csv_files) == 1:
            resolved = os.path.join(path, csv_files[0])
            logger.info(f"Auto-detected CSV file: {resolved}")
            return resolved
        elif len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        else:
            raise ValueError(
                f"Multiple CSV files found in directory: {path}\n"
                f"  Files: {csv_files}\n"
                f"  Please specify the exact file path."
            )

    raise FileNotFoundError(f"Path does not exist: {path}")


def _read_csv(filepath: str) -> pd.DataFrame:
    """
    Read the CSV file into a pandas DataFrame with basic error handling.
    """
    logger.info(f"Reading CSV file: {filepath}")
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, trying latin-1 encoding...")
        df = pd.read_csv(filepath, encoding="latin-1")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise

    logger.info(f"Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate the DataFrame's schema against the expected Kaggle dataset schema.

    Returns a validation report dictionary.
    """
    report = {
        "schema_valid": True,
        "missing_columns": [],
        "extra_columns": [],
        "column_count_match": False,
        "dtype_issues": [],
    }

    expected_cols = set(EXPECTED_SCHEMA.keys())
    actual_cols = set(df.columns.tolist())

    # ── Missing columns ──
    missing = expected_cols - actual_cols
    if missing:
        report["missing_columns"] = sorted(missing)
        report["schema_valid"] = False
        logger.warning(f"Missing {len(missing)} expected column(s): {sorted(missing)}")

    # ── Extra / unexpected columns ──
    extra = actual_cols - expected_cols
    if extra:
        report["extra_columns"] = sorted(extra)
        logger.info(f"Found {len(extra)} extra column(s) not in schema: {sorted(extra)}")

    # ── Column count ──
    report["column_count_match"] = (len(actual_cols) == len(expected_cols))
    if report["column_count_match"]:
        logger.info(f"Column count matches expected: {len(expected_cols)}")
    else:
        logger.warning(
            f"Column count mismatch — expected {len(expected_cols)}, got {len(actual_cols)}"
        )

    # ── Data type checks (only for columns present in both) ──
    common_cols = expected_cols & actual_cols
    for col in sorted(common_cols):
        expected_type = EXPECTED_SCHEMA[col]
        actual_dtype = str(df[col].dtype)

        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try coercing to check if it's convertible
                try:
                    pd.to_numeric(df[col], errors="raise")
                except (ValueError, TypeError):
                    non_numeric_count = pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
                    if non_numeric_count > 0:
                        report["dtype_issues"].append({
                            "column": col,
                            "expected": "numeric",
                            "actual": actual_dtype,
                            "non_numeric_values": int(non_numeric_count),
                        })
                        logger.warning(
                            f"Column '{col}': expected numeric, got {actual_dtype} "
                            f"({non_numeric_count} non-numeric values)"
                        )

        elif expected_type == "date":
            # Date columns are usually loaded as object/string — that's expected
            pass

    if report["dtype_issues"]:
        report["schema_valid"] = False

    return report


def _check_critical_columns(df: pd.DataFrame) -> bool:
    """
    Verify that all columns critical for the 3 prediction tasks are present.
    """
    missing_critical = [col for col in CRITICAL_COLUMNS if col not in df.columns]

    if missing_critical:
        logger.error(
            f"CRITICAL: Missing columns required for prediction tasks: {missing_critical}"
        )
        return False

    # Check shark investment columns for multi-label prediction
    missing_shark_cols = []
    for shark in SHARKS:
        inv_col = f"{shark} Investment Amount"
        if inv_col not in df.columns:
            missing_shark_cols.append(inv_col)

    if missing_shark_cols:
        logger.error(
            f"CRITICAL: Missing shark investment columns for participation prediction: "
            f"{missing_shark_cols}"
        )
        return False

    logger.info("All critical columns for prediction tasks are present ✓")
    return True


def _analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and report missing values across the dataset.

    Returns a DataFrame summarizing missing values per column.
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_pct.round(2),
        "dtype": df.dtypes,
    })

    # Filter to only columns with missing values, sorted descending
    missing_report = missing_report[missing_report["missing_count"] > 0]
    missing_report = missing_report.sort_values("missing_percentage", ascending=False)

    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()

    logger.info(f"Missing value analysis:")
    logger.info(f"  Total cells:   {total_cells:,}")
    logger.info(f"  Missing cells: {total_missing:,} ({(total_missing/total_cells)*100:.2f}%)")
    logger.info(f"  Columns with missing values: {len(missing_report)} / {df.shape[1]}")

    if not missing_report.empty:
        logger.info(f"\n  Top columns with missing values:")
        for col, row in missing_report.head(10).iterrows():
            logger.info(
                f"    • {col:<40s} → {int(row['missing_count']):>4d} missing "
                f"({row['missing_percentage']:.1f}%)"
            )

    return missing_report


def _check_duplicates(df: pd.DataFrame) -> dict:
    """
    Check for duplicate rows and duplicate pitch entries.
    """
    report = {
        "exact_duplicate_rows": 0,
        "duplicate_pitch_numbers": 0,
        "duplicate_startup_names": 0,
    }

    # Exact duplicate rows
    exact_dupes = df.duplicated().sum()
    report["exact_duplicate_rows"] = int(exact_dupes)
    if exact_dupes > 0:
        logger.warning(f"Found {exact_dupes} exact duplicate row(s)")
    else:
        logger.info("No exact duplicate rows found ✓")

    # Duplicate pitch numbers (should be unique per season)
    if "Pitch Number" in df.columns and "Season Number" in df.columns:
        pitch_dupes = df.duplicated(subset=["Season Number", "Pitch Number"]).sum()
        report["duplicate_pitch_numbers"] = int(pitch_dupes)
        if pitch_dupes > 0:
            logger.warning(
                f"Found {pitch_dupes} duplicate (Season Number, Pitch Number) combination(s)"
            )
        else:
            logger.info("No duplicate pitch numbers within seasons ✓")

    return report


def _check_consistency(df: pd.DataFrame) -> list:
    """
    Perform logical consistency checks on the dataset.
    Returns a list of issue descriptions.
    """
    issues = []

    # 1. Received Offer vs Deal Amount consistency
    if "Received Offer" in df.columns and "Total Deal Amount" in df.columns:
        no_offer_but_deal = df[
            (df["Received Offer"] == 0) &
            (pd.to_numeric(df["Total Deal Amount"], errors="coerce") > 0)
        ]
        if len(no_offer_but_deal) > 0:
            msg = (
                f"Inconsistency: {len(no_offer_but_deal)} row(s) have 'Received Offer' = 0 "
                f"but 'Total Deal Amount' > 0"
            )
            issues.append(msg)
            logger.warning(msg)

    # 2. Accepted Offer cannot be 1 if Received Offer is 0
    if "Received Offer" in df.columns and "Accepted Offer" in df.columns:
        accepted_no_offer = df[
            (df["Received Offer"] == 0) &
            (df["Accepted Offer"] == 1)
        ]
        if len(accepted_no_offer) > 0:
            msg = (
                f"Inconsistency: {len(accepted_no_offer)} row(s) have 'Accepted Offer' = 1 "
                f"but 'Received Offer' = 0"
            )
            issues.append(msg)
            logger.warning(msg)

    # 3. Number of presenters = Male + Female + Transgender
    presenter_cols = ["Number of Presenters", "Male Presenters", "Female Presenters", "Transgender Presenters"]
    if all(col in df.columns for col in presenter_cols):
        numeric_df = df[presenter_cols].apply(pd.to_numeric, errors="coerce")
        calculated = numeric_df["Male Presenters"] + numeric_df["Female Presenters"] + numeric_df["Transgender Presenters"]
        mismatch = (numeric_df["Number of Presenters"] != calculated) & numeric_df["Number of Presenters"].notna()
        mismatch_count = mismatch.sum()
        if mismatch_count > 0:
            msg = (
                f"Inconsistency: {mismatch_count} row(s) where 'Number of Presenters' ≠ "
                f"Male + Female + Transgender presenters"
            )
            issues.append(msg)
            logger.warning(msg)

    # 4. Negative values in financial columns that should be non-negative
    non_negative_cols = [
        "Original Ask Amount", "Original Offered Equity", "Number of Presenters",
        "Total Deal Equity", "Number of sharks in deal",
    ]
    for col in non_negative_cols:
        if col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            neg_count = (numeric_vals < 0).sum()
            if neg_count > 0:
                msg = f"Found {neg_count} negative value(s) in '{col}'"
                issues.append(msg)
                logger.warning(msg)

    # 5. Season number range check
    if "Season Number" in df.columns:
        seasons = pd.to_numeric(df["Season Number"], errors="coerce").dropna()
        if seasons.min() < 1 or seasons.max() > 10:
            msg = f"Season numbers out of expected range: min={seasons.min()}, max={seasons.max()}"
            issues.append(msg)
            logger.warning(msg)
        else:
            logger.info(f"Seasons found: {sorted(seasons.unique().astype(int).tolist())} ✓")

    if not issues:
        logger.info("All consistency checks passed ✓")

    return issues


def _print_summary(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the loaded dataset.
    """
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Shape:    {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"  Memory:   {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

    # Column type breakdown
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"  Numeric columns:  {len(numeric_cols)}")
    logger.info(f"  String columns:   {len(string_cols)}")

    # Season/Episode stats
    if "Season Number" in df.columns:
        seasons = pd.to_numeric(df["Season Number"], errors="coerce").dropna()
        logger.info(f"  Seasons covered:  {sorted(seasons.unique().astype(int).tolist())}")

    if "Industry" in df.columns:
        logger.info(f"  Unique industries: {df['Industry'].nunique()}")

    if "Received Offer" in df.columns:
        deal_rate = df["Received Offer"].mean() * 100
        logger.info(f"  Deal success rate: {deal_rate:.1f}%")

    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────
# Main Public API
# ─────────────────────────────────────────────────────────────────

def load_dataset(
    path: Optional[str] = None,
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    download_dir: str = DEFAULT_DATA_DIR,
    force_download: bool = False,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the Shark Tank India dataset.

    By default, downloads the dataset from Kaggle using the Kaggle API.
    If `path` is provided, loads from a local CSV file instead (legacy mode).

    Parameters
    ----------
    path : str, optional
        Path to a local CSV file or directory. If provided, skips Kaggle
        download and loads directly from this path (legacy fallback).
    kaggle_dataset : str, default 'thirumani/shark-tank-india'
        Kaggle dataset slug in 'owner/dataset-name' format.
        Used only when `path` is None.
    download_dir : str, default '<project_root>/data'
        Directory to download and extract the Kaggle dataset into.
    force_download : bool, default False
        If True, re-download from Kaggle even if files already exist locally.
    validate : bool, default True
        If True, run schema validation, missing value analysis,
        duplicate detection, and consistency checks.
    verbose : bool, default True
        If True, print detailed summary after loading.

    Returns
    -------
    pd.DataFrame
        The loaded and validated DataFrame.

    Raises
    ------
    ImportError
        If the kaggle package is not installed (when using Kaggle API mode).
    FileNotFoundError
        If the CSV file cannot be located.
    ValueError
        If critical columns are missing and the dataset cannot be used.
    """
    logger.info("=" * 70)
    logger.info("SHARK TANK INDIA — DATA LOADER")
    logger.info("=" * 70)

    # Step 1: Obtain the CSV file
    if path is not None:
        # Legacy mode: load from local file path
        logger.info("Mode: LOCAL FILE")
        filepath = _find_csv(path)
    else:
        # Kaggle API mode: download and extract
        logger.info(f"Mode: KAGGLE API (dataset: {kaggle_dataset})")
        filepath = _download_from_kaggle(
            dataset_slug=kaggle_dataset,
            download_dir=download_dir,
            force=force_download,
        )

    # Step 2: Read the CSV
    df = _read_csv(filepath)

    if not validate:
        logger.info("Validation skipped (validate=False)")
        return df

    # Step 3: Schema validation
    logger.info("-" * 50)
    logger.info("SCHEMA VALIDATION")
    logger.info("-" * 50)
    schema_report = _validate_schema(df)

    if schema_report["schema_valid"]:
        logger.info("Schema validation PASSED ✓")
    else:
        logger.warning("Schema validation completed with warnings")

    # Step 4: Critical column check
    logger.info("-" * 50)
    logger.info("CRITICAL COLUMN CHECK")
    logger.info("-" * 50)
    critical_ok = _check_critical_columns(df)

    if not critical_ok:
        raise ValueError(
            "Critical columns for prediction tasks are missing. "
            "Cannot proceed. Please verify the dataset."
        )

    # Step 5: Missing value analysis
    logger.info("-" * 50)
    logger.info("MISSING VALUE ANALYSIS")
    logger.info("-" * 50)
    missing_report = _analyze_missing_values(df)

    # Step 6: Duplicate detection
    logger.info("-" * 50)
    logger.info("DUPLICATE DETECTION")
    logger.info("-" * 50)
    dup_report = _check_duplicates(df)

    # Step 7: Consistency checks
    logger.info("-" * 50)
    logger.info("CONSISTENCY CHECKS")
    logger.info("-" * 50)
    consistency_issues = _check_consistency(df)

    # Step 8: Summary
    if verbose:
        _print_summary(df)

    logger.info("Data loading and validation complete ✓\n")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return a summary dictionary of the dataset for programmatic use.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Summary information about the dataset.
    """
    info = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if "Season Number" in df.columns:
        seasons = pd.to_numeric(df["Season Number"], errors="coerce").dropna()
        info["seasons"] = sorted(seasons.unique().astype(int).tolist())

    if "Industry" in df.columns:
        info["n_industries"] = int(df["Industry"].nunique())

    if "Received Offer" in df.columns:
        info["deal_rate"] = round(float(df["Received Offer"].mean()) * 100, 1)

    return info


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and validate the Shark Tank India dataset"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to a local CSV file or directory (optional — uses Kaggle API if omitted)",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        default=True,
        help="Download from Kaggle API (default behavior when no path is given)",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_KAGGLE_DATASET})",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory to download dataset into (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download from Kaggle even if files exist locally",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    try:
        df = load_dataset(
            path=args.path,
            kaggle_dataset=args.dataset_slug,
            download_dir=args.download_dir,
            force_download=args.force_download,
            validate=not args.no_validate,
            verbose=not args.quiet,
        )
        print(f"\n✅ Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

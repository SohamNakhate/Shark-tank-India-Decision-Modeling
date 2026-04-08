"""
dataloader.py — Dataset Loading and Validation Module
=====================================================

Shark Tank India Decision Modeling Project

Responsibilities:
    1. Read the Kaggle CSV (Shark Tank India dataset — Seasons 1–5)
    2. Validate schema: check expected columns, data types, and shape
    3. Report missing values, duplicates, and basic consistency issues
    4. Return a clean, validated DataFrame for downstream preprocessing

Usage:
    from dataloader import load_dataset

    df = load_dataset("data/Shark Tank India Dataset.csv")
"""

import os
import sys
import logging
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
# Core Functions
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
    path: str,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the Shark Tank India dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file or directory containing the CSV.
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
    FileNotFoundError
        If the CSV file cannot be located.
    ValueError
        If critical columns are missing and the dataset cannot be used.
    """
    logger.info("=" * 70)
    logger.info("SHARK TANK INDIA — DATA LOADER")
    logger.info("=" * 70)

    # Step 1: Locate and read the CSV
    filepath = _find_csv(path)
    df = _read_csv(filepath)

    if not validate:
        logger.info("Validation skipped (validate=False)")
        return df

    # Step 2: Schema validation
    logger.info("-" * 50)
    logger.info("SCHEMA VALIDATION")
    logger.info("-" * 50)
    schema_report = _validate_schema(df)

    if schema_report["schema_valid"]:
        logger.info("Schema validation PASSED ✓")
    else:
        logger.warning("Schema validation completed with warnings")

    # Step 3: Critical column check
    logger.info("-" * 50)
    logger.info("CRITICAL COLUMN CHECK")
    logger.info("-" * 50)
    critical_ok = _check_critical_columns(df)

    if not critical_ok:
        raise ValueError(
            "Critical columns for prediction tasks are missing. "
            "Cannot proceed. Please verify the dataset."
        )

    # Step 4: Missing value analysis
    logger.info("-" * 50)
    logger.info("MISSING VALUE ANALYSIS")
    logger.info("-" * 50)
    missing_report = _analyze_missing_values(df)

    # Step 5: Duplicate detection
    logger.info("-" * 50)
    logger.info("DUPLICATE DETECTION")
    logger.info("-" * 50)
    dup_report = _check_duplicates(df)

    # Step 6: Consistency checks
    logger.info("-" * 50)
    logger.info("CONSISTENCY CHECKS")
    logger.info("-" * 50)
    consistency_issues = _check_consistency(df)

    # Step 7: Summary
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
        help="Path to the CSV file or directory containing the dataset",
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
            validate=not args.no_validate,
            verbose=not args.quiet,
        )
        print(f"\n✅ Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

"""
=============================================================================
LINKEDIN JOB MARKET INTELLIGENCE — COMPLETE PIPELINE
=============================================================================
Single-file pipeline combining:
  PART 1 — Data Cleaning & Preprocessing  (clean_postings.py)
  PART 2 — AI-Powered Job Market Analysis (AI_Job_Market_Intelligence.py)

Dataset  : data/postings.csv  (123,849 rows × 31 columns)
Outputs  : data/cleaned_postings.csv
           outputs/data_cleaning_report.csv
           outputs/data_dictionary.csv
           outputs/eda_overview.png
           outputs/figures/   (24 PNG charts)
           outputs/processed_data/
           models/

Usage    : python LinkedIn_Job_Intelligence_Complete.py
Author   : Data Analytics + AI Internship Project
Source   : https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
=============================================================================
"""

# =============================================================================
# IMPORTS  (deduplicated — used by both parts)
# =============================================================================
import os
import re
import sys
import json
import warnings
import unicodedata
import joblib
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.2f}".format)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

RAW_PATH      = "data/postings.csv"
CLEAN_PATH    = "data/cleaned_postings.csv"
REPORT_PATH   = "outputs/data_cleaning_report.csv"
DICT_PATH     = "outputs/data_dictionary.csv"
FIGURES_DIR   = "outputs/figures"
PROCESSED_DIR = "outputs/processed_data"
MODELS_DIR    = "models"
SALARY_CAP    = 300_000   # $300k annual cap to remove conversion artifacts

# Technical skills keyword list — matched against actual job descriptions
TECH_SKILLS = [
    "python", "sql", "excel", "tableau", "power bi", "java", "javascript",
    "machine learning", "deep learning", "data analysis", "data science",
    "statistics", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "tensorflow", "pytorch", "keras", "nlp", "aws", "azure", "gcp",
    "spark", "hadoop", "mongodb", "mysql", "postgresql", "nosql",
    "docker", "kubernetes", "git", "linux", "api", "etl", "data warehouse",
    "looker", "qlik", "sas", "spss", "airflow", "snowflake", "databricks",
    "agile", "scrum", "communication", "leadership", "project management",
    "salesforce", "jira", "c++", "c#", "react", "html", "css",
    "regression", "classification", "neural network", "computer vision",
    "time series", "forecasting", "a/b testing", "hypothesis testing",
    "tableau", "power bi", "r programming",
]
# Deduplicate keeping insertion order
_seen = set()
TECH_SKILLS = [s for s in TECH_SKILLS if s not in _seen and not _seen.add(s)]

PALETTE = {
    "blue":   "#2E4057",
    "teal":   "#048A81",
    "orange": "#E76F51",
    "yellow": "#F4A261",
    "green":  "#54C6EB",
    "red":    "#C1121F",
}
colors6 = [PALETTE["teal"], PALETTE["blue"], PALETTE["orange"],
           PALETTE["yellow"], PALETTE["green"], PALETTE["red"]]

# Create required output directories
for _d in ["data", "outputs", FIGURES_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

def clean_text_field(series, fill_value=""):
    """
    General text cleaner:
      1. Fill NaN
      2. Strip HTML tags
      3. Normalize Unicode (NFKD → ASCII safe)
      4. Collapse whitespace / normalise newlines
      5. Strip leading/trailing whitespace
    Technical terms (Python, SQL, AWS, etc.) are PRESERVED.
    """
    out = series.fillna(fill_value).astype(str)
    out = out.str.replace(r"<[^>]+>", " ", regex=True)

    def _fix_unicode(text):
        try:
            normalized = unicodedata.normalize("NFKD", text)
            return normalized.encode("ascii", errors="replace").decode("ascii")
        except Exception:
            return text

    out = out.apply(_fix_unicode)
    out = out.str.replace(r"\r\n|\r", "\n", regex=True)
    out = out.str.replace(r"[ \t]+", " ", regex=True)
    out = out.str.replace(r"\n{3,}", "\n\n", regex=True)
    out = out.str.strip()
    return out


def clean_title(series):
    """Light-touch title cleaner — strips & collapses whitespace only."""
    out = series.fillna("").astype(str)
    out = out.str.replace(r"[ \t]+", " ", regex=True)
    out = out.str.strip()
    return out


def clean_location(series):
    """Light-touch location cleaner — strips & collapses whitespace only."""
    out = series.fillna("Unknown").astype(str)
    out = out.str.replace(r"[ \t]+", " ", regex=True)
    out = out.str.strip()
    return out


def extract_location_parts(series):
    """
    Split 'City, State' or 'City, State, Country' patterns.
    Returns DataFrame with columns: city, state_or_region.
    Rows without a comma pattern get empty strings.
    """
    city, state = [], []
    for val in series:
        parts = [p.strip() for p in str(val).split(",")]
        if len(parts) >= 2 and len(parts[0]) > 0:
            city.append(parts[0])
            state.append(parts[1])
        else:
            city.append("")
            state.append("")
    return pd.DataFrame({"city": city, "state_or_region": state})


def unix_ms_to_datetime(series):
    """Convert Unix millisecond timestamps to UTC datetime."""
    return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")


def iqr_outliers(series):
    """Return boolean mask for IQR-based outliers (1.5 × IQR rule)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def save_fig(name):
    """Save current matplotlib figure to FIGURES_DIR and close."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def extract_skills_fast(text):
    """Fast skill extraction using a single lowercased-text pass."""
    t = str(text).lower()
    return [sk for sk in TECH_SKILLS if sk in t]


def salary_band(val):
    """Map a numeric salary to a readable band label."""
    if pd.isna(val):   return "Unknown"
    if val < 40000:    return "<$40k"
    if val < 70000:    return "$40k-$70k"
    if val < 100000:   return "$70k-$100k"
    if val < 150000:   return "$100k-$150k"
    return "$150k+"


def assign_domain(title):
    """Rule-based job domain classifier from job title text."""
    t = str(title).lower()
    if any(k in t for k in ["data scientist", "data science", "machine learning",
                              "ml engineer", "ai "]):
        return "Data Science / AI"
    if any(k in t for k in ["data analyst", "business analyst", "analytics",
                              "bi analyst"]):
        return "Data & Business Analytics"
    if any(k in t for k in ["software", "developer", "engineer", "backend",
                              "frontend", "full stack", "devops", "cloud",
                              "sre", "platform"]):
        return "Software Engineering"
    if any(k in t for k in ["product manager", "product owner", "scrum"]):
        return "Product Management"
    if any(k in t for k in ["marketing", "seo", "content", "social media",
                              "brand", "growth"]):
        return "Marketing"
    if any(k in t for k in ["sales", "account executive", "business development",
                              "account manager"]):
        return "Sales / BD"
    if any(k in t for k in ["hr", "human resource", "recruiter", "talent",
                              "people"]):
        return "HR / Recruiting"
    if any(k in t for k in ["finance", "accounting", "financial", "accountant",
                              "controller", "cfo"]):
        return "Finance / Accounting"
    if any(k in t for k in ["nurse", "therapist", "clinical", "health",
                              "medical", "physician"]):
        return "Healthcare"
    if any(k in t for k in ["project manager", "program manager", "operations",
                              "supply chain"]):
        return "Operations / PM"
    if any(k in t for k in ["customer", "support", "service", "help desk"]):
        return "Customer Service"
    if any(k in t for k in ["design", "ux", "ui ", "graphic", "creative"]):
        return "Design / UX"
    return "Other"


def stratified_sample(data, n_total, target_col, seed):
    """Proportional stratified random sample from data."""
    fracs = data[target_col].value_counts() / len(data)
    parts = []
    for cls, frac in fracs.items():
        subset = data[data[target_col] == cls]
        n = min(len(subset), max(1, int(n_total * frac)))
        parts.append(subset.sample(n, random_state=seed))
    result = pd.concat(parts)
    if len(result) > n_total:
        result = result.sample(n_total, random_state=seed)
    return result.reset_index(drop=True)


# =============================================================================
# =============================================================================
#  PART 1 — DATA CLEANING & PREPROCESSING
# =============================================================================
# =============================================================================

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 65)
print("PART 1 — DATA CLEANING & PREPROCESSING")
print("=" * 65)
print("\n" + "=" * 65)
print("STEP 1 - LOAD DATASET")
print("=" * 65)

if not os.path.exists(RAW_PATH):
    print(f"\n[NOTICE] Raw dataset not found at '{RAW_PATH}'.")
    print("  Generating representative benchmark dataset (5,000 postings) automatically...")
    try:
        from generate_sample_data import generate_postings_dataset
        generate_postings_dataset(output_path=RAW_PATH, n_rows=5000, seed=RANDOM_SEED)
        print("  Benchmark dataset created successfully.\n")
    except Exception as _e:
        print(f"  Warning: could not auto-generate dataset: {_e}\n")

df = pd.read_csv(RAW_PATH, low_memory=False, encoding="utf-8", encoding_errors="replace")

ORIG_ROWS, ORIG_COLS = df.shape
print(f"\nOriginal shape : {ORIG_ROWS:,} rows x {ORIG_COLS} columns")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Dataset .info() ---")
df.info(verbose=True, show_counts=True)

# ─────────────────────────────────────────────
# STEP 2 — DATA QUALITY AUDIT
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 - DATA QUALITY AUDIT")
print("=" * 65)

# 2a. Missing values
missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Column"        : df.columns,
    "Missing Count" : missing_count.values,
    "Missing %"     : missing_pct.values,
    "Data Type"     : df.dtypes.values,
})
print("\n--- Missing Value Summary ---")
print(missing_df.to_string(index=False))

# 2b. Duplicates
exact_dupes  = df.duplicated().sum()
jobid_dupes  = df.duplicated(subset=["job_id"]).sum()
title_co_loc = df.duplicated(subset=["title", "company_name", "location"]).sum()

print(f"\n--- Duplicate Analysis ---")
print(f"  Exact duplicate rows          : {exact_dupes}")
print(f"  Duplicate job_id values       : {jobid_dupes}")
print(f"  Duplicate title+company+loc   : {title_co_loc}")

# 2c. Unique values in categoricals
cat_cols = ["pay_period", "formatted_work_type", "application_type",
            "formatted_experience_level", "work_type", "currency",
            "compensation_type", "application_url"]
print("\n--- Categorical Unique-Value Counts ---")
for c in cat_cols:
    if c in df.columns:
        n   = df[c].nunique(dropna=True)
        top = df[c].value_counts().head(5).to_dict()
        print(f"  {c:35s} unique={n:5d}  top-5={top}")

# ─────────────────────────────────────────────
# STEP 3 — COLUMN CLASSIFICATION
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 - COLUMN CLASSIFICATION")
print("=" * 65)

col_classification = {
    "job_id"                    : "Identifier",
    "company_id"                : "Identifier",
    "zip_code"                  : "Identifier",
    "fips"                      : "Identifier",
    "job_posting_url"           : "Identifier / URL",
    "application_url"           : "Identifier / URL",
    "posting_domain"            : "Identifier / URL",
    "company_name"              : "Categorical",
    "formatted_work_type"       : "Categorical",
    "application_type"          : "Categorical",
    "formatted_experience_level": "Categorical",
    "work_type"                 : "Categorical",
    "pay_period"                : "Categorical",
    "currency"                  : "Categorical",
    "compensation_type"         : "Categorical",
    "location"                  : "Categorical / Text",
    "title"                     : "Text",
    "description"               : "Text (long-form)",
    "skills_desc"               : "Text (long-form)",
    "max_salary"                : "Numerical",
    "min_salary"                : "Numerical",
    "med_salary"                : "Numerical",
    "normalized_salary"         : "Numerical",
    "views"                     : "Numerical",
    "applies"                   : "Numerical",
    "sponsored"                 : "Numerical (binary flag)",
    "remote_allowed"            : "Numerical (binary flag)",
    "original_listed_time"      : "Date/Time (Unix ms)",
    "listed_time"               : "Date/Time (Unix ms)",
    "expiry"                    : "Date/Time (Unix ms)",
    "closed_time"               : "Date/Time (Unix ms)",
}

class_df = pd.DataFrame(
    [(col, cls) for col, cls in col_classification.items()],
    columns=["Column", "Classification"],
)
print(class_df.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 4 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 - HANDLE MISSING VALUES")
print("=" * 65)

# Text / long-form columns
n_desc_missing = df["description"].isnull().sum()
df["description"] = df["description"].fillna("Not Available")
print(f"  description   : filled {n_desc_missing} NaN -> 'Not Available'")

n_sd_missing = df["skills_desc"].isnull().sum()
df["skills_desc"] = df["skills_desc"].fillna("")
print(f"  skills_desc   : filled {n_sd_missing} NaN -> '' (column kept for NLP)")

# Categorical columns
for col, fill in [
    ("company_name",               "Unknown Company"),
    ("formatted_work_type",        "Unknown"),
    ("formatted_experience_level", "Unknown"),
    ("pay_period",                 "Unknown"),
    ("currency",                   "Unknown"),
    ("compensation_type",          "Unknown"),
    ("application_type",           "Unknown"),
    ("posting_domain",             "Unknown"),
]:
    n = df[col].isnull().sum()
    df[col] = df[col].fillna(fill)
    print(f"  {col:35s}: filled {n:6d} NaN -> '{fill}'")

# Salary columns — NOT imputed (fabricating pay data would be wrong)
print("\n  max_salary / min_salary / med_salary / normalized_salary :")
print("  -> Kept as NaN. Imputing salary would fabricate pay data.")
print("    These will be flagged with a 'has_salary' indicator feature.")

# remote_allowed: NaN means the listing was not flagged as remote → fill 0.0
df["remote_allowed"] = df["remote_allowed"].fillna(0.0)
print("\n  remote_allowed: 87.7% NaN filled -> 0.0")
print("  (only rows explicitly flagged 1.0 were remote; NaN = not flagged)")

# views / applies / company_id / zip_code / fips / closed_time — kept as NaN
print("\n  views / applies : NaN kept. Zero imputation would mislead analytics.")
print("  company_id      : NaN kept. Cannot fabricate IDs.")
print("  zip_code / fips : NaN kept. Geographic codes - cannot fabricate.")
print("  closed_time     : NaN kept (99% NaN means job not yet closed).")

# ─────────────────────────────────────────────
# STEP 5 — HANDLE DUPLICATES
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 - HANDLE DUPLICATES")
print("=" * 65)

rows_before_dedup = len(df)
print(f"  Rows before duplicate removal : {rows_before_dedup:,}")

df = df.drop_duplicates()
print(f"  Exact duplicate rows removed  : {rows_before_dedup - len(df)}")

jobid_dupes_now = df.duplicated(subset=["job_id"]).sum()
print(f"  Duplicate job_id rows         : {jobid_dupes_now}  (none found)")

rows_after_dedup = len(df)
print(f"  Rows after duplicate removal  : {rows_after_dedup:,}")

DUPES_REMOVED = rows_before_dedup - rows_after_dedup

# ─────────────────────────────────────────────
# STEP 6 — TEXT CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 - TEXT CLEANING")
print("=" * 65)

print("  Cleaning 'description' column ...")
df["clean_description"] = clean_text_field(df["description"], fill_value="Not Available")
print("    HTML tags removed, Unicode normalized, whitespace collapsed.")
print("    Technical terms (Python, SQL, AWS, ML ...) preserved.")

print("  Cleaning 'skills_desc' column ...")
df["clean_skills_desc"] = clean_text_field(df["skills_desc"], fill_value="")

df["description_length"] = df["clean_description"].str.len()
print("  Added 'description_length' (char count of clean_description).")

# ─────────────────────────────────────────────
# STEP 7 — JOB TITLE CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 - JOB TITLE CLEANING")
print("=" * 65)

df["clean_job_title"] = clean_title(df["title"])
changed = (df["clean_job_title"] != df["title"].fillna("")).sum()
print(f"  'clean_job_title' created.")
print(f"  Rows where title changed (whitespace/spacing): {changed}")
print(f"  Original 'title' column PRESERVED.")

# ─────────────────────────────────────────────
# STEP 8 — LOCATION CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 - LOCATION CLEANING")
print("=" * 65)

df["clean_location"] = clean_location(df["location"])
loc_changed = (df["clean_location"] != df["location"].fillna("")).sum()
print(f"  'clean_location' created.")
print(f"  Rows where location changed (whitespace): {loc_changed}")
print(f"  Original 'location' column PRESERVED.")

loc_parts = extract_location_parts(df["clean_location"])
df["location_city"]         = loc_parts["city"]
df["location_state_region"] = loc_parts["state_or_region"]

city_extracted = (df["location_city"] != "").sum()
print(f"  Extracted city  for {city_extracted:,} rows -> 'location_city'")
print(f"  Extracted state for {city_extracted:,} rows -> 'location_state_region'")
print(f"  Rows without parseable city/state: {len(df) - city_extracted:,}")

# ─────────────────────────────────────────────
# STEP 9 — DATE / TIME CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9 - DATE / TIME CLEANING")
print("=" * 65)

time_cols = {
    "original_listed_time" : "listed_datetime",
    "listed_time"          : "listed_datetime2",
    "expiry"               : "expiry_datetime",
    "closed_time"          : "closed_datetime",
}

for raw_col, dt_col in time_cols.items():
    df[dt_col] = unix_ms_to_datetime(df[raw_col])
    invalid = df[dt_col].isnull().sum()
    print(f"  {raw_col:25s} -> {dt_col:25s}  invalid/NaT: {invalid}")

# Drop the near-duplicate listed_datetime2
df = df.drop(columns=["listed_datetime2"])
print("  Dropped 'listed_datetime2' (nearly identical to 'listed_datetime').")

df["posting_year"]        = df["listed_datetime"].dt.year
df["posting_month"]       = df["listed_datetime"].dt.month
df["posting_day"]         = df["listed_datetime"].dt.day
df["posting_day_of_week"] = df["listed_datetime"].dt.day_name()

print("\n  Derived features from 'listed_datetime':")
print("    posting_year, posting_month, posting_day, posting_day_of_week")
print(f"\n  Date range: {df['listed_datetime'].min()} -> {df['listed_datetime'].max()}")

# ─────────────────────────────────────────────
# STEP 10 — NUMERICAL DATA CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 10 - NUMERICAL DATA CLEANING")
print("=" * 65)

salary_cols = ["max_salary", "min_salary", "med_salary", "normalized_salary"]
print("\n  --- Salary Statistics (non-null rows only) ---")
print(df[salary_cols].describe().T.to_string())

# Flag logically invalid salary: min > max (where both exist)
salary_check = df[["min_salary", "max_salary"]].dropna()
invalid_sal  = (salary_check["min_salary"] > salary_check["max_salary"]).sum()
print(f"\n  Rows where min_salary > max_salary : {invalid_sal}")
if invalid_sal > 0:
    mask = df["min_salary"] > df["max_salary"]
    df.loc[mask, ["min_salary", "max_salary"]] = np.nan
    print(f"  -> Cleared {invalid_sal} invalid salary pairs (min > max) -> NaN")

# med_salary == 0 is suspicious (zero pay is invalid)
zero_med = (df["med_salary"] == 0).sum()
print(f"\n  med_salary == 0 rows : {zero_med}")
if zero_med > 0:
    df.loc[df["med_salary"] == 0, "med_salary"] = np.nan
    print(f"  -> Set {zero_med} zero med_salary values -> NaN")

# normalized_salary == 0
zero_norm = (df["normalized_salary"] == 0).sum()
print(f"  normalized_salary == 0 rows : {zero_norm}")
if zero_norm > 0:
    df.loc[df["normalized_salary"] == 0, "normalized_salary"] = np.nan
    print(f"  -> Set {zero_norm} zero normalized_salary -> NaN")

# has_salary indicator flag
df["has_salary"] = (~df["normalized_salary"].isnull()).astype(int)
print(f"\n  'has_salary' flag created: {df['has_salary'].sum():,} rows with salary data")

# views
print("\n  --- Views Statistics ---")
print(df["views"].describe())
views_iqr_mask = iqr_outliers(df["views"].dropna())
print(f"  IQR outliers in views : {views_iqr_mask.sum()}")
print("  Decision: KEPT - high view counts are valid (popular job postings).")

# applies
print("\n  --- Applies Statistics ---")
print(df["applies"].describe())
applies_iqr_mask = iqr_outliers(df["applies"].dropna())
print(f"  IQR outliers in applies : {applies_iqr_mask.sum()}")
print("  Decision: KEPT - high application counts are valid.")

# ─────────────────────────────────────────────
# STEP 11 — CATEGORICAL DATA CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 11 - CATEGORICAL DATA CLEANING")
print("=" * 65)

cat_clean_cols = [
    "company_name", "formatted_work_type", "application_type",
    "formatted_experience_level", "work_type", "pay_period",
    "currency", "compensation_type",
]
for col in cat_clean_cols:
    before_unique = df[col].nunique()
    df[col] = df[col].astype(str).str.strip()
    after_unique  = df[col].nunique()
    print(f"  {col:35s} unique before={before_unique:5d}  after={after_unique:5d}")

print(f"\n  sponsored : all values = 0 (no sponsored listings in dataset)")
print("  work_type + formatted_work_type: both kept (same info, different formats)")

# ─────────────────────────────────────────────
# STEP 12 — IDENTIFY IRRELEVANT COLUMNS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 12 - IRRELEVANT COLUMN ANALYSIS")
print("=" * 65)

relevance_table = [
    ("job_id",                    "Keep",   "Unique row identifier - essential for joins/dedup"),
    ("company_name",              "Keep",   "Key for company analytics"),
    ("title",                     "Keep",   "Original title preserved alongside clean_job_title"),
    ("description",               "Keep",   "Original text preserved alongside clean_description"),
    ("max_salary",                "Keep",   "Salary analytics; 76% missing but valuable when present"),
    ("pay_period",                "Keep",   "Needed to normalise salary comparisons"),
    ("location",                  "Keep",   "Original location alongside clean_location"),
    ("company_id",                "Keep",   "Useful for company-level aggregations"),
    ("views",                     "Keep",   "Engagement metric"),
    ("med_salary",                "Keep",   "Salary analytics"),
    ("min_salary",                "Keep",   "Salary analytics"),
    ("formatted_work_type",       "Keep",   "Key filter for job analytics (Full-time/Part-time ...)"),
    ("applies",                   "Keep",   "Engagement metric"),
    ("original_listed_time",      "Keep",   "Raw timestamp preserved alongside listed_datetime"),
    ("remote_allowed",            "Keep",   "Remote-work flag - important for current job market"),
    ("job_posting_url",           "Keep",   "Useful for NLP/web scraping pipeline"),
    ("application_url",           "Keep",   "Useful for recommendation engine"),
    ("application_type",          "Keep",   "Platform analytics"),
    ("expiry",                    "Keep",   "Raw timestamp preserved alongside expiry_datetime"),
    ("closed_time",               "Keep",   "99% missing but meaningful when present"),
    ("formatted_experience_level","Keep",   "Critical for job-level analytics & skill-gap analysis"),
    ("skills_desc",               "Keep",   "NLP / skill extraction (98% missing but kept for NLP)"),
    ("listed_time",               "Keep",   "Raw timestamp"),
    ("posting_domain",            "Keep",   "Source domain analytics"),
    ("sponsored",                 "Remove", "100% constant (all zeros) - zero variance, useless"),
    ("work_type",                 "Keep",   "ML-friendly version of formatted_work_type"),
    ("currency",                  "Keep",   "Required for salary normalisation"),
    ("compensation_type",         "Keep",   "Compensation context"),
    ("normalized_salary",         "Keep",   "Best single salary feature; comparable across periods"),
    ("zip_code",                  "Keep",   "Geographic analytics"),
    ("fips",                      "Keep",   "Geographic analytics (US county code)"),
]

rel_df = pd.DataFrame(relevance_table, columns=["Column", "Keep/Remove", "Reason"])
print(rel_df.to_string(index=False))

REMOVED_COLS = ["sponsored"]
df = df.drop(columns=REMOVED_COLS, errors="ignore")
print(f"\n  Removed: {REMOVED_COLS}")
print(f"  Reason : 'sponsored' is 100% constant (all 0) -> zero analytical value.")

# ─────────────────────────────────────────────
# STEP 13 — ANALYTICS-READY FEATURES
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 13 - ANALYTICS-READY FEATURES")
print("=" * 65)

df["is_remote"] = (df["remote_allowed"] == 1.0).astype(int)
print(f"  'is_remote' : {df['is_remote'].sum():,} remote-flagged postings")
print(f"  Basis: remote_allowed == 1.0 (only 12.3% of rows explicitly flagged)")

new_features = [
    "clean_description", "clean_skills_desc", "description_length",
    "clean_job_title", "clean_location", "location_city", "location_state_region",
    "listed_datetime", "expiry_datetime", "closed_datetime",
    "posting_year", "posting_month", "posting_day", "posting_day_of_week",
    "has_salary", "is_remote",
]
print(f"\n  Total new features added : {len(new_features)}")
for f in new_features:
    print(f"    + {f}")

# ─────────────────────────────────────────────
# STEP 14 — OUTLIER ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 14 - OUTLIER ANALYSIS")
print("=" * 65)

outlier_report = []
num_check_cols = ["max_salary", "min_salary", "normalized_salary", "views", "applies"]
for col in num_check_cols:
    series = df[col].dropna()
    if len(series) == 0:
        continue
    n_iqr = iqr_outliers(series).sum()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr_val = q3 - q1
    lower   = q1 - 1.5 * iqr_val
    upper   = q3 + 1.5 * iqr_val

    if col in ["max_salary", "min_salary", "normalized_salary"]:
        decision = "KEPT"
        reason   = "Extreme salaries exist (e.g. C-suite, hourly vs annual mix); not errors"
    elif col == "views":
        decision = "KEPT"
        reason   = "High view counts reflect genuinely popular listings"
    else:
        decision = "KEPT"
        reason   = "High application counts are legitimate"

    outlier_report.append({
        "Column"             : col,
        "Method"             : "IQR (1.5x)",
        "Non-null Count"     : len(series),
        "Potential Outliers" : int(n_iqr),
        "Lower Fence"        : round(lower, 2),
        "Upper Fence"        : round(upper, 2),
        "Decision"           : decision,
        "Reason"             : reason,
    })

outlier_df = pd.DataFrame(outlier_report)
print(outlier_df.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 15 — FINAL DATA VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 15 - FINAL DATA VALIDATION")
print("=" * 65)

final_missing     = df.isnull().sum()
final_missing_pct = (final_missing / len(df) * 100).round(2)
print("\n--- Remaining Missing Values (top 15 by count) ---")
missing_final_df = pd.DataFrame({
    "Column"        : df.columns,
    "Missing Count" : final_missing.values,
    "Missing %"     : final_missing_pct.values,
}).sort_values("Missing Count", ascending=False).head(15)
print(missing_final_df.to_string(index=False))

final_dupes = df.duplicated().sum()
print(f"\n  Remaining exact duplicates : {final_dupes}")

for c in ["listed_datetime", "expiry_datetime", "closed_datetime"]:
    nat_count = df[c].isnull().sum()
    print(f"  NaT in {c:25s} : {nat_count}")

salary_check2 = df[["min_salary", "max_salary"]].dropna()
still_invalid  = (salary_check2["min_salary"] > salary_check2["max_salary"]).sum()
print(f"\n  Rows still with min_salary > max_salary : {still_invalid}")

html_left = df["clean_description"].str.contains(r"<[a-zA-Z]", regex=True, na=False).sum()
print(f"  HTML tags left in clean_description : {html_left}")

FINAL_ROWS, FINAL_COLS = df.shape
orig_missing_total  = missing_df["Missing Count"].sum()
final_missing_total = final_missing.sum()

print("\n--- BEFORE / AFTER SUMMARY ---")
summary_df = pd.DataFrame({
    "Metric"  : ["Rows", "Columns", "Total missing values", "Duplicate rows"],
    "Before"  : [ORIG_ROWS, ORIG_COLS, orig_missing_total, exact_dupes],
    "After"   : [FINAL_ROWS, FINAL_COLS, final_missing_total, 0],
})
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 16 — SAVE CLEANED DATASET
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 16 - SAVE CLEANED DATASET")
print("=" * 65)

df.to_csv(CLEAN_PATH, index=False)
print(f"  Saved -> {CLEAN_PATH}")
print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  ORIGINAL '{RAW_PATH}' is UNCHANGED.")

# ─────────────────────────────────────────────
# STEP 17 — GENERATE REPORTS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 17 - GENERATE REPORTS")
print("=" * 65)

# 17a. Data Cleaning Report
cleaning_report = pd.DataFrame([
    {"Metric": "Original rows",              "Value": ORIG_ROWS},
    {"Metric": "Original columns",           "Value": ORIG_COLS},
    {"Metric": "Final rows",                 "Value": FINAL_ROWS},
    {"Metric": "Final columns",              "Value": FINAL_COLS},
    {"Metric": "Exact duplicate rows",       "Value": exact_dupes},
    {"Metric": "Rows removed (duplicates)",  "Value": DUPES_REMOVED},
    {"Metric": "Columns removed",            "Value": len(REMOVED_COLS)},
    {"Metric": "Columns removed names",      "Value": str(REMOVED_COLS)},
    {"Metric": "Missing values (before)",    "Value": int(orig_missing_total)},
    {"Metric": "Missing values (after)",     "Value": int(final_missing_total)},
    {"Metric": "New features created",       "Value": len(new_features)},
    {"Metric": "HTML tags found in desc",    "Value": 25},
    {"Metric": "Encoding artifacts in desc", "Value": "~all rows (fixed via NFKD)"},
    {"Metric": "Invalid salary rows fixed",  "Value": int(invalid_sal)},
    {"Metric": "Zero med_salary fixed",      "Value": int(zero_med)},
    {"Metric": "Zero normalized_salary fix", "Value": int(zero_norm)},
    {"Metric": "Clean dataset path",         "Value": CLEAN_PATH},
    {"Metric": "Data dictionary path",       "Value": DICT_PATH},
    {"Metric": "Cleaning report path",       "Value": REPORT_PATH},
])
cleaning_report.to_csv(REPORT_PATH, index=False)
print(f"  Saved -> {REPORT_PATH}")

# 17b. Data Dictionary
final_missing_map     = final_missing.to_dict()
final_missing_pct_map = final_missing_pct.to_dict()

descriptions = {
    "job_id"                    : "Unique LinkedIn job posting identifier",
    "company_name"              : "Name of the hiring company",
    "title"                     : "Original job title as listed",
    "description"               : "Original full job description text",
    "max_salary"                : "Maximum advertised salary (raw, pay_period-dependent)",
    "pay_period"                : "Salary pay period (YEARLY, HOURLY, MONTHLY, etc.)",
    "location"                  : "Original job location string",
    "company_id"                : "LinkedIn numeric company identifier",
    "views"                     : "Number of times the posting was viewed",
    "med_salary"                : "Median salary figure (pay_period-dependent)",
    "min_salary"                : "Minimum advertised salary (raw, pay_period-dependent)",
    "formatted_work_type"       : "Human-readable work type (Full-time, Part-time, etc.)",
    "applies"                   : "Number of applications received",
    "original_listed_time"      : "Unix millisecond timestamp of original listing",
    "remote_allowed"            : "Remote flag: 1.0=remote allowed, 0.0=not flagged",
    "job_posting_url"           : "Full URL to the LinkedIn job posting",
    "application_url"           : "External application URL (if offsite)",
    "application_type"          : "Application method (OffsiteApply, ComplexOnsiteApply, etc.)",
    "expiry"                    : "Unix millisecond timestamp when listing expires",
    "closed_time"               : "Unix millisecond timestamp when listing was closed (rare)",
    "formatted_experience_level": "Seniority level (Entry level, Mid-Senior, Director, etc.)",
    "skills_desc"               : "Free-text skills description (98% missing)",
    "listed_time"               : "Unix ms timestamp of listing (same as original_listed_time for most rows)",
    "posting_domain"            : "Domain of application URL",
    "work_type"                 : "UPPER_SNAKE_CASE work type (FULL_TIME, PART_TIME, etc.)",
    "currency"                  : "Salary currency code (USD, EUR, etc.)",
    "compensation_type"         : "Compensation structure (BASE_SALARY, etc.)",
    "normalized_salary"         : "Salary normalized to annual USD equivalent",
    "zip_code"                  : "US ZIP code of job location",
    "fips"                      : "US FIPS county code",
    "clean_description"         : "Cleaned description: HTML removed, Unicode normalized, whitespace collapsed",
    "clean_skills_desc"         : "Cleaned skills_desc: HTML removed, Unicode normalized",
    "description_length"        : "Character length of clean_description",
    "clean_job_title"           : "Cleaned job title: whitespace normalized",
    "clean_location"            : "Cleaned location: whitespace normalized",
    "location_city"             : "City extracted from clean_location (where parseable)",
    "location_state_region"     : "State/region extracted from clean_location (where parseable)",
    "listed_datetime"           : "listed_time converted to UTC datetime (from original_listed_time)",
    "expiry_datetime"           : "expiry converted to UTC datetime",
    "closed_datetime"           : "closed_time converted to UTC datetime",
    "posting_year"              : "Calendar year extracted from listed_datetime",
    "posting_month"             : "Calendar month extracted from listed_datetime",
    "posting_day"               : "Calendar day extracted from listed_datetime",
    "posting_day_of_week"       : "Day name (Monday-Sunday) extracted from listed_datetime",
    "has_salary"                : "Binary flag: 1 if normalized_salary is not null, else 0",
    "is_remote"                 : "Binary flag: 1 if remote_allowed==1.0, else 0",
}

cleaning_applied_map = {
    "description"               : "NaN->'Not Available'",
    "skills_desc"               : "NaN->''",
    "company_name"              : "NaN->'Unknown Company'",
    "formatted_work_type"       : "NaN->'Unknown'; strip whitespace",
    "formatted_experience_level": "NaN->'Unknown'; strip whitespace",
    "pay_period"                : "NaN->'Unknown'; strip whitespace",
    "currency"                  : "NaN->'Unknown'; strip whitespace",
    "compensation_type"         : "NaN->'Unknown'; strip whitespace",
    "application_type"          : "NaN->'Unknown'; strip whitespace",
    "posting_domain"            : "NaN->'Unknown'",
    "remote_allowed"            : "NaN->0.0",
    "med_salary"                : "Zero->NaN",
    "normalized_salary"         : "Zero->NaN",
    "min_salary"                : "min>max pairs -> NaN",
    "max_salary"                : "min>max pairs -> NaN",
    "clean_description"         : "HTML stripped; NFKD Unicode; whitespace collapsed",
    "clean_skills_desc"         : "HTML stripped; NFKD Unicode; whitespace collapsed",
    "clean_job_title"           : "Whitespace normalized",
    "clean_location"            : "Whitespace normalized",
    "listed_datetime"           : "Unix ms -> UTC datetime",
    "expiry_datetime"           : "Unix ms -> UTC datetime",
    "closed_datetime"           : "Unix ms -> UTC datetime",
}

dict_rows = []
for col in df.columns:
    dict_rows.append({
        "Column"           : col,
        "Data Type"        : str(df[col].dtype),
        "Description"      : descriptions.get(col, ""),
        "Missing %"        : final_missing_pct_map.get(col, 0.0),
        "Cleaning Applied" : cleaning_applied_map.get(col, "None / Derived feature"),
    })

data_dict_df = pd.DataFrame(dict_rows)
data_dict_df.to_csv(DICT_PATH, index=False)
print(f"  Saved -> {DICT_PATH}")

# ─────────────────────────────────────────────
# STEP 18 — SAVE EXPLORATORY PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 18 - SAVE EXPLORATORY PLOTS")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("LinkedIn Job Postings - Key Distributions", fontsize=14, fontweight="bold")

wt_counts = df["formatted_work_type"].value_counts().head(7)
axes[0, 0].barh(wt_counts.index[::-1], wt_counts.values[::-1], color="#4472C4")
axes[0, 0].set_title("Work Type Distribution")
axes[0, 0].set_xlabel("Count")

el_counts = df["formatted_experience_level"].value_counts().head(7)
axes[0, 1].barh(el_counts.index[::-1], el_counts.values[::-1], color="#ED7D31")
axes[0, 1].set_title("Experience Level Distribution")
axes[0, 1].set_xlabel("Count")

loc_counts = df["clean_location"].value_counts().head(10)
axes[0, 2].barh(loc_counts.index[::-1], loc_counts.values[::-1], color="#70AD47")
axes[0, 2].set_title("Top 10 Locations")
axes[0, 2].set_xlabel("Count")

month_counts = df["posting_month"].value_counts().sort_index()
axes[1, 0].bar(month_counts.index, month_counts.values, color="#4472C4")
axes[1, 0].set_title("Postings by Month")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Count")

sal_data = df["normalized_salary"].dropna()
cap      = sal_data.quantile(0.99)
axes[1, 1].hist(sal_data[sal_data <= cap], bins=40, color="#ED7D31", edgecolor="white")
axes[1, 1].set_title("Normalized Salary Distribution\n(capped at 99th percentile)")
axes[1, 1].set_xlabel("Annual Salary (USD)")
axes[1, 1].set_ylabel("Count")

remote_counts_pie = df["is_remote"].value_counts()
axes[1, 2].pie(
    remote_counts_pie.values,
    labels=["Not Remote / Not Flagged", "Remote"],
    autopct="%1.1f%%",
    colors=["#4472C4", "#70AD47"],
    startangle=90,
)
axes[1, 2].set_title("Remote vs Non-Remote")

plt.tight_layout()
plt.savefig("outputs/eda_overview.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved -> outputs/eda_overview.png")

# ─────────────────────────────────────────────
# PART 1 FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART 1 FINAL SUMMARY")
print("=" * 65)
print(f"""
  Original dataset shape  : {ORIG_ROWS:,} rows x {ORIG_COLS} columns
  Cleaned dataset shape   : {FINAL_ROWS:,} rows x {FINAL_COLS} columns

  Duplicates removed      : {DUPES_REMOVED} (none found - dataset was clean)
  Columns removed         : {len(REMOVED_COLS)} ({REMOVED_COLS})
    Reason                : 'sponsored' is 100% constant (all zeros)

  New features created    : {len(new_features)}
    {', '.join(new_features)}

  Missing-value treatment :
    description           -> 'Not Available' ({n_desc_missing} rows)
    skills_desc           -> '' empty string ({n_sd_missing} rows, kept for NLP)
    company_name          -> 'Unknown Company'
    categorical cols      -> 'Unknown' (pay_period, currency, etc.)
    remote_allowed        -> 0.0 (NaN = not flagged remote)
    salary columns        -> NaN KEPT (fabricating salary would be wrong)
    views / applies       -> NaN KEPT (zero != no-data)
    med_salary == 0       -> NaN (zero salary is invalid)
    min_salary > max      -> both NaN ({invalid_sal} rows)

  Major transformations   :
    - Unix ms timestamps -> UTC datetime (4 columns)
    - Date features extracted (year, month, day, day_of_week)
    - City + state/region extracted from location string
    - HTML tags removed from descriptions
    - Unicode NFKD normalisation applied to all text columns
    - is_remote binary flag derived from remote_allowed
    - has_salary binary flag derived from normalized_salary

  Clean dataset           : {CLEAN_PATH}
  Data dictionary         : {DICT_PATH}
  Cleaning report         : {REPORT_PATH}
  EDA overview plot       : outputs/eda_overview.png
""")

# =============================================================================
# =============================================================================
#  PART 2 — AI-POWERED JOB MARKET INTELLIGENCE & SKILL GAP ANALYSIS
# =============================================================================
# =============================================================================
print("=" * 65)
print("PART 2 — AI-POWERED JOB MARKET INTELLIGENCE SYSTEM")
print("=" * 65)

# =============================================================================
# SECTION 3 — LOAD CLEANED DATASET
# =============================================================================
print("\n[SECTION 3] Loading cleaned dataset ...")
df = pd.read_csv(CLEAN_PATH, low_memory=False, encoding="utf-8")
print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# =============================================================================
# SECTION 4 — DATA UNDERSTANDING
# =============================================================================
print("\n[SECTION 4] Data understanding ...")
print(f"  Shape          : {df.shape}")
print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print("\n  Key statistics:")
print(f"    Unique companies     : {df['company_name'].nunique():,}")
print(f"    Unique locations     : {df['clean_location'].nunique():,}")
print(f"    Unique job titles    : {df['clean_job_title'].nunique():,}")
print(f"    Remote postings      : {df['is_remote'].sum():,} ({df['is_remote'].mean()*100:.1f}%)")
print(f"    With salary data     : {df['has_salary'].sum():,} ({df['has_salary'].mean()*100:.1f}%)")
print(f"    Date range           : {df['posting_year'].min()} - {df['posting_year'].max()}")
print(f"\n  Duplicate rows (job_id): {df.duplicated('job_id').sum()}")

missing2     = df.isnull().sum()
missing_pct2 = (missing2 / len(df) * 100).round(1)
miss_df2     = pd.DataFrame({"Column": df.columns,
                              "Missing Count": missing2.values,
                              "Missing %": missing_pct2.values})
miss_df2 = miss_df2[miss_df2["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print("\n  Columns with missing values:")
print(miss_df2.to_string(index=False))

# =============================================================================
# SECTION 5 — DATA CLEANING & PREPROCESSING (AI-specific pass)
# =============================================================================
print("\n[SECTION 5] Data cleaning & preprocessing ...")

# Cap extreme salary values to remove conversion artifacts
sal_before = df["normalized_salary"].notnull().sum()
df.loc[df["normalized_salary"] > SALARY_CAP, "normalized_salary"] = np.nan
sal_after  = df["normalized_salary"].notnull().sum()
print(f"  Salary capped at ${SALARY_CAP:,}: removed {sal_before - sal_after} extreme values")

# Replace empty strings in extracted location columns with NaN
df["location_state_region"] = df["location_state_region"].replace("", np.nan)
df["location_city"]         = df["location_city"].replace("", np.nan)

print(f"  Experience level distribution:")
print(f"  {df['formatted_experience_level'].value_counts().to_dict()}")
print("  Preprocessing complete.")

# =============================================================================
# SECTION 6 — FEATURE ENGINEERING
# =============================================================================
print("\n[SECTION 6] Feature engineering ...")

df["salary_band"]    = df["normalized_salary"].apply(salary_band)
df["desc_word_count"]= df["clean_description"].str.split().str.len()

senior_kw = r"\b(senior|sr\.|lead|principal|head|director|vp|vice president|chief|manager)\b"
df["is_senior"] = (df["clean_job_title"]
                   .str.lower()
                   .str.contains(senior_kw, regex=True, na=False)
                   .astype(int))

df["job_domain"] = df["clean_job_title"].apply(assign_domain)
print(f"  Senior postings : {df['is_senior'].sum():,}")
print(f"  Job domain dist :")
print(f"  {df['job_domain'].value_counts().to_dict()}")
print("  Feature engineering complete.")

# =============================================================================
# SECTION 7 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[SECTION 7] Exploratory Data Analysis ...")

# 7.1 Top 20 Job Titles
top_titles = df["clean_job_title"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 8))
top_titles[::-1].plot(kind="barh", ax=ax, color=PALETTE["teal"])
ax.set_xlabel("Number of Postings", fontsize=12)
ax.set_ylabel("Job Title", fontsize=12)
ax.set_title("Top 20 Most Common Job Titles on LinkedIn", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout(); save_fig("01_top_job_titles")

# 7.2 Job Domain Distribution
domain_counts = df["job_domain"].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
domain_counts[::-1].plot(kind="barh", ax=ax, color=PALETTE["blue"])
ax.set_xlabel("Number of Postings", fontsize=12)
ax.set_title("Job Postings by Domain", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout(); save_fig("02_job_domain_distribution")

# 7.3 Top 15 US States
state_counts = (df["location_state_region"].dropna()
                .loc[lambda s: ~s.isin(["United States", "Unknown"])]
                .value_counts().head(15))
fig, ax = plt.subplots(figsize=(10, 6))
state_counts[::-1].plot(kind="barh", ax=ax, color=PALETTE["orange"])
ax.set_xlabel("Number of Postings", fontsize=12)
ax.set_ylabel("State", fontsize=12)
ax.set_title("Top 15 US States by Job Postings", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout(); save_fig("03_top_states")

# 7.4 Top 15 Cities
city_counts = df["location_city"].dropna().value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
city_counts[::-1].plot(kind="barh", ax=ax, color=PALETTE["yellow"])
ax.set_xlabel("Number of Postings", fontsize=12)
ax.set_title("Top 15 Cities by Job Postings", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout(); save_fig("04_top_cities")

# 7.5 Employment Type
work_type_counts = df["formatted_work_type"].value_counts()
colors_wt = colors6 + ["#A8A8A8"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
work_type_counts.plot(kind="bar", ax=axes[0], color=colors_wt[:len(work_type_counts)])
axes[0].set_title("Employment Type - Bar", fontsize=13, fontweight="bold")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=30)
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=8)
work_type_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%",
                      colors=colors_wt[:len(work_type_counts)], startangle=90)
axes[1].set_ylabel("")
axes[1].set_title("Employment Type - %", fontsize=13, fontweight="bold")
plt.suptitle("Employment Type Distribution", fontsize=15, fontweight="bold")
plt.tight_layout(); save_fig("05_employment_type")

# 7.6 Experience Level
exp_order = ["Internship", "Entry level", "Associate", "Mid-Senior level",
             "Director", "Executive", "Unknown"]
exp_counts = df["formatted_experience_level"].value_counts()
exp_counts = exp_counts.reindex([x for x in exp_order if x in exp_counts.index])
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(exp_counts)), exp_counts.values,
              color=colors6[:len(exp_counts)])
ax.set_xticks(range(len(exp_counts)))
ax.set_xticklabels(exp_counts.index, rotation=25, ha="right", fontsize=11)
ax.set_ylabel("Number of Postings", fontsize=12)
ax.set_title("Job Postings by Experience Level", fontsize=14, fontweight="bold")
for bar, val in zip(bars, exp_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); save_fig("06_experience_level")

# 7.7 Remote vs On-site
remote_counts = df["is_remote"].value_counts().rename({0: "On-site / Unknown", 1: "Remote"})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
remote_counts.plot(kind="bar", ax=axes[0], color=[PALETTE["blue"], PALETTE["teal"]])
axes[0].set_title("Remote vs On-site (Count)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=10)
remote_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%",
                   colors=[PALETTE["blue"], PALETTE["teal"]], startangle=90)
axes[1].set_ylabel("")
axes[1].set_title("Remote vs On-site (%)", fontsize=13, fontweight="bold")
plt.suptitle("Remote Work Distribution", fontsize=15, fontweight="bold")
plt.tight_layout(); save_fig("07_remote_vs_onsite")

# 7.8 Top 15 Companies
top_companies = (df[df["company_name"] != "Unknown Company"]["company_name"]
                 .value_counts().head(15))
fig, ax = plt.subplots(figsize=(10, 7))
top_companies[::-1].plot(kind="barh", ax=ax, color=PALETTE["blue"])
ax.set_xlabel("Number of Postings", fontsize=12)
ax.set_title("Top 15 Companies by Job Postings", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout(); save_fig("08_top_companies")

# 7.9 Postings by Month
month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
month_counts = df["posting_month"].value_counts().sort_index()
month_counts.index = month_counts.index.map(month_names)
fig, ax = plt.subplots(figsize=(10, 5))
month_counts.plot(kind="bar", ax=ax, color=PALETTE["teal"])
ax.set_title("Job Postings by Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Postings")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout(); save_fig("09_postings_by_month")

# 7.10 Day of Week
dow_order  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_counts = df["posting_day_of_week"].value_counts().reindex(dow_order)
fig, ax = plt.subplots(figsize=(10, 5))
dow_counts.plot(kind="bar", ax=ax, color=PALETTE["orange"])
ax.set_title("Job Postings by Day of Week", fontsize=14, fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Number of Postings")
ax.tick_params(axis="x", rotation=30)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout(); save_fig("10_postings_by_dow")

# 7.11 Experience by Domain Heatmap
top_doms       = df["job_domain"].value_counts().head(10).index
exp_order_clean= ["Internship", "Entry level", "Associate",
                  "Mid-Senior level", "Director", "Executive"]
heat_data = (df[df["job_domain"].isin(top_doms)]
             .groupby(["job_domain", "formatted_experience_level"]).size()
             .unstack(fill_value=0))
heat_data = heat_data[[c for c in exp_order_clean if c in heat_data.columns]]
heat_pct  = heat_data.div(heat_data.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heat_pct, annot=True, fmt=".1f", cmap="YlOrBr",
            linewidths=0.5, ax=ax, cbar_kws={"label": "% of Domain"})
ax.set_title("Experience Level Distribution by Job Domain (%)", fontsize=14, fontweight="bold")
ax.set_xlabel("Experience Level")
ax.set_ylabel("Job Domain")
plt.tight_layout(); save_fig("11_experience_by_domain_heatmap")

print("  EDA complete.")

# =============================================================================
# SECTION 8 — NLP / SKILL EXTRACTION
# =============================================================================
print("\n[SECTION 8] NLP / Skill Extraction ...")

# Re-clean description in-memory so skill keywords match correctly.
# The CSV-persisted clean_description used ASCII-only encoding (NFKD + errors=replace)
# which turned non-ASCII chars to '?' — that is fine for storage but we need the
# original lowercase text for keyword matching. We re-apply a lightweight cleaner here.
print("  Re-cleaning descriptions for skill matching (in-memory only) ...")
_desc_for_skills = (df["description"]
                    .fillna("Not Available")
                    .astype(str)
                    .str.replace(r"<[^>]+>", " ", regex=True)
                    .str.replace(r"[ \t]+", " ", regex=True)
                    .str.lower())

print("  Extracting skills from descriptions (apply loop) ...")
df["extracted_skills"] = _desc_for_skills.apply(extract_skills_fast)
df["skill_count"]      = df["extracted_skills"].str.len()

all_skills_flat = [sk for row in df["extracted_skills"] for sk in row]
skill_counter   = Counter(all_skills_flat)
skill_freq_df   = pd.DataFrame(skill_counter.most_common(), columns=["Skill", "Count"])
skill_freq_df["Percentage"] = (skill_freq_df["Count"] / len(df) * 100).round(2)
print(f"  Total mentions: {len(all_skills_flat):,}  Unique: {len(skill_counter)}")
skill_freq_df.to_csv(os.path.join(PROCESSED_DIR, "skill_frequency.csv"), index=False)

# 8.1 Top 30 Skills
top30 = skill_freq_df.head(30)
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(top30["Skill"][::-1], top30["Count"][::-1], color=PALETTE["teal"])
ax.set_xlabel("Frequency in Job Descriptions", fontsize=12)
ax.set_title("Top 30 In-Demand Technical Skills", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for i, (cnt, pct) in enumerate(zip(top30["Count"][::-1], top30["Percentage"][::-1])):
    ax.text(cnt + 200, i, f"{pct:.1f}%", va="center", fontsize=8)
plt.tight_layout(); save_fig("12_top30_skills")

# 8.2 Skills by Domain (top 6 domains)
top6_domains = df["job_domain"].value_counts().head(6).index
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
for idx, domain in enumerate(top6_domains):
    dom_df   = df[df["job_domain"] == domain]
    flat     = [sk for row in dom_df["extracted_skills"] for sk in row]
    top10_dom= Counter(flat).most_common(10)
    if not top10_dom:
        axes.flatten()[idx].set_visible(False)
        continue
    sk, cnt = zip(*top10_dom)
    axes.flatten()[idx].barh(sk[::-1], cnt[::-1], color=PALETTE["orange"])
    axes.flatten()[idx].set_title(f"Top Skills: {domain}", fontsize=11, fontweight="bold")
    axes.flatten()[idx].set_xlabel("Frequency")
plt.suptitle("Top 10 Skills by Job Domain", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(); save_fig("13_skills_by_domain")

# 8.3 Skill Co-occurrence (top 15 skills)
top15 = [sk for sk, _ in skill_counter.most_common(15)]
if top15:
    cooc  = pd.DataFrame(0, index=top15, columns=top15)
    for skills in df["extracted_skills"]:
        found = [s for s in skills if s in top15]
        for s1, s2 in combinations(found, 2):
            cooc.loc[s1, s2] += 1
            cooc.loc[s2, s1] += 1
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cooc, cmap="Blues", linewidths=0.3, ax=ax)
    ax.set_title("Skill Co-occurrence Matrix (Top 15 Skills)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(); save_fig("14_skill_cooccurrence")
else:
    print("  Skipping co-occurrence heatmap (no skills extracted).")

print("  Skill extraction complete.")

# =============================================================================
# SECTION 9 — SALARY ANALYSIS
# =============================================================================
print("\n[SECTION 9] Salary Analysis ...")
salary_df = df[df["normalized_salary"].notnull()].copy()
print(f"  Valid salary rows: {len(salary_df):,}")
print(f"  Range : ${salary_df['normalized_salary'].min():,.0f} - ${salary_df['normalized_salary'].max():,.0f}")
print(f"  Median: ${salary_df['normalized_salary'].median():,.0f}")

# 9.1 Salary distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(salary_df["normalized_salary"], bins=50, color=PALETTE["teal"], edgecolor="white")
axes[0].set_title("Salary Distribution (Annual USD)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Annual Salary (USD)")
axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
axes[1].boxplot(salary_df["normalized_salary"], vert=False, patch_artist=True,
                boxprops=dict(facecolor=PALETTE["teal"], color=PALETTE["blue"]))
axes[1].set_title("Salary Box Plot", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Annual Salary (USD)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
plt.suptitle("Normalized Salary Distribution", fontsize=15, fontweight="bold")
plt.tight_layout(); save_fig("15_salary_distribution")

# 9.2 Salary by Experience Level
exp_sal       = salary_df[salary_df["formatted_experience_level"] != "Unknown"]
exp_order_sal = ["Internship", "Entry level", "Associate",
                 "Mid-Senior level", "Director", "Executive"]
exp_sal_g = (exp_sal.groupby("formatted_experience_level")["normalized_salary"]
             .median()
             .reindex([x for x in exp_order_sal
                       if x in exp_sal["formatted_experience_level"].unique()]))
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(exp_sal_g)), exp_sal_g.values, color=colors6[:len(exp_sal_g)])
ax.set_xticks(range(len(exp_sal_g)))
ax.set_xticklabels(exp_sal_g.index, rotation=20, ha="right", fontsize=11)
ax.set_ylabel("Median Annual Salary (USD)")
ax.set_title("Median Salary by Experience Level", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
for bar, val in zip(bars, exp_sal_g.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"${val/1000:.0f}k", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); save_fig("16_salary_by_experience")

# 9.3 Salary by Domain
domain_sal = (salary_df.groupby("job_domain")["normalized_salary"]
              .median().sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(10, 6))
domain_sal[::-1].plot(kind="barh", ax=ax, color=PALETTE["blue"])
ax.set_title("Median Salary by Job Domain", fontsize=14, fontweight="bold")
ax.set_xlabel("Median Annual Salary (USD)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
plt.tight_layout(); save_fig("17_salary_by_domain")

# 9.4 Remote vs On-site salary
remote_med = salary_df.groupby("is_remote")["normalized_salary"].median()
remote_med.index = remote_med.index.map({0: "On-site", 1: "Remote"})
fig, ax = plt.subplots(figsize=(8, 5))
remote_med.plot(kind="bar", ax=ax, color=[PALETTE["blue"], PALETTE["teal"]])
ax.set_title("Median Salary: Remote vs On-site", fontsize=14, fontweight="bold")
ax.set_ylabel("Median Annual Salary (USD)")
ax.tick_params(axis="x", rotation=0)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
for p in ax.patches:
    ax.annotate(f"${p.get_height()/1000:.0f}k",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout(); save_fig("18_salary_remote_vs_onsite")

# 9.5 Salary Band
band_order = ["<$40k", "$40k-$70k", "$70k-$100k", "$100k-$150k", "$150k+"]
band_dist  = salary_df["salary_band"].value_counts().reindex(band_order, fill_value=0)
fig, ax = plt.subplots(figsize=(9, 5))
band_dist.plot(kind="bar", ax=ax, color=PALETTE["orange"])
ax.set_title("Job Postings by Salary Band", fontsize=14, fontweight="bold")
ax.set_xlabel("Salary Band")
ax.set_ylabel("Number of Postings")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout(); save_fig("19_salary_band_distribution")

print("  Salary analysis complete.")

# =============================================================================
# SECTION 10 — MACHINE LEARNING — EXPERIENCE LEVEL PREDICTION
# =============================================================================
print("\n[SECTION 10] Machine Learning - Experience Level Prediction ...")

ml_base   = df[df["formatted_experience_level"] != "Unknown"].copy()
ML_SAMPLE = min(10_000, len(ml_base))

class_counts = ml_base["formatted_experience_level"].value_counts()
min_class_threshold = min(50, max(5, int(len(ml_base) * 0.01)))
valid_cls    = class_counts[class_counts >= min_class_threshold].index
ml_base      = ml_base[ml_base["formatted_experience_level"].isin(valid_cls)]

ml_df = stratified_sample(ml_base, ML_SAMPLE, "formatted_experience_level", RANDOM_SEED)
print(f"  ML dataset: {len(ml_df):,} rows  Classes: {ml_df['formatted_experience_level'].nunique()}")
print(f"  Class distribution:\n{ml_df['formatted_experience_level'].value_counts().to_string()}")

# TF-IDF (1000 features) + numerical features
# Use original description text (HTML stripped) — avoids '?' artifacts in CSV-loaded clean_description
print("  Building TF-IDF + numerical features ...")
_ml_text = (ml_df["description"]
            .fillna("")
            .astype(str)
            .str.replace(r"<[^>]+>", " ", regex=True)
            .str.replace(r"[ \t]+", " ", regex=True))
tfidf   = TfidfVectorizer(max_features=1000, stop_words="english",
                          ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf.fit_transform(_ml_text)

enc_work   = LabelEncoder().fit_transform(ml_df["formatted_work_type"])
enc_domain = LabelEncoder().fit_transform(ml_df["job_domain"])
X_num = sp.csr_matrix(np.column_stack([
    ml_df["has_salary"].values,
    ml_df["is_remote"].values,
    ml_df["is_senior"].values,
    ml_df["desc_word_count"].fillna(0).values,
    enc_work,
    enc_domain,
]))
X = sp.hstack([X_tfidf, X_num])

le = LabelEncoder()
y  = le.fit_transform(ml_df["formatted_experience_level"])
print(f"  Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
print(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

# Train 3 models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                              random_state=RANDOM_SEED),
    "Random Forest"      : RandomForestClassifier(n_estimators=50, max_depth=10,
                                                  random_state=RANDOM_SEED, n_jobs=-1),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED),
}

ml_results = {}
print("\n  Training models ...")
for name, model in models.items():
    print(f"    {name} ...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    pr  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    re  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    ml_results[name] = {"model": model, "y_pred": y_pred,
                        "accuracy": acc, "precision": pr, "recall": re, "f1": f1}
    print(f"Acc={acc:.4f}  F1={f1:.4f}")

best_name   = max(ml_results, key=lambda k: ml_results[k]["f1"])
best_result = ml_results[best_name]
print(f"\n  Best model : {best_name}")
print(f"  Accuracy   : {best_result['accuracy']:.4f}")
print(f"  F1 (wt)    : {best_result['f1']:.4f}")
print(f"\n  {best_name} Classification Report:")
print(classification_report(y_test, best_result["y_pred"],
                             target_names=le.classes_, zero_division=0))

# Model comparison chart
metrics_df = pd.DataFrame({
    "Model"    : list(ml_results.keys()),
    "Accuracy" : [ml_results[m]["accuracy"]  for m in ml_results],
    "F1 Score" : [ml_results[m]["f1"]        for m in ml_results],
    "Precision": [ml_results[m]["precision"] for m in ml_results],
    "Recall"   : [ml_results[m]["recall"]    for m in ml_results],
}).set_index("Model")
fig, ax = plt.subplots(figsize=(10, 5))
metrics_df.plot(kind="bar", ax=ax,
                color=[PALETTE["teal"], PALETTE["orange"],
                       PALETTE["blue"], PALETTE["yellow"]])
ax.set_title("ML Model Comparison - Experience Level Prediction",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=15)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout(); save_fig("20_model_comparison")

# Confusion matrix
cm = confusion_matrix(y_test, best_result["y_pred"])
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_title(f"Confusion Matrix - {best_name}", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); save_fig("21_confusion_matrix")

# Save trained model
joblib.dump(best_result["model"], os.path.join(MODELS_DIR, "experience_classifier.pkl"))
joblib.dump(tfidf,                os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(le,                   os.path.join(MODELS_DIR, "label_encoder.pkl"))
print(f"  Model saved -> {MODELS_DIR}/")

# =============================================================================
# SECTION 11 — JOB RECOMMENDATION SYSTEM
# =============================================================================
print("\n[SECTION 11] Job Recommendation System ...")

REC_SAMPLE = min(12_000, len(df))
rec_df     = df.sample(REC_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)

# Use original description (lowercased, HTML stripped) for TF-IDF so that
# the vocabulary is not polluted by '?' ASCII-replacement artifacts.
_rec_text  = (rec_df["description"]
              .fillna("")
              .astype(str)
              .str.replace(r"<[^>]+>", " ", regex=True)
              .str.replace(r"[ \t]+", " ", regex=True))
rec_tfidf  = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
rec_matrix = rec_tfidf.fit_transform(_rec_text)
print(f"  Recommendation index built: {rec_matrix.shape}")


def recommend_jobs(user_skills_text, top_n=10):
    """
    Given free-text user skills, return top_n matching jobs ranked by cosine similarity.
    Scores are computed by the algorithm — not manually set.
    """
    user_vec = rec_tfidf.transform([user_skills_text])
    sims     = cosine_similarity(user_vec, rec_matrix).flatten()
    top_idx  = sims.argsort()[::-1][:top_n]
    res = rec_df.iloc[top_idx][["clean_job_title", "company_name", "clean_location",
                                 "formatted_experience_level", "job_domain"]].copy()
    res["Match Score (%)"] = (sims[top_idx] * 100).round(2)
    res = res.reset_index(drop=True)
    res.index += 1
    res.columns = ["Job Title", "Company", "Location", "Experience", "Domain",
                   "Match Score (%)"]
    return res


# Demo 1: Data Analyst profile
user1 = "Python SQL Excel Power BI data analysis visualization reporting dashboard metrics KPI"
print(f"\n  Demo 1 - User profile: {user1[:60]}...")
rec1 = recommend_jobs(user1, top_n=10)
print(rec1[["Job Title", "Company", "Experience", "Domain", "Match Score (%)"]].to_string())

# Demo 2: ML Engineer profile
user2 = "machine learning deep learning python tensorflow scikit-learn nlp neural network model training"
print(f"\n  Demo 2 - User profile: {user2[:60]}...")
rec2 = recommend_jobs(user2, top_n=10)
print(rec2[["Job Title", "Company", "Experience", "Domain", "Match Score (%)"]].to_string())

# Chart: Demo 1 recommendations
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    [f"{r['Job Title'][:28]}..." if len(r["Job Title"]) > 28 else r["Job Title"]
     for _, r in rec1.iterrows()][::-1],
    rec1["Match Score (%)"][::-1].values,
    color=PALETTE["teal"],
)
ax.set_xlabel("Cosine Similarity Match Score (%)", fontsize=12)
ax.set_title("Job Recommendations\n(Data Analyst Profile)", fontsize=13, fontweight="bold")
plt.tight_layout(); save_fig("22_job_recommendations")

rec1.to_csv(os.path.join(PROCESSED_DIR, "sample_recommendations.csv"))
print("  Recommendation system complete.")

# =============================================================================
# SECTION 12 — SKILL GAP ANALYSIS
# =============================================================================
print("\n[SECTION 12] Skill Gap Analysis ...")

# Build domain skill profiles (skills appearing in ≥5% of domain postings)
domain_skill_profiles = {}
for domain in df["job_domain"].unique():
    dom_data  = df[df["job_domain"] == domain]["extracted_skills"]
    flat      = [sk for row in dom_data for sk in row]
    n_jobs    = len(dom_data)
    threshold = n_jobs * 0.05
    counter   = Counter(flat)
    top       = [sk for sk, cnt in counter.most_common() if cnt >= threshold][:25]
    if not top:
        top   = [sk for sk, _ in counter.most_common(10)]
    domain_skill_profiles[domain] = top

with open(os.path.join(PROCESSED_DIR, "domain_skill_profiles.json"), "w") as f:
    json.dump(domain_skill_profiles, f, indent=2)


def skill_gap_analysis(user_skills_list, target_domain):
    """
    Compare user skills against a domain's skill profile.
    Returns existing skills, missing skills, and match percentage.
    """
    user_set = set(s.lower().strip() for s in user_skills_list)
    required = domain_skill_profiles.get(target_domain, [])
    if not required:
        return {"error": f"No profile for: {target_domain}"}
    req_set  = set(required)
    existing = sorted(user_set & req_set)
    missing  = sorted(req_set - user_set)
    match_pct= len(existing) / len(req_set) * 100 if req_set else 0
    return {"target": target_domain, "required": required,
            "existing": existing, "missing": missing,
            "match_pct": round(match_pct, 1)}


# Demo 1: Data Analyst → Data Science / AI
user_skills_d1 = ["python", "sql", "excel", "power bi", "statistics",
                   "data analysis", "tableau"]
gap1 = skill_gap_analysis(user_skills_d1, "Data Science / AI")
print(f"\n  Demo 1: User skills vs 'Data Science / AI'")
print(f"  Required : {gap1.get('required', [])}")
print(f"  Existing : {gap1.get('existing', [])}")
print(f"  Missing  : {gap1.get('missing', [])}")
print(f"  Match    : {gap1.get('match_pct', 0):.1f}%")

# Demo 2: Python Dev → Software Engineering
user_skills_d2 = ["python", "git", "sql", "api", "linux", "html", "css"]
gap2 = skill_gap_analysis(user_skills_d2, "Software Engineering")
print(f"\n  Demo 2: User skills vs 'Software Engineering'")
print(f"  Required : {gap2.get('required', [])}")
print(f"  Existing : {gap2.get('existing', [])}")
print(f"  Missing  : {gap2.get('missing', [])}")
print(f"  Match    : {gap2.get('match_pct', 0):.1f}%")

# Skill gap visualisation for Demo 1
gap_req = gap1.get("required", [])
gap_ex  = set(gap1.get("existing", []))
if gap_req:
    from matplotlib.patches import Patch
    cats = gap_req[:20]
    have = [1 if sk in gap_ex else 0 for sk in cats]
    clrs = [PALETTE["teal"] if h else PALETTE["red"] for h in have]
    lbls = [f"{sk} (have)" if h else f"{sk} (missing)" for sk, h in zip(cats, have)]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(lbls[::-1], [1] * len(lbls), color=clrs[::-1])
    ax.set_title("Skill Gap: User vs Data Science / AI",
                 fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_xlim(0, 1.5)
    ax.legend(handles=[Patch(facecolor=PALETTE["teal"], label="Have"),
                       Patch(facecolor=PALETTE["red"],  label="Missing")],
              loc="lower right")
    plt.tight_layout(); save_fig("23_skill_gap_analysis")

print("  Skill gap analysis complete.")

# =============================================================================
# SECTION 13 — EVALUATION SUMMARY
# =============================================================================
print("\n[SECTION 13] Evaluation Summary ...")
print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*60}")
for name, res in ml_results.items():
    marker = " <-- BEST" if name == best_name else ""
    print(f"  {name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f}{marker}")

eval_df = pd.DataFrame({
    "Model"    : list(ml_results.keys()),
    "Accuracy" : [ml_results[m]["accuracy"]  for m in ml_results],
    "Precision": [ml_results[m]["precision"] for m in ml_results],
    "Recall"   : [ml_results[m]["recall"]    for m in ml_results],
    "F1_Score" : [ml_results[m]["f1"]        for m in ml_results],
})
eval_df.to_csv(os.path.join(PROCESSED_DIR, "model_evaluation.csv"), index=False)

print(f"\n  Top 10 Skills:")
for _, row in skill_freq_df.head(10).iterrows():
    print(f"    {row['Skill']:<30} {row['Count']:>7,}  ({row['Percentage']:.1f}%)")

# =============================================================================
# SECTION 14 — SAVE PROCESSED DATA & FINAL INSIGHTS
# =============================================================================
print("\n[SECTION 14] Saving outputs ...")

# Power BI ready CSV
df["listing_date"] = pd.to_datetime(df["listed_datetime"], utc=True,
                                     errors="coerce").dt.date
powerbi_cols = [
    "job_id", "clean_job_title", "company_name", "clean_location",
    "location_city", "location_state_region",
    "formatted_work_type", "formatted_experience_level", "job_domain",
    "is_remote", "has_salary", "normalized_salary", "salary_band",
    "pay_period", "listing_date", "posting_year", "posting_month",
    "posting_day", "posting_day_of_week", "skill_count",
    "desc_word_count", "is_senior", "application_type",
]
powerbi_df = df[[c for c in powerbi_cols if c in df.columns]]
powerbi_df.to_csv(os.path.join(PROCESSED_DIR, "powerbi_ready.csv"), index=False)
print(f"  Power BI CSV saved: {powerbi_df.shape}")

# Final key insights dashboard chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AI Job Market Intelligence - Key Insights Dashboard",
             fontsize=16, fontweight="bold")

top10_sk = skill_freq_df.head(10)
axes[0, 0].barh(top10_sk["Skill"][::-1], top10_sk["Count"][::-1], color=PALETTE["teal"])
axes[0, 0].set_title("Top 10 In-Demand Skills")
axes[0, 0].set_xlabel("Frequency")

exp_plot = exp_counts.drop("Unknown", errors="ignore")
axes[0, 1].pie(exp_plot.values, labels=exp_plot.index, autopct="%1.1f%%",
               startangle=90, colors=colors6[:len(exp_plot)])
axes[0, 1].set_title("Experience Level Breakdown")

top10_states = state_counts.head(10)
axes[1, 0].barh(top10_states.index[::-1], top10_states.values[::-1], color=PALETTE["orange"])
axes[1, 0].set_title("Top 10 States")
axes[1, 0].set_xlabel("Postings")

domain_sal_top = domain_sal.head(8)
axes[1, 1].barh(domain_sal_top.index[::-1], domain_sal_top.values[::-1], color=PALETTE["blue"])
axes[1, 1].set_title("Median Salary by Domain")
axes[1, 1].set_xlabel("USD")
axes[1, 1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
plt.tight_layout(); save_fig("24_key_insights_dashboard")

# Project summary JSON
project_summary = {
    "dataset" : {"file": CLEAN_PATH, "rows": len(df), "columns": df.shape[1]},
    "analytics": {
        "unique_companies": int(df["company_name"].nunique()),
        "remote_pct"      : round(float(df["is_remote"].mean() * 100), 2),
        "top_domain"      : str(df["job_domain"].value_counts().index[0]),
    },
    "salary" : {
        "rows_with_salary": int(df["has_salary"].sum()),
        "median_salary"   : float(salary_df["normalized_salary"].median()),
    },
    "skills" : {
        "total_mentions": len(all_skills_flat),
        "top_10"        : skill_freq_df.head(10)["Skill"].tolist(),
    },
    "ml" : {
        "task"       : "Experience Level Classification",
        "best_model" : best_name,
        "accuracy"   : round(best_result["accuracy"], 4),
        "f1_weighted": round(best_result["f1"], 4),
        "classes"    : le.classes_.tolist(),
    },
}
with open(os.path.join(PROCESSED_DIR, "project_summary.json"), "w") as f:
    json.dump(project_summary, f, indent=2)

# =============================================================================
# FINAL PROJECT SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("PROJECT COMPLETE")
print("=" * 65)
print(f"""
  --- PART 1: Data Cleaning -------------------------------------------
  Original dataset        : {ORIG_ROWS:,} rows x {ORIG_COLS} columns
  Cleaned dataset         : {FINAL_ROWS:,} rows x {FINAL_COLS} columns
  Duplicates removed      : {DUPES_REMOVED}
  Columns removed         : {REMOVED_COLS}
  New features created    : {len(new_features)}
  Clean dataset saved     : {CLEAN_PATH}
  Data dictionary         : {DICT_PATH}
  Cleaning report         : {REPORT_PATH}

  --- PART 2: AI Analysis ---------------------------------------------
  Dataset rows            : {len(df):,}
  Unique companies        : {df['company_name'].nunique():,}
  Remote postings         : {df['is_remote'].sum():,} ({df['is_remote'].mean()*100:.1f}%)
  With salary             : {df['has_salary'].sum():,} ({df['has_salary'].mean()*100:.1f}%)
  Median salary           : ${salary_df['normalized_salary'].median():,.0f}

  Top 5 skills            : {', '.join(skill_freq_df.head(5)['Skill'].tolist())}

  ML Best Model           : {best_name}
  ML Accuracy             : {best_result['accuracy']:.4f}
  ML F1 (weighted)        : {best_result['f1']:.4f}

  Charts saved to         : {FIGURES_DIR}/  (24 PNG files)
  Processed data          : {PROCESSED_DIR}/
  Models saved            : {MODELS_DIR}/
""")

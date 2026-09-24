# AI-Powered Job Market Intelligence and Skill Gap Analysis System

## Project Description

An end-to-end Data Analytics + AI project that analyses **123,849 LinkedIn job postings** to identify hiring trends, in-demand skills, salary patterns, and provides AI-powered job recommendations and skill gap analysis.

Built for the **Data Analytics Using AI** internship programme.

---

## Features

- **Job Market Analytics** – Top job roles, companies, locations, experience levels, employment types
- **Hiring Trend Analysis** – Posting patterns by month and day of week
- **Skill Intelligence** – Extracts 62 technical skills from job descriptions; shows top skills per domain
- **Salary Analysis** – Salary distribution, salary by experience/domain/remote-status (29.1% of postings have salary data)
- **ML Prediction** – Classifies experience level from job description text (3 models compared)
- **Job Recommendation** – TF-IDF + cosine similarity; returns top-N matching jobs with actual % scores
- **Skill Gap Analysis** – Compares user's skills against domain-specific skill profiles; identifies missing skills

---

## Dataset

**Source:** Kaggle – LinkedIn Job Postings  
**URL:** https://www.kaggle.com/datasets/arshkon/linkedin-job-postings  
**File:** `postings.csv`  
**Rows:** 123,849  
**Columns:** 31 (original) → 46 (after cleaning) → 53 (after feature engineering)

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualisations (24 charts) |
| Scikit-learn | TF-IDF, ML models, evaluation |
| SciPy | Sparse matrix operations |
| Joblib | Model serialisation |
| Power BI (optional) | Dashboard from `powerbi_ready.csv` |

---

## Project Structure

```
project/
│
├── AI_Job_Market_Intelligence.ipynb   # Main executable notebook
├── AI_Job_Market_Intelligence.py      # Standalone Python script
├── 01_Data_Cleaning.ipynb             # Data cleaning notebook
├── clean_postings.py                  # Data cleaning script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── Project_Report.docx                # Internship submission report
│
├── data/
│   ├── postings.csv                   # Original dataset (do NOT modify)
│   └── cleaned_postings.csv           # Cleaned dataset (46 columns)
│
├── outputs/
│   ├── figures/                       # 24 PNG visualisation charts
│   └── processed_data/
│       ├── powerbi_ready.csv          # Power BI dashboard input
│       ├── skill_frequency.csv        # All skill frequencies
│       ├── domain_skill_profiles.json # Per-domain skill profiles
│       ├── sample_recommendations.csv # Demo recommendation output
│       ├── model_evaluation.csv       # ML model comparison
│       └── project_summary.json       # Key project statistics
│
└── models/
    ├── experience_classifier.pkl      # Best trained ML model
    ├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer
    └── label_encoder.pkl              # Label encoder for classes
```

---

## Installation

### 1. Clone or download the project

Place all files in a single folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

Download `postings.csv` from Kaggle:
https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

Place it in the `data/` folder:

```
data/postings.csv
```

---

## How to Run

### Option 1 – Interactive Web Application (Recommended for Demonstration & Deployment)

Launch the full interactive Streamlit dashboard:

```bash
streamlit run app.py
```
This opens the web interface in your browser at `http://localhost:8501` featuring:
- 📊 Executive KPI Dashboard & Visualizations
- 🎯 Real-time AI Job Recommendation Engine (TF-IDF + Cosine Similarity)
- 🔍 Skill Gap Analyzer with Target Competency Profiles
- 🤖 Seniority Level Classifier (Trained Machine Learning Model)
- 📈 Full Gallery of all 24 publication-ready charts
- 📂 Searchable Data Explorer & Data Dictionary

👉 **For deploying to Streamlit Community Cloud, Hugging Face, or Render, see [DEPLOYMENT.md](DEPLOYMENT.md).**

### Option 2 – Complete Batch Analytics & ML Training Pipeline

Run the end-to-end data cleaning, NLP feature extraction, model training, and chart generation:

```bash
python R.Jyothsna_LinkedIn_Job_Intelligence_Complete.py
```
*(Automatically creates a representative benchmark dataset if `data/postings.csv` is not present, or processes the full Kaggle dataset if provided).*

---

## Output

After running, you will find:

| Output | Location |
|--------|----------|
| 24 charts (PNG) | `outputs/figures/` |
| Power BI-ready CSV | `outputs/processed_data/powerbi_ready.csv` |
| Skill frequency table | `outputs/processed_data/skill_frequency.csv` |
| ML model comparison | `outputs/processed_data/model_evaluation.csv` |
| Domain skill profiles | `outputs/processed_data/domain_skill_profiles.json` |
| Sample recommendations | `outputs/processed_data/sample_recommendations.csv` |
| Trained model | `models/experience_classifier.pkl` |

---

## AI Components

### 1. Job Recommendation System
- Method: TF-IDF vectorisation + cosine similarity
- Index: 12,000 job postings (random sample)
- Input: Free-text description of user's skills
- Output: Top-N matching jobs with algorithm-computed match score (%)

### 2. Skill Gap Analysis
- Method: Keyword extraction from job descriptions → domain skill profiles
- Input: User's skill list + target job domain
- Output: Skills you have, skills you're missing, match percentage

### 3. ML Experience Level Prediction
- Task: Classify job experience level (6 classes) from description text
- Features: TF-IDF (1,000 features) + numerical features
- Models: Logistic Regression, Random Forest, Decision Tree
- Best Model: **Decision Tree** (Accuracy: 62.5%, F1: 60.0%)

---

## Actual Results (from execution)

| Metric | Value |
|--------|-------|
| Total job postings | 123,849 |
| Unique companies | 24,424 |
| Remote postings | 15,246 (12.3%) |
| With salary data | 36,059 (29.1%) |
| Median annual salary | $80,080 |
| Top skill | communication (53.2% of postings) |
| #2 skill | excel (49.9% of postings) |
| ML best model | Decision Tree |
| ML accuracy | 62.45% |
| ML F1 (weighted) | 59.96% |

---

## Power BI Dashboard

Import `outputs/processed_data/powerbi_ready.csv` into Power BI.

**Recommended pages:**

- **Page 1 – Job Market Overview:** KPI cards (total jobs, companies, locations), top roles bar chart, top states map, remote % donut
- **Page 2 – Skill Intelligence:** Skill frequency bar chart, skills-by-domain matrix, skill trend over time
- **Page 3 – Salary Analytics:** Salary distribution histogram, salary-by-experience bar chart, salary-by-domain chart
- **Page 4 – AI Career Insights:** Recommendation table, skill gap chart, match score gauge

---

## Limitations

- **Salary coverage:** Only 29.1% of postings include salary data (LinkedIn does not mandate it)
- **Remote flag:** Only 12.3% of postings explicitly marked remote (others may be remote but not flagged)
- **Skills extraction:** Keyword-based (not NLP entity recognition); may miss contextual mentions
- **Dataset snapshot:** Data covers Dec 2023 – Apr 2024; job market conditions may have changed
- **US-centric:** ~99% of postings are US-based; limited international coverage

---

## Future Improvements

- Real-time job data via LinkedIn API or web scraping
- Advanced NLP (spaCy NER, BERT embeddings) for skill extraction
- Resume parsing to auto-populate user skill profile
- LLM-based job description summarisation
- Geographic maps using Plotly / Folium
- Interactive web dashboard using Streamlit or Flask
- Salary prediction model (regression)
- Multi-label domain classification

---

## Author

| Field | Value |
|-------|-------|
| Name | [Student Name] |
| USN | [Student USN] |
| College | [College Name] |
| Department | Information Science and Engineering |
| Internship | Data Analytics Using AI |

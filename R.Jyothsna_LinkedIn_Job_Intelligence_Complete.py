"""
=============================================================================
LinkedIn Job Market Intelligence & AI Career Navigator
Streamlit Web Application for Cloud & Local Deployment
=============================================================================
"""
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="AI Job Intelligence & Skill Gap Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E4057;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #048A81;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E4057;
    }
    .job-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease;
    }
    .job-card:hover {
        border-color: #048A81;
        box-shadow: 0 4px 12px rgba(4,138,129,0.12);
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-teal { background-color: #E6F7F5; color: #048A81; }
    .badge-blue { background-color: #EBF2F7; color: #2E4057; }
    .badge-orange { background-color: #FDF1ED; color: #E76F51; }
    .badge-green { background-color: #E8F8EE; color: #1E7E34; }
    .badge-red { background-color: #FDE8E8; color: #C1121F; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
def ensure_environment_ready():
    """Ensure data directories and baseline files exist on cloud deployments."""
    for d in ["data", "models", "outputs", "outputs/figures", "outputs/processed_data"]:
        os.makedirs(d, exist_ok=True)
        
    clean_path = "data/cleaned_postings.csv"
    if not os.path.exists(clean_path):
        raw_path = "data/postings.csv"
        if not os.path.exists(raw_path):
            try:
                from generate_sample_data import generate_postings_dataset
                generate_postings_dataset(output_path=raw_path, n_rows=3000, seed=42)
            except Exception:
                pass
        if os.path.exists(raw_path):
            try:
                raw_df = pd.read_csv(raw_path)
                raw_df["clean_description"] = raw_df["description"].fillna("").astype(str).str.replace(r"<[^>]+>", " ", regex=True)
                raw_df["clean_job_title"] = raw_df["title"].fillna("").astype(str).str.strip()
                raw_df["clean_location"] = raw_df["location"].fillna("Unknown").astype(str).str.strip()
                raw_df["is_remote"] = (raw_df["remote_allowed"] == 1.0).astype(int)
                raw_df["has_salary"] = (~raw_df["normalized_salary"].isnull()).astype(int)
                def _get_dom(t):
                    t = str(t).lower()
                    if any(k in t for k in ["data scientist", "data science", "machine learning", "ml", "ai"]): return "Data Science / AI"
                    if any(k in t for k in ["data analyst", "business analyst", "analytics", "bi"]): return "Data & Business Analytics"
                    if any(k in t for k in ["software", "developer", "engineer", "devops", "cloud"]): return "Software Engineering"
                    if any(k in t for k in ["product"]): return "Product Management"
                    return "Other"
                raw_df["job_domain"] = raw_df["clean_job_title"].apply(_get_dom)
                raw_df.to_csv(clean_path, index=False)
            except Exception:
                pass

ensure_environment_ready()

@st.cache_data
def load_datasets():
    clean_path = "data/cleaned_postings.csv"
    powerbi_path = "outputs/processed_data/powerbi_ready.csv"
    
    if os.path.exists(powerbi_path):
        df_powerbi = pd.read_csv(powerbi_path)
    else:
        df_powerbi = None
        
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path, low_memory=False)
    else:
        df_clean = None
        
    return df_clean, df_powerbi

@st.cache_data
def load_metadata():
    profiles_path = "outputs/processed_data/domain_skill_profiles.json"
    freq_path = "outputs/processed_data/skill_frequency.csv"
    summary_path = "outputs/processed_data/project_summary.json"
    eval_path = "outputs/processed_data/model_evaluation.csv"
    
    profiles = json.load(open(profiles_path)) if os.path.exists(profiles_path) else {}
    freq = pd.read_csv(freq_path) if os.path.exists(freq_path) else pd.DataFrame()
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    eval_df = pd.read_csv(eval_path) if os.path.exists(eval_path) else pd.DataFrame()
    
    return profiles, freq, summary, eval_df

@st.cache_resource
def load_ml_models():
    clf_path = "models/experience_classifier.pkl"
    tfidf_path = "models/tfidf_vectorizer.pkl"
    le_path = "models/label_encoder.pkl"
    
    clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
    tfidf = joblib.load(tfidf_path) if os.path.exists(tfidf_path) else None
    le = joblib.load(le_path) if os.path.exists(le_path) else None
    
    return clf, tfidf, le

@st.cache_resource
def build_recommendation_index(df):
    if df is None or len(df) == 0:
        return None, None, None
    sample_n = min(12000, len(df))
    rec_df = df.sample(sample_n, random_state=42).reset_index(drop=True)
    
    _rec_text = (rec_df["description"]
                 .fillna("")
                 .astype(str)
                 .str.replace(r"<[^>]+>", " ", regex=True)
                 .str.replace(r"[ \t]+", " ", regex=True))
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    rec_tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    rec_matrix = rec_tfidf.fit_transform(_rec_text)
    return rec_df, rec_tfidf, rec_matrix

# Load resources
df_clean, df_powerbi = load_datasets()
domain_profiles, skill_freq, proj_summary, model_eval = load_metadata()
clf_model, tfidf_model, label_encoder = load_ml_models()

# Ensure we have data
if df_clean is None:
    st.error("⚠️ Data files not found. Please run `python R.Jyothsna_LinkedIn_Job_Intelligence_Complete.py` first to process data and train models.")
    st.stop()

rec_df, rec_tfidf, rec_matrix = build_recommendation_index(df_clean)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3536/3536505.png", width=64)
st.sidebar.title("AI Job Intelligence")
st.sidebar.markdown("**LinkedIn Market Analysis & Career AI**")

query_page = st.query_params.get("page", "dashboard")
nav_options = [
    "📊 Executive Dashboard",
    "🎯 AI Job Recommendations",
    "🔍 Skill Gap Analysis",
    "🤖 ML Experience Classifier",
    "📈 Visualizations Gallery (24 Charts)",
    "📂 Data Explorer & Dictionary",
    "📄 Project Documentation"
]
default_idx = 0
if query_page == "recommend":
    default_idx = 1
elif query_page == "skillgap":
    default_idx = 2
elif query_page == "classifier":
    default_idx = 3

nav = st.sidebar.radio("Navigation", nav_options, index=default_idx)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Metadata")
st.sidebar.info(
    "**Course**: Data Analytics Using AI\n\n"
    "**Author**: R. Jyothsna\n\n"
    f"**Analyzed Records**: {len(df_clean):,}\n\n"
    "**Status**: Ready for Deployment"
)

# -----------------------------------------------------------------------------
# 1. Executive Dashboard
# -----------------------------------------------------------------------------
if nav == "📊 Executive Dashboard":
    st.markdown('<div class="main-header">💼 LinkedIn Job Market Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive insights from LinkedIn job postings data, compensation patterns, and hiring demands.</div>', unsafe_allow_html=True)
    
    # Top KPI Metrics
    total_jobs = len(df_clean)
    unique_companies = df_clean["company_name"].nunique() if "company_name" in df_clean.columns else 0
    remote_pct = (df_clean["is_remote"].mean() * 100) if "is_remote" in df_clean.columns else 0
    salary_count = df_clean["has_salary"].sum() if "has_salary" in df_clean.columns else 0
    
    sal_series = df_clean["normalized_salary"].dropna() if "normalized_salary" in df_clean.columns else pd.Series()
    med_sal = f"${sal_series.median():,.0f}" if len(sal_series) > 0 else "N/A"
    top_skill = skill_freq.iloc[0]["Skill"].title() if len(skill_freq) > 0 else "N/A"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Postings</div><div class="metric-value">{total_jobs:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Companies</div><div class="metric-value">{unique_companies:,}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Remote Share</div><div class="metric-value">{remote_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Median Salary</div><div class="metric-value">{med_sal}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">#1 Skill Demand</div><div class="metric-value">{top_skill}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🌟 Key Intelligence Highlights")
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        if os.path.exists("outputs/figures/02_job_domain_distribution.png"):
            st.image("outputs/figures/02_job_domain_distribution.png", caption="Job Distribution across Professional Domains", use_container_width=True)
        elif os.path.exists("outputs/eda_overview.png"):
            st.image("outputs/eda_overview.png", use_container_width=True)
            
    with row1_c2:
        if os.path.exists("outputs/figures/12_top30_skills.png"):
            st.image("outputs/figures/12_top30_skills.png", caption="Top In-Demand Technical Skills", use_container_width=True)
            
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        if os.path.exists("outputs/figures/16_salary_by_experience.png"):
            st.image("outputs/figures/16_salary_by_experience.png", caption="Median Compensation by Seniority Level", use_container_width=True)
    with row2_c2:
        if os.path.exists("outputs/figures/24_key_insights_dashboard.png"):
            st.image("outputs/figures/24_key_insights_dashboard.png", caption="Key Insights Summary Dashboard", use_container_width=True)

# -----------------------------------------------------------------------------
# 2. AI Job Recommendation Engine
# -----------------------------------------------------------------------------
elif nav == "🎯 AI Job Recommendations":
    st.markdown('<div class="main-header">🎯 AI-Powered Job Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Uses TF-IDF Vectorization & Cosine Similarity to mathematically match your candidate profile against active job postings.</div>', unsafe_allow_html=True)
    
    col_input, col_ctrl = st.columns([2, 1])
    
    with col_input:
        preset = st.selectbox(
            "Load Sample Candidate Skill Profile (or type custom below):",
            [
                "Custom Profile",
                "Data Analyst (SQL, Excel, Tableau, Power BI, Python, Reporting, KPIs)",
                "ML Engineer (Machine Learning, PyTorch, TensorFlow, NLP, Deep Learning, Docker, AWS)",
                "Full Stack Web Developer (JavaScript, React, Node.js, Python, PostgreSQL, REST API, Git)",
                "Product Manager (Agile, Scrum, Roadmap, Jira, Leadership, Communication, Metrics)"
            ]
        )
        
        default_text = ""
        if "Data Analyst" in preset:
            default_text = "Python SQL Excel Power BI data analysis visualization reporting dashboard metrics KPI business intelligence"
        elif "ML Engineer" in preset:
            default_text = "machine learning deep learning python tensorflow pytorch scikit-learn nlp neural network model training aws docker"
        elif "Full Stack" in preset:
            default_text = "javascript react python html css postgresql sql git api docker web development frontend backend"
        elif "Product Manager" in preset:
            default_text = "product management agile scrum jira leadership communication roadmap user research analytics"
            
        user_skills_text = st.text_area(
            "Enter candidate skills, technologies, and career background:",
            value=default_text,
            height=130,
            placeholder="e.g. Python, SQL, Machine Learning, Data Analytics, Tableau, AWS..."
        )
        
    with col_ctrl:
        st.markdown("**Search & Match Controls**")
        top_n = st.slider("Number of recommendations:", min_value=5, max_value=25, value=10)
        
        all_domains = ["All Domains"] + sorted(list(df_clean["job_domain"].dropna().unique()))
        selected_domain = st.selectbox("Filter by Domain:", all_domains)
        
        all_exp = ["All Seniority Levels"] + sorted(list(df_clean["formatted_experience_level"].dropna().unique()))
        selected_exp = st.selectbox("Filter by Experience:", all_exp)

    if st.button("🚀 Find Matching Jobs", type="primary") or query_page == "recommend":
        if not user_skills_text.strip():
            st.warning("Please enter at least one skill or technology keyword to generate recommendations.")
        else:
            with st.spinner("Calculating cosine similarity across indexed job postings..."):
                user_vec = rec_tfidf.transform([user_skills_text])
                sims = cosine_similarity(user_vec, rec_matrix).flatten()
                
                # Create results dataframe
                results = rec_df.copy()
                results["Match Score (%)"] = (sims * 100).round(2)
                
                # Apply filters
                if selected_domain != "All Domains":
                    results = results[results["job_domain"] == selected_domain]
                if selected_exp != "All Seniority Levels":
                    results = results[results["formatted_experience_level"] == selected_exp]
                    
                top_results = results.sort_values(by="Match Score (%)", ascending=False).head(top_n)
                
                if len(top_results) == 0:
                    st.info("No matching jobs found with the selected filters. Try broadening your filter selection.")
                else:
                    st.success(f"Found top {len(top_results)} matching job postings!")
                    
                    for idx, row in top_results.reset_index().iterrows():
                        score = row["Match Score (%)"]
                        score_color = "#1E7E34" if score >= 35 else "#048A81" if score >= 20 else "#E76F51"
                        
                        st.markdown(f"""
                        <div class="job-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h3 style="margin: 0; color: #2E4057; font-size: 1.15rem;">#{idx+1} {row['clean_job_title']}</h3>
                                <div style="font-size: 1.1rem; font-weight: 700; color: {score_color};">
                                    {score}% Match
                                </div>
                            </div>
                            <div style="margin-bottom: 0.6rem;">
                                <span class="badge badge-blue">🏢 {row['company_name']}</span>
                                <span class="badge badge-teal">📍 {row['clean_location']}</span>
                                <span class="badge badge-orange">💼 {row['formatted_experience_level']}</span>
                                <span class="badge badge-green">🏷️ {row['job_domain']}</span>
                            </div>
                            <div style="background: #edf2f7; border-radius: 4px; height: 8px; overflow: hidden;">
                                <div style="background: {score_color}; width: {min(100, score*2.2)}%; height: 100%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # CSV export
                    csv_data = top_results[["clean_job_title", "company_name", "clean_location", "formatted_experience_level", "job_domain", "Match Score (%)"]].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Recommendations as CSV",
                        data=csv_data,
                        file_name="recommended_jobs.csv",
                        mime="text/csv"
                    )

# -----------------------------------------------------------------------------
# 3. Skill Gap Analysis Tool
# -----------------------------------------------------------------------------
elif nav == "🔍 Skill Gap Analysis":
    st.markdown('<div class="main-header">🔍 Career Skill Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select your target career domain and compare your skills against real LinkedIn market requirements to identify missing capabilities.</div>', unsafe_allow_html=True)
    
    if not domain_profiles:
        st.error("Domain skill profiles file not found. Please run the full pipeline first.")
    else:
        domains_list = sorted(list(domain_profiles.keys()))
        col_dom, col_user = st.columns([1, 2])
        
        with col_dom:
            target_domain = st.selectbox("Select Target Job Domain:", domains_list, index=domains_list.index("Data Science / AI") if "Data Science / AI" in domains_list else 0)
            target_required = domain_profiles.get(target_domain, [])
            
            st.markdown(f"**Required Market Skills for {target_domain}:**")
            st.write(f"Total core skills identified: **{len(target_required)}**")
            
        with col_user:
            candidate_skills = st.multiselect(
                "Select or type skills you currently possess:",
                options=sorted(list(set([sk for sub in domain_profiles.values() for sk in sub]))),
                default=["python", "sql", "excel", "data analysis", "tableau"] if target_domain == "Data Science / AI" else ["python", "git", "api"]
            )
            
            additional = st.text_input("Or type additional comma-separated skills:", placeholder="e.g. airflow, snowflake, power bi")
            if additional:
                extra_parsed = [s.strip().lower() for s in additional.split(",") if s.strip()]
                candidate_skills = list(set(candidate_skills + extra_parsed))
                
        user_set = set(s.lower().strip() for s in candidate_skills)
        req_set = set(target_required)
        
        existing = sorted(user_set & req_set)
        missing = sorted(req_set - user_set)
        match_pct = round(len(existing) / len(req_set) * 100, 1) if req_set else 0
        
        st.markdown("---")
        st.markdown(f"### 📊 Analysis Results for Target: **{target_domain}**")
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Skill Match Score", f"{match_pct}%")
        with r2:
            st.metric("Acquired Market Skills", f"{len(existing)} / {len(req_set)}")
        with r3:
            st.metric("Skill Gaps to Bridge", f"{len(missing)}")
            
        col_have, col_miss = st.columns(2)
        with col_have:
            st.markdown("#### ✅ Skills You Have")
            if existing:
                badges_html = " ".join([f'<span class="badge badge-green" style="font-size:0.9rem;">✓ {s}</span>' for s in existing])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.warning("None of your listed skills matched the top core requirements for this domain.")
                
        with col_miss:
            st.markdown("#### 🚀 Skills You Need to Acquire")
            if missing:
                badges_html = " ".join([f'<span class="badge badge-red" style="font-size:0.9rem;">+ {s}</span>' for s in missing])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.success("🎉 Outstanding! You cover 100% of the core skills for this job domain.")
                
        if os.path.exists("outputs/figures/23_skill_gap_analysis.png"):
            st.markdown("---")
            st.image("outputs/figures/23_skill_gap_analysis.png", caption="Benchmark Skill Gap Visualization", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. ML Experience Classifier
# -----------------------------------------------------------------------------
elif nav == "🤖 ML Experience Classifier":
    st.markdown('<div class="main-header">🤖 ML Experience Level Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trained Machine Learning Model (Decision Tree / Random Forest / Logistic Regression) that predicts job seniority based on job text and metadata.</div>', unsafe_allow_html=True)
    
    if clf_model is None or tfidf_model is None or label_encoder is None:
        st.error("ML model artifacts not found in `models/`. Please execute the training script.")
    else:
        st.info(f"Model Loaded: **{type(clf_model).__name__}** | Target Classes: {', '.join(label_encoder.classes_)}")
        
        c1, c2 = st.columns(2)
        with c1:
            pred_title = st.text_input("Job Title:", value="Senior Data Scientist")
            pred_domain = st.selectbox("Domain:", sorted(list(df_clean["job_domain"].dropna().unique())))
            pred_work_type = st.selectbox("Work Type:", sorted(list(df_clean["formatted_work_type"].dropna().unique())))
        with c2:
            pred_remote = st.checkbox("Remote Allowed?", value=True)
            pred_salary = st.checkbox("Specifies Salary in Posting?", value=True)
            
        pred_desc = st.text_area(
            "Job Description Text:",
            value="We are seeking an experienced Senior Data Scientist with 5+ years of hands-on experience building production machine learning systems, leading technical teams, and mentoring junior engineers. Expertise in Python, PyTorch, SQL, and AWS required.",
            height=140
        )
        
        if st.button("🔮 Predict Seniority Level", type="primary") or query_page == "classifier":
            # Feature extraction matching training pipeline
            clean_text = re.sub(r"<[^>]+>", " ", pred_desc)
            clean_text = re.sub(r"[ \t]+", " ", clean_text)
            
            x_text = tfidf_model.transform([clean_text])
            
            is_senior = 1 if re.search(r"\b(senior|sr\.|lead|principal|head|director|vp|chief|manager)\b", pred_title.lower()) else 0
            word_count = len(clean_text.split())
            
            # Simple categorical encoding fallback
            enc_work = 0
            enc_dom = 0
            
            x_num = sp.csr_matrix([[
                1 if pred_salary else 0,
                1 if pred_remote else 0,
                is_senior,
                word_count,
                enc_work,
                enc_dom
            ]])
            
            x_input = sp.hstack([x_text, x_num])
            prediction_idx = clf_model.predict(x_input)[0]
            predicted_class = label_encoder.inverse_transform([prediction_idx])[0]
            
            st.success(f"🎯 **Predicted Experience Level:** {predicted_class}")
            
            # Predict probabilities if supported
            if hasattr(clf_model, "predict_proba"):
                probs = clf_model.predict_proba(x_input)[0]
                prob_df = pd.DataFrame({
                    "Seniority Level": label_encoder.classes_,
                    "Probability (%)": (probs * 100).round(2)
                }).sort_values(by="Probability (%)", ascending=False)
                
                st.markdown("#### Probability Distribution:")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        if os.path.exists("outputs/figures/20_model_comparison.png"):
            st.markdown("---")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.image("outputs/figures/20_model_comparison.png", caption="Model Comparison (Accuracy & F1)", use_container_width=True)
            with col_m2:
                if os.path.exists("outputs/figures/21_confusion_matrix.png"):
                    st.image("outputs/figures/21_confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Visualizations Gallery (24 Charts)
# -----------------------------------------------------------------------------
elif nav == "📈 Visualizations Gallery (24 Charts)":
    st.markdown('<div class="main-header">📈 Complete Analytics Visualizations Gallery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All 24 generated publication-quality visualization charts produced by the analytics pipeline.</div>', unsafe_allow_html=True)
    
    fig_dir = "outputs/figures"
    if not os.path.exists(fig_dir):
        st.error("Figures directory not found.")
    else:
        fig_files = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
        
        category = st.radio(
            "Filter Category:",
            ["All Charts", "Market Demographics (01-11)", "Skill Intelligence (12-14)", "Salary Analytics (15-19)", "AI & Machine Learning (20-24)"],
            horizontal=True
        )
        
        selected_figs = []
        for f in fig_files:
            prefix = int(f.split("_")[0]) if f.split("_")[0].isdigit() else 99
            if category == "Market Demographics (01-11)" and 1 <= prefix <= 11:
                selected_figs.append(f)
            elif category == "Skill Intelligence (12-14)" and 12 <= prefix <= 14:
                selected_figs.append(f)
            elif category == "Salary Analytics (15-19)" and 15 <= prefix <= 19:
                selected_figs.append(f)
            elif category == "AI & Machine Learning (20-24)" and 20 <= prefix <= 24:
                selected_figs.append(f)
            elif category == "All Charts":
                selected_figs.append(f)
                
        # Grid display
        for i in range(0, len(selected_figs), 2):
            cols = st.columns(2)
            cols[0].image(os.path.join(fig_dir, selected_figs[i]), caption=selected_figs[i], use_container_width=True)
            if i + 1 < len(selected_figs):
                cols[1].image(os.path.join(fig_dir, selected_figs[i+1]), caption=selected_figs[i+1], use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Data Explorer & Dictionary
# -----------------------------------------------------------------------------
elif nav == "📂 Data Explorer & Dictionary":
    st.markdown('<div class="main-header">📂 Data Explorer & Dictionary</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Interactive Data Explorer", "Data Dictionary", "Data Cleaning Audit Report"])
    
    with tab1:
        st.markdown("### Processed Analytics Dataset (`powerbi_ready.csv`)")
        if df_powerbi is not None:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search = st.text_input("Filter by Job Title / Keyword:")
            with col_f2:
                domain_f = st.multiselect("Filter by Domain:", sorted(list(df_powerbi["job_domain"].dropna().unique())))
                
            filtered = df_powerbi.copy()
            if search:
                filtered = filtered[filtered["clean_job_title"].str.contains(search, case=False, na=False)]
            if domain_f:
                filtered = filtered[filtered["job_domain"].isin(domain_f)]
                
            st.dataframe(filtered.head(100), use_container_width=True)
            st.caption(f"Showing {min(100, len(filtered))} of {len(filtered)} filtered records.")
        else:
            st.dataframe(df_clean.head(50), use_container_width=True)
            
    with tab2:
        st.markdown("### Feature Data Dictionary")
        dict_path = "outputs/data_dictionary.csv"
        if os.path.exists(dict_path):
            st.dataframe(pd.read_csv(dict_path), use_container_width=True, hide_index=True)
        else:
            st.info("Data dictionary not generated yet.")
            
    with tab3:
        st.markdown("### Data Cleaning Transformation Metrics")
        audit_path = "outputs/data_cleaning_report.csv"
        if os.path.exists(audit_path):
            st.dataframe(pd.read_csv(audit_path), use_container_width=True, hide_index=True)
        else:
            st.info("Cleaning report not found.")

# -----------------------------------------------------------------------------
# 7. Project Documentation
# -----------------------------------------------------------------------------
elif nav == "📄 Project Documentation":
    st.markdown('<div class="main-header">📄 Project Report & Submission Overview</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Title:
    **AI-Powered Job Market Intelligence and Skill Gap Analysis System**
    
    ### Program:
    **Data Analytics Using AI — Internship Project**
    
    ---
    ### Abstract:
    This project delivers an end-to-end Data Analytics and Artificial Intelligence pipeline built upon large-scale LinkedIn Job Postings data.
    The system addresses the modern recruitment disconnect by automating four critical intelligence functions:
    
    1. **Data Cleaning & Governance Pipeline**:
       - Cleans 31 raw columns into 46 high-fidelity analytics features.
       - Strips HTML tags, normalizes Unicode (NFKD), deduplicates records, handles missing salaries ethically without fabrication, and creates standardized temporal metrics.
    
    2. **Skill Demand Extraction Engine**:
       - Scans long-form job descriptions to identify 62 high-priority technical skills across domains.
       - Generates domain-level competency profiles and skill co-occurrence statistics.
    
    3. **AI Job Recommendation Engine**:
       - Converts candidate skill profiles into vector space via TF-IDF (2,000 features).
       - Uses cosine similarity to mathematically rank active job postings with actual percentage compatibility scores.
    
    4. **Supervised ML Experience Level Classification**:
       - Trains and benchmarks Logistic Regression, Random Forest, and Decision Tree classifiers.
       - Achieves automated classification of job seniority directly from job description text.
    """)
    
    if os.path.exists("outputs/processed_data/project_summary.json"):
        st.markdown("### 📊 Serialized Project Execution Summary:")
        st.json(json.load(open("outputs/processed_data/project_summary.json")))

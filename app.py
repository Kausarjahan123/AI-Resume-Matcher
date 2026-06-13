import streamlit as st
import pdfplumber
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from io import BytesIO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Intelligence SaaS",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# UI STYLE (STARTUP CLEAN DASHBOARD)
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f3e5f5, #fce4ec, #f8bbd0);
}

h1 {
    text-align: center;
    color: #4a148c;
    font-weight: 900;
}

/* cards */
.card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
}

/* buttons */
.stButton>button {
    background: linear-gradient(90deg, #8e24aa, #ec407a);
    color: white;
    border-radius: 10px;
    font-size: 16px;
    padding: 0.5rem 1rem;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("💼 AI Resume Intelligence SaaS")
st.write("Startup-grade Resume ATS + AI Career Analyzer")

# -----------------------------
# INPUTS
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("🧾 Paste Job Description")

run = st.button("🚀 Run AI Analysis")

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_words(text):
    return re.findall(r'\b\w+\b', text.lower())

def categorize(words):
    tech = {"python","java","sql","aws","docker","git","linux"}
    ml = {"machine","learning","deep","neural","nlp","pandas","numpy","tensorflow","pytorch","sklearn"}
    soft = {"communication","leadership","team","management"}

    counts = Counter()

    for w in words:
        if w in tech:
            counts["Tech"] += 1
        elif w in ml:
            counts["AI/ML"] += 1
        elif w in soft:
            counts["Soft"] += 1

    return counts

def job_predictor(text):
    t = text.lower()
    if "machine" in t or "ai" in t or "data" in t:
        return "AI/ML Engineer"
    elif "frontend" in t:
        return "Frontend Developer"
    elif "backend" in t:
        return "Backend Developer"
    return "Software Engineer"

def ai_feedback(score):
    if score > 0.75:
        return "🔥 Strong ATS Profile (High Hiring Chance)"
    elif score > 0.5:
        return "⚠️ Good Profile (Needs Optimization)"
    else:
        return "❌ Weak Profile (Major Improvements Needed)"

def make_chart(skill_counts):
    labels = list(skill_counts.keys())
    values = list(skill_counts.values())

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    ax.set_title("Skill Distribution")
    return fig

# -----------------------------
# MAIN
# -----------------------------
if run and uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    semantic_score = cosine_similarity([resume_emb], [job_emb])[0][0]

    resume_words = set(clean_words(resume_text))
    job_words = set(clean_words(job_description))

    overlap = resume_words & job_words
    missing = job_words - resume_words

    skill_score = len(overlap) / (len(job_words) + 1)

    final_score = (0.65 * semantic_score) + (0.35 * skill_score)

    predicted_role = job_predictor(job_description)
    skill_counts = categorize(clean_words(resume_text))

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("🤖 AI Match", f"{final_score*100:.2f}%")
    col2.metric("🧠 Semantic", f"{semantic_score*100:.2f}%")
    col3.metric("📊 ATS Score", f"{skill_score*100:.2f}%")

    st.progress(float(final_score))

    st.success(ai_feedback(final_score))

    st.info(f"🎯 Predicted Role: {predicted_role}")

    # -----------------------------
    # SKILL GAP
    # -----------------------------
    st.markdown("## 📌 Missing Skills")
    st.write(list(missing)[:25])

    # -----------------------------
    # CHART
    # -----------------------------
    st.markdown("## 📊 Skill Breakdown")
    st.pyplot(make_chart(skill_counts))

    # -----------------------------
    # RAW TEXT
    # -----------------------------
    with st.expander("📄 View Resume Text"):
        st.text(resume_text[:4000])

    # -----------------------------
    # PDF REPORT GENERATOR (SIMPLE)
    # -----------------------------
    st.download_button(
        label="📥 Download Report (Text)",
        data=f"""
AI Resume Report

Match Score: {final_score*100:.2f}%
Semantic Score: {semantic_score*100:.2f}%
ATS Score: {skill_score*100:.2f}%

Predicted Role: {predicted_role}

Missing Skills:
{list(missing)[:30]}
""",
        file_name="resume_report.txt"
    )

else:
    st.info("Upload resume and click RUN to generate AI analysis 🚀")

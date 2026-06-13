import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Intelligence",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# PREMIUM UI STYLE (CLEAN SAAS)
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(120deg, #f8bbd0, #fce4ec, #f3e5f5);
}

/* Title */
h1 {
    text-align: center;
    color: #6a1b9a;
    font-weight: 900;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #8e24aa, #ec407a);
    color: white;
    border-radius: 10px;
    font-size: 16px;
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
st.title("💼 AI Resume Intelligence Platform")
st.write("Analyze Resume vs Job Description with ATS-level intelligence")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")
show_raw = st.sidebar.checkbox("Show Raw Text")
show_keywords = st.sidebar.checkbox("Show Keywords")

# -----------------------------
# INPUT
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("🧾 Paste Job Description")

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
    return set(text.lower().replace("\n", " ").split())

def ats_score(resume_text, job_text):
    resume_words = clean_words(resume_text)
    job_words = clean_words(job_text)

    overlap = resume_words & job_words
    missing = job_words - resume_words

    skill_score = len(overlap) / (len(job_words) + 1)
    gap_score = len(missing) / (len(job_words) + 1)

    return skill_score, gap_score, missing

def ai_feedback(score):
    if score > 0.75:
        return "🔥 Excellent ATS Match – Strong Hire Potential"
    elif score > 0.5:
        return "⚠️ Good Match – Improve Some Skills"
    else:
        return "❌ Weak Match – Resume Needs Optimization"

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    # Embeddings
    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    semantic_score = cosine_similarity([resume_emb], [job_emb])[0][0]

    skill_score, gap_score, missing_skills = ats_score(resume_text, job_description)

    # FINAL SCORE (SMART WEIGHTING)
    final_score = (0.6 * semantic_score) + (0.4 * skill_score)

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🤖 AI Match Score", f"{round(final_score*100,2)}%")

    with col2:
        st.metric("🧠 Semantic Score", f"{round(semantic_score*100,2)}%")

    with col3:
        st.metric("📊 ATS Skill Score", f"{round(skill_score*100,2)}%")

    st.progress(float(final_score))

    st.markdown("## 💡 AI Feedback")
    st.success(ai_feedback(final_score))

    # -----------------------------
    # SKILL GAP
    # -----------------------------
    st.markdown("## 📌 Missing Skills (AI Detected)")
    st.write(list(missing_skills)[:25])

    # -----------------------------
    # RAW DATA (OPTIONAL)
    # -----------------------------
    if show_raw:
        st.markdown("## 📄 Resume Text")
        st.text(resume_text[:3000])

    if show_keywords:
        st.markdown("## 🔑 Keywords Extracted")
        st.write(list(clean_words(resume_text))[:50])

# -----------------------------
# EMPTY STATE
# -----------------------------
else:
    st.info("Upload resume and paste job description to start analysis 🚀")

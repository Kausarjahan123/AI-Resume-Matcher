import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("🧠 AI Resume Matcher")
st.write("Upload your resume and compare it with a job description")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Paste Job Description")

if uploaded_file and job_description:

    # Extract resume text
    with pdfplumber.open(uploaded_file) as pdf:
        resume_text = ""
        for page in pdf.pages:
            resume_text += page.extract_text() or ""

    # Embeddings
    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    # Similarity score
    score = cosine_similarity([resume_emb], [job_emb])[0][0]

    st.subheader("Match Score")
    st.write(f"{round(score * 100, 2)} %")

    # Result interpretation
    if score > 0.7:
        st.success("Strong Match 💚")
    elif score > 0.4:
        st.warning("Moderate Match ⚠️")
    else:
        st.error("Weak Match ❌")

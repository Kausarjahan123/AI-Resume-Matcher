# AI Resume Matcher

AI Resume Matcher is a machine learning-based web application that evaluates the compatibility between a candidate’s resume and a job description. It uses natural language processing and transformer-based embeddings to compute semantic similarity and generate an ATS-style match report.

---

## Live Demo

https://ai-resume-matcher-ms57xeu8lvcn3f227u4vnh.streamlit.app/

---

## Project Overview

This project automates the resume screening process by comparing resumes with job descriptions using AI techniques. It provides an estimated match score along with skill gap analysis to help understand candidate suitability.

The system simulates modern Applicant Tracking Systems (ATS) used in recruitment pipelines.

---

## Key Features

### Resume Processing
- PDF resume text extraction
- Text cleaning and preprocessing
- Structured skill identification

### AI Matching Engine
- SentenceTransformer embeddings
- Cosine similarity calculation
- Hybrid scoring (semantic + keyword matching)

### ATS Analysis
- Resume-to-job match score
- Skill gap detection
- Missing keyword identification
- Final candidate evaluation score

### Output Dashboard
- Clean Streamlit interface
- Real-time analysis results
- Structured ATS-style report

---

## Tech Stack

- Python
- Streamlit
- SentenceTransformers (all-MiniLM-L6-v2)
- Scikit-learn
- NumPy
- PDFplumber

---

## System Architecture

Resume (PDF)
→ Text Extraction
→ Preprocessing
→ Transformer Embeddings
→ Job Description Embeddings
→ Cosine Similarity
→ ATS Scoring Engine
→ Streamlit UI Output

---

## Project Structure
ai-resume-matcher/
│
├── app.py
├── requirements.txt
├── README.md
│
└── AI_Resume_Matcher.ipynb

---

## How It Works

1. Upload a resume in PDF format
2. Paste a job description
3. System extracts and processes text
4. AI generates embeddings for both inputs
5. Cosine similarity is computed
6. ATS scoring logic evaluates match strength
7. Results are displayed in a structured dashboard

---

## Core Concepts Used

- Natural Language Processing (NLP)
- Transformer-based embeddings
- Cosine similarity
- Feature extraction from text
- Rule-based ATS scoring system
- Hybrid ML evaluation approach

---

## Future Improvements

- GPT-based AI feedback engine
- Multi-resume ranking system
- Resume improvement suggestions
- Database storage for candidates
- User authentication system
- Downloadable ATS report (PDF)

---

## Author

Kausar Jahan  
M.Tech Computer Science  
AI/ML Portfolio Project

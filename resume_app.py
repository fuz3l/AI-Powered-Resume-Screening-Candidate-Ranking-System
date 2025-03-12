import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import subprocess
import plotly.express as px

# Ensure spaCy model is installed (fix for Streamlit deployment)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Expanded skill set (Can be modified or fetched from a database)
SKILL_SET = {
    "Python", "JavaScript", "Machine Learning", "Data Science", "Artificial Intelligence",
    "Scikit Learn", "Deep Learning", "TensorFlow", "NLP", "SQL"
}

def preprocess_text(text):
    """Cleans text by removing special characters and extra spaces."""
    return re.sub(r'[^\w\s]', '', text).lower().strip()

def extract_skills(text):
    """Extracts skills from text using keyword matching."""
    doc = nlp(text)
    return {token.text for token in doc if token.text.lower() in {skill.lower() for skill in SKILL_SET}}

def compare_skills(job_desc, resume_text):
    """Compares skills in job description and resume."""
    job_skills = extract_skills(preprocess_text(job_desc))
    resume_skills = extract_skills(preprocess_text(resume_text))
    return job_skills & resume_skills, job_skills - resume_skills

def extract_text_from_pdf(file):
    """Extracts text from PDF file."""
    pdf = PdfReader(file)
    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip() if text else "No text extracted"

def rank_resumes(job_description, resumes):
    """Ranks resumes based on job description similarity."""
    vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], vectors[1:]).flatten()

# Streamlit UI
st.title("🚀 AI Resume Screening & Candidate Ranking System")

# Job Description Input
st.header("📌 Job Description")
job_description = st.text_area("Enter the job description")

# File Uploader
st.header("📂 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")

    resumes, extracted_skills_data = [], []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

        # Skill Matching
        matched_skills, missing_skills = compare_skills(job_description, text)
        extracted_skills_data.append((file.name, matched_skills, missing_skills))

    # Rank Resumes
    scores = rank_resumes(job_description, resumes)

    # Display Ranked Results
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.dataframe(results)  # More interactive table

    # Skill Analysis
    st.header("📝 Detailed Resume Skill Analysis")
    for file_name, matched_skills, missing_skills in extracted_skills_data:
        st.subheader(f"📄 Resume: {file_name}")
        st.write(f"✅ **Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
        st.write(f"❌ **Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

    # Visualization
    st.header("📊 Candidate Ranking Visualization")
    fig = px.bar(results, x="Resume", y="Score", title="Candidate Ranking", text="Score", color="Score")
    st.plotly_chart(fig)

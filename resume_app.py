import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Expanded skill set (Can be modified or fetched from a database)
SKILL_SET = {"Python", "JavaScript", "Machine Learning", "Data Science", "Artificial Intelligence",  "Scikit Learn", "Deep Learning", "TensorFlow", "NLP", "SQL"}

def preprocess_text(text):
    """Cleans text by removing special characters and extra spaces."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()  # Convert to lowercase
    return text

def extract_skills(text):
    """Extracts skills from text using keyword matching."""
    doc = nlp(text)
    extracted_skills = set()

    for token in doc:
        token_text = token.text.lower()  # Case insensitive matching
        if token_text in [skill.lower() for skill in SKILL_SET]:  
            extracted_skills.add(token.text)

    return extracted_skills

def compare_skills(job_desc, resume_text):
    """Compares skills in job description and resume."""
    job_desc = preprocess_text(job_desc)
    resume_text = preprocess_text(resume_text)

    job_skills = extract_skills(job_desc)
    resume_skills = extract_skills(resume_text)

    missing_skills = job_skills - resume_skills
    matched_skills = job_skills & resume_skills

    return matched_skills, missing_skills

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    extracted_skills_data = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

        # Skill Matching
        matched_skills, missing_skills = compare_skills(job_description, text)
        extracted_skills_data.append((file.name, matched_skills, missing_skills))

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)

    # Display detailed skill analysis
    st.header("Detailed Resume Skill Analysis")
    for file_name, matched_skills, missing_skills in extracted_skills_data:
        st.subheader(f"Resume: {file_name}")
        st.write(f"✅ **Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
        st.write(f"❌ **Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

import plotly.express as px
fig = px.bar(results, x="Resume", y="Score", title="Candidate Ranking")
st.plotly_chart(fig)

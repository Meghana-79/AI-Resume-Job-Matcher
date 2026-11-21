import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="🎯", layout="wide")

SKILLS = [
    "python", "java", "c++", "javascript", "typescript", "html", "css",
    "react", "node.js", "sql", "mysql", "mongodb", "flask", "django",
    "streamlit", "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "data analysis", "data science", "git", "github",
    "linux", "rest api", "docker", "aws", "azure", "power bi", "tableau",
    "excel", "communication", "problem solving"
]

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def find_skills(text):
    normalized = normalize(text)
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9+#.])" + re.escape(skill) + r"(?![a-z0-9+#.])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(set(found))

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([resume_text, job_text])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)

st.title("🎯 AI-Powered Resume & Job Matcher")
st.caption("A student-built NLP project that compares a resume with a job description.")
st.info("This tool provides an indicative text-similarity score. It is not an automated hiring decision.")

with st.sidebar:
    st.header("About this project")
    st.write("Uses PDF text extraction, TF-IDF, cosine similarity, and keyword-based skill matching.")
    st.write("**Privacy:** Your uploaded resume is processed during this session and is not intentionally saved by this app.")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area(
        "Paste the job description",
        height=220,
        placeholder="Example: Looking for a Python developer with machine learning, SQL, Git and communication skills..."
    )

if st.button("Analyze Resume", type="primary", use_container_width=True):
    if not resume_file:
        st.error("Please upload a PDF resume.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            resume_text = extract_pdf_text(resume_file)
            if len(resume_text) < 30:
                st.error("Could not extract enough text. Please use a text-based PDF rather than a scanned image.")
            else:
                score = calculate_similarity(resume_text, job_description)
                resume_skills = set(find_skills(resume_text))
                job_skills = set(find_skills(job_description))
                matched = sorted(resume_skills & job_skills)
                missing = sorted(job_skills - resume_skills)

                st.divider()
                st.subheader("Analysis results")
                a, b, c = st.columns(3)
                a.metric("Text similarity", f"{score:.1f}%")
                b.metric("Matched skills", len(matched))
                c.metric("Skills to consider", len(missing))

                st.progress(min(int(score), 100))
                left, right = st.columns(2)
                with left:
                    st.markdown("#### ✅ Skills found in both")
                    if matched:
                        st.write(", ".join(matched))
                    else:
                        st.write("No exact skill keyword matches found.")
                with right:
                    st.markdown("#### 📌 Job skills not detected")
                    if missing:
                        st.write(", ".join(missing))
                    else:
                        st.write("All detected job-description skills also appear in the resume.")

                st.markdown("#### 💡 Suggestions")
                suggestions = []
                if missing:
                    suggestions.append("If you genuinely have experience with these skills, add them clearly to the relevant project or skills sections: " + ", ".join(missing) + ".")
                if score < 35:
                    suggestions.append("Tailor your project descriptions and summary to the responsibilities and terminology in the job description.")
                suggestions.append("Use measurable project outcomes where possible, and never add skills or experience you do not have.")
                for item in suggestions:
                    st.write("• " + item)

                with st.expander("View extracted resume text"):
                    st.text(resume_text[:12000])
        except Exception as exc:
            st.error(f"Unable to analyze this PDF: {exc}")

st.divider()
st.caption("Built with Python, Streamlit, PyPDF, scikit-learn and TF-IDF.")

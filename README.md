# AI-Powered Resume & Job Matcher

A beginner-friendly NLP web application that compares a PDF resume with a job description.

## Features
- Extracts text from text-based PDF resumes.
- Calculates TF-IDF cosine similarity between the resume and job description.
- Detects technical and professional skill keywords.
- Shows matched skills and skills not detected in the resume.
- Provides practical resume-improvement suggestions.

> The similarity score is an indicative text-similarity measure, not a prediction of employability or an automated hiring decision.

## Tech stack
- Python
- Streamlit
- PyPDF
- scikit-learn
- TF-IDF and cosine similarity

## Run locally

1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app:

```bash
streamlit run app.py
```

6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploy for free with Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Select **Create app** and choose this repository.
4. Set the main file path to `app.py`.
5. Deploy.

## How it works

1. `pypdf` extracts text from the uploaded resume.
2. `TfidfVectorizer` converts the resume and job description into TF-IDF feature vectors.
3. Cosine similarity measures how similar the two text vectors are.
4. A small editable keyword list detects skills and identifies job-description skills not found in the resume.

## Limitations
- Scanned/image-only PDFs may not produce extractable text; OCR is not included.
- Skill detection uses a predefined keyword list and may miss synonyms or new technologies.
- TF-IDF measures text similarity, not actual candidate ability.
- This is a portfolio/learning project, not a production hiring system.

## Future improvements
- Add OCR for scanned resumes.
- Add synonym-aware skill extraction.
- Support DOCX resumes.
- Add downloadable analysis reports.
- Add multilingual resume support.

## Author
Add your name, course, college, and GitHub profile here before publishing.

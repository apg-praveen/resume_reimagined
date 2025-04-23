import os
import re
import fitz
import tempfile
import pandas as pd
import streamlit as st
import google.generativeai as genai
import kagglehub

# ✅ Streamlit configuration (Mason colors + dark mode)
st.set_page_config(page_title="AI Career Navigator", page_icon="🎓", layout="wide")

# ✅ Custom CSS for George Mason University theme + layout polish
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #0B3D91;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .highlight {
            background-color: #1a1a1a;
            padding: 16px;
            border-radius: 10px;
            border-left: 6px solid #FFD200;
            color: #ffffff;
            text-align: justify;
        }
        .stSelectbox > div, .stTextArea label, .stTextInput label, .stMarkdown h3, .stMarkdown h4, .stMarkdown h2 {
            color: #FFD200 !important;
        }
        .stButton > button {
            background-color: #FFD200;
            color: black;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Gemini API key setup
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBHeirb_J5_EFU6Itzy49j93xAQ__aiFnM"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_16LQ2WnnHYhDF5W3DHZgIlVPEA5F4Po"
os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ PDF text extraction
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    try:
        with fitz.open(temp_file_path) as doc:
            return " ".join([page.get_text("text") for page in doc if page.get_text("text")])
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        return ""

# ✅ Gemini category prediction
def predict_job_category(resume_text):
    prompt = f"""
    Classify the following resume as either INFORMATION-TECHNOLOGY or BUSINESS-DEVELOPMENT.
    Only return one of the two labels.

    Resume:
    {resume_text[:1500]}
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        result = response.text.strip().upper()

        if "INFORMATION" in result:
            return "INFORMATION-TECHNOLOGY"
        elif "BUSINESS" in result:
            return "BUSINESS-DEVELOPMENT"
        else:
            return "UNKNOWN"
    except Exception as e:
        st.error(f"Gemini classification failed: {e}")
        return "UNKNOWN"

# ✅ Gemini AI response
def get_ai_response(user_input, resume_text, job_category):
    prompt = f"""
    Act as a career coach. Based on this resume and category '{job_category}', answer:

    Resume:
    {resume_text[:1500]}

    Query:
    {user_input}

    Return response in this format:
    Heading
    • Bullet 1
    • Bullet 2
    Next Heading
    • Bullet 1
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        if not response or not response.text:
            return "⚠️ No response generated."

        text = response.text
        text = text.replace("**", "").replace("- ", "• ").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"

# ✅ Suggest queries
def suggest_queries(category):
    suggestions = {
        "BUSINESS-DEVELOPMENT": [
            "What career advice can you give based on my resume?",
            "What roles align with my experience in business development?",
            "How can I grow my network in the industry?",
            "What skills do I need to improve for a promotion?"
        ],
        "INFORMATION-TECHNOLOGY": [
            "What career path suits my technical background?",
            "How do I transition to a cloud-related IT role?",
            "What certifications would benefit my IT career?",
            "What tools or platforms should I master based on my resume?"
        ]
    }
    return suggestions.get(category.upper(), [
        "What job roles suit my experience and resume?",
        "What certifications should I pursue next?",
        "How can I position myself better in the job market?",
        "What are my strengths and areas of improvement?"
    ])

# ✅ UI layout
st.title("🎓 GMU AI Career Navigator")

with st.sidebar:
    st.header("📄 Upload Your Resume")
    doc_upload = st.file_uploader("Upload PDF", type=["pdf"])
    st.markdown("Once uploaded, we’ll analyze your resume and provide tailored advice using AI ✨")

if doc_upload:
    resume_text = extract_text_from_pdf(doc_upload)[:1500]
    if resume_text:
        job_category = predict_job_category(resume_text)
        st.success(f"✅ Resume analyzed. Predicted category: `{job_category}`")

        st.subheader("💬 Ask Your Career Question")
        suggestions = suggest_queries(job_category)
        question_option = st.selectbox("💡 Choose a suggested question or type your own:", suggestions + ["Other"], index=0)

        user_input = ""
        if question_option == "Other":
            user_input = st.text_area("✍️ Enter your custom question")
        else:
            user_input = question_option

        if st.button("🔍 Get Career Advice"):
            if user_input:
                with st.spinner("Thinking like a career coach..."):
                    ai_response = get_ai_response(user_input, resume_text, job_category)

                st.subheader("🧠 AI Career Advice")

                lines = ai_response.splitlines()
                current_heading = None
                current_bullets = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("• "):
                        current_bullets.append(line[2:].strip())
                    else:
                        if current_heading:
                            st.markdown(f"### 🔹 {current_heading}")
                            for bullet in current_bullets:
                                st.markdown(f"- {bullet}")
                            st.markdown("---")
                        current_heading = line
                        current_bullets = []

                if current_heading and current_bullets:
                    st.markdown(f"### 🔹 {current_heading}")
                    for bullet in current_bullets:
                        st.markdown(f"- {bullet}")
            else:
                st.warning("Please enter a question to receive advice.")
else:
    st.info("📥 Please upload your resume from the sidebar to begin.")

st.markdown("---")
st.caption("Built at George Mason University | Built by Prince Alikana, Rohan Sardar, Nattawut Promkam | Advised by Dr. Hadi Rezazad")

import os
import re
import fitz
import joblib
import torch
import tempfile
import pandas as pd
import streamlit as st
import google.generativeai as genai
import kagglehub
import requests
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# ✅ Environment keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_16LQ2WnnHYhDF5W3DHZgIlVPEA5F4Po"
os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Caching setup
@st.cache_resource(show_spinner="🔄 Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource(show_spinner="🔄 Loading trained ML model...")
def load_trained_model():
    return joblib.load('career_advice_model.pkl')

@st.cache_resource(show_spinner="🔄 Loading BERT model and tokenizer...")
def load_bert_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

@st.cache_data(show_spinner="📥 Loading resume dataset...")
def load_resume_data(csv_path):
    df = pd.read_csv(csv_path, usecols=["Resume_str", "Category"])
    df = df[df['Category'].isin(['INFORMATION-TECHNOLOGY', 'BUSINESS-DEVELOPMENT'])]
    df['Job_Title'] = df['Resume_str'].str[:50].str.strip()
    df = df.dropna(subset=['Job_Title'])
    return df

@st.cache_data(show_spinner="🧠 Generating embeddings...")
def get_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False, batch_size=32)

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

# ✅ Train model if needed
def train_ml_model():
    texts = df_csv["Resume_str"].values
    labels = df_csv["Category"].values
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    embedding_model = load_embedding_model()
    X_train_embeddings = get_embeddings(embedding_model, X_train)
    X_test_embeddings = get_embeddings(embedding_model, X_test)
    models = {"SVM": SVC(kernel='linear', probability=True), "Random Forest": RandomForestClassifier(n_estimators=200)}
    best_model, best_accuracy = None, 0
    for name, model in models.items():
        model.fit(X_train_embeddings, y_train)
        y_pred = model.predict(X_test_embeddings)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.2f}")
        if acc > best_accuracy:
            best_accuracy, best_model = acc, model
    joblib.dump(best_model, 'career_advice_model.pkl')
    print(f"✅ Model trained and saved with accuracy: {best_accuracy:.2f}")

# ✅ Load dataset
path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
csv_file_path = path + "/Resume/Resume.csv"
df_csv = load_resume_data(csv_file_path)

if not os.path.exists("career_advice_model.pkl"):
    with st.spinner("🛠️ Training model (first run only)..."):
        train_ml_model()

embedding_model = load_embedding_model()
model = load_trained_model()
bert_tokenizer, bert_model = load_bert_model_and_tokenizer()

# ✅ Predict category
def predict_job_category(resume_text):
    text_embedding = embedding_model.encode([resume_text], show_progress_bar=False)
    return model.predict(text_embedding)[0]

# ✅ Gemini output
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
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)

        if not response or not response.text:
            return "⚠️ No response generated."

        # ✅ Basic string cleanup
        text = response.text
        text = text.replace("**", "")
        text = text.replace("- ", "• ")
        text = text.strip()
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

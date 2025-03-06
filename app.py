import streamlit as st
import pickle
import os

# Load the trained model and vectorizer
model_path = "models/spam_classifier.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model or vectorizer file missing. Train the model first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Set page configuration
st.set_page_config(page_title="Spam Detector", layout="centered")

# Modernized Title
st.markdown(
    """
    <h1 style='text-align: center; color: #3349ff;'>📧 Email Spam Detector</h1>
    <p style='text-align: center; color: #7F8C8D;'>Enter an email below to check if it's Spam or Not.</p>
    """,
    unsafe_allow_html=True,
)

# Modernized Text Box
user_input = st.text_area(
    "✍️ Email Content:", 
    height=150, 
    placeholder="Type or paste your email here...",
)

# Predict Button with Styling
if st.button("🚀 Predict", use_container_width=True):
    if user_input.strip():
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        result = "🚨 Spam" if prediction == 1 else "✅ Not Spam"
        result_color = "#E74C3C" if prediction == 1 else "#27AE60"

        st.markdown(
            f"<h3 style='text-align: center; color: {result_color};'>{result}</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Please enter some text before predicting.")

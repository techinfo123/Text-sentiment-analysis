import streamlit as st
from transformers import pipeline

# Load sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit app
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Text Sentiment Analyzer")

st.markdown("Enter text below to analyze its sentiment using a Hugging Face model.")

# Text input
user_input = st.text_area("Text to analyze", height=150)

# Run analysis
if st.button("Analyze") and user_input.strip():
    with st.spinner("Analyzing..."):
        result = sentiment_pipeline(user_input)[0]
        st.markdown(f"**Sentiment:** `{result['label']}`")
        st.markdown(f"**Confidence:** `{result['score']:.4f}`")

st.markdown("---")
st.caption("Powered by Hugging Face 🤗 and Streamlit 🚀")

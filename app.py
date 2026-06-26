import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# 2. Cache Model Loading (Ensures it only loads into memory once on startup)
@st.cache_resource
def load_local_model():
    model_path = "./movie_sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

try:
    tokenizer, model = load_local_model()
except Exception as e:
    st.error(f"Could not load model weights. Verify the folder path. Error: {e}")
    st.stop()

# 3. Streamlit Simple UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Input a review below to evaluate its positive or negative orientation with exact confidence scores.")
st.write("---")

# User Input Text Area
user_review = st.text_area("Enter Review Text:", placeholder="Type your movie review here...", height=120)

# Run Inference on Button Click
if st.button("Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Computing probabilities..."):
            # Step A: Tokenize raw input text
            inputs = tokenizer(user_review, padding=True, truncation=True, max_length=256, return_tensors="pt")
            
            # Step B: Model Forward Pass (No gradients needed for inference)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Step C: Convert raw logits to probability scores via Softmax
                probabilities = F.softmax(logits, dim=1).flatten().tolist()
            
            neg_score = probabilities[0]
            pos_score = probabilities[1]
            
            # Determine Dominant Class
            if pos_score > neg_score:
                st.success(f"**Result: POSITIVE** ({pos_score * 100:.2f}% Confidence)")
            else:
                st.error(f"**Result: NEGATIVE** ({neg_score * 100:.2f}% Confidence)")
            
            # Display Full Distribution Layout
            st.write("### Probability Breakdown:")
            st.write(f"Positive Intensity: {pos_score * 100:.2f}%")
            st.progress(pos_score)
            
            st.write(f"Negative Intensity: {neg_score * 100:.2f}%")
            st.progress(neg_score)
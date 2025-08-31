import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import get_config
from src.predict import predict_single_sentence

@st.cache_resource
def load_model_and_dependencies():
    """Tải model, tokenizer và config"""
    local_model_path = "models/final_model"
    if not os.path.exists(local_model_path):
        st.error("Không tìm thấy model tại '{local_model_path}'")
        return None, None, None

    config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

    print("Model, tokenizer và config đã được tải thành công")
    return model, tokenizer, config

st.set_page_config(page_title="Sentiment Analysis App", layout="centered", page_icon=":memo:")
st.title("Sentiment Analysis App")
st.write(
    "This is a sentiment analysis app that uses a model which was finetuned by LoRA on the Sentiment dataset to predict the sentiment of a sentence."
)
st.markdown("---")

model, tokenizer, config = load_model_and_dependencies()

if model is None or tokenizer is None or config is None:
    st.error("Lỗi! Không tìm thấy model.")
else:
    # Tạo form để nhập liệu
    with st.form(key="sentiment_form"):
        user_input = st.text_area(
            "Enter your sentence here...",
            "",
            height=100,
            placeholder="e.g., This movie was amazing!"       
        )
        submitted = st.form_submit_button("Predict")
    
    # Xử lý khi người dùng nhấn nút
    if submitted and user_input:
        with st.spinner("Predicting..."):
            prediction = predict_single_sentence(model, tokenizer, config, user_input)
        
        # Hiển thị kết quả
        st.success(f"The predicted sentiment is: {prediction}")
    elif submitted and not user_input:
        st.warning("Please enter a sentence to predict.")

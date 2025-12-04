import streamlit as st
import joblib
import numpy as np

# Load model & vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

label_map = {0: "FAKE", 1: "REAL"}

def predict_news(text):
    clean_text = text.lower()
    # (You can reuse your full clean_text function here)
    vec = tfidf.transform([clean_text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return label_map[pred], float(np.max(prob))

st.title("📰 Fake vs Real News Classifier")
st.write("Enter a news article or headline and I’ll predict if it is likely **Fake** or **Real**.")

user_input = st.text_area("Paste news text here:", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        label, confidence = predict_news(user_input)
        if label == "FAKE":
            st.error(f"Prediction: {label} (confidence: {confidence:.2f})")
        else:
            st.success(f"Prediction: {label} (confidence: {confidence:.2f})")

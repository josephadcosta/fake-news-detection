import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return " ".join(words)

# UI Design
st.title("📰 Fake News Detection System")
st.write("Enter a news article and click Detect")

# Input box
user_text = st.text_area("Enter News Text")

# Button
if st.button("Detect News"):
    if user_text:

        cleaned = clean_text(user_text)

        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)

        probability = model.predict_proba(vectorized)

        confidence = max(probability[0]) * 100

        if prediction[0] == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.write("Confidence:", round(confidence, 2), "%")

    else:
        st.warning("Please enter text first")
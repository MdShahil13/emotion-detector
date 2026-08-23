import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Emotion mapping
emotion_names = {
    0: "sadness 😢",
    1: "anger 😠",
    2: "love ❤️",
    3: "surprise 😲",
    4: "fear 😨",
    5: "joy 😄"
}

# Page configuration
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊"
)

st.title("😊 Emotion Detector")
st.write("Enter a sentence and the AI will predict its emotion.")

# User input
sentence = st.text_area(
    "Enter your sentence:",
    placeholder="Example: I am very happy today!"
)

# Predict button
if st.button("Predict Emotion"):

    if sentence.strip() == "":
        st.warning("Please enter a sentence.")

    else:
        # Convert sentence to TF-IDF
        sentence_tfidf = vectorizer.transform([sentence])

        # Predict
        prediction = model.predict(sentence_tfidf)[0]

        # Convert number to emotion
        emotion = emotion_names[prediction]

        # Display result
        st.success(f"Predicted Emotion: {emotion}")
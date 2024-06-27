import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Function to load Sklearn models
def load_model_sklearn(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to load Deep Learning models
def load_model_dl(model_path):
    model = load_model(model_path)
    return model

# Load the TfidfVectorizer
with open('Models/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the Tokenizer
with open('Models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Preprocessing Function for Deep Learning Models
def preprocess_text_for_dl_model(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=1000, padding='post')
    return padded_sequences

# Dictionary of models
models = {
    "Linear SVM": "Models/linear_svm.pkl",
    "Logistic Regression": "Models/logistic_regression.pkl",
    "Naive Bayes": "Models/naive_bayes.pkl",
    "Random Forest": "Models/random_forest.pkl",
    "LSTM": "Models/lstm_model.keras",
    "Simple RNN": "Models/simple_rnn_model.keras"
}

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Streamlit UI
st.title("Fake News Classifier")

# User input
title = st.text_input("Title of the news")
description = st.text_area("Description of the news")
model_name = st.selectbox("Choose a model", list(models.keys()))

if st.button("Classify"):
    if title and description:

        # Vectorize the input text for sklearn models
        if model_name in ["Linear SVM", "Logistic Regression", "Naive Bayes", "Random Forest"]:
            text = title
            model = load_model_sklearn(models[model_name])
            text_vectorized = vectorizer.transform([text])
            binary_prediction = model.predict(text_vectorized)[0]
            print(binary_prediction)

        # Preprocess and predict for deep learning models
        else:
            text = title + " " + description
            model = load_model_dl(models[model_name])
            text_preprocessed = preprocess_text_for_dl_model(text)
            prediction = model.predict(text_preprocessed)
            print(prediction)
            binary_prediction = (prediction >= 0.5).astype(int)
            print(binary_prediction)

        # Display result
        result = "Fake" if binary_prediction == 1 else "True"
        st.write(f"The news is: {result}")
    else:
        st.write("Please enter both title and description.")

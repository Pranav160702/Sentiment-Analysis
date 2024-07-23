# Step 1: Import libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# load the IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}


# Load Pretrained model with ReLU Activation
model = load_model('simple_rnn_imdb.h5')


# Step 2: Helper Function
# Function to decode Reviews
def decode_reviews(encoded_reviews):
    return ' '.join([reversed_word_index.get(i - 3,'?')for i in encoded_reviews])

# Function to Preprocess User Input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    encoded_review = [index if index < 10000 else 2 for index in encoded_review]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# # Step 3: Prediction Function
# def predict_sentiment(review):
#     preprocessed_input = preprocess_text(review)

#     prediction = model.predict(preprocessed_input)

#     sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

#     return sentiment, prediction[0][0]


# *********************************************************************

# #Streamlit App
import streamlit as st

st.write('IMDB Movie Review Sentiment Analysis')
st.write('Enter a Movie Review to Classify it as Positive or Negative.')

# User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    # Make Prediction
    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"


    #  Display Results
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please Enter a Movie Review')


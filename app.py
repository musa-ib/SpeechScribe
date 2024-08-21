import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

# Define your mappings and load the model
mappings = ["bed","bird","cat","dog","down","eight","five","four","go","happy",
            "house","left","marvin","nine","no","off","on","one","right","seven","sheila",
            "six","stop","three","tree","two","up","wow","yes","zero"]

keras_model = keras.models.load_model("SpeechCom.keras")

def preprocess(file):
    signal, sr = librosa.load(file)
    current_length = len(signal)
    desired_length = 22050
    if current_length < desired_length:
        padding = desired_length - current_length
        signal = np.pad(signal, (0, padding), 'constant')
    elif current_length > desired_length:
        signal = signal[:desired_length]
    
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, hop_length=512, n_fft=2048)
    mfccs = mfccs.T.tolist()
    mfccs = np.array(mfccs)
    input_data = mfccs[np.newaxis, ..., np.newaxis]
    return input_data

# Streamlit app
st.title("Speech Recognition")
st.write("Upload an audio file to get the word.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Preprocess the file and make prediction
    x = preprocess(uploaded_file)
    p = keras_model.predict(x)
    p_i = p.argmax()
    st.write(f"Recognized word: {mappings[p_i]}")

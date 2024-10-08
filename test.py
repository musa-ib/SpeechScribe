import bentoml
import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa

mappings = ["bed","bird","cat","dog","down","eight","five","four","go","happy",
           "house","left","marvin","nine","no","off","on","one","right","seven","sheila",
            "six","stop","three","tree","two","up","wow","yes","zero"]

keras_model = keras.models.load_model("D:\Speech\SpeechCom.keras")


def preprocess(file):
    # Load the audio file
    signal, sr = librosa.load(file)
    print(len(signal))
    
    # Preprocess the audio
    current_length = len(signal)
    desired_length = 22050
    if current_length < desired_length:
        padding = desired_length - current_length
        # Pad with zeros
        signal = np.pad(signal, (0, padding), 'constant')
    elif current_length > desired_length:
        # Optionally, you can trim the signal if it's longer than desired length
        signal = signal[:desired_length]
    
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, hop_length=512, n_fft=2048)
    mfccs = mfccs.T.tolist()
    mfccs = np.array(mfccs)
    input_data = mfccs[np.newaxis, ..., np.newaxis]
    print(input_data.shape)

    # Run the model prediction
    # Return the prediction
    return input_data
file_path = '00b01445_nohash_0.wav'
x = preprocess(file_path)
p = keras_model.predict(x)
p_i = p.argmax()
print(f"predicted word: {mappings[p_i]}")




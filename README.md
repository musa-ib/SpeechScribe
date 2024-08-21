# SpeechScribe
SpeechScribe is a Python-based application that uses Convolutional Neural Networks (CNNs) to transcribe audio files by recognizing the spoken words.
Automatic Speech Recognition: SpeechScribe uses Mel-Frequency Cepstral Coefficients (MFCC) feature extraction and a CNN model to convert audio files into text transcripts.
# How it Works
MFCC Feature Extraction: SpeechScribe uses the LIBROSA library to extract Mel-Frequency Cepstral Coefficients (MFCC) from the input audio files. MFCC features capture the unique characteristics of the speech signal and are commonly used in speech recognition tasks.
CNN Model: The extracted MFCC features are fed into a Convolutional Neural Network (CNN) model that has been trained to recognize the spoken words. The CNN model learns the patterns in the MFCC features and maps them to the corresponding text transcripts.
# Steps to run this project:
1. Clone the repository:
Copygit clone https://github.com/musa-ib/SpeechScribe.git

2. Install the required dependencies:
cd SpeechScribe
pip install -r requirements.txt
3. Download dataset :
   https://research.google/blog/launching-the-speech-commands-dataset/
4. Run prepare_dataset.py and data.json will be created in current folder
5. Run train.py to train CNN model and save the model
6. Test model by running test.py file
7. Run the app:
   In command propmt type streamlit run app.py
8. Alternatively, you can run SpeechScribe in a Docker container:

Build the Docker image:
docker build -t speechscribe .

Run the Docker container:
docker run -d 8501:8501 speechscribe

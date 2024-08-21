import librosa
import numpy as np
import json
import os

dataset_path = 'C:/Users/Lenovo/Documents/Deep Learning Deploy/speech_commands_v0.01'
json_path = 'data.json'
samples = 22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
          category = dirpath.split('/')[-1]
          data['mapping'].append(category)
          print(f'Processing: {category}')

          for f in filenames:
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path)

            if len(signal) >= samples:
              signal = signal[:samples]

              mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc,  hop_length=hop_length, n_fft=n_fft)

              data['labels'].append(i-1)
              data['mfcc'].append(mfccs.T.tolist())
              data['files'].append(file_path)
              print(f'{file_path}: {i-1}')

    with open(json_path, 'w') as fp:
      json.dump(data, fp, indent=4)
      
prepare_dataset(dataset_path, json_path)
import numpy as np 
import bentoml
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
json_data = 'D:\Speech\data.json'

def data_splits(json_path):
  with open(json_path, 'r') as fp:
    data = json.load(fp)
  print(data.keys())

  X = np.array(data['mfcc'])
  y = np.array(data['labels'])

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
  X_train = X_train[..., np.newaxis]
  X_val = X_val[..., np.newaxis]
  X_test = X_test[..., np.newaxis]
  return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = data_splits(json_data)
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

inputs = keras.Input(shape=input_shape)
x = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)

outputs = keras.layers.Dense(30, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=45)

test_error, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Error: {test_error}, Test Accuracy: {test_accuracy}')


# keras_model = tf.keras.models.load_model("D:SpeechCom.keras")
bento_model = bentoml.keras.save_model("speech_com_model", model)
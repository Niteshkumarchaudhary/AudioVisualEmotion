"""
Version: 02
Train our LSTM model for video sequence emotion classification.
Nitesh Kumar Chaudhary
niteshku001@e.ntu.edu.sg

"""

import numpy as np
import matplotlib.pyplot as plt
from model_video import LSTM_model, LSTM_model_v1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from data import DataSet
from extractor import Extractor
from tqdm import tqdm
import pickle


data = DataSet(seq_length=60, class_limit=8, image_shape=(720, 1280, 3))
data_type = 'features'
X_train, y_train = data.get_all_sequences_in_memory('train', data_type)
X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
print("Train X: ", X_train.shape)
print("Train y: ", y_train.shape)

print("Test X: ", X_test.shape)
print("Test y: ", y_test.shape)




LSTM_model = LSTM_model_v1((60, 2048))
LSTM_model.summary()



LSTM_history = LSTM_model.fit(X_train, y_train,
                               batch_size=16, epochs=200,
                               validation_data=(X_test, y_test))


predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
new_y_test = y_test.astype(int)
matrix = confusion_matrix(new_y_test, predictions)

print(classification_report(new_y_test, predictions))
print(matrix)



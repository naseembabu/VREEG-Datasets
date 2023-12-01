
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import glob
from glob import glob
import os
import numpy as np
import pandas as pd
import antropy as ant
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Bidirectional, Dropout, Conv1D, MaxPooling1D, BatchNormalization
import mne
import pywt
import pywt.data
from scipy.signal import stft, spectrogram, welch, butter, lfilter
from scipy.stats import kurtosis, t
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import math
from nolds import sampen
import json
import tensorflow as tf
from spektral.layers import GCNConv
import os


all_files_path=glob('eeg_emotions/*.edf')
#print(len(all_files_path))

fear = glob('new_eeg/*.edf')

#fear = [i for i in all_files_path if  'd' in i.split('/')[-1]]
#fear = [i for i in fear1 if  'd' in i.split('/')[-1]]
sad = [i for i in all_files_path if  's' in i.split('/')[-1]]
neutral = [i for i in all_files_path if  'n' in i.split('/')[-1]]
happy = [i for i in all_files_path if  'h' in i.split('/')[-1]]


def read_data(file_path):
    raw=mne.io.read_raw_edf(file_path,preload=True)
    
    # Define the names of the electrodes to select
    electrode_names = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
    # Select the data for the desired electrodes and remove all other channels
    raw.pick_channels(ch_names=electrode_names)
    # Set the EEG 10/20 electrode layout
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)  
    duration = raw.times[-1]		
    # Define the desired duration of the signal in seconds
    T_desired = 180  # seconds
    # Crop the signal to the desired duration
    croped = raw.crop(tmin=0, tmax=T_desired)
    datax = croped	
    datax.set_eeg_reference()
    epochs = mne.make_fixed_length_epochs(datax, duration=1, overlap=0, preload=True)
    return epochs #trials,channel,length

fear_epochs_array=[read_data(subject) for subject in fear]
sad_epochs_array=[read_data(subject) for subject in sad]
neutral_epochs_array=[read_data(subject) for subject in neutral]
happy_epochs_array=[read_data(subject) for subject in happy]

print(len(fear_epochs_array), len(sad_epochs_array), len(neutral_epochs_array), len(happy_epochs_array))


fear_epochs_labels = []
for i in fear_epochs_array:
    if i is not None:
        fear_epochs_labels.append(len(i) * [0])
    else:
        continue

sad_epochs_labels=[]
for i in sad_epochs_array:
    if i is not None:
        sad_epochs_labels.append(len(i) * [1])
    else:
        continue
        
        
neutral_epochs_labels=[]
for i in neutral_epochs_array:
    if i is not None:
        neutral_epochs_labels.append(len(i) * [2])
    else:
        continue


happy_epochs_labels=[]
for i in happy_epochs_array:
    if i is not None:
        happy_epochs_labels.append(len(i) * [3])
    else:
        continue 
print(len(fear_epochs_labels), len(sad_epochs_labels), len(neutral_epochs_labels), len(happy_epochs_labels))

data_list=fear_epochs_array+sad_epochs_array+neutral_epochs_array+happy_epochs_array
label_list=fear_epochs_labels+sad_epochs_labels+neutral_epochs_labels+happy_epochs_labels


data_array=np.vstack(data_list)
label_array=np.hstack(label_list)
#group_array=np.hstack(groups_list)
print(data_array.shape,label_array.shape)


# Load the saved stacked_array
np.save('raw_eeg.npy', data_array)
# Load the saved stacked_array
np.save('raw_eeg_label.npy', label_array)

# Load the saved stacked_array
data_array = np.load('raw_eeg.npy')
# Load the saved stacked_array
label_array = np.load('raw_eeg_label.npy')

#print(data_array.shape)
#print(label_array.shape)
# Generate synthetic EEG data and labels for demonstration
num_segments = data_array.shape[0]
num_channels = data_array.shape[1]
num_samples_in_one_file = data_array.shape[2]
#print(num_segments)
#print(num_samples_in_one_file)
num_classes = 4

# Split the decomposed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, test_size=0.2, random_state=42)

# Flatten the data while keeping the samples
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# Standardization (Z-score normalization)
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_2d)
X_test_normalized = scaler.transform(X_test_2d)

# Reshape the data back to the original shape
X_train_normalized = X_train_normalized.reshape(X_train.shape)
X_test_normalized = X_test_normalized.reshape(X_test.shape)

y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)
# Split the testing data into a new testing set and a validation set
X_test, X_val, y_test, y_val = train_test_split(X_test_normalized, y_test_categorical, test_size=0.2, random_state=42)

input_shape = (X_train_normalized.shape[1], X_train_normalized.shape[2], 1)

model = Sequential([
    Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
])
print(model.summary())

learning_rate = 0.0001  # You can adjust this value as needed

# Create a custom optimizer with the desired learning rate
custom_optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_normalized, y_train_categorical, epochs=100, batch_size=32, validation_data=(X_val, y_val))

with open('2DCNN.json', 'w') as json_file:
    json.dump(history.history, json_file)

model.save('2DCNN')


# Assuming model is saved as 'your_model.h5'
model = tf.keras.models.load_model('2DCNN')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_normalized, y_test_categorical)
print(f"Test Accuracy: {accuracy:.2f}")

# Load the training history
with open('2DCNN.json', 'r') as json_file:
    loaded_history = json.load(json_file)

# Plot training accuracy and validation accuracy curves
plt.plot(loaded_history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(loaded_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
# Customize the y-axis limits
plt.ylim(0.2, 1.2)
# Add a legend to the plot
legend = plt.legend()
for text in legend.get_texts():
    text.set_fontweight('bold')
# Customize the plot
plt.title('Raw EEG signal', fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.savefig('1dcnn_lstm_02.png')
plt.show()
plt.close()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Validation Accuracy:", test_accuracy)

# Make predictions on the test set
test_probabilities = model.predict(X_test)
test_predictions = np.argmax(test_probabilities, axis=1)

# Calculate precision, recall, and F1-score
precision = precision_score(np.argmax(y_test, axis=1), test_predictions, average='macro')
recall = recall_score(np.argmax(y_test, axis=1), test_predictions, average='macro')
f1 = f1_score(np.argmax(y_test, axis=1), test_predictions, average='macro')
miss_rate = 1 - recall

# Print the evaluation metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Miss Rate: {miss_rate:.4f}')
print(f'F1 Score: {f1:.4f}')

# Calculate the confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), test_predictions)
print('Confusion Matrix:')
print(conf_matrix)

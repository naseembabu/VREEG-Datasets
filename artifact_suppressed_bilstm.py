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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Bidirectional, Dropout
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
    #electrode_names = ['Fp1', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'P7', 'P8', 'T7', 'T8', 'O1', 'O2','Fp2']
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
    epochs = mne.make_fixed_length_epochs(datax, duration=0.5, preload=True)
    epochs.filter(l_freq=1, h_freq=None)
    n_components = 32  # Adjust the number of components as needed
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=32, max_iter=1000)
    ica.fit(epochs)
    ica_components = ica.get_components()


    confidence_level = 0.98
    alpha = 1 - confidence_level
    degrees_of_freedom = 31  # Replace with your actual degrees of freedom
    
    # Calculate the critical t-value for a two-tailed test
    t_critical = t.ppf(1 - alpha / 2, degrees_of_freedom)

    sample_entropies = []
    for component in ica_components:
        # Check if the component contains meaningful information
        if np.all(np.abs(component) == 0):
            entropy_value = 0.0  # Set entropy to 0.0 for components with no information
        else:
            # Calculate Sample Entropy
            entropy_value = sampen(component, emb_dim=2, tolerance=0.2)  # Adjust parameters as needed
        sample_entropies.append(entropy_value)
    
    # Calculate the mean and standard deviation of the sample entropy values
    mean_entropy = np.mean(sample_entropies)
    std_entropy = np.std(sample_entropies)
    
    
    margin_of_error = t_critical * (std_entropy / np.sqrt(len(sample_entropies)))
    lower_limit = mean_entropy - margin_of_error  # Lower limit for thresholding
    #if lower_limit==nan:
        #lower_limit = minimum_value
    # Identify ICs with mMSE below the lower limit
    print("lower_limit:", lower_limit)
    blink_artifact_entropy = [idx for idx, entropy in enumerate(sample_entropies) if entropy < lower_limit]

    # Calculate kurtosis for each independent component
    kurtosis_values = []  # Initialize an array to store ICA kurtosis values
    for component in ica_components:
        kurtosis_value = kurtosis(component)
        kurtosis_values.append(kurtosis_value)

    mean_kurtosis = np.mean(kurtosis_values)
    std_kurtosis = np.std(kurtosis_values)    
    

    margin_of_error = t_critical * (std_kurtosis / np.sqrt(len(kurtosis_values)))
    upper_limit = mean_kurtosis + margin_of_error  # Upper limit for thresholding
    # Identify ICs with mMSE below the lower limit
    print("upper_limit:", upper_limit)
    blink_artifact_kurtosis = [idx for idx, kurtosis in enumerate(kurtosis_values) if kurtosis > upper_limit]
    artifactual_components = np.union1d(blink_artifact_kurtosis, blink_artifact_entropy)
    artifactual_components = [int(x) for x in artifactual_components]

    level = 2
    def custom_threshold(data, level, method='hard'):
        # Calculate the threshold k
        N = len(data)
        threshold_multiplier = 0.6745  # Constant for Gaussian noise
        sigma = np.sqrt(np.median(np.abs(data) / threshold_multiplier))
        k = np.sqrt(2 * np.log(N) * sigma)

        # Set the threshold as a multiple of k based on the specified method
        if method == 'hard':
            threshold = k
        elif method == 'soft':
            threshold = k
        else:
            raise ValueError("Invalid thresholding method")
        return threshold

        # Adjust the level as needed
        threshold_method = 'hard'  # Choose 'hard' or 'soft'

    def wavelet_denoising(x, wavelet='bior4.4', level=level, method='hard'):
        threshold = custom_threshold(x, level, method)
        coeff = pywt.wavedec(x, wavelet, mode="per", level=level)
        coeff[1:] = (pywt.threshold(i, value=threshold, mode=method) for i in coeff[1:])
        denoised_ic = pywt.waverec(coeff, wavelet, mode='per')
        return denoised_ic
    
    denoised_components = []
    # For each index of the component to be denoised
    print("artifactual_components:", artifactual_components)
    for idx in artifactual_components:
        # Access the component from ica_components using the index
        original_component = ica_components[idx]      
        denoised_component = wavelet_denoising(original_component)  
        denoised_components.append(denoised_component)
        
        
    # Loop through the artifactual components and replace them with denoised components
    for component_idx, denoised_component in zip(artifactual_components, denoised_components):
        ica.mixing_matrix_[component_idx] = denoised_component
    ica.apply(epochs)
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
np.save('raw_eeg_artifact_suppression.npy', data_array)
# Load the saved stacked_array
np.save('raw_eeg_label_artifact_suppression.npy', label_array)


# Load the saved stacked_array
data_array = np.load('raw_eeg_artifact_suppression.npy')
# Load the saved stacked_array
label_array = np.load('raw_eeg_label_artifact_suppression.npy')

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
print("X_train_normalized:",X_train_normalized.shape)
print("y_train_categorical:",y_train_categorical.shape)

# Build a Bidirectional LSTM (BiLSTM) model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(16, return_sequences=False)),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
])

learning_rate = 0.00001  # You can adjust this value as needed

# Create a custom optimizer with the desired learning rate
custom_optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_normalized, y_train_categorical, epochs=250, batch_size=32, validation_split=0.1)
with open('lstm_training_history_denoise.json', 'w') as json_file:
    json.dump(history.history, json_file)
model.save('lstm_artifact')

model = tf.keras.models.load_model('lstm_artifact')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_normalized, y_test_categorical)
print(f"Test Accuracy: {accuracy:.2f}")

# Load the training history
with open('lstm_training_history_denoise.json', 'r') as json_file:
    loaded_history = json.load(json_file)

# Plot training accuracy curve in blue
plt.plot(loaded_history['accuracy'], label='Training Accuracy', linewidth=3)
plt.plot(loaded_history['val_accuracy'], label='Validation Accuracy', linewidth=3)
plt.ylim(0.29, 1.1)
plt.title('Artifact-suppressed EEG signal', fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.savefig('lstm_artifact_suppresed.png')
plt.show()
plt.close()


X_test = X_test_normalized
y_test = y_test_categorical

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

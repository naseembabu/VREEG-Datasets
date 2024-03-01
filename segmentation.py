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
from scipy.signal import stft, spectrogram, welch, butter, lfilter, filtfilt
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

negative = glob('negative/*.edf')
neutral = glob('neutral/*.edf')
positive = glob('positive/*.edf')

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
    epochs = mne.make_fixed_length_epochs(datax, duration=30, overlap=0, preload=True)
    return epochs #trials,channel,length

negative_epochs_array=[read_data(subject) for subject in negative]
neutral_epochs_array=[read_data(subject) for subject in neutral]
positive_epochs_array=[read_data(subject) for subject in positive]

print(len(negative_epochs_array), len(neutral_epochs_array), len(positive_epochs_array))


               
positive_epochs_labels=[]
for i in positive_epochs_array:
    if i is not None:
        positive_epochs_labels.append(len(i) * [0])
    else:
        continue
        
neutral_epochs_labels=[]
for i in neutral_epochs_array:
    if i is not None:
        neutral_epochs_labels.append(len(i) * [1])
    else:
        continue 
        
negative_epochs_labels = []
for i in negative_epochs_array:
    if i is not None:
        negative_epochs_labels.append(len(i) * [2])
    else:
        continue

        
print(len(negative_epochs_labels), len(neutral_epochs_labels), len(positive_epochs_array))

data_list=negative_epochs_array+neutral_epochs_array+positive_epochs_array
label_list=negative_epochs_labels+neutral_epochs_labels+positive_epochs_labels


data_array=np.vstack(data_list)
label_array=np.hstack(label_list)
#group_array=np.hstack(groups_list)
print(data_array.shape,label_array.shape)


# Load the saved stacked_array
np.save('3class_30s.npy', data_array)
# Load the saved stacked_array
np.save('3class_label_30s.npy', label_array)

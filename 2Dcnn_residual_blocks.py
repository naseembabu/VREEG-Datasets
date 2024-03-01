from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import pandas as pd

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Bidirectional, Dropout, Conv1D, MaxPooling1D, BatchNormalization

import matplotlib.pyplot as plt
import math
import json
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

data_array = np.load('3class_05s.npy')
# Load the saved stacked_array
label_array = np.load('3class_label_05s.npy')
print(data_array.shape,label_array.shape)
num_classes = 3

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

# Assuming X_train_normalized has shape (num_samples, num_data_points, num_channels)
num_samples, num_data_points, num_channels = X_train_normalized.shape
# Reshape data for ResNet50
X_train_reshaped = np.repeat(X_train_normalized[..., np.newaxis], 3, axis=-1)

# Reshape data for ResNet50
X_test_reshaped = np.repeat(X_test_normalized[..., np.newaxis], 3, axis=-1)



# Lists to store confusion matrix metrics for each fold
conf_matrices = []
test_accuracies = []
precision_values = []
recall_values = []
f1_score_values = []
missrate_values = []
class_level_accuracies = []
# Stratified k-fold cross-validation
num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
# List to store model history for each fold
history_list = []
# List to store models for each fold
models_list = []
for fold, (train_idx, test_idx) in enumerate(skf.split(data_array, label_array), 1):
    print(f"Fold {fold}")
    X_train, X_test = data_array[train_idx], data_array[test_idx]
    y_train, y_test = label_array[train_idx], label_array[test_idx]
    
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
    
    # Assuming X_train_normalized has shape (num_samples, num_data_points, num_channels)
    num_samples, num_data_points, num_channels = X_train_normalized.shape
    # Reshape data for ResNet50
    X_train_reshaped = np.repeat(X_train_normalized[..., np.newaxis], 3, axis=-1)
    
    # Reshape data for ResNet50
    X_test_reshaped = np.repeat(X_test_normalized[..., np.newaxis], 3, axis=-1)
    
    def residual_block(x, filters, kernel_size=5, stride=1):
        #with tf.device(f'/GPU:2'):
        # Shortcut
        shortcut = x
    
        # First convolution
        x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        # Second convolution
        x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
    
        # Adjust shortcut if necessary
        if stride > 1:
            shortcut = Conv2D(filters, 5, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
    
        # Add shortcut to the output
        x = Add()([x, shortcut])
        x = ReLU()(x)
    
        return x
    
    def build_resnet18(input_shape, num_classes):
        #with tf.device(f'/GPU:2'):
        input_tensor = Input(shape=input_shape)
    
        # Initial convolution
        x = Conv2D(64, 7, strides=2, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
    
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
    
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
    
        #x = residual_block(x, 512, stride=2)
        #x = residual_block(x, 512)
    
        # Global average pooling and fully connected layer
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        output_tensor = Dense(num_classes, activation='softmax')(x)

        # Build the model
        model = Model(inputs=input_tensor, outputs=output_tensor)
        return model
    
    # Example usage
    input_shape=(num_data_points, num_channels, 3)  # Adjust based on your input image size and channels
    # Adjust based on your task
    model = build_resnet18(input_shape, num_classes)
    
    # Display the model summary
    #print(model.summary())
    learning_rate = 0.0001  # You can adjust this value as needed
    # Create a custom optimizer with the desired learning rate
    custom_optimizer = Adam(learning_rate=learning_rate)
    # Compile the model
    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #with tf.device('/device:GPU:0'):
    history = model.fit(X_train_reshaped, y_train_categorical, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test_categorical))


    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_categorical)
    print("Test Accuracy:", test_accuracy)
    # Save the test accuracy to the list
    test_accuracies.append(test_accuracy)
    
    
    # Evaluate the model on the test set
    y_pred_proba = model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # True labels
    y_true = np.argmax(y_test_categorical, axis=1)

    # Calculate precision, recall, F1-score, and miss rate
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    miss_rate = 1 - recall
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Print and store the confusion matrix
    print(f"Confusion Matrix (Fold {fold}):\n", conf_matrix)

    
    # Print and store the metrics
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Miss Rate:", miss_rate)

    # Store the metrics in lists
    precision_values.append(precision)
    recall_values.append(recall)
    f1_score_values.append(f1)
    missrate_values.append(miss_rate)
    conf_matrices.append(conf_matrix)
    
    # Calculate and store class-level accuracy
    class_accuracy_fold = []
    for class_label in range(num_classes):
        indices = y_true == class_label
        class_predictions = y_pred[indices]
        class_true_labels = y_true[indices]
        class_acc = accuracy_score(class_true_labels, class_predictions)
        class_accuracy_fold.append(class_acc)

    print("Class-Level Accuracy:", class_accuracy_fold)
    class_level_accuracies.append(class_accuracy_fold)


np.savez('3class_metrics_05s.npz', 
         test_accuracies=test_accuracies,
         precision_values=precision_values,
         recall_values=recall_values,
         f1_score_values=f1_score_values,
         missrate_values=missrate_values,
         class_level_accuracies=class_level_accuracies,
         conf_matrices=conf_matrices)


tf.keras.backend.clear_session()

# Load lists from the file
loaded_data = np.load('3class_metrics_04s.npz')
test_accuracies = loaded_data['test_accuracies']
precision_values = loaded_data['precision_values']
recall_values = loaded_data['recall_values']
f1_score_values = loaded_data['f1_score_values']
missrate_values = loaded_data['missrate_values']
class_level_accuracies = loaded_data['class_level_accuracies']
conf_matrices = loaded_data['conf_matrices']

# Calculate and print the average metrics
avg_precision = np.mean(precision_values)
avg_recall = np.mean(recall_values)
avg_f1_score = np.mean(f1_score_values)
avg_miss_rate = np.mean(missrate_values)
# Calculate the average test accuracy
average_test_accuracy = np.mean(test_accuracies)
# Calculate and print the average class-level accuracy
avg_class_level_accuracy = np.mean(class_level_accuracies, axis=0)
# Calculate the average confusion matrix
avg_conf_matrix = np.mean(conf_matrices, axis=0)


print("Average Test Accuracy across Folds:", average_test_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1 Score:", avg_f1_score)
print("Average Miss Rate:", avg_miss_rate)
print("Average Class-Level Accuracy:", avg_class_level_accuracy)
print("Average Confusion Matrix:\n", avg_conf_matrix)

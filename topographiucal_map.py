import mne
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import norm, t
import pywt
from nolds import sampen
import matplotlib.pyplot as plt
# Load your EEG data (replace with your own data)
eeg_file = '/content/drive/MyDrive/eeg_emotions/s01.edf'
raw = mne.io.read_raw_edf(eeg_file, preload=True)
electrode_names = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
# Select the data for the desired electrodes and remove all other channels
raw.pick_channels(ch_names=electrode_names)
# Set the EEG 10/20 electrode layout
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Define the start and end time for the fixed duration segment (in seconds)
start_time = 0 # Start at 60 seconds
end_time = 0.5 # End at 90 seconds

# Extract the segment of data
raw = raw.copy().crop(tmin=start_time, tmax=end_time)
segment = raw
segment.filter(l_freq=1, h_freq=None)
# Create an ICA object
n_components = 32  # Number of EEG channels
ica = mne.preprocessing.ICA(n_components=n_components, random_state=37,max_iter=1000)
ica.fit(segment)
# # Obtain the independent components
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

        # Check for inf
        if np.isinf(entropy_value):
            entropy_value = 0.0  # Set to a default value

    sample_entropies.append(entropy_value)
    # print('entropy_value:', entropy_value)

mean_entropy = np.mean(sample_entropies)
std_entropy = np.std(sample_entropies)
margin_of_error = t_critical * (std_entropy / np.sqrt(len(sample_entropies)))
lower_limit = mean_entropy -  margin_of_error  # Lower limit for thresholding
blink_artifact_entropy = [idx for idx, entropy in enumerate(sample_entropies) if entropy < lower_limit]


# Calculate kurtosis for each independent component
kurtosis_values = []  # Initialize an array to store ICA kurtosis values
for component in ica_components:
    kurtosis_value = kurtosis(component)
    kurtosis_values.append(kurtosis_value)
    
mean_kurtosis = np.mean(kurtosis_values)
std_kurtosis = np.std(kurtosis_values)
margin_of_error = t_critical * (std_kurtosis / np.sqrt(len(kurtosis_values)))
upper_limit = mean_kurtosis +  margin_of_error  # Upper limit for thresholding
# Identify ICs with mMSE below the lower limit
blink_artifact_kurtosis = [idx for idx, kurtosis in enumerate(kurtosis_values) if kurtosis > upper_limit]

artifactual_components = np.union1d(blink_artifact_kurtosis, blink_artifact_entropy)
artifactual_components = [int(x) for x in artifactual_components]

level = 4
def custom_threshold(data, level, method='hard'):
    # Calculate the threshold k
    N = len(data)
    threshold_multiplier = 0.6745  # Constant for Gaussian noise
    sigma = np.sqrt(np.median(np.abs(data) / threshold_multiplier))
    k = np.sqrt(2 * np.log(N)) * sigma
    # Set the threshold as a multiple of k based on the specified method
    if method == 'hard':
        threshold = k
    elif method == 'soft':
        threshold = k
    else:
        raise ValueError("Invalid thresholding method")
    return threshold
    threshold_method = 'hard'  # Choose 'hard' or 'soft'

def wavelet_denoising(x, wavelet='bior4.4', level=level, method='hard'):
    threshold = custom_threshold(x, level, method)
    coeff = pywt.wavedec(x, wavelet, mode="per", level=level)
    coeff[1:] = [pywt.threshold(i, value=threshold, mode=method) for i in coeff[1:]]
    denoised_ic = pywt.waverec(coeff, wavelet, mode='per')
    return denoised_ic


denoised_components = []
for idx in artifactual_components:
    # Access the component from ica_components using the index
    original_component = ica_components[idx]
    denoised_component = wavelet_denoising(original_component)
    denoised_components.append(denoised_component)

# Loop through the artifactual components and replace them with denoised components
for component_idx, denoised_component in zip(artifactual_components, denoised_components):
    ica.mixing_matrix_[component_idx, :] = denoised_component

# Check the number of components in the ICA object
ica.apply(segment)
# Define colors for the channels
colors = ['r', 'g', 'b', 'c']
# Create a box plot with specified colors for channels
plt.figure(figsize=(12, 5))

# Create a list of colors that repeats as needed
channel_colors = colors * (len(electrode_names) // len(colors))

# Create the box plot
bp = plt.boxplot(segment.get_data().T, patch_artist=True, labels=electrode_names, boxprops=dict(facecolor='white'))

# Assign colors to the boxes
for box, color in zip(bp['boxes'], channel_colors):
    box.set_facecolor(color)
# Set the y-axis limits
plt.ylim(-0.0002, 0.00015)
# Customize the plot
plt.title('Artifact-suppressed EEG signal', fontweight='bold')
plt.xlabel('Channels', fontweight='bold')
plt.ylabel('Signal Amplitude', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
# Show the plot
plt.show()

segment.plot(n_channels=32, title='EEG Data', scalings={'eeg': 50e-6}, show_scrollbars=False)


# Get the topographies of all components
topographies = ica.get_components()
# Calculate the mean topography across all components
mean_topography = topographies.mean(axis=0)
# Plot the average topographical map
fig, ax = plt.subplots()
mne.viz.plot_topomap(mean_topography, segment.info, axes=ax, show=False)
plt.show()

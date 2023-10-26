import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd

# Read the CSV file
data = pd.read_csv('wrist_movement.csv')

# Define filter parameters
order = 4  # Order of the filter
fs = 31.0  # Sample rate, adjust according to your data
nyquist = 0.5 * fs
cutoff_freq = 12.0  # Desired cutoff frequency

# Calculate normalized cutoff frequency
normal_cutoff = cutoff_freq / nyquist

# Design a Butterworth filter
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Define the columns to filter
columns_to_filter = ['right_wrist_x', 'right_wrist_y', 'right_wrist_z']  # Adjust with your actual column names

# Apply the filter to each column
for column in columns_to_filter:
    data_column = data[column]
    filtered_data = filtfilt(b, a, data_column)
    data[f'filtered_{column}'] = filtered_data

# Save the filtered data back to a new CSV file
data.to_csv('filtered_right_wrist.csv', index=False)


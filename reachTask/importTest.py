import math
import sys
import os
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

# Read the CSV file
fileOne = pd.read_csv(r"C:/Users/romyu/OneDrive - University of Calgary/Freemocap 2023/freemocap_data/recording_sessions/session_2024-01-25_12_01_45/recording_12_37_15_gmt-7__trial3/recording_12_37_15_gmt-7__trial3_by_trajectory.csv", skiprows=range(1, 25), nrows=850)

# Extract relevant columns
x_data_wrist = fileOne['right_wrist_x']
y_data_wrist = fileOne['right_wrist_y']
z_data_wrist = fileOne['right_wrist_z']
time_unix = fileOne['timestamp']

# Calculate absolute position
fileOne['wrist_abs_position'] = np.sqrt(x_data_wrist**2 + y_data_wrist**2 + z_data_wrist**2)
abs_position = fileOne['wrist_abs_position']

# Convert Unix time to seconds from the start of the file
start_time = time_unix.iloc[0]  # Get the timestamp of the first row
time_seconds = (time_unix - start_time) / 1000  # Convert to seconds

# Plot the data
plt.plot(time_seconds, abs_position)
plt.xlabel('Time (seconds from start of file)')
plt.ylabel('Absolute Position')
plt.title('Absolute Position vs. Time')
plt.grid(True)
plt.show()

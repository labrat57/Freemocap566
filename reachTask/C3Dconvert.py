import c3d
import numpy as np
import os
import csv
import freemocapAnalysis as fa
import matplotlib.pyplot as plt

# Set data path
datadir = fa.setdatapath('rom')
fname = os.path.join(datadir, 'PhaseSpace-2024-01-25', 'trial3.c3d')

# Open the file
with open(fname, 'rb') as file:
    reader = c3d.Reader(file)
    all_frames = list(reader.read_frames())

# Remove initial and final frames
all_frames = all_frames[2000:-2000]

# Initialize lists to store data
puck1_data = []
puck2_data = []
finger1_data = []
wrist1_data = []
wrist2_data = []
elbow_data = []
sho_data = []
timestamps = []

for frame in all_frames:
    index, p, analog = frame
    puck1_data.append(p[5, :])
    puck2_data.append(p[6, :])
    finger1_data.append(p[0, :])
    wrist1_data.append(p[1, :])
    wrist2_data.append(p[2, :])
    elbow_data.append(p[3, :])
    sho_data.append(p[4, :])

    # Extract timestamp
    timestamp = frame[0]  # Assuming the first element of the frame tuple is the timestamp
    timestamps.append(timestamp)

# Stack data arrays
puck1_data = np.array(puck1_data)
puck2_data = np.array(puck2_data)
finger1_data = np.array(finger1_data)
wrist1_data = np.array(wrist1_data)
wrist2_data = np.array(wrist2_data)
elbow_data = np.array(elbow_data)
sho_data = np.array(sho_data)
timestamps = np.array(timestamps)

# Define headers for the CSV file
headers = ['timestamp', 'puck1x', 'puck1y', 'puck1z', 'puck2x', 'puck2y', 'puck2z',
           'finger1x', 'finger1y', 'finger1z', 'wrist1x', 'wrist1y', 'wrist1z',
           'wrist2x', 'wrist2y', 'wrist2z', 'elbowx', 'elbowy', 'elbowz',
           'shox', 'shoy', 'shoz']

# Combine data and timestamps
combined_data = np.column_stack((timestamps, puck1_data, puck2_data, finger1_data, wrist1_data, wrist2_data, elbow_data, sho_data))

# Define output CSV file path
output_file = 'output.csv'

# Write data into the CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers
    writer.writerows(combined_data)  # Write combined data

print("CSV file with timestamp has been successfully created.")

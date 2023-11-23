import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

emc = pd.read_csv('comparison/calibrationOfFreemocap_WristOnly.csv', skiprows=[0])
fmc = pd.read_csv('comparison/mediapipe_right_hand_3d_xyz.csv')

# Set the start indices for the emc and fmc data
start_index_emc = 800
start_index_fmc = 200

# Subtract the start indices from the indices of the emc and fmc data
emc.index = emc.index - start_index_emc
fmc.index = fmc.index - start_index_fmc


# Create a new index for the fmc data that is x the original index
fmc.index = fmc.index * 8

# takes every x point of the data
emc = emc.iloc[::50]

# Plot the data
plt.plot(emc.index, emc[' pos_x'], label='Expensive Mocap')
plt.plot(fmc.index, fmc['right_hand_wrist_x'], label='Freemocap')

plt.xlim([0, max(max(emc.index), max(fmc.index))])

plt.xlabel('Index')
plt.ylabel('wrist position (mm)')
plt.title('wrist position')
plt.legend()

plt.show()
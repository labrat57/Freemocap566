
#%%
import freemocapAnalysis as fm
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from freemocapAnalysis import get_list_subject_files , tan_vel, setdatapath
from reach_fmc import __init__

# indeces we need
fileOne = pd.read_csv(r"C:/Users/romyu/OneDrive - University of Calgary/Freemocap 2023/freemocap_data/recording_sessions/session_2024-01-25_12_01_45/recording_12_37_15_gmt-7__trial3/recording_12_37_15_gmt-7__trial3_by_trajectory.csv", skiprows=range(1, 25), nrows=850)
    
x_data_wrist = fileOne['right_wrist_x']
y_data_wrist = fileOne['right_wrist_y']
z_data_wrist = fileOne['right_wrist_z']

sr_fixed = 30.0 #new sampling rate?

fmc = fileOne

# compute the time vector.
ts = fileOne['timestamp']
ts0 = ts - ts[0]
#now ts0 is in nanoseconds, conver to seconds
ts_sec = ts0 / 1000000000
#plt.plot(ts_sec)
time_temp = np.array(ts_sec)

shotemp = np.array([fmc['right_shoulder_x'], fmc['right_shoulder_y'], fmc['right_shoulder_z']])
elbtemp = np.array([fmc['right_elbow_x'], fmc['right_elbow_y'], fmc['right_elbow_z']])
writemp = np.array([fmc['right_wrist_x'], fmc['right_wrist_y'], fmc['right_wrist_z']])

time, sho = fm.resample_data(time_temp, shotemp, sr_fixed)
time, elb = fm.resample_data(time_temp, elbtemp, sr_fixed)
time, wri = fm.resample_data(time_temp, writemp, sr_fixed)
print("Shape of resampled time array:", time.shape)
print("Shape of resampled data array:", wri.shape)

# new stuff for plotting ---- might need to change
# Calculate the differences between consecutive positions
diffs = np.diff(wri, axis=1)

# Calculate the Euclidean distances between consecutive positions
distances = np.sqrt((diffs**2).sum(axis=0))

# Calculate the cumulative sum of the distances
cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

# Plot the cumulative distance over time

plt.plot(time[1:], cumulative_distance[1:])
plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Distance Traveled')
plt.title('Wrist Distance Traveled Over Time')
plt.show()
#%%
# in this cell we will do the phase space data
import c3d
import numpy as np
import os, sys
import freemocapAnalysis as fa
import reach_fmc as rf
import matplotlib.pyplot as plt

pp_file = pd.read_csv("C:/code/Freemocap566/output.csv", skiprows=range(1, 25), nrows=850)
    
ppx_data_wrist = pp_file['wrist1x']
ppy_data_wrist = pp_file['wrist1y']
ppz_data_wrist = pp_file['wrist1z']


fmc = pp_file

sr_fixed = 30.0 #new sampling rate?


# Assuming `npoints` is the length of your data
npoints = len(fmc)  # You should replace `all_frames` with your actual data

# Calculate time duration
time_duration = npoints / 960.0  # Duration in seconds

# Create time vector
ts = np.linspace(0, time_duration, npoints)  # Generate a linearly spaced array of time values

# Convert to numpy array if needed
pp_time_temp = np.array(ts_sec)

pp_writemp = np.array([fmc['wrist1x'], fmc['wrist1y'], fmc['wrist1z']])

pp_time, pp_wri = fm.resample_data(pp_time_temp, pp_writemp, sr_fixed)

# new stuff for plotting ---- might need to change
# Calculate the differences between consecutive positions
pp_diffs = np.diff(pp_wri, axis=1)

# Calculate the Euclidean distances between consecutive positions
pp_distances = np.sqrt((pp_diffs**2).sum(axis=0))

# Calculate the cumulative sum of the distances
pp_cumulative_distance = np.concatenate(([0], np.cumsum(pp_distances)))

# Plot the cumulative distance over time

plt.plot(pp_time[1:], pp_cumulative_distance[1:])
plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Distance Traveled')
plt.title('Wrist Distance Traveled Over Time')
plt.show()
#%%

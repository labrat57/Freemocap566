# this is a module for pulling functions from into the code late.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def tanVel(fmc:pd.DataFrame):
    # freemocap data
    body_parts = ['wrist', 'elbow', 'shoulder']

    # changes in distance between points
    for part in body_parts:
        fmc[f'right_{part}_delta_x'] = fmc[f'right_{part}_x'].diff()
        fmc[f'right_{part}_delta_y'] = fmc[f'right_{part}_y'].diff()
        fmc[f'right_{part}_delta_z'] = fmc[f'right_{part}_z'].diff()

    for part in body_parts:
        ts = fmc['timestamp']
        ts0 = ts - ts[0]
        #now ts0 is in nanoseconds, conver to seconds
        ts_sec = ts0 / 1000000000
        #plt.plot(ts_sec)
        fmc['time_s'] = ts_sec

    # velocity of points
    for part in body_parts:
        fmc[f'right_{part}_velocity_x'] = np.gradient(fmc[f'right_{part}_x'], fmc['time_s'])
        fmc[f'right_{part}_velocity_y'] = np.gradient(fmc[f'right_{part}_y'], fmc['time_s'])
        fmc[f'right_{part}_velocity_z'] = np.gradient(fmc[f'right_{part}_z'], fmc['time_s'])

    # tangential velocity of points
    for part in body_parts:
        fmc[f'right_{part}_tangential_velocity'] = (
            (fmc[f'right_{part}_velocity_x'] ** 2 + 
            fmc[f'right_{part}_velocity_y'] ** 2 + 
            fmc[f'right_{part}_velocity_z'] ** 2) ** 0.5
        ) 
      
    return fmc
           

# this is a butterworth filter
def butterfilter(fmc, order=4, fs=31.0, cutoff_freq=12.0):
    # Read the CSV file
    fmc

    # Calculate normalized cutoff frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # Design a Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each column
    for column in fmc:
        data_column = fmc[column]
        filtered_data = filtfilt(b, a, data_column)
        fmc[f'filtered_{column}'] = filtered_data

    return fmc
    
# add a few plotting one
# add the animation one here


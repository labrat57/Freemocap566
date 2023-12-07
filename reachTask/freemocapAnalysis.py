# this is a module for pulling functions from into the code late.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def tanVel(fmc):
    # freemocap data
    body_parts = ['wrist', 'elbow', 'shoulder']

    # changes in distance between points
    for part in body_parts:
        fmc[f'right_{part}_delta_x'] = fmc[f'right_{part}_x'].diff()
        fmc[f'right_{part}_delta_y'] = fmc[f'right_{part}_y'].diff()
        fmc[f'right_{part}_delta_z'] = fmc[f'right_{part}_z'].diff()

    # Convert Unix timestamp to datetime
    fmc['timestamp'] = pd.to_datetime(fmc['timestamp'], unit='ms')

    # Calculate the time interval in seconds
    fmc['time_interval'] = fmc['timestamp'].diff().dt.total_seconds()

    # Handle the first row which will have a NaN time interval
    fmc.loc[fmc.index[0], 'time_interval'] = fmc['time_interval'].iloc[1]

    # velocity of points
    for part in body_parts:
        fmc[f'right_{part}_velocity_x'] = fmc[f'right_{part}_delta_x'] / fmc['time_interval'] / 1000
        fmc[f'right_{part}_velocity_y'] = fmc[f'right_{part}_delta_y'] / fmc['time_interval'] / 1000
        fmc[f'right_{part}_velocity_z'] = fmc[f'right_{part}_delta_z'] / fmc['time_interval'] / 1000

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

    # Save the filtered data back to a new CSV file
    fmc.to_csv(fmc, index=False)
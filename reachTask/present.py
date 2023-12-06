import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# this opens the files
fileOne = pd.read_csv('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)
fileTwo = pd.read_csv('reachTask/recording_11_37_04_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)
fileThree = pd.read_csv('reachTask/recording_11_52_41_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)

def calculate_abs_location(file):
    # We just need the right wrist data
    x_data_wrist = file['right_wrist_x']
    y_data_wrist = file['right_wrist_y']
    z_data_wrist = file['right_wrist_z']

    # Calculate absolute location
    file['wrist_abs_location'] = (x_data_wrist**2 + y_data_wrist**2 + z_data_wrist**2)**0.5

# Call the function for each file
calculate_abs_location(fileOne)
calculate_abs_location(fileTwo)
calculate_abs_location(fileThree)

# gets the length of the file 
def get_length(file):
    # Get the length of the file
    num_points = len(file)
    return num_points

# Call the function for each file
length_fileOne = get_length(fileOne)
length_fileTwo = get_length(fileTwo)
length_fileThree = get_length(fileThree)


#need a velocity over time graph


# Assuming 'timestamp' is a column of Unix timestamps
timestamps = fileTwo['timestamp'] / 1000  # Convert to seconds

# Convert Unix timestamps to elapsed time starting from 0 seconds
start_time = timestamps.min()
fileTwo['elapsed_time'] = timestamps - start_time

# Calculate the time interval in seconds
fileTwo['time_interval'] = fileTwo['elapsed_time'].diff()

# Calculate the change in absolute location
fileTwo['location_change'] = fileTwo['wrist_abs_location'].diff()

# Calculate velocity
fileTwo['velocity'] = fileTwo['location_change'] / fileTwo['time_interval']

# Plot velocity over time
plt.plot(fileTwo['elapsed_time'], fileTwo['velocity'])
plt.xlabel('Time (s)')
plt.ylabel('Velocity (units/s)')
plt.xlim(fileTwo['elapsed_time'].min(), fileTwo['elapsed_time'].max())
plt.show()


# need a velocity over distance graph
# Plot velocity over distance
def plot_velocity_over_distance(file):
    # Convert the timestamp to datetime
    file['timestamp'] = pd.to_datetime(file['timestamp'], unit='s')

    # Calculate the time interval in seconds
    file['time_interval'] = file['timestamp'].diff().dt.total_seconds()

    # Calculate the change in absolute location
    file['location_change'] = file['wrist_abs_location'].diff()

    # Calculate velocity
    file['velocity'] = file['location_change'] / file['time_interval']

    # Only include rows where 'velocity' is greater than or equal to 0
    file = file[file['velocity'] >= 0]

    # Plot velocity over distance
    plt.scatter(file['wrist_abs_location'], file['velocity'])

# Call the function for each file
plot_velocity_over_distance(fileOne)
plot_velocity_over_distance(fileTwo)
plot_velocity_over_distance(fileThree)

plt.xlabel('Distance (units)')
plt.ylabel('Velocity (units/s)')
plt.show()

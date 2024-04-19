
#%%
import math
import sys, os
sys.path.append(os.path.join(os.getcwd()))
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
from scipy.signal import find_peaks



fileOne = pd.read_csv(r"C:/Users/romyu/OneDrive - University of Calgary/Freemocap 2023/freemocap_data/recording_sessions/session_2024-01-25_12_01_45/recording_12_37_15_gmt-7__trial3/recording_12_37_15_gmt-7__trial3_by_trajectory.csv", skiprows=range(1, 25), nrows=850)

# Drop the first 200 rows
#fileOne = fileOne.drop(fileOne.index[-200:50])

x_data_wrist = fileOne['right_wrist_x']
y_data_wrist = fileOne['right_wrist_y']
z_data_wrist = fileOne['right_wrist_z']

length = len(x_data_wrist)
print(length)
time = fileOne['timestamp']

# Calculate absolute position
fileOne['wrist_abs_position'] = np.sqrt(x_data_wrist**2 + y_data_wrist**2 + z_data_wrist**2)

abs_position = fileOne['wrist_abs_position']

plt.plot(time, abs_position)



##############################
import c3d
import numpy as np
import os, sys
import freemocapAnalysis as fa
import reach_fmc as rf
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

datadir = fa.setdatapath('rom')
fname = os.path.join(datadir,'PhaseSpace-2024-01-25','trial3.c3d')



# Open the file
with open(fname, 'rb') as file:
    reader = c3d.Reader(file)

    all_frames = list(reader.read_frames())

all_frames = all_frames[2000:-2000]

    # Read the rest of the frames
npoints = len(all_frames)
#for i, p, analog in all_frames:
#    npoints = i
#    p.shape

#the lines below work if stuff above doesnt
#reader = c3d.Reader(open(fname, 'rb'))
#npoints = 0
#for i, p, analog in reader.read_frames():
  #print('frame {}: point {}, analog {}'.format(i, p.shape, analog.shape))
#  npoints = i
#  p.shape



puck1 = np.ndarray((npoints,5))
puck2 = np.ndarray((npoints,5))
finger1 = np.ndarray((npoints,5))
wrist1 = np.ndarray((npoints,5))
wrist2 = np.ndarray((npoints,5))
elbow = np.ndarray((npoints,5))
sho  = np.ndarray((npoints,5))

#change back all_frames to reader.read_frames() to revert to uncut file
for i, frame in enumerate(all_frames):
  index, p, analog = frame
  puck1[i-1,:] = p[5,:]
  puck2[i-1,:] = p[6,:]
  finger1[i-1,:] = p[0,:]
  wrist1[i-1,:] = p[1,:]
  wrist2[i-1,:] = p[2,:]
  elbow[i-1,:] = p[3,:]
  sho[i-1,:] = p[4,:]


# magnitude of the position vector of puck1
puck1_magnitude = np.sqrt(puck1[:,0]**2 + puck1[:,1]**2 + puck1[:,2]**2)

collection_frequency = 960.00
time_duration = npoints / collection_frequency

# time array for plotting
time_array = np.linspace(0, time_duration, npoints)


def interpolate_to_size(source, target_size):
    """
    Interpolates the source variable to match the size of the target variable.

    Parameters:
        source: list or numpy array
            The source variable to be interpolated.
        target_size: int
            The size to which the source variable will be interpolated.

    Returns:
        interpolated_source: list or numpy array
            The interpolated source variable.
    """
    import numpy as np

    # Ensure source is a numpy array
    source = np.array(source)

    # Get the size of the source and target variables
    source_size = len(source)

    # Check if interpolation is needed
    if source_size == target_size:
        return source

    # Calculate the interpolation indices
    indices = np.linspace(0, source_size - 1, target_size)

    # Perform linear interpolation
    interpolated_source = np.interp(indices, np.arange(source_size), source)

    return interpolated_source

# Example usage:
# Assuming you have imported the necessary libraries and read the data

# Find the start time for each dataset
start_time_abs_position = time.iloc[0]  # Start time of abs_position
start_time_puck1 = time_array[0]        # Start time of puck1_magnitude

# Find the time offset between the start times
time_offset = start_time_puck1 - start_time_abs_position

# Adjust the time vector of abs_position to align with puck1_magnitude
time_abs_position_adjusted = time + time_offset

# Interpolate puck1_magnitude to match the size of abs_position
interpolated_puck1_magnitude = interpolate_to_size(puck1_magnitude, len(abs_position))

# Plot the adjusted abs_position and interpolated puck1_magnitude
plt.plot(time_abs_position_adjusted, abs_position, label='Adjusted abs_position')
plt.plot(time, interpolated_puck1_magnitude, label='Interpolated Puck1 Magnitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Aligned and Interpolated Data')
plt.legend()
plt.show()

# magnitude over time
plt.plot(time_array, puck1_magnitude)  
plt.xlabel('Time (seconds)')
plt.ylabel('Puck1 Position Magnitude')
plt.title('Puck1 Position Magnitude over Time')
plt.show()

# Assuming you have imported the necessary libraries and read the data

# Find the start time for each dataset
start_time_abs_position = time.iloc[0]  # Start time of abs_position
start_time_puck1 = time_array[0]        # Start time of puck1_magnitude

# Find the time offset between the start times
time_offset = start_time_puck1 - start_time_abs_position

# Adjust the time vector of abs_position to align with puck1_magnitude
time_abs_position_adjusted = time + time_offset

# Plot the adjusted abs_position and puck1_magnitude
plt.plot(time_abs_position_adjusted, abs_position, label='Adjusted abs_position')
plt.plot(time_array, puck1_magnitude, label='Puck1 Magnitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Aligned Data')
plt.legend()
plt.show()

#%%

#using wrist data now

reader = c3d.Reader(open(fname, 'rb'))
npoints = 0
for i, p, analog in reader.read_frames():
  #print('frame {}: point {}, analog {}'.format(i, p.shape, analog.shape))
  npoints = i
  p.shape


# magnitude of the position vector of puck1

wrist1_magnitude = np.sqrt(wrist1[:,0]**2 + wrist1[:,1]**2 + wrist1[:,2]**2)

collection_frequency = 960.00
time_duration = npoints / collection_frequency


def interpolate_to_size(source, target_size):
    """
    Interpolates the source variable to match the size of the target variable.

    Parameters:
        source: list or numpy array
            The source variable to be interpolated.
        target_size: int
            The size to which the source variable will be interpolated.

    Returns:
        interpolated_source: list or numpy array
            The interpolated source variable.
    """
    import numpy as np

    # Ensure source is a numpy array
    source = np.array(source)

    # Get the size of the source and target variables
    source_size = len(source)

    # Check if interpolation is needed
    if source_size == target_size:
        return source

    # Calculate the interpolation indices
    indices = np.linspace(0, source_size - 1, target_size)

    # Perform linear interpolation
    interpolated_source = np.interp(indices, np.arange(source_size), source)

    return interpolated_source


# Interpolate puck1_magnitude to match the size of abs_position
#interpolated_wrist1_magnitude = interpolate_to_size(wrist1_magnitude, len(abs_position))
interpolated_wrist1_magnitude = interpolate_to_size(abs_position, len(wrist1_magnitude))
# Plot the adjusted abs_position and interpolated puck1_magnitude
plt.plot(time_abs_position_adjusted, abs_position, label='Adjusted abs_position')
plt.plot(time_array, interpolated_wrist1_magnitude, label='Interpolated wrist1 Magnitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Aligned and Interpolated Data')
plt.legend()
plt.show()


# magnitude over time
#plt.plot(time, wrist1_magnitude)  
#plt.xlabel('Time (seconds)')
#plt.ylabel('wrist1 Position Magnitude')
#plt.title('wrist1 Position Magnitude over Time')
#plt.show()




#%%
# Initialize an array to store the cumulative distance traveled
fmc_cumulative_distance = np.zeros_like(abs_position)
pp_cumulative_distance = np.zeros_like(interpolated_wrist1_magnitude)

# Calculate the distance traveled at each time point
for i in range(1, len(abs_position)):
    # Calculate the distance between the current and previous positions
    distance = abs_position[i] - abs_position[i-1]
    # Update the cumulative distance traveled array
    fmc_cumulative_distance[i] = fmc_cumulative_distance[i-1] + abs(distance)

# Calculate the distance traveled at each time point
for i in range(1, len(interpolated_wrist1_magnitude)):
    # Calculate the distance between the current and previous positions
    distance = interpolated_wrist1_magnitude[i] - interpolated_wrist1_magnitude[i-1]
    # Update the cumulative distance traveled array
    pp_cumulative_distance[i] = pp_cumulative_distance[i-1] + abs(distance)

# Plot the cumulative distance traveled over time
plt.plot(time_abs_position_adjusted, fmc_cumulative_distance)
plt.plot(time_array, pp_cumulative_distance)
plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Distance Traveled')
plt.title('Cumulative Distance Traveled over Time')
plt.show

#%%


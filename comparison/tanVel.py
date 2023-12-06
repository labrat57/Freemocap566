# this file will be used to calculate the tangential velocity of the data collected
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# the following files are from the same recording, but different methods of collecting data
fmc = pd.read_csv('comparison/recording_14_01_05_gmt-7_by_trajectory.csv')#, skiprows=range(1, 300))
emc = pd.read_csv('comparison/filteredEMCData_wristOnly_001.csv')#, skiprows=range(1, 300))

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
    fmc[f'right_{part}_velocity_x'] = fmc[f'right_{part}_delta_x'] / fmc['time_interval']
    fmc[f'right_{part}_velocity_y'] = fmc[f'right_{part}_delta_y'] / fmc['time_interval']
    fmc[f'right_{part}_velocity_z'] = fmc[f'right_{part}_delta_z'] / fmc['time_interval']

# tangential velocity of points
for part in body_parts:
    fmc[f'right_{part}_tangential_velocity'] = (
        (fmc[f'right_{part}_velocity_x'] ** 2 + 
         fmc[f'right_{part}_velocity_y'] ** 2 + 
         fmc[f'right_{part}_velocity_z'] ** 2) ** 0.5
    ) 


# __________________expenisve mocap data________________
# changes in distance between points
# Define the points

points = ['0_pos']
# if theres more than 1 point use the line below
#points = [f'{i}_pos' for i in range(17)]  # Adjust the range as needed


#grad 5 to smooth out curve
# changes in distance between points
for point in points:
    emc[f'{point}_delta_x'] = emc[f'{point}_x'].diff()
    emc[f'{point}_delta_y'] = emc[f'{point}_y'].diff()
    emc[f'{point}_delta_z'] = emc[f'{point}_z'].diff()


# either 120 or 240 fps ex: 1/120
time_int = 1/240  
# velocity of points
for point in points:
    emc[f'{point}_velocity_x'] = emc[f'{point}_delta_x'] / time_int
    emc[f'{point}_velocity_y'] = emc[f'{point}_delta_y'] / time_int
    emc[f'{point}_velocity_z'] = emc[f'{point}_delta_z'] / time_int

# tangential velocity of points
for point in points:
    emc[f'{point}_tangential_velocity'] = (
        (emc[f'{point}_velocity_x'] ** 2 + 
         emc[f'{point}_velocity_y'] ** 2 + 
         emc[f'{point}_velocity_z'] ** 2) ** 0.5
    )



# outputs plot
plt.figure(figsize=(10, 6))
#for part in body_parts:
#    plt.plot(fmc.index, fmc[f'right_{part}_tangential_velocity'], label=f'Right {part}')

plt.plot(fmc.index, fmc[f'right_wrist_tangential_velocity'], label=f'Right wrist (FMC)')
plt.plot(emc.index, emc[f'0_pos_tangential_velocity'], label=f'Right wrist (EMC)')

plt.xlabel('Time')
plt.ylabel('Tangential Velocity')
plt.legend()
plt.show()


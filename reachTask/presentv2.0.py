import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

filenames = ['fileOne_chunk_1.csv',
'fileOne_chunk_2.csv',
'fileOne_chunk_3.csv',
'fileOne_chunk_4.csv',
'fileOne_chunk_6.csv',
'fileOne_chunk_5.csv',
'fileOne_chunk_7.csv',
'fileThree_chunk_1.csv',
'fileThree_chunk_2.csv',
'fileThree_chunk_3.csv',
'fileThree_chunk_4.csv',
'fileThree_chunk_5.csv',
'fileThree_chunk_6.csv',
'fileThree_chunk_7.csv',
'fileTwo_chunk_1.csv',
'fileTwo_chunk_2.csv',
'fileTwo_chunk_3.csv',
'fileTwo_chunk_4.csv',
'fileTwo_chunk_5.csv',
'fileTwo_chunk_6.csv',
'fileTwo_chunk_7.csv']


import pandas as pd

dataframes = []

for filename in filenames:
    df = pd.read_csv(filename)
    dataframes.append(df)

def calculate_abs_location_velocity_and_tangential_velocity(df):
    # We just need the right wrist data
    x_data_wrist = df['right_wrist_x']
    y_data_wrist = df['right_wrist_y']
    z_data_wrist = df['right_wrist_z']

    # Calculate absolute location
    df['wrist_abs_location'] = (x_data_wrist**2 + y_data_wrist**2 + z_data_wrist**2)**0.5

    # Calculate the time difference
    df['time_diff'] = df['timestamp'].diff()

    # Calculate the distance difference
    df['distance_diff'] = df['wrist_abs_location'].diff()

    # Calculate velocity
    df['velocity'] = df['distance_diff'] / df['time_diff']

   # Calculate tangential velocity for the wrist
    df['right_wrist_tangential_velocity'] = (
        (df['right_wrist_velocity_x'] ** 2 + 
         df['right_wrist_velocity_y'] ** 2 + 
         df['right_wrist_velocity_z'] ** 2) ** 0.5
    )

# Call the function for each DataFrame in the dataframes list
for df in dataframes:
    calculate_abs_location_velocity_and_tangential_velocity(df)

import matplotlib.pyplot as plt

# velocity over distance
plt.figure()

# Plot the positive velocity over the absolute location for each DataFrame
for i, df in enumerate(dataframes):
    positive_velocity_df = df[df['velocity'] > 0]
    plt.scatter(positive_velocity_df['wrist_abs_location'], positive_velocity_df['velocity'], label=f'DataFrame {i+1}')

plt.xlabel('Absolute Location')
plt.ylabel('Positive Velocity')
plt.title('Positive Velocity over Absolute Location for All DataFrames')
plt.legend()

plt.show()

#====================================================================================================
#figure for velocity over time
plt.figure()

# Plot the positive velocity over time for each DataFrame
for i, df in enumerate(dataframes):
    positive_velocity_df = df[df['velocity'] > 0]
    plt.plot(positive_velocity_df['timestamp'], positive_velocity_df['velocity'], label=f'DataFrame {i+1}')

plt.xlabel('Time')
plt.ylabel('Positive Velocity')
plt.title('Positive Velocity over Time for All DataFrames')
plt.legend()

plt.show()

#====================================================================================================
#figure for duration over distance
plt.figure()
# Plot the duration over distance for each DataFrame
for i, df in enumerate(dataframes):
    plt.plot(df['wrist_abs_location'], df['time_diff'], label=f'DataFrame {i+1}')

plt.xlabel('Distance')
plt.ylabel('Duration')
plt.title('Duration over Distance for All DataFrames')
plt.legend()

plt.show()


#====================================================================================================
#tanvel
# Create a new figure for tangential velocity over time
plt.figure()

# Plot the tangential velocity over time for each DataFrame
for i, df in enumerate(dataframes):
    plt.plot(df['timestamp'], df['right_wrist_tangential_velocity'], label=f'DataFrame {i+1}')

plt.xlabel('Time')
plt.ylabel('Tangential Velocity')
plt.title('Tangential Velocity over Time for All DataFrames')
plt.legend()

plt.show()
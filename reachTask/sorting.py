import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# this opens the files
fileOne = pd.read_csv('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)
fileTwo = pd.read_csv('reachTask/recording_11_37_04_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)
fileThree = pd.read_csv('reachTask/recording_11_52_41_gmt-7_by_trajectory.csv', skiprows=range(1, 150), nrows=300)


def calculate_abs_location(file):
    # We just need the right wrist data
    x_data_wrist = file['right_wrist_x']
    y_data_wrist = file['right_wrist_y']
    z_data_wrist = file['right_wrist_z']

    # Calculate absolute location
    file['wrist_abs_location'] = (x_data_wrist**2 + y_data_wrist**2 + z_data_wrist**2)**0.5


# Calculate absolute location for fileThree
calculate_abs_location(fileThree)

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(fileThree.index, fileThree['wrist_abs_location'], linewidth=3)
# Set the titles for the axes
ax.set_xlabel('Index')
ax.set_ylabel('Wrist Abs Location')
ax.set_zlabel('Z')
plt.show()
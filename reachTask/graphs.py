import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# this opens the files
fileOne = pd.read_csv('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)
fileTwo = pd.read_csv('reachTask/recording_11_37_04_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)
fileThree = pd.read_csv('reachTask/recording_11_52_41_gmt-7_by_trajectory.csv') #, skiprows=range(1, 300))# nrows=50)

#we just need the right wrist data
x_data_wrist = fileThree['right_wrist_x']
y_data_wrist = fileThree['right_wrist_y']
z_data_wrist = fileThree['right_wrist_z']

# gets the length of the file 
num_points = len(x_data_wrist)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([x_data_wrist.min(), x_data_wrist.max()])
ax.set_ylim([y_data_wrist.min(), y_data_wrist.max()])
ax.set_zlim([z_data_wrist.min(), z_data_wrist.max()])
pltdata_wrist, = ax.plot(x_data_wrist[:1], y_data_wrist[:1], z_data_wrist[:1], '-r', linewidth = 3)
lastPoint_wrist, = ax.plot(x_data_wrist[0], y_data_wrist[0], z_data_wrist[0], 'r', marker='o')



def update(i):
    pltdata_wrist.set_data(x_data_wrist[:i+1], y_data_wrist[:i+1])
    pltdata_wrist.set_3d_properties(z_data_wrist[:i+1])
    lastPoint_wrist.set_data(x_data_wrist[i:i+1], y_data_wrist[i:i+1])
    lastPoint_wrist.set_3d_properties(z_data_wrist[i:i+1])

    return pltdata_wrist, lastPoint_wrist,

ani = FuncAnimation(fig, func=update, frames=num_points, interval=10, blit=True, repeat=False)

plt.show()

#need a velocity over time graph
# need a velocity over distance graph

# need a time over distnae graph
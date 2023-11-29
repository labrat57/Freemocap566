import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

emc = pd.read_csv('comparison/filteredEMCData_004.csv')#, skiprows=range(1, 300))# nrows=50)

num_points = len(emc.index)

def plot_positional_data(id_num):
    if f'{id_num}_pos_x' not in emc.columns or f'{id_num}_pos_y' not in emc.columns or f'{id_num}_pos_z' not in emc.columns:
        return None

    x_data = emc[f'{id_num}_pos_x']
    y_data = emc[f'{id_num}_pos_y']
    z_data = emc[f'{id_num}_pos_z']

    pltdata, = ax.plot(x_data[:1], y_data[:1], z_data[:1], '-r', linewidth = 3)
    lastPoint, = ax.plot(x_data[0], y_data[0], z_data[0], 'r', marker='o')

    return pltdata, lastPoint, x_data, y_data, z_data


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize lists to hold the plot data and positional data for each id
plot_data = []
positional_data = []


# Call the function for each id from 1 to 16
for id_num in range(1, 17):
    result = plot_positional_data(id_num)
    if result is not None:
        pltdata, lastPoint, x_data, y_data, z_data = result
        plot_data.append((pltdata, lastPoint))
        positional_data.append((x_data, y_data, z_data))
    else:
        continue

# Set the axis limits
ax.set_xlim([emc.min().min(), emc.max().max()])
ax.set_ylim([emc.min().min(), emc.max().max()])
ax.set_zlim([emc.min().min(), emc.max().max()])


def update(i):
    artists = []
    for j in range(len(plot_data)):
        pltdata, lastPoint = plot_data[j]
        x_data, y_data, z_data = positional_data[j]

        pltdata.set_data(x_data[:i+1], y_data[:i+1])
        pltdata.set_3d_properties(z_data[:i+1])
        lastPoint.set_data(x_data[i:i+1], y_data[i:i+1])
        lastPoint.set_3d_properties(z_data[i:i+1])

        artists.append(pltdata)
        artists.append(lastPoint)

    return artists


ani = FuncAnimation(fig, func=update, frames=num_points, interval=10, blit=True, repeat=False)

plt.show()






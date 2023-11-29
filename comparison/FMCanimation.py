import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fmc = pd.read_csv('comparison/recording_13_58_54_gmt-7_by_trajectory.csv', skiprows=range(1, 300))# nrows=50)

x_data_wrist = fmc['right_wrist_x']
y_data_wrist = fmc['right_wrist_y']
z_data_wrist = fmc['right_wrist_z']

x_data_elbow = fmc['right_elbow_x']
y_data_elbow = fmc['right_elbow_y']
z_data_elbow = fmc['right_elbow_z']

x_data_shoulder = fmc['right_shoulder_x']
y_data_shoulder = fmc['right_shoulder_y']
z_data_shoulder = fmc['right_shoulder_z']

# Subtract the first shoulder point from all points
x_data_wrist -= x_data_shoulder[0]
y_data_wrist -= y_data_shoulder[0]
z_data_wrist -= z_data_shoulder[0]

x_data_elbow -= x_data_shoulder[0]
y_data_elbow -= y_data_shoulder[0]
z_data_elbow -= z_data_shoulder[0]

x_data_shoulder -= x_data_shoulder[0]
y_data_shoulder -= y_data_shoulder[0]
z_data_shoulder -= z_data_shoulder[0]

num_points = len(x_data_wrist)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([min(min(x_data_wrist), min(x_data_elbow), min(x_data_shoulder)), max(max(x_data_wrist), max(x_data_elbow))])
ax.set_ylim([min(min(y_data_wrist), min(y_data_elbow), min(x_data_shoulder)), max(max(y_data_wrist), max(y_data_elbow))])
ax.set_zlim([min(min(z_data_wrist), min(z_data_elbow), min(x_data_shoulder)), max(max(z_data_wrist), max(z_data_elbow))])

pltdata_wrist, = ax.plot(x_data_wrist[:1], y_data_wrist[:1], z_data_wrist[:1], '-r', linewidth = 3)
lastPoint_wrist, = ax.plot(x_data_wrist[0], y_data_wrist[0], z_data_wrist[0], 'r', marker='o')

pltdata_elbow, = ax.plot(x_data_elbow[:1], y_data_elbow[:1], z_data_elbow[:1], '-y', linewidth = 3)
lastPoint_elbow, = ax.plot(x_data_elbow[0], y_data_elbow[0], z_data_elbow[0], 'y', marker='o')

pltdata_shoulder, = ax.plot(x_data_shoulder[:1], y_data_shoulder[:1], z_data_shoulder[:1], '-b', linewidth = 3)
lastPoint_shoulder, = ax.plot(x_data_shoulder[0], y_data_shoulder[0], z_data_shoulder[0], 'bo', marker='o')

#line plots for the connecting bars
pltdata_wrist_elbow, = ax.plot([x_data_wrist[0], x_data_elbow[0]], [y_data_wrist[0], y_data_elbow[0]], [z_data_wrist[0], z_data_elbow[0]], '-m')
pltdata_elbow_shoulder, = ax.plot([x_data_elbow[0], x_data_shoulder[0]], [y_data_elbow[0], y_data_shoulder[0]], [z_data_elbow[0], z_data_shoulder[0]], '-c')


def update(i):
    # Update the data for the connecting bars
    pltdata_wrist_elbow.set_data([x_data_wrist[i], x_data_elbow[i]], [y_data_wrist[i], y_data_elbow[i]])
    pltdata_wrist_elbow.set_3d_properties([z_data_wrist[i], z_data_elbow[i]])

    pltdata_elbow_shoulder.set_data([x_data_elbow[i], x_data_shoulder[i]], [y_data_elbow[i], y_data_shoulder[i]])
    pltdata_elbow_shoulder.set_3d_properties([z_data_elbow[i], z_data_shoulder[i]])

    pltdata_wrist.set_data(x_data_wrist[:i+1], y_data_wrist[:i+1])
    pltdata_wrist.set_3d_properties(z_data_wrist[:i+1])
    lastPoint_wrist.set_data(x_data_wrist[i:i+1], y_data_wrist[i:i+1])
    lastPoint_wrist.set_3d_properties(z_data_wrist[i:i+1])

    pltdata_elbow.set_data(x_data_elbow[:i+1], y_data_elbow[:i+1])
    pltdata_elbow.set_3d_properties(z_data_elbow[:i+1])
    lastPoint_elbow.set_data(x_data_elbow[i:i+1], y_data_elbow[i:i+1])
    lastPoint_elbow.set_3d_properties(z_data_elbow[i:i+1])

    pltdata_shoulder.set_data(x_data_shoulder[:i+1], y_data_shoulder[:i+1])
    pltdata_shoulder.set_3d_properties(z_data_shoulder[:i+1])
    lastPoint_shoulder.set_data(x_data_shoulder[i:i+1], y_data_shoulder[i:i+1])
    lastPoint_shoulder.set_3d_properties(z_data_shoulder[i:i+1])

    return pltdata_wrist, lastPoint_wrist, pltdata_elbow, lastPoint_elbow, pltdata_shoulder, lastPoint_shoulder, pltdata_wrist_elbow, pltdata_elbow_shoulder

ani = FuncAnimation(fig, func=update, frames=num_points, interval=100, blit=True, repeat=False)

plt.show()


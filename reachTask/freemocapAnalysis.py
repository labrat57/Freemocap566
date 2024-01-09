# this is a module for pulling functions from into the code late.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


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
    
# 3d plotting of data
def x_y_z_plot(fmc, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(fmc[x], fmc[y], fmc[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f'{x}, {y}, {z}')

    plt.show()
    return fmc


# add the animation one here
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def animate_3d_plot(fmc, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot(fmc[x], fmc[y], fmc[z])

    def update(num):
        line.set_data(fmc[x][:num], fmc[y][:num])
        line.set_3d_properties(fmc[z][:num])
        return line,

    ani = FuncAnimation(fig, update, frames=len(fmc[x]), interval=100, blit=True)

    plt.show()

    return ani

# click detection

def click_starts_ends(fname):
    coordinates = []
    # Function to capture mouse clicks
    def onclick(event):
        coordinates.append((event.xdata, event.ydata))
        df = pd.DataFrame(coordinates, columns=['X'])
        df.to_csv(f'click_{fname}.csv', index=False)
        #print(f"Coordinates after {len(coordinates)} clicks saved to 'click_coordinates.csv'")

# Connect the click event to the function
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    return coordinates



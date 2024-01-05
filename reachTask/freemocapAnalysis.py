# this is a module for pulling functions from into the code late.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def addReachPath():
  str_path = sys.path[0]
  sys.path.append(os.path.join(str_path,'reachTask'))


def setdatapath(str_who):
  # add the path reachTask
  # get the name of the directory containing this file:
  if str_who == 'romeo':
    str_datadir = "hi"
  elif str_who == "jeremy":
    str_datadir = "/Users/jeremy/OneDrive - University of Calgary/Freemocap 2023/freemocap_data/recording_sessions"
  else:
    print('unknown user name %s' % (str_who))
  
  sys.path.append(str_datadir)
  return str_datadir

def tan_vel(fmc:pd.DataFrame,side='right',body_parts=['wrist', 'elbow', 'shoulder']):
    # freemocap data
    
    # changes in distance between points
    for part in body_parts:
        fmc[f'{side}_{part}_delta_x'] = fmc[f'right_{part}_x'].diff()
        fmc[f'{side}_{part}_delta_y'] = fmc[f'right_{part}_y'].diff()
        fmc[f'{side}_{part}_delta_z'] = fmc[f'right_{part}_z'].diff()

    for part in body_parts:
        ts = fmc['timestamp']
        ts0 = ts - ts[0]
        #now ts0 is in nanoseconds, conver to seconds
        ts_sec = ts0 / 1000000000
        #plt.plot(ts_sec)
        fmc['time_s'] = ts_sec

    # velocity of points
    for part in body_parts:
        fmc[f'{side}_{part}_velocity_x'] = np.gradient(fmc[f'{side}_{part}_x'], fmc['time_s'])
        fmc[f'{side}_{part}_velocity_y'] = np.gradient(fmc[f'{side}_{part}_y'], fmc['time_s'])
        fmc[f'{side}_{part}_velocity_z'] = np.gradient(fmc[f'{side}_{part}_z'], fmc['time_s'])

    # tangential velocity of points
    for part in body_parts:
        fmc[f'right_{part}_tangential_velocity'] = (
            (fmc[f'{side}_{part}_velocity_x'] ** 2 + 
            fmc[f'{side}_{part}_velocity_y'] ** 2 + 
            fmc[f'{side}_{part}_velocity_z'] ** 2) ** 0.5
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

# this is a class for the reach data. 
# i find it really difficult to keep typing fmc['right_shoulder_x'] and so on.
class reachData:
  sho = []
  elb = []
  wri = []
  tanvelsho = []
  tanvelelb = []
  tanvelwri = []
  velsho = []
  velelb = []
  velwri = []
  time = []

# constructor for reach data, receiving a pandas dataframe
  def __init__(self, fmc:pd.DataFrame):
    # get the x and y data for the shoulder, elbow, and wrist
    self.sho = np.array([fmc['right_shoulder_x'], fmc['right_shoulder_y'], fmc['right_shoulder_z']])
    self.elb = np.array([fmc['right_elbow_x'], fmc['right_elbow_y'], fmc['right_elbow_z']])
    self.wri = np.array([fmc['right_wrist_x'], fmc['right_wrist_y'], fmc['right_wrist_z']])
    # get the velocity data for the shoulder, elbow, and wrist
    self.velsho = np.array([fmc['right_shoulder_velocity_x'], fmc['right_shoulder_velocity_y'], fmc['right_shoulder_velocity_z']])
    self.velelb = np.array([fmc['right_elbow_velocity_x'], fmc['right_elbow_velocity_y'], fmc['right_elbow_velocity_z']])
    self.velwri = np.array([fmc['right_wrist_velocity_x'], fmc['right_wrist_velocity_y'], fmc['right_wrist_velocity_z']])

    # get tanvel data for sho elb wrist
    self.tanvelsho = np.array(fmc['right_shoulder_tangential_velocity'])
    self.tanvelelb = np.array(fmc['right_elbow_tangential_velocity'])
    self.tanvelwri = np.array(fmc['right_wrist_tangential_velocity'])
    # get the time data
    self.time = np.array(fmc['time_s'])


    
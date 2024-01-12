# this is a module for pulling functions from into the code late.
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# list of functions in this module:
# get_list_subject_files(sname)
# addReachPath()
# setdatapath(str_who)
# resample_data(data, time, sr)
# x_y_z_plot(fmc, x, y, z)
# tan_vel(fmc:pd.DataFrame,side='right',body_parts=['wrist', 'elbow', 'shoulder'])
# vel3D(time:np.array, data:np.array)
# butterfilter(fmc, order=4, fs=31.0, cutoff_freq=12.0)
# lowpass(data:np.array, order=4, fs=31.0, cutoff_freq=12.0)
# lowpass_cols(datarows:np.array, order=4, fs=31.0, cutoff_freq=12.0)
# animate_3d_plot(fmc, x, y, z)
# click_starts_ends(fname)

def addReachPath():
  str_path = sys.path[0]
  sys.path.append(os.path.join(str_path,'reachTask'))

def setdatapath(str_who):
  # add the path reachTask
  # get the name of the directory containing this file:
  if str_who == 'rom':
    str_datadir = r"C:\Users\romyu\OneDrive - University of Calgary\Freemocap 2023\freemocap_data\recording_sessions"
  elif str_who == "jer":
    str_datadir = "/Users/jeremy/OneDrive - University of Calgary/Freemocap 2023/freemocap_data/recording_sessions"
  else:
    print('unknown user name %s' % (str_who))
  
  if os.path.exists(str_datadir):
    sys.path.append(str_datadir)
  else:
    print('path %s does not exist. Check var in setdatapath() is either jer or rom.' % (str_datadir))
  return str_datadir

def resample_data(time, data, sr):
  # new time
  time_resamp = np.arange(time[0], time[-1], 1/sr)

  # Resample the data using linear interpolation
  data_resamp = np.zeros((3, len(time_resamp)))
  for i in range(data.shape[0]):
    data_resamp[i, :] = np.interp(time_resamp, time, data[i, :])
  return time_resamp, data_resamp

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

def vel3D(time:np.array, data:np.array):
  # def vel3D(time:np.array, data:np.array):
  # take derivative of 3 rows of input data nparray i.e. data[0,:],data[1,:],data[2,:] wrt time vector
  # use np.gradient

  # initialize an empty array to store the velocity data
  vel = np.empty_like(data)
  # now fill each of vel rows
  for i, row in enumerate(data):
    vel[i] = np.gradient(row, time)

  return vel
        
          # this is a butterworth filter
def butterfilter(fmc, order=4, fs=31.0, cutoff_freq=12.0):
    # Read the CSV file

    # Calculate normalized cutoff frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # Design a Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each column
    for column in fmc:
        data_column = fmc[column]
        filtered_data = filtfilt(b, a, data_column)
        fmc[f'{column}_f'] = filtered_data

    return fmc
    
#lowpass and return a particular data column
def lowpass(data:np.array, order=4, fs=31.0, cutoff_freq=12.0):

    # Calculate normalized cutoff frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # Design a Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each column
    data_f = filtfilt(b, a, data)

    return data_f

def lowpass_cols(datarows:np.array, order=4, fs=31.0, cutoff_freq=12.0):
  # Initialize an empty array to store the filtered data
  filtered_data = np.empty_like(datarows)

  # Apply lowpass filter to each row in the input array
  for i, row in enumerate(datarows):
    filtered_data[i] = lowpass(row, order, fs, cutoff_freq)

  return filtered_data

# add the animation one here
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

# %%
# click detection

def click_starts_ends(fname):
    coordinates = []
    # Function to capture mouse clicks
    def onclick(event):
        coordinates.append((event.xdata))
        df = pd.DataFrame(coordinates, columns=['X'])
        df.to_csv(f'click_{fname}.csv', index=False)
        #print(f"Coordinates after {len(coordinates)} clicks saved to 'click_coordinates.csv'")

# Connect the click event to the function
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    return 

def get_list_subject_files(sname,datapath):
  fnames = []
  
  if sname == 'who':
    print("na, le, je, ro, cal_romnov10, cal_jernov10")
    print("but, mostly je and ro are clean, everything else noisy.")
  
  elif sname == 'je':
    name_session    = "session_2023-12-11_11_07_49"
    
    name_recording  = "recording_11_13_39_gmt-7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_11_17_02_gmt-7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'ro':    
    name_session    = "session_2023-12-11_11_27_27"
    name_recording  = "recording_11_31_45_gmt-7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'nadec':
    name_session    = "session_2023-12-11_10_07_53"
    name_recording  = "recording_10_11_01_gmt-7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'le':
    name_session = 'session_2023-11-09_10_44_13'
    name_recording = 'recording_11_19_02_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

    name_recording = 'recording_11_38_11_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'na':
    name_session = 'session_2023-11-09_10_44_13'
    name_recording = 'recording_11_54_33_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

    name_recording = 'recording_11_56_55_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'cal_romnov10':
    name_session = 'session_2023-11-10_12_35_57'
    name_recording = 'recording_13_02_50_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)
    
    name_recording = 'recording_13_04_20_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

    name_recording = 'recording_13_08_48_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

    name_recording = 'recording_13_13_36_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'cal_jernov10':
    name_session = 'session_2023-11-10_13_41_10'
    name_recording = 'recording_13_58_54_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)

    name_recording = 'recording_14_01_05_gmt-7'
    name_file = f'{name_recording}_by_trajectory.csv'
    fname_full = os.path.join(datapath, name_session, name_recording, name_file)
    fnames.append(fname_full)
  return fnames
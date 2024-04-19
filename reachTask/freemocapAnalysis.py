# this is a module for pulling functions from into the code late.
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.io
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
  if str_who == "rom":
    str_datadir = "/Users/romyu/OneDrive - University of Calgary/Freemocap 2023/freemocap_data/recording_sessions"
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
    if len(time) != len(data[i, :]):
      print(f"Mismatched lengths at i={i}: length of time: {len(time)}, length of data[{i}]: {len(data[i, :])}")
    data_resamp[i, :] = np.interp(time_resamp, time, data[i, :])
  return time_resamp, data_resamp

#position over time plot of data
# y should be something like this 'right_wrist_y'
def pos_time_plot(fmc, x,y,z):
    x_data = fmc[x]
    y_data= fmc[y]
    z_data= fmc[z]

    # Calculate absolute location
    fmc['wrist_abs_location'] = (x_data**2 + y_data**2 + z_data**2)**0.5

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
    print("paper0x, where x is 1-8, but 5 6 bad. ro03= paper08. Previous: na, le, je, ro, cal_romnov10, cal_jernov10, hpl_trial1,2,3,4,5")
  
  elif sname == 'paper09': #jake
    name_session     = "session_2024-04-03_15_04_11"
    name_recording   = "recording_15_08_20_gmt-6__trial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_15_10_24_gmt-6__trial2"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_15_12_58_gmt-6__trial3"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_15_18_17_gmt-6__trial4"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_15_21_28_gmt-6__trial5"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_15_24_55_gmt-6__trial6"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_15_28_22_gmt-6__trial7"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper07': #osman
    name_session     = "session_2024-03-25_11_57_43"
    name_recording   = "recording_13_35_45_gmt-6__osmantrial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_37_51_gmt-6__osmantrial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_40_04_gmt-6__osmantrial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_42_16_gmt-6__osmantrial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_45_06_gmt-6__osmantrial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_48_23_gmt-6__osmantrial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_51_39_gmt-6__osmantrial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
     
  elif sname == 'paper06': #kira

    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_13_49_59_gmt-6__kiratrial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_52_14_gmt-6__kiratrial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_54_25_gmt-6__kiratrial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_56_31_gmt-6__kiratrial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_59_29_gmt-6__kiratrial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_14_02_54_gmt-6__kiratrial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_14_06_14_gmt-6__kiratrial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper05': # hazel

    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_13_27_10_gmt-6__hazeltrial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_29_33_gmt-6__hazeltrial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_31_50_gmt-6__hazeltrial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_33_54_gmt-6__hazeltrial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    # missing trial 5

    name_recording   = "recording_13_41_01_gmt-6__hazeltrial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_45_00_gmt-6__hazeltrial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper04': #ashna
    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_12_52_34_gmt-6__trial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_12_54_42_gmt-6__trial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_12_56_54_gmt-6__trial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_12_59_13_gmt-6__trial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_02_09_gmt-6__trial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_05_42_gmt-6__trial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording   = "recording_13_09_05_gmt-6__trial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper03': #pat

    name_session    = "session_2024-03-22_14_28_34"
    name_recording   = "recording_14_22_55_gmt-6__pattrial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full       = os.path.join(datapath, name_session, name_recording, name_file)
    print(fname_full)
    fnames.append(fname_full)

    # the trial 2 for pat did not save
    name_recording  = "recording_14_29_47_gmt-6__pattrial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_14_31_57_gmt-6__pattrial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_14_34_54_gmt-6__pattrial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_14_38_17_gmt-6__pattrial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_14_41_52_gmt-6__pattrial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper02':#jer
     
    name_session   = "session_2024-03-19_10_52_13"
    name_recording = "recording_11_30_31_gmt-6__trial"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full       = os.path.join(datapath, name_session, name_recording, name_file)
    print(fname_full)
    fnames.append(fname_full)

    name_recording = "recording_11_34_30_gmt-6__trial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording = "recording_11_38_37_gmt-6__trial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording = "recording_11_40_57_gmt-6__trial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording = "recording_11_45_38_gmt-6__trial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording = "recording_11_51_20_gmt-6__trial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording = "recording_11_55_52_gmt-6__trial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper01': #anna

    name_session     ="session_2024-03-22_15_08_13"
    name_recording   = "recording_15_10_38_gmt-6__annatrial1"
    name_file        = f"{name_recording}_by_trajectory.csv"
    fname_full       = os.path.join(datapath, name_session, name_recording, name_file)
    print(fname_full)
    fnames.append(fname_full)

    name_recording  = "recording_15_12_50_gmt-6__annatrial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
    
    name_recording  = "recording_15_14_58_gmt-6__annatrial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_15_17_08_gmt-6__annatrial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_15_19_55_gmt-6__annatrial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_15_23_16_gmt-6__annatrial6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_15_26_35_gmt-6__annatrial7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  #each 'sname' needs 
    # 1. name_session
    # 2. several name_recordings.
  elif (sname == 'ro03') | (sname == 'paper08'): #rom

    name_session    = "session_2024-03-19_10_52_13"
    name_recording  = "recording_12_15_43_gmt-6__romeo1"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    print(fname_full)
    fnames.append(fname_full)

    name_recording  = "recording_12_18_14_gmt-6__romeo2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_12_20_24_gmt-6__romeo3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
    
    name_recording  = "recording_12_22_21_gmt-6__romeo4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
    
    name_recording  = "recording_12_27_04_gmt-6__romeo5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
    
    name_recording  = "recording_12_30_27_gmt-6__romeo6"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

    name_recording  = "recording_12_33_44_gmt-6__romeo7"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif (sname == 'ro03_heel') | (sname == 'paper08_heel'):
    name_session    = "session_2024-03-19_10_52_13"
    name_recording  = "recording_11_11_31_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper01_heel':
    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_12_07_20_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper02_heel':
    name_session    = "session_2024-03-19_10_52_13"
    name_recording  = "recording_11_11_31_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'paper03_heel':
    name_session = "session_2024-03-22_11_58_00"
    name_recording = "recording_12_07_20_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
     
  elif sname == 'paper04_heel':
    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_12_07_20_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
     
  elif sname == 'paper05_heel':
    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_12_07_20_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
     
  elif sname == 'paper06_heel':
    name_session     = "session_2024-03-22_11_58_00"
    name_recording   = "recording_12_07_20_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
     
  elif sname == 'paper07_heel':
    name_session     = "session_2024-03-25_11_57_43"
    name_recording   = "recording_13_07_24_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
  
  elif sname == 'paper09_heel':
    name_session     = "session_2024-04-03_15_04_11"
    name_recording   = "recording_15_35_24_gmt-6__heel"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)


  elif sname == 'hpl_threepcalib3':    
    name_session    = "session_2024-01-25_12_01_45"
    name_recording  = "recording_12_03_15_gmt-7__threepcalib3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'hpl_trial1':    
    name_session    = "session_2024-01-25_12_01_45"
    name_recording  = "recording_12_11_12_gmt-7__trial1"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'hpl_trial2':    
    name_session    = "session_2024-01-25_12_01_45"
    name_recording  = "recording_12_26_11_gmt-7__trial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)


  elif sname == 'hpl_trial3':    
    name_session    = "session_2024-01-25_12_01_45"
    name_recording  = "recording_12_37_15_gmt-7__trial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  elif sname == 'hpl_trial4':    
    name_session    = "session_2024-01-25_12_01_45"
    name_recording  = "recording_12_47_24_gmt-7__trial4"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)
  
  elif sname == 'hpl_trial5':    
    name_session    = "session_2024-01-25_12_01_45"
    name_recording  = "recording_12_54_48_gmt-7__trial5"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  # this is jer's versions of trial_2 analysis. 
  elif sname == 'ro_0125_X':
    name_session    = "session_2024-01-25_12_01_45"
    
    name_recording  = "recording_12_26_11_gmt-7__trial2"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

  # this is jer's versions of trial_3 analysis.  
  elif sname == 'ro_0125_Y':
    name_session    = "session_2024-01-25_12_01_45"
  
    name_recording  = "recording_12_37_15_gmt-7__trial3"
    name_file       = f"{name_recording}_by_trajectory.csv"
    fname_full      = os.path.join(datapath, name_session,name_recording, name_file)
    fnames.append(fname_full)

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
  else:
    print('unknown sname %s' % (sname))
  return fnames

def get_cached_R(sname):
  current_module_directory = os.path.dirname(__file__)
  subjname_rot = sname + '_heel'
  fname_full = os.path.join(current_module_directory,'processed_clicks',subjname_rot + '.mat')
  if os.path.exists(fname_full):
    dat = scipy.io.loadmat(fname_full)
    R = dat["R"]
    return R
  else:
    print('R matrix not found for %s' % (sname))
    return None

def get_list_conditions(sname):
  cond = []
  if sname == 'paper01':
    cond = ['p','p2','f','s', 'l', 'm', 't']
  elif sname == 'paper02':
    cond = ['p','p2','f','s', 't', 'm', 'l']
  elif sname == 'paper03':
    cond = ['p','f','s', 't', 'm', 'l']
  elif sname == 'paper07':
    cond = ['p','p2','f','s', 'm', 'l', 't']
  elif sname == 'paper08':
    cond = ['p','p2','f','s', 'l', 'm', 't']
  elif sname == 'paper09':
    cond = ['p','p2','f','s', 'm', 'l', 't']

  return cond

def color_from_condition(cond):
  if cond == 's':
    return '#deebf7'
  elif (cond == 'p') | (cond == 'p2'):
    return '#9ecae1'
  elif cond == 'f':
    return '#3182bd'
  elif cond == 't':
    return '#efedf5'
  elif cond == 'm':
    return '#bcbddc'
  elif cond == 'l':
    return '#756bb1'

def index_from_condition(cond):
  if cond == 's':
    return 2
  elif cond == 'p':
    return 0
  elif cond == 'p2':
    return 1
  elif cond == 'f':
    return 3
  elif cond == 't':
    return 4
  elif cond == 'm':
    return 5
  elif cond == 'l':
    return 6


def get_cached_clicks(fname):
  git_directory = os.path.dirname(__file__)
  fname_full = os.path.join(git_directory,'processed_clicks',f'{fname[:-4]}' + '_savedclicks.csv')
  if os.path.exists(fname_full):
    clickpd = pd.read_csv(fname_full)
    indices = clickpd['indices'].tolist()
    
  else:
    print('clicks not found for %s' % (sname))
    indices = None
  return indices

def get_cached_mainsequence(fname):
  current_module_directory = os.path.dirname(__file__)
  fname_full = os.path.join(current_module_directory,'processed_clicks',f'{fname[:-4]}' + '_mainsequence.mat')
  if os.path.exists(fname_full):
    ms = scipy.io.loadmat(fname_full)
    return ()
  else:
    print('clicks not found for %s' % (sname))
    return (None, None, None, None)
# %%

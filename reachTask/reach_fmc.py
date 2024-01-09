
import pandas as pd
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import freemocapAnalysis as fm
#%%
# this is a class for the reach data. 
# i find it really difficult to keep typing fmc['right_shoulder_x'] and so on.
# we also typically want to work with the data in numpy arrays, not pandas dataframes.
# and have the xyz data in a single array, not three separate arrays.
class reachData:
  time = []
  sho = []
  elb = []
  wri = []
  tanvelsho = []
  tanvelelb = []
  tanvelwri = []
  velsho = []
  velelb = []
  velwri = []

  mov_starts = []
  mov_ends = []

# constructor for reach data, receiving a pandas dataframe
  def __init__(self, fmc:pd.DataFrame, sr_fixed = 30.0):
    # compute the time vector.
    ts = fmc['timestamp']
    ts0 = ts - ts[0]
    #now ts0 is in nanoseconds, conver to seconds
    ts_sec = ts0 / 1000000000
    #plt.plot(ts_sec)
    time_temp = np.array(ts_sec)
    
    # get the x and y data for the shoulder, elbow, and wrist
    # make temp variables for sho elb wri
    shotemp = np.array([fmc['right_shoulder_x'], fmc['right_shoulder_y'], fmc['right_shoulder_z']])
    elbtemp = np.array([fmc['right_elbow_x'], fmc['right_elbow_y'], fmc['right_elbow_z']])
    writemp = np.array([fmc['right_wrist_x'], fmc['right_wrist_y'], fmc['right_wrist_z']])

    # resample the data                   
    time, sho = fm.resample_data(time_temp, shotemp, sr_fixed)
    time, elb = fm.resample_data(time_temp, elbtemp, sr_fixed)
    time, wri = fm.resample_data(time_temp, writemp, sr_fixed)

    self.time = time
    self.sho = sho
    self.elb = elb
    self.wri = wri
    
    self.wri_f = fm.lowpass_cols(self.wri)
    self.sho_f = fm.lowpass_cols(self.sho)
    self.elb_f = fm.lowpass_cols(self.elb)

    # get the time data
    self.wri_ddt = fm.vel3D(self.time, self.wri_f)
    self.elb_ddt = fm.vel3D(self.time, self.elb_f)
    self.sho_ddt = fm.vel3D(self.time, self.sho_f)
      
  def click_add_starts_ends(self, the_time, the_field, k=2):
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(the_time, the_field[0, :])
    ax[1].plot(the_time, the_field[1, :])
    ax[2].plot(the_time, the_field[2, :])

    clicks = []
    def onclick(event):
      if len(clicks) < k:
        clicks.append((event.xdata, event.ydata))

      if len(clicks) == k:
        fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    
    # Extract x-values from clicks and round them
    x_values = [round(click[0]) for click in clicks]
    
    # Find the closest time value in reachdat.time for each click
    indices = [np.abs(the_time - x).argmin() for x in x_values]

    # close the plot with plt.close(fig)
    plt.close(fig) 

    self.mov_starts = indices[::2]
    self.mov_ends = indices[1::2]
    return indices[::2], indices[1::2]

# %%


import pandas as pd
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import freemocapAnalysis as fm
import pandas as pd
import os
from scipy.signal import find_peaks
#%%
def peaks_and_valleys(tv_sub):
  tv_thresh_mms = 250 # mm/s

  # for i in range(len(reachr.mov_starts)):
  tv_sub_f = fm.lowpass(tv_sub,fs=30,cutoff_freq= 4)

  # Find peaks above tv_thresh
  ind_peaks, _ = find_peaks(tv_sub_f, height=tv_thresh_mms)
  # Loop between each pair of peaks and find the minima between each
  ind_valleys = []
  for i in range(len(ind_peaks)-1):
    start_index = ind_peaks[i]
    end_index = ind_peaks[i+1]
    minima_index = np.argmin(tv_sub_f[start_index:end_index]) + start_index
    ind_valleys.append(minima_index)

  ind_valleys = np.array(ind_valleys)

  plt.plot(tv_sub)
  plt.plot(ind_peaks, tv_sub[ind_peaks], "x")
  plt.plot(ind_valleys, tv_sub[ind_valleys], "o")
  plt.show()
  # 
  return ind_peaks, ind_valleys


#%%
# this is a class for the reach data. 
# i find it really difficult to keep typing fmc['right_shoulder_x'] and so on.
# we also typically want to work with the data in numpy arrays, not pandas dataframes.
# and have the xyz data in a single array, not three separate arrays.
class reachData:
  time = []
  sho_r = []
  elb_r = []
  wri_r = []
  sho_f = []
  elb_f = []
  wri_f = []
  tanvel_sho = []
  tanvel_elb = []
  tanvel_wri = []
  vel_sho = []
  vel_elb = []
  vel_wri = []
  vel_sho_r = []
  vel_elb_r = []
  vel_wri_r = []

  mov_starts = []
  mov_ends = []

# constructor for reach data, receiving a pandas dataframe
  def __init__(self, fmc:pd.DataFrame, sr_fixed = 30.0, cutoff_freq = 12.0):
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

    self.time   = time
    self.sho_r  = sho
    self.elb_r  = elb
    self.wri_r  = wri
    
    self.wri_f = fm.lowpass_cols(self.wri_r, cutoff_freq = cutoff_freq)
    self.sho_f = fm.lowpass_cols(self.sho_r, cutoff_freq = cutoff_freq)
    self.elb_f = fm.lowpass_cols(self.elb_r, cutoff_freq = cutoff_freq)

    # get the time data
    self.vel_wri = fm.vel3D(self.time, self.wri_f)
    self.vel_elb = fm.vel3D(self.time, self.elb_f)
    self.vel_sho = fm.vel3D(self.time, self.sho_f)

    # raw velocity, which probably we won't use given fmc noise. 
    self.vel_wri_r = fm.vel3D(self.time, self.wri_f)
    self.vel_elb_r = fm.vel3D(self.time, self.elb_f)
    self.vel_sho_r = fm.vel3D(self.time, self.sho_f)

    # add tanvel wrist
    self.tanvel_wri = np.sqrt(self.vel_wri[0,:]**2 + self.vel_wri[1,:]**2 + self.vel_wri[2,:]**2)
    self.tanvel_elb = np.sqrt(self.vel_elb[0,:]**2 + self.vel_elb[1,:]**2 + self.vel_elb[2,:]**2)
    self.tanvel_sho = np.sqrt(self.vel_sho[0,:]**2 + self.vel_sho[1,:]**2 + self.vel_sho[2,:]**2)
  
  def mainsequence(self):
    distances = list()
    durations = list()
    peakspeeds = list()
    valleys    = list()
    for i in range(len(self.mov_starts)):
      tv    = self.tanvel_wri[self.mov_starts[i]:self.mov_ends[i]]
      wrist = self.wri_f[:,self.mov_starts[i]:self.mov_ends[i]]
      time = self.time[self.mov_starts[i]:self.mov_ends[i]]

      ind_peaks, ind_valleys = peaks_and_valleys(tv)
      print(ind_peaks, ind_valleys)

      mid_reach_wrist = wrist[:,ind_valleys[0]:ind_valleys[1]]
      dist_wrist = np.sqrt(np.sum((self.wri_f[:,ind_valleys[0]] - self.wri_f[:,ind_valleys[1]])**2))
      distances.append(dist_wrist)
      peakspeeds.append(max(tv[ind_valleys[0]:ind_valleys[1]]))
      durations.append(time[ind_valleys[1]] - time[ind_valleys[0]])
      #append to valleys the ind_valleys, corrected for the start of the reach
      valleys.append(ind_valleys + self.mov_starts[i])

    return np.array(distances), np.array(durations), np.array(peakspeeds),valleys

  def click_add_wrist_starts_ends(self, numclicks=-1, sname=None):
    
    indices = []
    if sname is not None:
      filename = f'{sname}_savedclicks.csv'
      #fnamefull = os.path.join('..\\processed_clicks',filename) # this is to allow it to find the right path to the outside directory
      # Path to the current file
      current_file_path = __file__
      # Directory of the current module
      current_module_directory = os.path.dirname(current_file_path)
      
      fnamefull = os.path.join(current_module_directory,'processed_clicks',filename) 
      print(fnamefull)
      if os.path.exists(fnamefull):
        # Load the file
        clickpd = pd.read_csv(fnamefull)
        indices = clickpd['indices'].tolist()

    if len(indices)==0:
      fig, ax = plt.subplots(4, 1)
      ax[0].plot(self.time, self.vel_wri[0, :])
      ax[0].set_ylabel('v (mm/s)')
      ax[1].plot(self.time, self.vel_wri[1, :])
      ax[1].set_ylabel('v (mm/s)')
      ax[2].plot(self.time, self.vel_wri[2, :])
      ax[2].set_ylabel('v (mm/s)')
      ax[3].plot(self.time, self.tanvel_wri)
      ax[3].set_xlabel('time (s)')
      ax[3].set_ylabel('v (mm/s)')


      clicks = []
      def onclick(event):
        if len(clicks) < numclicks:
          clicks.append((event.xdata, event.ydata))

        if numclicks > 0:
          if len(clicks) == numclicks:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

      cid = fig.canvas.mpl_connect('button_press_event', onclick)

      plt.show()
      
      # Extract x-values from clicks and round them
      x_values = [click[0] for click in clicks]
      
      # Find the closest time value in reachdat.time for each click;
      # these are the indices in the time array.
      indices = [np.abs(self.time - x).argmin() for x in x_values]

      # Save the indices to a csv file
      if numclicks > 0:
        if len(clicks) == numclicks:
          df = pd.DataFrame(indices, columns=['indices'])
          fsave = os.path.join('processed_clicks',f'{sname}_savedclicks.csv')
          df.to_csv(fsave, index=False)
        else:
          print("Not saving indices, because you didn't click enough times.")
      else:
        df = pd.DataFrame(indices, columns=['indices'])
        fsave = os.path.join('processed_clicks',f'{sname}_savedclicks.csv')
        df.to_csv(fsave, index=False)
            
    self.mov_starts = indices[::2]
    self.mov_ends = indices[1::2]
    return indices[::2], indices[1::2]

  

  
# %%


import pandas as pd
import numpy as np
from scipy.signal import resample,find_peaks
import scipy.io
import matplotlib.pyplot as plt
import freemocapAnalysis as fm
import pandas as pd
import os
import statsmodels as sm
import statsmodels.formula.api as smf

class mainSeq():
  D = []
  V = []
  T = []

  def __init__(self, D=[], T=[],V=[]):
    self.D = D
    self.V = V
    self.T = T

def fit_ct(ms, kv = 0.902,kt = 2.264, normV = .3, normT =1.0, verbose = True):
  """
  Returns a CT that fits peakspeed V and duration T for distance D simultaneously (D, V, and T [i.e. reach_fmc.mainsequence() output). 
  
  Inputs:
  ms -> mainsequence object.
  D: np.array
    the distances of the reaches
  V: np.array
    the peak speeds of the reaches
  T: np.array
    the duration of the reaches

  Parameters:
  kv: float, default 0.902
    coefficient in the speed equation:
    V = kv*(D .^3 * ct)^(1/4)
  kt: float, default 2.264
    coefficient in the time equation:
    T = kt*(D ./  ct)^(1/4)
  normV: float, default 0.3
    normalization factor for V
  normT: float, default 1.0
    normalization factor for T
  
  Returns:
  c_t: float
    the estimate of the cost of time
  pvalue: float
    the p-value of the estimate
  results: statsmodels object
  f_v: the speed function
  f_t: the duration function

  Explained math:
  We think people minimize energy and time.
  Energy for reaching is Force-rate (i.e. of units M*D/T^3)
  so the cost function that people minimize is 
  J(T) = c * M*D/T^3 + c_t*T (for integrated f-rate across time) 
  then
  dJdT = c*M*D/T^4 + c_t = 0 
  rearranging to solve for V = D/T or T, we have two propotionalities:
  V = kv* CT^1/4 * D ^(3/4)
  T = kt*(D / CT).^(1/4)

  We would like to solve using ordinary least-squares, i.e A*CT = b.
  To do this linearly, and scaling V and T errors with normV and normT, 
  we rearrange: 
  Aspd = (1/normV)^4 * kv^4*D.^3 
  bspd = (V ./ normV).^4 
  Adur = (kt^4 * D) ./normT 
  bdur = 1./(T ./ normT) 
  Then stack [Aspd;Adur] and [bspd;bdur] and solve.

  Example:
  D = np.array([0.1,0.2, 0.3,0.4, 0.5 ,0.60,0.7])
  T = np.array([0.7,0.75,0.8,0.85,0.90,0.95,1.0])
  V = np.array([0.3,0.50,0.7,0.90,1.1 ,1.30,1.5])
  fit_ct(D,V,T)
  """
  D = ms.D
  V = ms.V
  T = ms.T

  Adur = (T / normT) **4
  bdur = (kt / normT) **4 * D

  Aspd = (1/normV)**4 * kv**4 * D**3
  bspd = (V / normV)**4

  A = np.concatenate((Adur,Aspd))
  b = np.concatenate((bdur,bspd))

  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  c_t     = results.params.A #our estimate of the cost of time. 
  pvalue  = results.pvalues.A

  # V = kv* CT^1/4 * D ^(3/4)
  # T = kt*(D / CT).^(1/4)
  def f_v(D,ct): return kv*c_t**(1/4)*D**(3/4)
  def f_t(D,ct): return kt*(1/c_t)**(1/4) * D**(1/4) 

  if verbose:
    print(results.summary())

  return c_t, pvalue, results, f_v, f_t

def peaks_and_valleys(tv_sub,tv_thresh_mms=80, domanual=False):
  """
  Find the peaks and valleys in the speed (i.e. tanvel) data.
  
  Each reach-to-grasp is three reaches, each of which have peak speeds.
  The middle reach is the one we are most interested in because it positions the object.
  The peaks and valleys are used to define the start and end of the middle reach:
  the two valleys define the start and end of the middle reach.
  
  Inputs:
  tv_sub: np.array
    the speed data, in mm/s (as per freemocap, units of mm/s)
  tv_thresh_mms: float, default 80
    the threshold for the speed data in mmps

  Returns:
  ind_peaks: np.array
    the indices of the peaks
  ind_valleys: np.array
    the indices of the valleys

  """

  ind_peaks   = []
  ind_valleys = [] # valleys define the start and end of the middle movement.

  if tv_sub.shape[0] > 15: # then it cannot be filtered at 2 Hz.
    # for i in range(len(self.mov_starts)):
    tv_sub_f = fm.lowpass(tv_sub,fs=30,cutoff_freq= 2)

    # Find peaks above tv_thresh
    ind_peaks, _ = find_peaks(tv_sub_f, height=tv_thresh_mms,distance = 10)
    # Loop between each pair of peaks and find the minima between each
    
    for i in range(len(ind_peaks)-1):
      start_index = ind_peaks[i]
      end_index = ind_peaks[i+1]
      minima_index = np.argmin(tv_sub_f[start_index:end_index]) + start_index
      ind_valleys.append(minima_index)

    ind_valleys = np.array(ind_valleys)
  else:
    tv_sub_f = tv_sub
    print("Warning: not enough data to filter.")
    print("Likely this is a double-click by accident. delete the processedclicks.csv file and try again.")

  # currently this does not run. but we could insert a manual flag that checks for each. 
  if domanual:
    plt.plot(tv_sub)
    # plot with dashed line tv_sub_f
    plt.plot(tv_sub_f, '--', label='lowpass')
    plt.plot(ind_peaks, tv_sub[ind_peaks], "x")
    plt.plot(ind_valleys, tv_sub[ind_valleys], "o")
    plt.show()
    print("Enter 'm' to manually score, anything else to continue.")
    answer = input()
    if answer == 'm':
      domanual = True
  
  # if we do not have the right number of peaks/valleys, switch to manual.
  if (len(ind_peaks) + len(ind_valleys) != 5) or domanual:
    print("Warning: not enough peaks and valleys found.")
    print("switching to manual.")
    coordinates = []
    # while len(coordinates) is not equal! to 5, not smaller or not larger.
    while (len(coordinates) + len(coordinates) != 5): 
      print("switching to manual. Click 5 peaks/valleys in sequence, then close the figure.")
      f,ax = plt.subplots()
      plt.plot(tv_sub)
      plt.plot(tv_sub_f)
      
      # Function to capture mouse clicks
      def onclick(event):
        coordinates.append((event.xdata))

      # Connect the click event to the function
      cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
      plt.show()
      ind_peaks = [coordinates[0],coordinates[2],[coordinates[4]]]
      ind_peaks = [int(np.round(ind_peaks[i])) for i in range(3)]
      ind_valleys = [coordinates[1],coordinates[3]]
      ind_valleys = [int(np.round(ind_valleys[i])) for i in range(2)]
      ind_peaks = np.array(ind_peaks)
      ind_valleys = np.array(ind_valleys)
    
    # print that we manually scored correctly
    print("Five (peaks + valleys) manually scored.")
    domanual = False

  plt.plot(tv_sub)
  # plot with dashed line tv_sub_f
  plt.plot(tv_sub_f, '--', label='lowpass')
  plt.plot(ind_peaks, tv_sub[ind_peaks], "x")
  plt.plot(ind_valleys, tv_sub[ind_valleys], "o")
  plt.show()
  # 
  return ind_peaks, ind_valleys

def get_list_x_clicks_on_plot(data3d):
  f,ax = plt.subplots()
  plt.plot(data3d.T)
  coordinates = []
  # Function to capture mouse clicks
  def onclick(event):
    coordinates.append((event.xdata))

  # Connect the click event to the function
  cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
  plt.show()

  return coordinates

def get_rotmat(data3d):
  
  ind_clicks = get_list_x_clicks_on_plot(data3d)
  ind_clicks = [int(np.round(ind_clicks[i])) for i in range(3)]
  print(ind_clicks)
  pt1 = data3d[:,ind_clicks[0]]
  pt2 = data3d[:,ind_clicks[1]]
  pt3 = data3d[:,ind_clicks[2]]
  
  # define which order. assume standard:
  v1 = pt3 - pt2 # positive right
  v2 = pt2 - pt1 # positive forward
  # normalize v1 and v2 to be unit length
  v1 = v1/np.linalg.norm(v1)
  v2 = v2/np.linalg.norm(v2)
  # remove any component of v2 that is in the direction of v1
  v2 = v2 - np.dot(v2,v1)*v1
  v2 = v2/np.linalg.norm(v2)
  # now take the cross product
  v3 = np.cross(v1,v2)

  # compute the rotation matrix to go from the data frame to that defined by v1,v2,v3
  R = np.array([v1,v2,v3])
  return R
#%%
# this is a class for the reach data. 
# i find it really difficult to keep typing fmc['right_shoulder_x'] and so on.
# we also typically want to work with the data in numpy arrays, not pandas dataframes.
# and have the xyz data in a single array, not three separate arrays.
class reachData:
  fraw_name = ''
  path  = ''
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
  R         = []

  mov_starts = []
  mov_ends = []

# constructor for reach data, receiving a pandas dataframe
  def __init__(self, fmc:pd.DataFrame, path, sr_fixed = 30.0, cutoff_freq = 12.0):
    self.path = path
    self.fraw_name = os.path.basename(path)

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
    valleys    = list() # note the valleys are the middle movement, the one we sort of want most.

    # check for cached file _mainsequence.mat
    # if it exists, load it and return the data
    # if it doesn't exist, compute the data and save it to a file
    module_directory = os.path.dirname(__file__)
    fsave = os.path.join(module_directory,'processed_clicks',f'{self.fraw_name[:-4]}_mainsequence.mat')
    if os.path.exists(fsave):
      dat = scipy.io.loadmat(fsave)
      return dat['distances'], dat['durations'], dat['peakspeeds'], dat['valleys']
    
    else:
      print(f"File {fsave} not found. Computing main sequence data.")
      for i in range(len(self.mov_starts)):
        tv    = self.tanvel_wri[self.mov_starts[i]:self.mov_ends[i]]
        wrist = self.wri_f[:,self.mov_starts[i]:self.mov_ends[i]]
        time = self.time[self.mov_starts[i]:self.mov_ends[i]]

        ind_peaks, ind_valleys = peaks_and_valleys(tv)
        print(ind_peaks, ind_valleys)

        mid_reach_wrist = wrist[:,ind_valleys[0]:ind_valleys[1]]
        dist_wrist = np.sqrt(np.sum((mid_reach_wrist[:,0]-mid_reach_wrist[:,-1])**2))
        distances.append(dist_wrist)
        peakspeeds.append(max(tv[ind_valleys[0]:ind_valleys[1]]))
        durations.append(time[ind_valleys[1]] - time[ind_valleys[0]])
        #append to valleys the ind_valleys, corrected for the start of the reach
        valleys.append(ind_valleys + self.mov_starts[i])
        # save the data to a file
      scipy.io.savemat(fsave,{'distances':distances,'durations':durations,'peakspeeds':peakspeeds,'valleys':valleys})
      return np.array(distances), np.array(durations), np.array(peakspeeds),valleys

  def cut_middle_movements(self,inds_middle_start_end):
    # make a list of the reaches defined by ind_moves, which is really a list pair of indices
    cutreaches = list()
    for imov in range(len(inds_middle_start_end)):
      inds = np.arange(inds_middle_start_end[imov][0],inds_middle_start_end[imov][1])
      tzeroed = self.time[inds] - self.time[inds[0]]
      cutreaches.append((tzeroed,np.array(self.wri_f[:,inds])))
    return cutreaches

  def click_add_wrist_starts_ends(self, numclicks=-1, do_skip_figs = False):
    
    sname = self.fraw_name
    indices = []
    
    ### try to load an existing clicks file first.
    # Path to the current file
    current_module_path = __file__
    # Directory of the current module
    current_module_directory = os.path.dirname(current_module_path)
    fnamefull = os.path.join(current_module_directory,'processed_clicks',f'{self.fraw_name[:-4]}_savedclicks.csv') 
    print(fnamefull)
    if os.path.exists(fnamefull):
      # Load the file
      clickpd = pd.read_csv(fnamefull)
      indices = clickpd['indices'].tolist()
      if do_skip_figs == True:
        print("Skipping clicks, using saved clicks.")
        self.mov_starts = indices[::2]
        self.mov_ends = indices[1::2]
        return indices[::2], indices[1::2]

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
      # set ylim to be 1500
      ax[3].set_ylim([0, 1500])

      clicks = []
      def onclick(event):
        clicks.append((event.xdata, event.ydata))

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
        if len(clicks) > 0:
          df = pd.DataFrame(indices, columns=['indices'])
          module_directory = os.path.dirname(__file__)
          fsave = os.path.join(module_directory,'processed_clicks',f'{sname[:-4]}_savedclicks.csv')
          print(fsave)
          # check if fsave already exists. if it does, ask
          if os.path.exists(fsave):
            print(f"{fsave} already exists. Do you want to overwrite it?")
            print("Enter 'y' to overwrite, anything else to not overwrite.")
            answer = input()
            if answer == 'y':
              df.to_csv(fsave, index=False)
          else:
            df.to_csv(fsave, index=False)
        else:
          print("Not saving indices, because you did not click.")
            
    self.mov_starts = indices[::2]
    self.mov_ends = indices[1::2]
    return indices[::2], indices[1::2]

  

  
# %%

#%%
import sys, os
sys.path.append(os.path.join(os.getcwd()))
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
from scipy.signal import find_peaks
# %matplotlib widget
# then you can do importlib.reload(rf) to reload the module

## here are hopefully the only two things you need to set.
sname = 'ro_0125_Y' # at the moment, 'je' and 'ro' are the only clean datasets that i can tell. and ro is short. 
datapath = fa.setdatapath("jer") 
## /here are hopefully the only two things you need to set.

fnames = fa.get_list_subject_files(sname,datapath)
pddata = pd.read_csv(fnames[0])
reachr = rf.reachData(pddata)

# click the starts and ends
reachr.click_add_wrist_starts_ends(numclicks=24,sname=sname)

# plot the return values
plt.plot(reachr.time,reachr.tanvel_wri)
plt.plot(reachr.time[reachr.mov_starts], reachr.tanvel_wri[reachr.mov_starts], 'ro')
plt.plot(reachr.time[reachr.mov_ends], reachr.tanvel_wri[reachr.mov_ends], 'go')
plt.xlabel('Time')
plt.ylabel('tanvel_wri')
plt.legend(['tanvel_wri', 'Movement Starts', 'Movement Ends'])
plt.show()

# call reach_fmc.peaks_and_valleys to find the peaks and valleys of the mov_starts and mov_ends
distances,durations,peakspeeds, ind_mmoves = reachr.mainsequence()

#% plot the dist peakspeed
fig,ax = plt.subplots(2,1)
#set plot size
fig.set_size_inches(2,2)
ax[0].plot(distances/1000.0,durations,'o')
ax[0].set_xlabel('Distance (m)')
ax[0].set_ylabel('Duration (s)')
#set xlimit
ax[0].set_xlim([0,.5])
ax[0].set_ylim([0,1.0])
ax[1].plot(distances/1000,peakspeeds/1000,'o')
ax[1].set_xlabel('Distance (m)')
ax[1].set_ylabel('Peak Speed (m/s)')
ax[0].set_xlim([0,0.5])
ax[0].set_ylim([0,1.5])
#%
#%%
# plot wri_f[0,mov_starts[i]:mov_ends[i]] for all mov_starts
fig = plt.figure()
fig.set_size_inches(2,2)
tv_sm = fa.lowpass(reachr.tanvel_wri, fs=30, cutoff_freq = 8)
tgts = list()
ax3d = fig.add_subplot(221,projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
for i in range(len(reachr.mov_starts)):
  
  ind_m = ind_mmoves[i]
  inds = range(ind_m[0],ind_m[1])
  tgt_start = reachr.wri_f[:,inds[0]]
  tgt_end = reachr.wri_f[:,inds[-1]]
  ax3d.plot(reachr.wri_f[0,inds], reachr.wri_f[1,inds], reachr.wri_f[2,inds])
  ax3d.plot(tgt_start[0], tgt_start[1], tgt_start[2], 'ro')
  ax3d.plot(tgt_end[0], tgt_end[1], tgt_end[2], 'go')
  
  t = reachr.time[inds]
  t = t-t[0]
  
  ax2.plot(t,tv_sm[inds])
  
  ax3.plot(reachr.wri_f[0, inds])
  

plt.xlabel('Time')
plt.ylabel('wri_f')
plt.title(f'Movement {i+1}')
plt.show()
# %% sanity
fig, ax = plt.subplots(4, 1)
ax[0].plot(reachr.time, reachr.vel_wri[0, :])
ax[0].set_ylabel('v (mm/s)')
ax[1].plot(reachr.time, reachr.vel_wri[1, :])
ax[1].set_ylabel('v (mm/s)')
ax[2].plot(reachr.time, reachr.vel_wri[2, :])
ax[2].set_ylabel('v (mm/s)')
plt.show()
      

# %%

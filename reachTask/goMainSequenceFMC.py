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
datapath = fa.setdatapath("rom") 
print(datapath)

## /here are hopefully the only two things you need to set.

fnames = fa.get_list_subject_files(sname,datapath)
pddata = pd.read_csv(fnames[0])
reachr = rf.reachData(pddata, datapath)

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
ax[1].set_xlim([0,0.5])
ax[1].set_ylim([0,1.5])
#%
#%%
# plot wri_f[0,mov_starts[i]:mov_ends[i]] for all mov_starts
fig = plt.figure()
fig.set_size_inches(2,2)
tv_sm = fa.lowpass(reachr.tanvel_wri, fs=30, cutoff_freq = 8)
tgts = list()
ax_3d    = fig.add_subplot(221,projection='3d')
ax_3dr  = fig.add_subplot(222,projection='3d')
ax_tv   = fig.add_subplot(223) 

for i in range(len(reachr.mov_starts)):
  
  ind_m = ind_mmoves[i]
  inds = range(ind_m[0],ind_m[1])
  ax_3d.plot(reachr.wri_f[0,inds], reachr.wri_f[1,inds], reachr.wri_f[2,inds])
  
  tgt_start = reachr.wri_f[:,inds[0]]
  tgt_end = reachr.wri_f[:,inds[-1]]
  ax_3d.plot(tgt_start[0], tgt_start[1], tgt_start[2], 'ro')
  ax_3d.plot(tgt_end[0], tgt_end[1], tgt_end[2], 'go')
  
  R2calxy = np.array([[-0.689578  , -0.72413109, -0.01078618],
       [ 0.66407996, -0.62631008, -0.40833013],
       [ 0.28892905, -0.28873836,  0.9127706 ]])
  
  # zero the movements to the first shoulder position
  sho0 = reachr.sho_f[:,reachr.mov_starts[0:1]]
  wri_f = reachr.wri_f[:,inds]
  wri_f = wri_f - sho0
  # now rotate the vectors
  wri_r = np.dot(R2calxy.T,wri_f)
  ax_3dr.plot(wri_r[0,:], wri_r[1,:], wri_r[2,:])
  tgt_start = wri_r[:,0]
  tgt_end = wri_r[:,-1]
  ax_3dr.plot(tgt_start[0], tgt_start[1], tgt_start[2], 'ro')
  ax_3dr.plot(tgt_end[0], tgt_end[1], tgt_end[2], 'go')
  ax_3dr.set_xlabel('x (r+)')
  ax_3dr.set_ylabel('y (f+)')
  ax_3dr.set_zlabel('z (u+)')
  ax_3dr.set_xlim([-200,800])
  ax_3dr.set_ylim([-200,800])
  ax_3dr.set_zlim([-500,500])

  t = reachr.time[inds]
  t = t-t[0]
  
  ax_tv.plot(t,tv_sm[inds])
  
plt.xlabel('Time')
plt.ylabel('wri_f')
plt.show()
# %%

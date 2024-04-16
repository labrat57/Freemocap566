#%%
# python code: 
import sys, os
sys.path.append(os.path.join(os.getcwd()))
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import scipy.io
from scipy.signal import find_peaks

# if doing raw scoring, this needs to be set to 'rom' or 'jer'
datapath  = fa.setdatapath("jer") 

# file names:
subjname_rot     = 'paper02_heel'
subjname_trials  = 'paper02'

#### steps:
###1. get the rotation matrix from the heel-on-floor trial (either pre-loaded or from clicks). 
  # save matrix R in processed clicks as processed_clicks/fname_rot.mat. we could use this to rescale.
###2. get the reach data from the trials 2-7, scoring starts/ends of the triple movement. 
  # save as processed_clicks/f'{fname[:-4]}_processed_clicks.csv'. 
###3. plot results.
###4. get main sequence data.
  # save middle movement scoring as processed_clicks/f'{fname[:-4]}_mainsequence.mat'. 
###5. plot main sequence data.
### after this file has been used to run through every subject, 
# each of the trials will have a 'f{filename}_mainsequence.mat' file that we can use
# to merge the subjects together. 

#%% step 1. get rotation matrix for heel-trial file.

fnames    = fa.get_list_subject_files(subjname_rot, datapath)
pddata    = pd.read_csv(fnames[0]) # this list fnames has only 1 heel file for rotation.

lhip = np.array([pddata["left_hip_x"],pddata["left_hip_y"],pddata["left_hip_z"]])
rhip = np.array([pddata["right_hip_x"],pddata["right_hip_y"],pddata["right_hip_z"]])

# take the mean of the two hip detections, each is a 3, array
mhip = np.mean([lhip,rhip],axis=0)
# check to see if we have computed a rotation matrix.
curfile_directory = os.path.dirname(__file__)
fname_full = os.path.join(curfile_directory,'processed_clicks',subjname_rot + '.mat')
if os.path.exists(fname_full):
  dat = scipy.io.loadmat(fname_full)
  R = dat["R"]
else:
  R = rf.get_rotmat(mhip)
  # save R
  scipy.io.savemat(fname_full,{'R':R})

### print the rotation matrix
print(R)
# how plot the rotated left and right hip
lhip_rot = np.dot(R,lhip)
rhip_rot = np.dot(R,rhip)
f,ax = plt.subplots()
ax.plot(lhip_rot.T)
ax.plot(rhip_rot.T)
plt.show()

# %% Step 2 and 3: Click the start and end of each triple movement, and plot.
fnames_triallist = fa.get_list_subject_files(subjname_trials,datapath)
for i in range(len(fnames_triallist)):
  pathname_cur = fnames_triallist[i]
  pddata_cur = pd.read_csv(pathname_cur)
  reachr = rf.reachData(pddata_cur,pathname_cur)

  # assign Rotation matrix R to the current data.
  reachr.R = R
  reachr.click_add_wrist_starts_ends(os.path.basename(pathname_cur))

  f,ax = plt.subplots()

  #reachr now has 'mov_starts' and 'mov_ends' defined. let's plot to see them.
  ax.plot(reachr.time,reachr.tanvel_wri)
  ax.plot(reachr.time[reachr.mov_starts], reachr.tanvel_wri[reachr.mov_starts], 'go')
  ax.plot(reachr.time[reachr.mov_ends], reachr.tanvel_wri[reachr.mov_ends], 'rs')
  ax.set_xlabel('Time')
  ax.set_ylabel('speed wri (mm/s)')
  ax.set_ylim([0,1500])
  ax.legend(['tanvel_wri', 'Movement Starts', 'Movement Ends'])
  ax.set_title(reachr.fraw_name)
  f.show()
  # pause for input
  input('Press Enter to continue')

#%% step 4: mainsequence analysis.
# note: mainsequence comes from 'mainsequence' Bahill, ..., Stark 1975.
# score the middle movements using peaks_and_valleys, or manual if it's the wrong number.
for i in range(len(fnames_triallist)):
  pathname_cur = fnames_triallist[i]
  print(f'Main Sequence (speed, duration): iteration {i}; filename {pathname_cur}.')
  pddata_cur = pd.read_csv(pathname_cur)
  reachr = rf.reachData(pddata_cur, path=pathname_cur)

  # get just the file name from the path
  fname_cur = os.path.basename(pathname_cur)

  # assign Roation matrix R to the current data.
  reachr.R = R
  # read in the saved starts/ends.
  reachr.click_add_wrist_starts_ends()

  distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = reachr.mainsequence()

  # each element in this list is a tuple! time, then wrist.
  tpllist_time_wrist = reachr.cut_middle_movements(indlist_middle_mvmnt_start_end)

  
  #%% Step 5: plot mainsequence results.
  # plot in m/s the distance, duration, and speed. 
  #% plot the dist peakspeed
   #%
  #%% Plot 
  # 1. tangential velocity 
  # 2. hand position in 3D
  # 3. rotated hand position in 3D
  fig = plt.figure()
  fig.set_size_inches(7,7)

  tgts = list()
  ax_3dr  = fig.add_subplot(221,projection='3d')
  ax_tv   = fig.add_subplot(222) 
  ax_v    = fig.add_subplot(223) 
  ax_t    = fig.add_subplot(224) 

  for i in range(len(tpllist_time_wrist)):
    ind01      = indlist_middle_mvmnt_start_end[i] # the indices of the middle movement. can use to get sho/finger any other column.
    inds      = range(ind01[0],ind01[1])
    t         = tpllist_time_wrist[i][0]
    movwri    = tpllist_time_wrist[i][1]
    
    tgt_start = movwri[:,0]
    tgt_end = movwri[:,-1]
        
    # zero the movements to the first shoulder position of the first movement.
    sho0 = reachr.sho_f[:,reachr.mov_starts[0:1]]
    wri_f_fromsho = movwri - sho0
    # now rotate the vectors
    R2calxy = R
    wri_r = np.dot(R2calxy, wri_f_fromsho)
    ax_3dr.plot(wri_r[0,:], wri_r[1,:], wri_r[2,:])
    tgt_start = wri_r[:,0]
    tgt_end = wri_r[:,-1]
    ax_3dr.plot(tgt_start[0], tgt_start[1], tgt_start[2], 'ro')
    ax_3dr.plot(tgt_end[0], tgt_end[1], tgt_end[2], 'go')
    ax_3dr.set_xlabel('x (m)')
    ax_3dr.set_ylabel('y (m)')
    ax_3dr.set_zlabel('z (m)')
    ax_3dr.set_xlim([-500,800])
    ax_3dr.set_ylim([-200,800])
    ax_3dr.set_zlim([-500,500])

    t = reachr.time[inds]
    t = t-t[0]
    
    tv = reachr.tanvel_wri
    tv_f = fa.lowpass(tv, cutoff_freq = 10)
    ax_tv.plot(t,tv_f[inds]/1000)
    # end loop through movements  
  ax_tv.set_xlabel('Time (s)',fontsize = 24)
  ax_tv.set_ylabel('Speed (m/s)',fontsize = 24)
  
  ax_t.plot(distancelist/1000.0,durationlist,'o')
  ax_t.set_xlabel('Distance (m)',fontsize = 24)
  ax_t.set_ylabel('Duration (s)',fontsize = 24)
  #set xlimit
  ax_t.set_xlim([0,.8])
  ax_t.set_ylim([0,1.5])
  ax_v.plot(distancelist/1000,peakspeedlist/1000,'o')
  ax_v.set_xlabel('Distance (m)',fontsize = 24)
  ax_v.set_ylabel('Peak Speed (m/s)',fontsize = 24)
  ax_v.set_xlim([0,0.8])
  ax_v.set_ylim([0,1.5])

for label in (ax_t.get_xticklabels() + ax_t.get_yticklabels()+ax_v.get_xticklabels()+ax_v.get_yticklabels()+ax_tv.get_xticklabels() + ax_tv.get_yticklabels()):
  label.set_fontsize(18)
  
  # shift the position of ax_3dr slightly left
  f.tight_layout()
  pos1 = ax_3dr.get_position() # get the original position
  pos2 = [pos1.x0 - 0.1, pos1.y0, pos1.width, pos1.height]
  
  
  # plt.show()
  fbase = os.path.basename(pathname_cur)
  fsave = os.path.join(curfile_directory,'figures',fbase[:-4] + '_summary.pdf')
  fig.savefig(fsave)